"""Evaluate an AO6-lightness / BL5-chroma analytical transplant."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.filmmatch_strict_interior_fresh_confirmation import (
    AO6_ARM,
    CANDIDATE_ARM as BL5_ARM,
)
from src.eval.fresh_native_standard_confirmation import boundary_metrics
from src.eval.global_frontier import sha256_file
from src.preprocess import save_srgb16_png
from src.roll2film.factorized_boundary_guard import (
    apply_target_residual_boundary_guard,
)


SCHEMA = (
    "neuro_film.u5_r2bl11_filmmatch_luma_preserving_"
    "chroma_transplant_report.v1"
)


class LumaPreservingChromaTransplantError(RuntimeError):
    """Raised when BL11 provenance or image contracts drift."""


@dataclass(frozen=True)
class LumaPreservingChromaResult:
    output: np.ndarray
    residual_scale: np.ndarray
    base_lab: np.ndarray
    donor_lab: np.ndarray
    output_lab: np.ndarray


def _load_exact_json(root: Path, path: str, expected: str) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise LumaPreservingChromaTransplantError(f"hash mismatch: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LumaPreservingChromaTransplantError(f"expected JSON object: {path}")
    return payload


def _decode_rgb16(path: Path, expected_sha256: str) -> np.ndarray:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise LumaPreservingChromaTransplantError("parent output identity drift")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        decoded is None
        or decoded.dtype != np.uint16
        or decoded.ndim != 3
        or decoded.shape[-1] != 3
    ):
        raise LumaPreservingChromaTransplantError("expected RGB16 PNG parent output")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float64) / 65535.0


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def compose_luma_preserving_chroma(
    base_encoded: np.ndarray,
    donor_encoded: np.ndarray,
    *,
    hard_boundary_epsilon_encoded_srgb: float,
    guard_boundary_epsilon_encoded_srgb: float,
) -> LumaPreservingChromaResult:
    """Preserve AO6 Lab lightness while requesting BL5 chroma.

    The Lab target may be outside sRGB. The final transform follows the exact
    target direction in linear RGB and analytically shortens it per pixel to
    remain inside source-inclusive rails; it never clips the target.
    """

    base = np.asarray(base_encoded, dtype=np.float64)
    donor = np.asarray(donor_encoded, dtype=np.float64)
    if (
        base.ndim != 3
        or base.shape[-1] != 3
        or donor.shape != base.shape
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(donor))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
        or np.any(donor < 0.0)
        or np.any(donor > 1.0)
    ):
        raise ValueError("base and donor must be matching finite encoded RGB")

    base_linear = encoded_srgb_to_linear(base)
    donor_linear = encoded_srgb_to_linear(donor)
    base_lab = linear_rgb_to_lab(
        np.asarray(base_linear, dtype=np.float32), working_space="linear_srgb"
    )
    donor_lab = linear_rgb_to_lab(
        np.asarray(donor_linear, dtype=np.float32), working_space="linear_srgb"
    )
    target_lab = donor_lab.copy()
    target_lab[..., 0] = base_lab[..., 0]
    target_linear = lab_to_linear_rgb(target_lab, working_space="linear_srgb")
    guarded = apply_target_residual_boundary_guard(
        base_linear,
        target_linear,
        strength=1.0,
        hard_boundary_epsilon_encoded_srgb=hard_boundary_epsilon_encoded_srgb,
        guard_boundary_epsilon_encoded_srgb=guard_boundary_epsilon_encoded_srgb,
    )
    output = linear_srgb_to_encoded(guarded.output)
    output_lab = linear_rgb_to_lab(
        np.asarray(guarded.output, dtype=np.float32), working_space="linear_srgb"
    )
    return LumaPreservingChromaResult(
        output=output,
        residual_scale=guarded.residual_scale,
        base_lab=base_lab,
        donor_lab=donor_lab,
        output_lab=output_lab,
    )


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = config["parent"]
    report = _load_exact_json(root, parent["bl8_report"], parent["bl8_report_sha256"])
    decision = _load_exact_json(
        root, parent["bl8_decision"], parent["bl8_decision_sha256"]
    )
    candidate = config["candidate"]
    if (
        config.get("schema")
        != "neuro_film.u5_r2bl11_filmmatch_luma_preserving_chroma_transplant.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("source_feature_routing_allowed")
        or config.get("product_integration_allowed")
        or candidate["base_arm"] != AO6_ARM
        or candidate["chroma_donor_arm"] != BL5_ARM
        or candidate["hard_clipping_allowed"]
        or candidate["free_strength_parameter"]
        or candidate["per_image_fit_allowed"]
        or report["stable_evidence_id"] != parent["bl8_stable_evidence_id"]
        or decision["decision"] != parent["required_bl8_decision"]
        or decision["report_stable_evidence_id"] != report["stable_evidence_id"]
        or decision["output_sha256"]
        != {
            source_id: {
                arm_id: row["output_sha256"]
                for (candidate_source, arm_id), row in {
                    (item["source_id"], item["arm_id"]): item
                    for item in report["rows"]
                }.items()
                if candidate_source == source_id
            }
            for source_id in sorted({item["source_id"] for item in report["rows"]})
        }
    ):
        raise LumaPreservingChromaTransplantError("BL11 frozen contract drift")
    return report, decision


def run_luma_preserving_chroma_transplant(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("BL11 output is create-only")
    parent_report, _ = validate_contract(root, config)
    parent_dir = root / config["parent"]["bl8_output_dir"]
    parent_rows = {
        (row["source_id"], row["arm_id"]): row for row in parent_report["rows"]
    }
    source_ids = list(dict.fromkeys(row["source_id"] for row in parent_report["rows"]))
    candidate = config["candidate"]
    rows: list[dict[str, Any]] = []
    limited_fractions: list[float] = []
    retentions: list[float] = []
    lightness_drifts: list[float] = []
    styles: list[float] = []
    output_dir.mkdir(parents=True)

    for source_id in source_ids:
        base_row = parent_rows[(source_id, AO6_ARM)]
        donor_row = parent_rows[(source_id, BL5_ARM)]
        base = _decode_rgb16(parent_dir / base_row["output"], base_row["output_sha256"])
        donor = _decode_rgb16(
            parent_dir / donor_row["output"], donor_row["output_sha256"]
        )
        composed = compose_luma_preserving_chroma(
            base,
            donor,
            hard_boundary_epsilon_encoded_srgb=float(
                candidate["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                candidate["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        requested_chroma = composed.donor_lab[..., 1:] - composed.base_lab[..., 1:]
        actual_chroma = composed.output_lab[..., 1:] - composed.base_lab[..., 1:]
        requested_energy = float(np.sqrt(np.mean(np.square(requested_chroma))))
        actual_energy = float(np.sqrt(np.mean(np.square(actual_chroma))))
        retention = actual_energy / requested_energy if requested_energy > 0.0 else 1.0
        limited = float(np.mean(composed.residual_scale < 1.0 - 1e-12))
        lightness_drift = float(
            np.quantile(
                np.abs(composed.output_lab[..., 0] - composed.base_lab[..., 0]),
                0.95,
            )
        )
        style = float(np.sqrt(np.mean(np.square(composed.output - base))))
        path = output_dir / "renders" / candidate["candidate_arm"] / f"{source_id}.png"
        save_srgb16_png(np.asarray(composed.output, dtype=np.float32), path)
        rows.append(
            {
                "source_id": source_id,
                "make": base_row["make"],
                "output": path.relative_to(output_dir).as_posix(),
                "output_sha256": sha256_file(path),
                "requested_chroma_rms_delta_lab": requested_energy,
                "actual_chroma_rms_delta_lab": actual_energy,
                "requested_chroma_energy_retention": retention,
                "safety_limited_pixel_fraction": limited,
                "minimum_residual_scale": float(np.min(composed.residual_scale)),
                "p95_absolute_lightness_drift_delta_lstar": lightness_drift,
                "style_rgb_rmse_from_ao6": style,
                **boundary_metrics(composed.output, base),
            }
        )
        limited_fractions.append(limited)
        retentions.append(retention)
        lightness_drifts.append(lightness_drift)
        styles.append(style)

    gate = config["automatic_gate"]
    aggregate = {
        "output_count": len(rows),
        "maximum_output_code_boundary_fraction": max(
            row["output_code_boundary_fraction"] for row in rows
        ),
        "maximum_new_boundary_fraction_vs_ao6": max(
            row["new_boundary_fraction_vs_ao6"] for row in rows
        ),
        "p95_safety_limited_pixel_fraction": float(
            np.quantile(limited_fractions, 0.95)
        ),
        "median_requested_chroma_energy_retention": float(np.median(retentions)),
        "median_image_p95_absolute_lightness_drift_delta_lstar": float(
            np.median(lightness_drifts)
        ),
        "median_style_rgb_rmse_from_ao6": float(np.median(styles)),
    }
    passed = bool(
        aggregate["output_count"] == gate["expected_outputs"]
        and aggregate["maximum_output_code_boundary_fraction"]
        <= gate["maximum_output_code_boundary_fraction"]
        and aggregate["maximum_new_boundary_fraction_vs_ao6"]
        <= gate["maximum_new_boundary_fraction_vs_ao6"]
        and aggregate["p95_safety_limited_pixel_fraction"]
        <= gate["maximum_p95_safety_limited_pixel_fraction"]
        and aggregate["median_requested_chroma_energy_retention"]
        >= gate["minimum_median_requested_chroma_energy_retention"]
        and aggregate["median_image_p95_absolute_lightness_drift_delta_lstar"]
        <= gate["maximum_median_image_p95_absolute_lightness_drift_delta_lstar"]
        and aggregate["median_style_rgb_rmse_from_ao6"]
        >= gate["minimum_median_style_rgb_rmse_from_ao6"]
    )
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "parent_stable_evidence_id": parent_report["stable_evidence_id"],
        "rows": rows,
        "aggregate": aggregate,
        "automatic_gate_pass": passed,
        "visual_review_opened": passed,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_blind_review_sheets(
    *,
    root: Path,
    parent_output_dir: Path,
    candidate_output_dir: Path,
    source_rows: Mapping[str, Mapping[str, Any]],
    source_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Build a deterministic one-round AO6/candidate blind visual audit."""

    import random

    if output_dir.exists():
        raise FileExistsError("BL11 blind review is create-only")
    output_dir.mkdir(parents=True)
    font = ImageFont.load_default()
    tiles: list[Image.Image] = []
    mapping: list[dict[str, str]] = []
    candidate_arm = "ao6_lightness_bl5_chroma_analytical_safe"
    for source_id in source_ids:
        order = [AO6_ARM, candidate_arm]
        random.Random(
            hashlib.sha256(f"u5-r2bl11:{source_id}".encode()).digest()
        ).shuffle(order)
        mapping.append({"source_id": source_id, "A": order[0], "B": order[1]})
        with Image.open(root / source_rows[source_id]["decoded_path"]) as opened:
            source = opened.convert("RGB")
            source.thumbnail((520, 330), Image.Resampling.LANCZOS)
        compared: list[Image.Image] = []
        for arm_id in order:
            base_dir = parent_output_dir if arm_id == AO6_ARM else candidate_output_dir
            if arm_id == AO6_ARM:
                path = base_dir / "renders" / AO6_ARM / f"{source_id}.png"
            else:
                path = base_dir / "renders" / candidate_arm / f"{source_id}.png"
            with Image.open(path) as opened:
                rendered = opened.convert("RGB")
                rendered.thumbnail((520, 330), Image.Resampling.LANCZOS)
            compared.append(rendered)
        tile = Image.new("RGB", (1580, 375), "white")
        draw = ImageDraw.Draw(tile)
        for index, (label, image) in enumerate(
            zip(("Source", "A", "B"), (source, *compared))
        ):
            x = 5 + index * 525
            draw.text((x, 3), label, fill="black", font=font)
            tile.paste(image, (x, 23))
        draw.text((5, 357), source_id, fill="black", font=font)
        tiles.append(tile)
    part_hashes: list[str] = []
    for part_index, start in enumerate(range(0, len(tiles), 4), start=1):
        selected = tiles[start : start + 4]
        sheet = Image.new("RGB", (1580, len(selected) * 375 + 28), "white")
        ImageDraw.Draw(sheet).text(
            (5, 5), f"U5.R2BL11 blind part {part_index}", fill="black", font=font
        )
        for index, tile in enumerate(selected):
            sheet.paste(tile, (0, 28 + index * 375))
        path = output_dir / f"blind_part_{part_index}.png"
        sheet.save(path, "PNG")
        part_hashes.append(sha256_file(path))
    mapping_path = output_dir / "mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "parts": part_hashes,
        "mapping_sha256": sha256_file(mapping_path),
    }


__all__ = [
    "LumaPreservingChromaResult",
    "compose_luma_preserving_chroma",
    "build_blind_review_sheets",
    "run_luma_preserving_chroma_transplant",
    "validate_contract",
]
