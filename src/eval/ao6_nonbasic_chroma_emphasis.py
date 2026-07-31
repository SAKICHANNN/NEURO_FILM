"""Emphasize the non-basic chroma component of the fixed AO6 look."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.srgb_transfer import encoded_srgb_to_linear, linear_srgb_to_encoded
from src.eval.filmmatch_identity_residual_ood import _encode_scene_linear
from src.eval.filmmatch_strict_interior_fresh_confirmation import (
    AO6_ARM,
    validate_contract as validate_bl8_contract,
)
from src.eval.fresh_native_standard_confirmation import boundary_metrics
from src.eval.global_frontier import sha256_file
from src.preprocess import save_srgb16_png
from src.preprocess.raw_decode import load_raw_working_image
from src.roll2film.baselines import fit_joint_basic_adjustment
from src.roll2film.factorized_boundary_guard import apply_target_residual_boundary_guard


SCHEMA = "neuro_film.u5_r2bl13_ao6_nonbasic_chroma_emphasis_report.v1"


class AO6ChromaEmphasisError(RuntimeError):
    """Raised when frozen evidence or candidate execution drifts."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _decode_rgb16(path: Path, expected_sha256: str) -> np.ndarray:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise AO6ChromaEmphasisError("AO6 output identity drift")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        decoded is None
        or decoded.dtype != np.uint16
        or decoded.ndim != 3
        or decoded.shape[-1] != 3
    ):
        raise AO6ChromaEmphasisError("expected AO6 RGB16 PNG")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float64) / 65535.0


def _aligned_sample(*arrays: np.ndarray, maximum: int) -> tuple[np.ndarray, ...]:
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise AO6ChromaEmphasisError("aligned sample shape mismatch")
    flat = [np.asarray(array).reshape(-1, 3) for array in arrays]
    count = min(maximum, flat[0].shape[0])
    if count < 256:
        raise AO6ChromaEmphasisError("insufficient pixels")
    indices = np.linspace(0, flat[0].shape[0] - 1, count, dtype=np.int64)
    return tuple(value[indices] for value in flat)


def compose_ao6_nonbasic_chroma_emphasis(
    source_encoded: np.ndarray,
    ao6_encoded: np.ndarray,
    *,
    fit_pixel_budget: int,
    chroma_residual_strength: float,
    hard_boundary_epsilon_encoded_srgb: float,
    guard_boundary_epsilon_encoded_srgb: float,
) -> dict[str, np.ndarray]:
    """Build one deterministic AO6 residual-emphasis candidate."""

    source = np.asarray(source_encoded, dtype=np.float64)
    ao6 = np.asarray(ao6_encoded, dtype=np.float64)
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or ao6.shape != source.shape
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(ao6))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(ao6 < 0.0)
        or np.any(ao6 > 1.0)
        or fit_pixel_budget < 256
        or not 0.0 < chroma_residual_strength <= 1.0
    ):
        raise ValueError("invalid AO6 chroma-emphasis inputs")

    sampled_source, sampled_ao6 = _aligned_sample(
        source, ao6, maximum=fit_pixel_budget
    )
    basic_operator = fit_joint_basic_adjustment(sampled_source, sampled_ao6)
    basic = np.clip(basic_operator.apply(source), 0.0, 1.0)

    ao6_linear = encoded_srgb_to_linear(ao6)
    basic_linear = encoded_srgb_to_linear(basic)
    ao6_lab = linear_rgb_to_lab(
        np.asarray(ao6_linear, dtype=np.float32), working_space="linear_srgb"
    ).astype(np.float64)
    basic_lab = linear_rgb_to_lab(
        np.asarray(basic_linear, dtype=np.float32), working_space="linear_srgb"
    ).astype(np.float64)
    target_lab = ao6_lab.copy()
    target_lab[..., 1:] += chroma_residual_strength * (
        ao6_lab[..., 1:] - basic_lab[..., 1:]
    )
    target_linear = lab_to_linear_rgb(
        np.asarray(target_lab, dtype=np.float32), working_space="linear_srgb"
    ).astype(np.float64)
    guarded = apply_target_residual_boundary_guard(
        np.asarray(ao6_linear, dtype=np.float64),
        target_linear,
        strength=1.0,
        hard_boundary_epsilon_encoded_srgb=hard_boundary_epsilon_encoded_srgb,
        guard_boundary_epsilon_encoded_srgb=guard_boundary_epsilon_encoded_srgb,
    )
    output = linear_srgb_to_encoded(guarded.output)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("AO6 chroma emphasis escaped encoded RGB")
    return {
        "output": np.asarray(output, dtype=np.float64),
        "basic": basic,
        "ao6_lab": ao6_lab,
        "target_linear": target_linear,
        "ao6_linear": np.asarray(ao6_linear, dtype=np.float64),
        "residual_scale": guarded.residual_scale,
    }


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema")
        != "neuro_film.u5_r2bl13_ao6_nonbasic_chroma_emphasis.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("training_allowed")
        or config.get("product_integration_allowed")
        or config.get("production_default_changed")
    ):
        raise AO6ChromaEmphasisError("BL13 frozen contract drift")
    parent = config["parent"]
    bl8_config_path = root / parent["bl8_contract"]
    if sha256_file(bl8_config_path) != parent["bl8_contract_sha256"]:
        raise AO6ChromaEmphasisError("BL8 contract identity drift")
    bl8_config = json.loads(bl8_config_path.read_text(encoding="utf-8"))
    validated = validate_bl8_contract(root, bl8_config)
    report_path = root / parent["bl8_report"]
    if sha256_file(report_path) != parent["bl8_report_sha256"]:
        raise AO6ChromaEmphasisError("BL8 report identity drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not report["automatic_gate_pass"]
        or report["stable_evidence_id"] != parent["bl8_stable_evidence_id"]
    ):
        raise AO6ChromaEmphasisError("BL8 evidence gate is not open")
    records = {
        (str(row["source_id"]), str(row["arm_id"])): row
        for row in report["rows"]
    }
    expected = {(source_id, AO6_ARM) for source_id in validated["eligible_ids"]}
    if not expected.issubset(records):
        raise AO6ChromaEmphasisError("AO6 inventory drift")
    return {**validated, "records": records}


def run_ao6_nonbasic_chroma_emphasis(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise AO6ChromaEmphasisError("BL13 requires OMP_NUM_THREADS=1")
    if output_dir.exists():
        raise FileExistsError("BL13 output is create-only")
    output_dir.mkdir(parents=True)
    spec = config["candidate"]
    rows: list[dict[str, Any]] = []
    for source_id in validated["eligible_ids"]:
        source_row = validated["source_rows"][source_id]
        raw_path = root / source_row["raw_path"]
        if sha256_file(raw_path) != source_row["raw_sha256"]:
            raise AO6ChromaEmphasisError("RAW identity drift")
        working = load_raw_working_image(raw_path)
        source = _encode_scene_linear(working.pixels).astype(np.float64)
        record = validated["records"][(source_id, AO6_ARM)]
        ao6 = _decode_rgb16(
            root / config["parent"]["bl8_output_dir"] / record["output"],
            record["output_sha256"],
        )
        result = compose_ao6_nonbasic_chroma_emphasis(
            source,
            ao6,
            fit_pixel_budget=int(spec["fit_pixel_budget"]),
            chroma_residual_strength=float(spec["chroma_residual_strength"]),
            hard_boundary_epsilon_encoded_srgb=float(
                spec["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                spec["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        output = result["output"]
        output_path = output_dir / f"{source_id}.png"
        save_srgb16_png(np.asarray(output, dtype=np.float32), output_path)

        sample_source, sample_ao6, sample_output = _aligned_sample(
            source, ao6, output, maximum=int(spec["fit_pixel_budget"])
        )
        source_lab = linear_rgb_to_lab(
            np.asarray(encoded_srgb_to_linear(sample_source), dtype=np.float32),
            working_space="linear_srgb",
        ).astype(np.float64)
        ao6_lab = linear_rgb_to_lab(
            np.asarray(encoded_srgb_to_linear(sample_ao6), dtype=np.float32),
            working_space="linear_srgb",
        ).astype(np.float64)
        output_lab = linear_rgb_to_lab(
            np.asarray(encoded_srgb_to_linear(sample_output), dtype=np.float32),
            working_space="linear_srgb",
        ).astype(np.float64)
        ao6_style = np.linalg.norm(ao6_lab - source_lab, axis=1)
        candidate_style = np.linalg.norm(output_lab - source_lab, axis=1)
        difference = np.linalg.norm(output_lab - ao6_lab, axis=1)
        requested = result["target_linear"] - result["ao6_linear"]
        actual = encoded_srgb_to_linear(output) - result["ao6_linear"]
        requested_energy = float(np.sqrt(np.mean(requested**2)))
        actual_energy = float(np.sqrt(np.mean(actual**2)))
        limited = result["residual_scale"] < 1.0 - 1e-12
        metrics = boundary_metrics(output, ao6)
        rows.append(
            {
                "source_id": source_id,
                "make": source_row["make"],
                "output": output_path.name,
                "output_sha256": sha256_file(output_path),
                "ao6_output_sha256": record["output_sha256"],
                "median_ao6_style_delta_e76": float(np.median(ao6_style)),
                "median_candidate_style_delta_e76": float(
                    np.median(candidate_style)
                ),
                "style_ratio_vs_ao6": float(
                    np.median(candidate_style) / max(float(np.median(ao6_style)), 1e-12)
                ),
                "median_difference_from_ao6_delta_e76": float(
                    np.median(difference)
                ),
                "p95_absolute_lightness_drift_delta_lstar": float(
                    np.quantile(np.abs(output_lab[:, 0] - ao6_lab[:, 0]), 0.95)
                ),
                "requested_residual_energy_retention": actual_energy
                / max(requested_energy, 1e-12),
                "safety_limited_pixel_fraction": float(np.mean(limited)),
                **metrics,
            }
        )
        del working, source, ao6, result, output

    aggregate = {
        "output_count": len(rows),
        "maximum_output_code_boundary_fraction": max(
            row["output_code_boundary_fraction"] for row in rows
        ),
        "maximum_new_boundary_fraction_vs_ao6": max(
            row["new_boundary_fraction_vs_ao6"] for row in rows
        ),
        "p95_safety_limited_pixel_fraction": float(
            np.quantile(
                [row["safety_limited_pixel_fraction"] for row in rows], 0.95
            )
        ),
        "median_requested_residual_energy_retention": float(
            np.median([row["requested_residual_energy_retention"] for row in rows])
        ),
        "median_style_delta_e76_ratio_vs_ao6": float(
            np.median([row["style_ratio_vs_ao6"] for row in rows])
        ),
        "median_difference_from_ao6_delta_e76": float(
            np.median([row["median_difference_from_ao6_delta_e76"] for row in rows])
        ),
        "median_image_p95_absolute_lightness_drift_delta_lstar": float(
            np.median(
                [row["p95_absolute_lightness_drift_delta_lstar"] for row in rows]
            )
        ),
    }
    gate = config["automatic_gate"]
    checks = {
        "output_count": aggregate["output_count"] == int(gate["expected_outputs"]),
        "output_boundary": aggregate["maximum_output_code_boundary_fraction"]
        <= float(gate["maximum_output_code_boundary_fraction"]),
        "new_boundary": aggregate["maximum_new_boundary_fraction_vs_ao6"]
        <= float(gate["maximum_new_boundary_fraction_vs_ao6"]),
        "safety_limited_tail": aggregate["p95_safety_limited_pixel_fraction"]
        <= float(gate["maximum_p95_safety_limited_pixel_fraction"]),
        "residual_retention": aggregate["median_requested_residual_energy_retention"]
        >= float(gate["minimum_median_requested_residual_energy_retention"]),
        "style_gain": aggregate["median_style_delta_e76_ratio_vs_ao6"]
        >= float(gate["minimum_median_style_delta_e76_ratio_vs_ao6"]),
        "material_difference": aggregate["median_difference_from_ao6_delta_e76"]
        >= float(gate["minimum_median_difference_from_ao6_delta_e76"]),
        "lightness_drift": aggregate[
            "median_image_p95_absolute_lightness_drift_delta_lstar"
        ]
        <= float(gate["maximum_median_image_p95_absolute_lightness_drift_delta_lstar"]),
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "rows": rows,
        "aggregate": aggregate,
        "automatic_checks": checks,
        "automatic_gate_pass": all(checks.values()),
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "AO6ChromaEmphasisError",
    "compose_ao6_nonbasic_chroma_emphasis",
    "run_ao6_nonbasic_chroma_emphasis",
    "validate_contract",
]
