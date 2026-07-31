"""Fixed strict-interior FilmMatch equation on the BH1 CC0 population."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.eval.filmmatch_identity_residual_ood import (
    _encode_scene_linear,
    validate_bl3_contract,
)
from src.eval.fresh_native_standard_confirmation import boundary_metrics
from src.eval.global_frontier import sha256_file
from src.preprocess import save_srgb16_png
from src.preprocess.raw_decode import load_raw_working_image
from src.roll2film.identity_residual_sigmoid import IdentityResidualSigmoidOperator
from src.roll2film.strict_interior_sigmoid import StrictInteriorSigmoidOperator


SCHEMA = "neuro_film.u5_r2bl6_filmmatch_strict_interior_ood_report.v1"
AO6_ARM = "fixed_ao6_colour_only_t15_c35"
CANDIDATE_ARM = "fixed_bl5_strict_interior_sigmoid"


class StrictInteriorOODError(RuntimeError):
    """Raised when a BL6 identity or execution invariant drifts."""


def _load_exact_json(root: Path, path: str, expected: str) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise StrictInteriorOODError(f"hash mismatch: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StrictInteriorOODError(f"expected JSON object: {path}")
    return payload


def _operator_from_report(report: Mapping[str, Any]) -> StrictInteriorSigmoidOperator:
    payload = report["final_fit"]["operator"]
    return StrictInteriorSigmoidOperator(
        base=IdentityResidualSigmoidOperator(
            capture_matrix=np.asarray(payload["capture_matrix"], dtype=np.float64),
            response_midpoints=np.asarray(payload["response_midpoints"], dtype=np.float64),
            response_slopes=np.asarray(payload["response_slopes"], dtype=np.float64),
            scan_matrix=np.asarray(payload["scan_matrix"], dtype=np.float64),
            nonlinear_strength=float(payload["nonlinear_strength"]),
            exposure_floor=float(payload["exposure_floor"]),
        ),
        output_epsilon=float(payload["output_epsilon"]),
    )


def _apply_rows(operator: Any, image: np.ndarray, rows: int = 128) -> np.ndarray:
    output = np.empty_like(image, dtype=np.float32)
    for y0 in range(0, image.shape[0], rows):
        y1 = min(image.shape[0], y0 + rows)
        output[y0:y1] = operator.apply(
            np.asarray(image[y0:y1], dtype=np.float64)
        ).astype(np.float32)
    return output


def _decode_parent_rgb16(path: Path, expected_sha256: str) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise StrictInteriorOODError("AO6 parent output identity drift")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16 or decoded.shape[-1] != 3:
        raise StrictInteriorOODError("expected AO6 RGB16 PNG")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def validate_bl6_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rendering = config["rendering"]
    if (
        config.get("schema")
        != "neuro_film.u5_r2bl6_filmmatch_strict_interior_ood.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or rendering["per_image_fit_allowed"]
        or rendering["operator_refit_allowed"]
        or rendering["strength_retuning_allowed"]
        or rendering["routing_allowed"]
        or rendering["hard_clipping_allowed"]
    ):
        raise StrictInteriorOODError("BL6 frozen contract drift")
    candidate = config["candidate"]
    report = _load_exact_json(
        root, candidate["bl5_report"], candidate["bl5_report_sha256"]
    )
    decision = _load_exact_json(
        root, candidate["bl5_decision"], candidate["bl5_decision_sha256"]
    )
    if (
        report["stable_evidence_id"] != candidate["bl5_stable_evidence_id"]
        or not report["automatic_gate_passed"]
        or decision["decision"]["status"] != candidate["required_status"]
    ):
        raise StrictInteriorOODError("BL5 candidate gate drift")
    population = config["population"]
    bl3_config = _load_exact_json(
        root, population["bl3_contract"], population["bl3_contract_sha256"]
    )
    validated = validate_bl3_contract(root, bl3_config)
    if (
        len(validated["eligible_ids"]) != population["expected_sources"]
        or len({validated["source_rows"][key]["make"] for key in validated["eligible_ids"]})
        != population["expected_camera_makes"]
    ):
        raise StrictInteriorOODError("BL3 population drift")
    return {**validated, "operator_report": report}


def run_strict_interior_ood(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    parent_output_dir: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_bl6_contract(root, config)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise StrictInteriorOODError("BL6 requires OMP_NUM_THREADS=1")
    if output_dir.exists():
        raise FileExistsError("BL6 output is create-only")
    parent_report_path = parent_output_dir / "report.json"
    if sha256_file(parent_report_path) != config["population"]["bl3_parent_report_sha256"]:
        raise StrictInteriorOODError("BL3 parent report identity drift")
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    parent_rows = {(r["source_id"], r["arm_id"]): r for r in parent_report["rows"]}
    operator = _operator_from_report(validated["operator_report"])
    output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    styles: list[float] = []
    for source_id in validated["eligible_ids"]:
        source = validated["source_rows"][source_id]
        raw_path = root / source["raw_path"]
        if sha256_file(raw_path) != source["raw_sha256"]:
            raise StrictInteriorOODError("live RAW identity drift")
        working = load_raw_working_image(raw_path)
        if (
            working.transfer_state != "scene_linear"
            or working.working_space not in {"linear_srgb", "linear_srgb_d65"}
            or not working.orientation_applied
        ):
            raise StrictInteriorOODError("WorkingImage contract drift")
        encoded = _encode_scene_linear(working.pixels)
        candidate = _apply_rows(operator, encoded)
        ao6_row = parent_rows[(source_id, AO6_ARM)]
        ao6 = _decode_parent_rgb16(
            parent_output_dir / ao6_row["output"], ao6_row["output_sha256"]
        )
        style = float(np.sqrt(np.mean(np.square(candidate - encoded))))
        path = output_dir / "renders" / f"{source_id}.png"
        save_srgb16_png(candidate, path)
        rows.append(
            {
                "source_id": source_id,
                "source_raw_sha256": source["raw_sha256"],
                "make": source["make"],
                "output": path.relative_to(output_dir).as_posix(),
                "output_sha256": sha256_file(path),
                "style_rgb_rmse_from_source": style,
                **boundary_metrics(candidate, ao6),
            }
        )
        styles.append(style)
        del working, encoded, candidate, ao6
    gate = config["automatic_gate"]
    aggregate = {
        "output_count": len(rows),
        "maximum_output_code_boundary_fraction": max(
            row["output_code_boundary_fraction"] for row in rows
        ),
        "maximum_new_boundary_fraction_vs_ao6": max(
            row["new_boundary_fraction_vs_ao6"] for row in rows
        ),
        "median_style_rgb_rmse_from_source": float(np.median(styles)),
    }
    passed = bool(
        aggregate["output_count"] == gate["expected_outputs"]
        and aggregate["maximum_output_code_boundary_fraction"]
        <= gate["maximum_output_code_boundary_fraction"]
        and aggregate["maximum_new_boundary_fraction_vs_ao6"]
        <= gate["maximum_new_boundary_fraction_vs_ao6"]
        and aggregate["median_style_rgb_rmse_from_source"]
        >= gate["minimum_median_style_rgb_rmse_from_source"]
    )
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "operator_stable_evidence_id": validated["operator_report"]["stable_evidence_id"],
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


def build_bl6_blind_round(
    *,
    root: Path,
    parent_output_dir: Path,
    candidate_output_dir: Path,
    source_rows: Mapping[str, Mapping[str, Any]],
    eligible_ids: list[str],
    round_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    mapping: list[dict[str, str]] = []
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    for source_id in eligible_ids:
        order = [AO6_ARM, CANDIDATE_ARM]
        random.Random(
            hashlib.sha256(f"u5-r2bl6:{round_index}:{source_id}".encode()).digest()
        ).shuffle(order)
        mapping.append({"source_id": source_id, "A": order[0], "B": order[1]})
        with Image.open(root / source_rows[source_id]["decoded_path"]) as image:
            original = image.convert("RGB")
            original.thumbnail((520, 330), Image.Resampling.LANCZOS)
        candidates = []
        for arm_id in order:
            path = (
                parent_output_dir / "renders" / AO6_ARM / f"{source_id}.png"
                if arm_id == AO6_ARM
                else candidate_output_dir / "renders" / f"{source_id}.png"
            )
            with Image.open(path) as image:
                candidate = image.convert("RGB")
                candidate.thumbnail((520, 330), Image.Resampling.LANCZOS)
            candidates.append(candidate)
        tile = Image.new("RGB", (1580, 375), "white")
        draw = ImageDraw.Draw(tile)
        for index, (label, image) in enumerate(zip(("Source", "A", "B"), (original, *candidates))):
            x = 5 + index * 525
            draw.text((x, 3), label, fill="black", font=font)
            tile.paste(image, (x, 23))
        draw.text((5, 357), source_id, fill="black", font=font)
        tiles.append(tile)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = []
    for part_index, start in enumerate((0, 4, 8), start=1):
        sheet = Image.new("RGB", (1580, 4 * 375 + 28), "white")
        ImageDraw.Draw(sheet).text(
            (5, 5), f"U5.R2BL6 round {round_index} part {part_index}", fill="black", font=font
        )
        for index, tile in enumerate(tiles[start : start + 4]):
            sheet.paste(tile, (0, 28 + index * 375))
        path = output_dir / f"blind_round_{round_index}_part_{part_index}.png"
        sheet.save(path, "PNG")
        hashes.append(sha256_file(path))
    mapping_path = output_dir / f"blind_round_{round_index}_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"part_sha256": hashes, "mapping_sha256": sha256_file(mapping_path)}


__all__ = [
    "build_bl6_blind_round",
    "run_strict_interior_ood",
    "validate_bl6_contract",
]
