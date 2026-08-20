"""SF3.A0M deterministic capture-alignment stress on the A0K scene pack."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.real_film.scanner_nuisance import (
    ScannerNuisanceError,
    align_source_to_scan,
)

SCHEMA = "neuro-film.sf3-a0m-three-stock-capture-alignment-stress-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0m-three-stock-capture-alignment-stress-report.v1"


class ThreeStockCaptureAlignmentError(RuntimeError):
    """Raised when an SF3.A0M binding or numerical invariant fails."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ThreeStockCaptureAlignmentError("SF3.A0M paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    evaluation = value.get("evaluation", {})
    gates = value.get("gates", {})
    conditions = value.get("stress_conditions", [])
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != "SF3.A0M"
        or value.get("status") != "FROZEN_BEFORE_ALIGNMENT_STRESS"
        or len(value.get("expected_scene_ids", [])) != 12
        or len(set(value.get("expected_scene_ids", []))) != 12
        or [row.get("id") for row in conditions]
        != ["mild_warm_scan", "cool_low_contrast_scan", "dense_warp_neutral_scan"]
        or evaluation.get("formal_processes") != 2
        or evaluation.get("save_stressed_pixels") is not False
        or evaluation.get("stock_operator_fit_allowed") is not False
        or gates.get("required_rows") != 36
        or gates.get("require_exact_scientific_replay") is not True
    ):
        raise ThreeStockCaptureAlignmentError("SF3.A0M frozen contract drift")
    for parent in value["parent"].values():
        _relative(parent["path"])
    return value


def stress_image(source: np.ndarray, condition: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Apply one fixed scan-like stress and return its exact source-to-scan H."""

    image = np.asarray(source, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or not np.all(np.isfinite(image)):
        raise ThreeStockCaptureAlignmentError("invalid source image")
    height, width = image.shape[:2]
    source_corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    offsets = np.asarray(condition["corner_offsets_fraction"], dtype=np.float32)
    if offsets.shape != (4, 2):
        raise ThreeStockCaptureAlignmentError("corner offset shape drift")
    scale = np.asarray([width, height], dtype=np.float32)
    target_corners = source_corners + offsets * scale
    homography = cv2.getPerspectiveTransform(source_corners, target_corners)
    warped = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    sigma = float(condition["gaussian_sigma_pixels"])
    blurred = cv2.GaussianBlur(warped, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gain = np.asarray(condition["channel_gain"], dtype=np.float32)
    offset = np.asarray(condition["channel_offset"], dtype=np.float32)
    gamma = float(condition["gamma"])
    stressed = np.clip(np.power(np.clip(blurred, 0.0, 1.0), gamma) * gain + offset, 0.0, 1.0)
    return np.asarray(stressed, dtype=np.float32), np.asarray(homography, dtype=np.float64)


def truth_reprojection_errors(
    estimated: np.ndarray,
    truth: np.ndarray,
    shape: tuple[int, int],
    evaluation: Mapping[str, Any],
) -> np.ndarray:
    height, width = shape
    margin = float(evaluation["inner_margin_fraction"])
    xs = np.linspace(margin * width, (1.0 - margin) * (width - 1), int(evaluation["truth_grid_columns"]))
    ys = np.linspace(margin * height, (1.0 - margin) * (height - 1), int(evaluation["truth_grid_rows"]))
    points = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
    observed = cv2.perspectiveTransform(points, np.asarray(estimated, dtype=np.float64)).reshape(-1, 2)
    expected = cv2.perspectiveTransform(points, np.asarray(truth, dtype=np.float64)).reshape(-1, 2)
    return np.linalg.norm(observed - expected, axis=1)


def _bound(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    path = root / _relative(str(binding["path"]))
    if _sha(path) != binding["sha256"]:
        raise ThreeStockCaptureAlignmentError(f"SF3.A0M {label} hash drift")
    return path


def evaluate(contract: Mapping[str, Any], root: Path, *, reverse: bool = False) -> dict[str, Any]:
    evidence_path = _bound(root, contract["parent"]["stimulus_evidence"], "evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("decision") != contract["parent"]["stimulus_evidence"]["required_decision"]:
        raise ThreeStockCaptureAlignmentError("SF3.A0K parent decision drift")
    manifest_path = _bound(root, contract["parent"]["stimulus_manifest"], "manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_rows = manifest.get("scene_rows", [])
    ids = [row.get("scene_id") for row in scene_rows]
    if ids != contract["expected_scene_ids"]:
        raise ThreeStockCaptureAlignmentError("SF3.A0K scene order drift")

    work = [(row, condition) for row in scene_rows for condition in contract["stress_conditions"]]
    if reverse:
        work.reverse()
    results = []
    for row, condition in work:
        source_path = manifest_path.parent / _relative(row["stimulus_relative_path"])
        if _sha(source_path) != row["stimulus_sha256"]:
            raise ThreeStockCaptureAlignmentError(f"{row['scene_id']} stimulus hash drift")
        with Image.open(source_path) as image:
            source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        stressed, truth = stress_image(source, condition)
        try:
            estimated, diagnostics = align_source_to_scan(source, stressed, contract["alignment"])
            errors = truth_reprojection_errors(estimated, truth, source.shape[:2], contract["evaluation"])
            aligned = True
            reason = None
            median_error = float(np.median(errors))
            p95_error = float(np.quantile(errors, 0.95))
            maximum_error = float(np.max(errors))
        except ScannerNuisanceError as error:
            diagnostics = {}
            aligned = False
            reason = str(error)
            median_error = p95_error = maximum_error = None
        results.append(
            {
                "scene_id": row["scene_id"],
                "role": row["role"],
                "condition_id": condition["id"],
                "source_sha256": row["stimulus_sha256"],
                "aligned": aligned,
                "failure_reason": reason,
                "median_truth_reprojection_error_px": median_error,
                "p95_truth_reprojection_error_px": p95_error,
                "maximum_truth_reprojection_error_px": maximum_error,
                "alignment_diagnostics": diagnostics,
            }
        )
    order = {scene_id: index for index, scene_id in enumerate(contract["expected_scene_ids"])}
    condition_order = {row["id"]: index for index, row in enumerate(contract["stress_conditions"])}
    results.sort(key=lambda row: (order[row["scene_id"]], condition_order[row["condition_id"]]))
    successful = [row for row in results if row["aligned"]]
    gates = contract["gates"]
    automatic = {
        "row_count": len(results) == int(gates["required_rows"]),
        "all_rows_aligned": len(successful) == len(results),
        "population_median": bool(successful)
        and float(np.median([row["median_truth_reprojection_error_px"] for row in successful]))
        <= float(gates["maximum_population_median_truth_reprojection_error_px"]),
        "row_p95": bool(successful)
        and max(row["p95_truth_reprojection_error_px"] for row in successful)
        <= float(gates["maximum_row_p95_truth_reprojection_error_px"]),
        "row_max": bool(successful)
        and max(row["maximum_truth_reprojection_error_px"] for row in successful)
        <= float(gates["maximum_row_max_truth_reprojection_error_px"]),
    }
    automatic_pass = all(automatic.values())
    aggregate = {
        "rows": len(results),
        "aligned_rows": len(successful),
        "population_median_truth_reprojection_error_px": float(
            np.median([row["median_truth_reprojection_error_px"] for row in successful])
        ) if successful else None,
        "maximum_row_p95_truth_reprojection_error_px": max(
            (row["p95_truth_reprojection_error_px"] for row in successful), default=None
        ),
        "maximum_row_max_truth_reprojection_error_px": max(
            (row["maximum_truth_reprojection_error_px"] for row in successful), default=None
        ),
    }
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha(root / "configs/sf3_a0m_three_stock_capture_alignment_stress_v1.json"),
        "stimulus_manifest_sha256": contract["parent"]["stimulus_manifest"]["sha256"],
        "rows": results,
        "aggregate": aggregate,
        "gates": automatic,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass" if automatic_pass else "decision_if_fail"],
        "pixel_operator_fits": 0,
        "stock_operator_fits": 0,
        "stressed_pixels_retained": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha_bytes(_canonical(core))}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    return _sha_bytes(payload)
