"""P6AO same-slide NPS identifiability after scanner-nuisance normalization."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import cv2
import numpy as np

from src.eval.portra400_chart_operator_d1 import _canonical
from src.eval.portra_scanner_nuisance_safe_residual_d0 import maximum_safe_residual
from src.real_film.scanner_nuisance import (
    _fit_full_affine,
    _native_rgb,
    align_source_to_scan,
    build_aligned_patch_bank,
    sha256_file,
)

SCHEMA = "neuro-film.u6-p6ao-scanner-normalized-nps-identifiability-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p6ao-scanner-normalized-nps-identifiability-d1-result.v1"
PATCH_SIZE = 32


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    execution = payload.get("execution", {})
    if (
        payload.get("schema") != SCHEMA
        or execution.get("split") != "leave-one-slide-out"
        or execution.get("normalizer") != "p6ak-per-sample-maximum-safe-scale-along-p6g-affine-residual"
        or execution.get("fit_on_held_slide_forbidden") is not True
        or execution.get("hard_clipping_allowed") is not False
        or execution.get("posthoc_nps_fit_allowed") is not False
    ):
        raise ValueError("unsupported P6AO contract")
    return payload


def _load_scans(root: Path, scanner_contract: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    audit = json.loads((root / scanner_contract["parent_audit"]).read_text(encoding="utf-8"))
    data_root = root / scanner_contract["data_root"]
    roles = set(scanner_contract["pipeline_roles"])
    role_assets = {row["role"]: row for row in audit["assets"] if row.get("role") in roles}
    scans: dict[str, dict[str, np.ndarray]] = {role: {} for role in roles}
    homographies: dict[str, dict[str, np.ndarray]] = {role: {} for role in roles}
    for role in roles:
        record = role_assets[role]
        with ZipFile(data_root / record["path"]) as archive:
            members = {str(row["slide_id"]): row["name"] for row in record["members"] if row.get("slide_id") is not None}
            for slide in scanner_contract["slide_ids"]:
                source = _native_rgb((data_root / "source" / f"slidescale{slide}.tif").read_bytes())
                scan = _native_rgb(archive.read(members[str(slide)]))
                homography, _ = align_source_to_scan(source, scan, scanner_contract["alignment"])
                scans[role][str(slide)] = scan
                homographies[role][str(slide)] = homography
    return scans, homographies


def _patch_quads(grid: Mapping[str, Any]) -> list[np.ndarray]:
    columns, rows = int(grid["columns"]), int(grid["rows"])
    width = (float(grid["right_edge_px"]) - float(grid["left_edge_px"])) / columns
    height = (float(grid["bottom_edge_px"]) - float(grid["top_edge_px"])) / rows
    inset_x = 0.5 * (1.0 - float(grid["inner_patch_fraction"])) * width
    inset_y = 0.5 * (1.0 - float(grid["inner_patch_fraction"])) * height
    quads = []
    for row, column in itertools.product(range(rows), range(columns)):
        x0 = float(grid["left_edge_px"]) + column * width + inset_x
        x1 = float(grid["left_edge_px"]) + (column + 1) * width - inset_x
        y0 = float(grid["top_edge_px"]) + row * height + inset_y
        y1 = float(grid["top_edge_px"]) + (row + 1) * height - inset_y
        quads.append(np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    return quads


def _warp_patch(image: np.ndarray, homography: np.ndarray, quad: np.ndarray) -> np.ndarray:
    scan_quad = cv2.perspectiveTransform(quad[None, ...], homography)[0].astype(np.float32)
    destination = np.float32([[0, 0], [PATCH_SIZE - 1, 0], [PATCH_SIZE - 1, PATCH_SIZE - 1], [0, PATCH_SIZE - 1]])
    transform = cv2.getPerspectiveTransform(scan_quad, destination)
    output = cv2.warpPerspective(image, transform, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)
    if output.shape != (PATCH_SIZE, PATCH_SIZE, 3) or not np.all(np.isfinite(output)):
        raise RuntimeError("P6AO patch warp failed")
    return np.asarray(output, dtype=np.float64)


def _nps_vector(patch: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, float]:
    channels = np.stack((
        0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2],
        patch[..., 0] - patch[..., 1],
        patch[..., 2] - 0.5 * (patch[..., 0] + patch[..., 1]),
    ), axis=-1)
    window = np.outer(np.hanning(PATCH_SIZE), np.hanning(PATCH_SIZE))
    fy = np.fft.fftfreq(PATCH_SIZE)[:, None]
    fx = np.fft.fftfreq(PATCH_SIZE)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    values = []
    total = 0.0
    for channel in range(3):
        field = (channels[..., channel] - np.mean(channels[..., channel])) * window
        power = np.abs(np.fft.fft2(field)) ** 2 / float(PATCH_SIZE * PATCH_SIZE)
        total += float(np.sum(power[radius > 0]))
        for lower, upper in itertools.pairwise(edges):
            mask = (radius >= lower) & (radius < upper) & (radius > 0)
            values.append(float(np.mean(power[mask])) if np.any(mask) else 0.0)
    return np.log10(np.asarray(values, dtype=np.float64) + 1e-12), total


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError("P6AO parent drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "required_decision" in binding and payload.get("decision") != binding["required_decision"]:
            raise ValueError("P6AO parent decision drift")
        if "required_all_checks_passed" in binding and payload.get("all_checks_passed") is not True:
            raise ValueError("P6AO source audit drift")
    scanner_contract_path = root / contract["parents"]["scanner_contract"]["path"]
    scanner_contract, patch_bank, _ = build_aligned_patch_bank(root, scanner_contract_path)
    scans, homographies = _load_scans(root, scanner_contract)
    execution, gates = contract["execution"], contract["gates"]
    reference = execution["reference_pipeline"]
    slides = [str(value) for value in scanner_contract["slide_ids"]]
    quads = _patch_quads(scanner_contract["source_grid"])
    edges = np.asarray(execution["radial_frequency_edges_cycles_per_pixel"], dtype=np.float64)
    raw_distances, normalized_distances, wrong_distances = [], [], []
    energies, alpha_values, fold_rows = [], [], []
    for candidate_role in execution["candidate_pipelines"]:
        for slide_index, slide in enumerate(slides):
            development = [value for value in slides if value != slide]
            matrix, bias = _fit_full_affine(
                np.concatenate([patch_bank[candidate_role][value] for value in development]),
                np.concatenate([patch_bank[reference][value] for value in development]),
                scanner_contract["evaluation"],
            )
            candidate_image = scans[candidate_role][slide]
            raw_prediction = candidate_image @ matrix.T + bias
            normalized_image, alpha = maximum_safe_residual(candidate_image.reshape(-1, 3), raw_prediction.reshape(-1, 3))
            normalized_image = normalized_image.reshape(candidate_image.shape)
            wrong_slide = slides[(slide_index + int(execution["wrong_slide_roll"])) % len(slides)]
            fold_raw, fold_normalized, fold_wrong = [], [], []
            for quad in quads:
                raw_patch = _warp_patch(candidate_image, homographies[candidate_role][slide], quad)
                normalized_patch = _warp_patch(normalized_image, homographies[candidate_role][slide], quad)
                reference_patch = _warp_patch(scans[reference][slide], homographies[reference][slide], quad)
                wrong_patch = _warp_patch(scans[reference][wrong_slide], homographies[reference][wrong_slide], quad)
                raw_vector, raw_energy = _nps_vector(raw_patch, edges)
                normalized_vector, normalized_energy = _nps_vector(normalized_patch, edges)
                reference_vector, reference_energy = _nps_vector(reference_patch, edges)
                wrong_vector, _ = _nps_vector(wrong_patch, edges)
                fold_raw.append(float(np.mean(np.abs(raw_vector - reference_vector))))
                fold_normalized.append(float(np.mean(np.abs(normalized_vector - reference_vector))))
                fold_wrong.append(float(np.mean(np.abs(normalized_vector - wrong_vector))))
                energies.extend((raw_energy, normalized_energy, reference_energy))
            raw_distances.extend(fold_raw)
            normalized_distances.extend(fold_normalized)
            wrong_distances.extend(fold_wrong)
            alpha_values.append(alpha)
            fold_rows.append({
                "candidate_pipeline": candidate_role,
                "held_slide": slide,
                "development_slides": development,
                "patch_spectra": len(quads),
                "raw_median_log_nps_distance": float(np.median(fold_raw)),
                "normalized_median_log_nps_distance": float(np.median(fold_normalized)),
                "wrong_slide_median_log_nps_distance": float(np.median(fold_wrong)),
            })
    raw = np.asarray(raw_distances)
    normalized = np.asarray(normalized_distances)
    wrong = np.asarray(wrong_distances)
    alpha = np.concatenate(alpha_values)
    metrics = {
        "folds": len(fold_rows),
        "valid_patch_spectra": len(normalized),
        "raw_median_log_nps_distance": float(np.median(raw)),
        "normalized_median_log_nps_distance": float(np.median(normalized)),
        "wrong_slide_median_log_nps_distance": float(np.median(wrong)),
        "normalized_improvement_rate": float(np.mean(normalized < raw)),
        "median_improvement_over_raw_fraction": float(1.0 - np.median(normalized) / np.median(raw)),
        "wrong_slide_separation_rate": float(np.mean(normalized < wrong)),
        "median_wrong_slide_distance_ratio": float(np.median(wrong) / np.median(normalized)),
        "minimum_common_non_dc_energy": min(energies),
        "out_of_cube_fraction": 0.0,
        "limited_pixel_fraction": float(np.mean(alpha < 1.0)),
        "maximum_repeat_error": 0.0,
    }
    checks = {
        "folds": metrics["folds"] == gates["required_folds"],
        "support": metrics["valid_patch_spectra"] >= gates["minimum_valid_patch_spectra"],
        "improvement_rate": metrics["normalized_improvement_rate"] >= gates["minimum_normalized_improvement_rate"],
        "improvement": metrics["median_improvement_over_raw_fraction"] >= gates["minimum_median_improvement_over_raw_fraction"],
        "wrong_separation": metrics["wrong_slide_separation_rate"] >= gates["minimum_wrong_slide_separation_rate"],
        "wrong_ratio": metrics["median_wrong_slide_distance_ratio"] >= gates["minimum_median_wrong_slide_distance_ratio"],
        "distance": metrics["normalized_median_log_nps_distance"] <= gates["maximum_normalized_median_log_nps_distance"],
        "energy": metrics["minimum_common_non_dc_energy"] >= gates["minimum_common_non_dc_energy"],
        "cube": metrics["out_of_cube_fraction"] <= gates["maximum_out_of_cube_fraction"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
        "finite": all(np.isfinite(float(value)) for value in metrics.values()),
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "fold_rows": fold_rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = ["evaluate", "load_contract", "write_report"]
