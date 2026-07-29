"""U6.P6I per-device characterization against measured D50 target values."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.optimize import lsq_linear
from skimage.color import deltaE_ciede2000, xyz2lab

from src.real_film.scanner_nuisance import (
    ScannerNuisanceError,
    _native_rgb,
    align_source_to_scan,
    patch_medians,
)


SCHEMA = "neuro_film.u6_p6i_measured_scanner_characterization_contract.v1"
COLOUR_SAMPLE = re.compile(r"^([A-L])([1-9]|1[0-9]|2[0-2])$")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6I contract")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def _slide_member(
    names: list[str], target_set: int, slide: int
) -> str:
    pattern = re.compile(
        rf"(?i)(?:testscan|test)(?:_|-)?{target_set}(?:_|-){slide}"
    )
    matches = [
        name
        for name in names
        if Path(name).suffix.lower() in {".tif", ".tiff"}
        and pattern.search(Path(name).stem)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"target {target_set} slide {slide} has members {matches}"
        )
    return matches[0]


def _load_target_table(
    path: Path,
    target_sets: list[int],
) -> dict[int, dict[int, dict[str, np.ndarray | tuple[str, ...]]]]:
    rows: dict[int, dict[int, list[dict[str, str]]]] = {
        target_set: {slide: [] for slide in range(1, 6)}
        for target_set in target_sets
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            target_set = int(row["test_set"])
            slide = int(row["slide_index"])
            if target_set in rows and COLOUR_SAMPLE.fullmatch(row["sample_id"]):
                rows[target_set][slide].append(row)
    result: dict[
        int, dict[int, dict[str, np.ndarray | tuple[str, ...]]]
    ] = {}
    expected_ids = tuple(
        f"{letter}{column}"
        for letter in "ABCDEFGHIJKL"
        for column in range(1, 23)
    )
    for target_set, slide_rows in rows.items():
        result[target_set] = {}
        for slide, values in slide_rows.items():
            ids = tuple(row["sample_id"] for row in values)
            if ids != expected_ids:
                raise ValueError(
                    f"target {target_set} slide {slide} patch order mismatch"
                )
            result[target_set][slide] = {
                "sample_ids": ids,
                "source_rgb": np.asarray(
                    [
                        [
                            float(row["source_r"]),
                            float(row["source_g"]),
                            float(row["source_b"]),
                        ]
                        for row in values
                    ],
                    dtype=np.float64,
                ),
                "xyz": np.asarray(
                    [
                        [
                            float(row["xyz_x"]),
                            float(row["xyz_y"]),
                            float(row["xyz_z"]),
                        ]
                        for row in values
                    ],
                    dtype=np.float64,
                )
                / 100.0,
                "lab": np.asarray(
                    [
                        [
                            float(row["lab_l"]),
                            float(row["lab_a"]),
                            float(row["lab_b"]),
                        ]
                        for row in values
                    ],
                    dtype=np.float64,
                ),
            }
    return result


def _fit_positive_matrix(
    source: np.ndarray,
    target_xyz: np.ndarray,
    coefficient_interval: list[float],
) -> np.ndarray:
    lower, upper = (float(value) for value in coefficient_interval)
    rows = []
    for channel in range(3):
        fit = lsq_linear(
            source,
            target_xyz[:, channel],
            bounds=(lower, upper),
            method="trf",
            lsmr_tol="auto",
        )
        if not fit.success:
            raise ValueError(f"positive matrix fit failed: {fit.message}")
        rows.append(fit.x)
    return np.asarray(rows, dtype=np.float64)


def _select_common_power(
    source: np.ndarray,
    target_xyz: np.ndarray,
    powers: list[float],
    coefficient_interval: list[float],
) -> tuple[float, np.ndarray, float]:
    candidates = []
    for power in powers:
        transformed = np.power(source, float(power), dtype=np.float64)
        matrix = _fit_positive_matrix(
            transformed, target_xyz, coefficient_interval
        )
        prediction = transformed @ matrix.T
        loss = float(np.mean(np.linalg.norm(prediction - target_xyz, axis=1)))
        candidates.append((loss, float(power), matrix))
    loss, power, matrix = min(candidates, key=lambda row: (row[0], row[1]))
    return power, matrix, loss


def _metrics(
    predicted_xyz: np.ndarray,
    target_xyz: np.ndarray,
    target_lab: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    predicted_lab = xyz2lab(
        predicted_xyz.reshape(-1, 1, 3), illuminant="D50"
    ).reshape(-1, 3)
    delta_e76 = np.linalg.norm(predicted_lab - target_lab, axis=1)
    delta_e00 = deltaE_ciede2000(
        predicted_lab.reshape(-1, 1, 3),
        target_lab.reshape(-1, 1, 3),
    ).reshape(-1)
    xyz_l2 = np.linalg.norm(predicted_xyz - target_xyz, axis=1)
    return {
        "mean_delta_e76": float(np.mean(delta_e76)),
        "median_delta_e76": float(np.median(delta_e76)),
        "p95_delta_e76": float(np.percentile(delta_e76, 95.0)),
        "maximum_delta_e76": float(np.max(delta_e76)),
        "median_delta_e00": float(np.median(delta_e00)),
        "p95_delta_e00": float(np.percentile(delta_e00, 95.0)),
        "mean_xyz_l2": float(np.mean(xyz_l2)),
        "p95_xyz_l2": float(np.percentile(xyz_l2, 95.0)),
    }, delta_e76


def evaluate_measured_scanner_characterization(
    root: Path,
    contract_path: Path,
    alignment_retry_path: Path | None = None,
) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = load_contract(contract_path)
    parents = contract["parents"]
    for path_key, hash_key in (
        ("p6h_decision_path", "p6h_decision_sha256"),
        ("p6h_report_path", "p6h_report_sha256"),
        ("aq1_report_path", "aq1_report_sha256"),
        ("pair_table_path", "pair_table_sha256"),
        ("alignment_contract_path", "alignment_contract_sha256"),
    ):
        if _sha256(root / parents[path_key]) != parents[hash_key]:
            raise ValueError(f"{path_key} hash mismatch")
    p6h_decision = json.loads(
        (root / parents["p6h_decision_path"]).read_text(encoding="utf-8")
    )
    p6h_report = json.loads(
        (root / parents["p6h_report_path"]).read_text(encoding="utf-8")
    )
    if (
        p6h_decision.get("decision")
        != "source_pass_open_per_device_measured_reference_pilot_cross_target_generalization_closed"
        or p6h_report.get("automatic_pass") is not True
        or p6h_report["connectivity"][
            "scanner_target_graph_connected_across_sets"
        ]
        is not False
    ):
        raise ValueError("P6H source/closure decision is not exact")
    p6h_contract_path = root / p6h_decision["evidence"]["contract"]["path"]
    if _sha256(p6h_contract_path) != p6h_decision["evidence"]["contract"][
        "sha256"
    ]:
        raise ValueError("P6H contract hash mismatch")
    p6h_contract = json.loads(p6h_contract_path.read_text(encoding="utf-8"))
    acquisition = p6h_contract["acquisition"]
    data_root = root / acquisition["data_root"]

    alignment_contract = json.loads(
        (root / parents["alignment_contract_path"]).read_text(encoding="utf-8")
    )
    grid = alignment_contract["source_grid"]
    alignment = alignment_contract["alignment"]
    if alignment_retry_path is None:
        alignment_retry_path = (
            root / "configs" / "u6_p6i1_alignment_retry_v1.json"
        )
    retry_bytes = alignment_retry_path.read_bytes()
    retry_contract = json.loads(retry_bytes)
    if (
        retry_contract.get("schema")
        != "neuro_film.u6_p6i1_alignment_retry_contract.v1"
        or _sha256(root / retry_contract["parent_contract_path"])
        != retry_contract["parent_contract_sha256"]
        or _sha256(root / retry_contract["preflight_failure_path"])
        != retry_contract["preflight_failure_sha256"]
    ):
        raise ValueError("P6I1 alignment retry lineage mismatch")
    unchanged_gates = retry_contract["unchanged_acceptance_gates"]
    for key in (
        "minimum_good_matches",
        "minimum_ransac_inliers",
        "maximum_median_inlier_reprojection_error_px",
    ):
        if float(unchanged_gates[key]) != float(alignment[key]):
            raise ValueError(f"P6I1 changes alignment gate {key}")
    if float(retry_contract["primary"]["ratio_test"]) != float(
        alignment["ratio_test"]
    ):
        raise ValueError("P6I1 primary ratio mismatch")
    retry_alignment = dict(alignment)
    retry_alignment["ratio_test"] = float(
        retry_contract["retry"]["ratio_test"]
    )
    measurement = contract["measurement_contract"]
    target_sets = [int(value) for value in measurement["target_sets"]]
    target_table = _load_target_table(
        root / parents["pair_table_path"], target_sets
    )

    source_images: dict[int, np.ndarray] = {}
    source_patch_max_l2 = 0.0
    for row in contract["source_images"]:
        path = root / row["path"]
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"source slide {row['slide']} hash mismatch")
        slide = int(row["slide"])
        source_images[slide] = _native_rgb(path.read_bytes())
        source_patches = patch_medians(
            source_images[slide], np.eye(3, dtype=np.float64), grid
        )
        for target_set in target_sets:
            difference = np.linalg.norm(
                source_patches
                - np.asarray(
                    target_table[target_set][slide]["source_rgb"],
                    dtype=np.float64,
                ),
                axis=1,
            )
            source_patch_max_l2 = max(
                source_patch_max_l2, float(np.max(difference))
            )

    published_consistency = []
    for target_set in target_sets:
        for slide in range(1, 6):
            target_xyz = np.asarray(
                target_table[target_set][slide]["xyz"], dtype=np.float64
            )
            target_lab = np.asarray(
                target_table[target_set][slide]["lab"], dtype=np.float64
            )
            derived_lab = xyz2lab(
                target_xyz.reshape(-1, 1, 3), illuminant="D50"
            ).reshape(-1, 3)
            published_consistency.append(
                np.linalg.norm(derived_lab - target_lab, axis=1)
            )
    maximum_published_delta = float(
        np.max(np.concatenate(published_consistency))
    )

    assets_by_role = {row["role"]: row for row in p6h_report["assets"]}
    scan_bank: dict[str, dict[int, np.ndarray]] = {}
    pipeline_meta: dict[str, dict[str, Any]] = {}
    alignment_records: list[dict[str, Any]] = []
    for asset_contract in acquisition["archives"]:
        role = str(asset_contract["role"])
        asset = assets_by_role[role]
        archive_path = data_root / str(asset["path"])
        if _sha256(archive_path) != asset["sha256"]:
            raise ValueError(f"archive {role} hash mismatch")
        target_set = int(asset_contract["target_set"])
        scan_bank[role] = {}
        pipeline_meta[role] = {
            "target_set": target_set,
            "scanner": asset_contract["scanner"],
            "software": asset_contract["software"],
        }
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            for slide in range(1, 6):
                member = _slide_member(names, target_set, slide)
                scan = _native_rgb(archive.read(member))
                try:
                    homography, diagnostics = align_source_to_scan(
                        source_images[slide], scan, alignment
                    )
                    alignment_attempt = "primary"
                except ScannerNuisanceError:
                    homography, diagnostics = align_source_to_scan(
                        source_images[slide], scan, retry_alignment
                    )
                    alignment_attempt = "ratio-0.75-retry"
                patches = patch_medians(scan, homography, grid)
                expected_shape = (
                    int(measurement["colour_patches_per_slide"]),
                    3,
                )
                if patches.shape != expected_shape:
                    raise ValueError(f"{role}/{slide} shape mismatch")
                scan_bank[role][slide] = patches
                alignment_records.append(
                    {
                        "role": role,
                        "target_set": target_set,
                        "slide": slide,
                        "member": member,
                        "alignment_attempt": alignment_attempt,
                        **diagnostics,
                    }
                )

    candidate = contract["candidate_contract"]
    model_names = candidate["candidates"]
    pooled_errors: dict[str, list[np.ndarray]] = {
        name: [] for name in model_names
    }
    pipeline_errors: dict[str, dict[str, list[np.ndarray]]] = {
        role: {name: [] for name in model_names} for role in scan_bank
    }
    target_errors: dict[int, dict[str, list[np.ndarray]]] = {
        target_set: {name: [] for name in model_names}
        for target_set in target_sets
    }
    output_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in model_names
    }
    fold_records: list[dict[str, Any]] = []
    matrix_coefficients: list[np.ndarray] = []
    selected_powers: list[float] = []
    slides = tuple(range(1, 6))
    for role, by_slide in scan_bank.items():
        target_set = int(pipeline_meta[role]["target_set"])
        for held_slide in slides:
            train_slides = [slide for slide in slides if slide != held_slide]
            train_source = np.concatenate(
                [by_slide[slide] for slide in train_slides]
            )
            train_xyz = np.concatenate(
                [
                    np.asarray(
                        target_table[target_set][slide]["xyz"],
                        dtype=np.float64,
                    )
                    for slide in train_slides
                ]
            )
            test_source = by_slide[held_slide]
            test_xyz = np.asarray(
                target_table[target_set][held_slide]["xyz"],
                dtype=np.float64,
            )
            test_lab = np.asarray(
                target_table[target_set][held_slide]["lab"],
                dtype=np.float64,
            )
            linear_matrix = _fit_positive_matrix(
                train_source,
                train_xyz,
                candidate["matrix_coefficient_interval"],
            )
            power, power_matrix, development_loss = _select_common_power(
                train_source,
                train_xyz,
                candidate["common_power_grid"],
                candidate["matrix_coefficient_interval"],
            )
            predictions = {
                "identity-device-rgb-as-xyz-control": test_source,
                "nonnegative-bounded-3x3": test_source @ linear_matrix.T,
                "common-power-plus-nonnegative-bounded-3x3": (
                    np.power(test_source, power, dtype=np.float64)
                    @ power_matrix.T
                ),
            }
            metrics: dict[str, Any] = {}
            for name, prediction in predictions.items():
                summary, errors = _metrics(prediction, test_xyz, test_lab)
                metrics[name] = summary
                pooled_errors[name].append(errors)
                pipeline_errors[role][name].append(errors)
                target_errors[target_set][name].append(errors)
                output_chunks[name].append(prediction)
            matrix_coefficients.extend((linear_matrix, power_matrix))
            selected_powers.append(power)
            fold_records.append(
                {
                    "role": role,
                    "target_set": target_set,
                    "held_slide": held_slide,
                    "selected_power": power,
                    "development_mean_xyz_l2": development_loss,
                    "linear_matrix": linear_matrix.tolist(),
                    "power_matrix": power_matrix.tolist(),
                    "models": metrics,
                }
            )

    aggregate = {
        name: {
            "median_delta_e76": float(np.median(np.concatenate(chunks))),
            "p95_delta_e76": float(
                np.percentile(np.concatenate(chunks), 95.0)
            ),
            "maximum_delta_e76": float(np.max(np.concatenate(chunks))),
            "patch_predictions": int(np.concatenate(chunks).size),
        }
        for name, chunks in pooled_errors.items()
    }
    per_pipeline = {
        role: {
            name: {
                "median_delta_e76": float(np.median(np.concatenate(chunks))),
                "p95_delta_e76": float(
                    np.percentile(np.concatenate(chunks), 95.0)
                ),
            }
            for name, chunks in models.items()
        }
        for role, models in pipeline_errors.items()
    }
    per_target_set = {
        str(target_set): {
            name: {
                "median_delta_e76": float(np.median(np.concatenate(chunks))),
                "p95_delta_e76": float(
                    np.percentile(np.concatenate(chunks), 95.0)
                ),
            }
            for name, chunks in models.items()
        }
        for target_set, models in target_errors.items()
    }
    output_extrema = {
        name: {
            "minimum": float(np.min(np.concatenate(chunks, axis=0))),
            "maximum": float(np.max(np.concatenate(chunks, axis=0))),
        }
        for name, chunks in output_chunks.items()
    }
    coefficient_stack = np.stack(matrix_coefficients)
    power_counts = {
        str(power): selected_powers.count(power)
        for power in sorted(set(selected_powers))
    }
    primary = "common-power-plus-nonnegative-bounded-3x3"
    identity = "identity-device-rgb-as-xyz-control"
    linear = "nonnegative-bounded-3x3"
    identity_median = aggregate[identity]["median_delta_e76"]
    primary_median = aggregate[primary]["median_delta_e76"]
    improvement = (
        (identity_median - primary_median) / identity_median
        if identity_median > 0.0
        else 0.0
    )
    gates = contract["automatic_gates"]
    checks = {
        "pipeline_count": len(scan_bank) == int(gates["expected_pipelines"]),
        "alignment_cell_count": len(alignment_records)
        == int(gates["expected_alignment_cells"]),
        "all_alignment_gates_pass": len(alignment_records)
        == int(gates["expected_alignment_cells"]),
        "source_patch_index_exact": source_patch_max_l2
        <= float(gates["source_patch_to_pair_table_rgb_max_l2"]),
        "published_xyz_lab_consistency": maximum_published_delta
        <= float(gates["published_xyz_lab_consistency_max_delta_e76"]),
        "candidate_xyz_domain": all(
            interval["minimum"] >= float(gates["candidate_xyz_minimum"]) - 1e-15
            and interval["maximum"]
            <= float(gates["candidate_xyz_maximum"]) + 1e-15
            for name, interval in output_extrema.items()
            if name != identity
        ),
        "matrix_coefficient_domain": bool(
            np.min(coefficient_stack)
            >= float(candidate["matrix_coefficient_interval"][0]) - 1e-15
            and np.max(coefficient_stack)
            <= float(candidate["matrix_coefficient_interval"][1]) + 1e-15
        ),
        "primary_relative_improvement": improvement
        >= float(gates["common_power_vs_identity_median_improvement_min"]),
        "primary_aggregate_median": primary_median
        <= float(gates["common_power_aggregate_median_delta_e76_max"]),
        "primary_aggregate_p95": aggregate[primary]["p95_delta_e76"]
        <= float(gates["common_power_aggregate_p95_delta_e76_max"]),
        "primary_each_pipeline_median": all(
            row[primary]["median_delta_e76"]
            <= float(
                gates["common_power_each_pipeline_median_delta_e76_max"]
            )
            for row in per_pipeline.values()
        ),
        "primary_each_target_set_median": all(
            row[primary]["median_delta_e76"]
            <= float(
                gates["common_power_each_target_set_median_delta_e76_max"]
            )
            for row in per_target_set.values()
        ),
        "primary_not_worse_than_linear": primary_median
        <= aggregate[linear]["median_delta_e76"] + 1e-15,
    }
    stable_payload = {
        "schema": "neuro_film.u6_p6i_measured_scanner_characterization_report.v1",
        "node": contract["node"],
        "config_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "alignment_retry_sha256": hashlib.sha256(retry_bytes).hexdigest(),
        "parent_hashes_verified": {
            hash_key: parents[hash_key]
            for _, hash_key in (
                ("p6h_decision_path", "p6h_decision_sha256"),
                ("p6h_report_path", "p6h_report_sha256"),
                ("aq1_report_path", "aq1_report_sha256"),
                ("pair_table_path", "pair_table_sha256"),
                ("alignment_contract_path", "alignment_contract_sha256"),
            )
        },
        "support": {
            "pipelines": len(scan_bank),
            "target_sets": target_sets,
            "slides_per_pipeline": 5,
            "alignment_cells": len(alignment_records),
            "patches_per_slide": int(
                measurement["colour_patches_per_slide"]
            ),
            "scanner_target_graph_connected_across_sets": False,
        },
        "alignment_summary": {
            "minimum_ransac_inliers": min(
                int(row["ransac_inliers"]) for row in alignment_records
            ),
            "maximum_median_inlier_reprojection_error_px": max(
                float(row["median_inlier_reprojection_error_px"])
                for row in alignment_records
            ),
            "retry_cells": sum(
                row["alignment_attempt"] != "primary"
                for row in alignment_records
            ),
        },
        "reference_consistency": {
            "source_patch_max_l2": source_patch_max_l2,
            "published_xyz_to_lab_max_delta_e76": maximum_published_delta,
        },
        "aggregate": aggregate,
        "per_pipeline": per_pipeline,
        "per_target_set": per_target_set,
        "output_extrema": output_extrema,
        "fit_diagnostics": {
            "minimum_matrix_coefficient": float(np.min(coefficient_stack)),
            "maximum_matrix_coefficient": float(np.max(coefficient_stack)),
            "selected_common_power_counts": power_counts,
            "fold_fits": len(fold_records),
        },
        "primary_median_improvement_fraction": improvement,
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "claim_ceiling": contract["claim_ceiling"],
        "forbidden_claims": contract["forbidden_claims"],
    }
    return {
        **stable_payload,
        "stable_evidence_id": _stable_id(stable_payload),
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "alignment_records": alignment_records,
        "fold_records": fold_records,
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "_fit_positive_matrix",
    "_load_target_table",
    "_select_common_power",
    "_slide_member",
    "evaluate_measured_scanner_characterization",
    "load_contract",
    "write_report",
]
