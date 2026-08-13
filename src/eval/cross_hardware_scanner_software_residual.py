"""Cross-hardware test of one shared scanner-software residual."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.optimize import lsq_linear

from src.eval.physical_measured_scanner_characterization import (
    _sha256,
    _slide_member,
)
from src.film_physics.scanner import apply_scanner_safe_residual
from src.real_film.scanner_nuisance import (
    ScannerNuisanceError,
    _native_rgb,
    align_source_to_scan,
    patch_medians,
)

SCHEMA = "neuro-film.u6-p6aq-cross-hardware-scanner-software-residual-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p6aq-cross-hardware-scanner-software-residual-result.v1"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    pairs = value.get("hardware_pairs", [])
    roles = [
        role
        for row in pairs
        for role in (row.get("source_role"), row.get("target_role"))
    ]
    if (
        value.get("schema") != SCHEMA
        or len(pairs) != 4
        or len({row.get("id") for row in pairs}) != 4
        or len(set(roles)) != 8
        or value.get("candidate", {}).get("hard_clipping_allowed") is not False
        or value.get("candidate", {}).get("held_hardware_pixels_allowed_in_fit")
        is not False
    ):
        raise ValueError("unsupported P6AQ contract")
    return value


def _fit(
    source: np.ndarray, target: np.ndarray, candidate: Mapping[str, Any]
) -> tuple[float, np.ndarray, np.ndarray]:
    lo, hi = map(float, candidate["coefficient_interval"])
    blo, bhi = map(float, candidate["bias_interval"])
    fits = []
    for power in candidate["power_grid"]:
        transformed = np.power(source, float(power), dtype=np.float64)
        design = np.column_stack((transformed, np.ones(len(transformed))))
        matrix = np.empty((3, 3), dtype=np.float64)
        bias = np.empty(3, dtype=np.float64)
        for channel in range(3):
            result = lsq_linear(
                design, target[:, channel], bounds=([lo] * 3 + [blo], [hi] * 3 + [bhi])
            )
            if not result.success:
                raise ValueError("P6AQ affine fit failed")
            matrix[channel] = result.x[:3]
            bias[channel] = result.x[3]
        prediction = transformed @ matrix.T + bias
        loss = float(np.mean(np.linalg.norm(prediction - target, axis=1)))
        fits.append((loss, float(power), matrix, bias))
    _, power, matrix, bias = min(fits, key=lambda row: (row[0], row[1]))
    return power, matrix, bias


def _apply(
    source: np.ndarray, model: tuple[float, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, dict[str, float]]:
    power, matrix, bias = model
    raw = np.power(source, power, dtype=np.float64) @ matrix.T + bias
    output, receipt = apply_scanner_safe_residual(source, raw)
    return output, {
        "limited_fraction": receipt.limited_pixel_fraction,
        "median_scale": receipt.median_scale,
        "minimum_scale": receipt.minimum_scale,
        "direction_error": receipt.maximum_collinearity_error,
    }


def _load_bank(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[int, np.ndarray]]:
    parents = contract["parents"]
    p6h = json.loads(
        (root / parents["p6h_contract"]["path"]).read_text(encoding="utf-8")
    )
    report = json.loads(
        (root / parents["p6h_report"]["path"]).read_text(encoding="utf-8")
    )
    p6i = json.loads(
        (root / parents["p6i_contract"]["path"]).read_text(encoding="utf-8")
    )
    retry = json.loads(
        (root / parents["alignment_retry"]["path"]).read_text(encoding="utf-8")
    )
    alignment_contract = json.loads(
        (root / p6i["parents"]["alignment_contract_path"]).read_text(encoding="utf-8")
    )
    alignment = dict(alignment_contract["alignment"])
    retry_alignment = dict(alignment)
    retry_alignment["ratio_test"] = float(retry["retry"]["ratio_test"])
    grid = alignment_contract["source_grid"]
    source_images = {}
    for row in p6i["source_images"]:
        path = root / row["path"]
        if _sha256(path) != row["sha256"]:
            raise ValueError("P6AQ source image drift")
        source_images[int(row["slide"])] = _native_rgb(path.read_bytes())
    assets = {row["role"]: row for row in report["assets"]}
    wanted = {
        role
        for pair in contract["hardware_pairs"]
        for role in (pair["source_role"], pair["target_role"])
    }
    data_root = root / p6h["acquisition"]["data_root"]
    bank: dict[str, dict[int, np.ndarray]] = {}
    for row in p6h["acquisition"]["archives"]:
        role = row["role"]
        if role not in wanted:
            continue
        asset = assets[role]
        archive_path = data_root / asset["path"]
        if _sha256(archive_path) != asset["sha256"]:
            raise ValueError("P6AQ archive drift")
        target_set = int(row["target_set"])
        bank[role] = {}
        with ZipFile(archive_path) as archive:
            for slide in range(1, 6):
                member = _slide_member(archive.namelist(), target_set, slide)
                scan = _native_rgb(archive.read(member))
                try:
                    homography, _ = align_source_to_scan(
                        source_images[slide], scan, alignment
                    )
                except ScannerNuisanceError:
                    homography, _ = align_source_to_scan(
                        source_images[slide], scan, retry_alignment
                    )
                bank[role][slide] = patch_medians(scan, homography, grid)
    if set(bank) != wanted or any(len(value) != 5 for value in bank.values()):
        raise ValueError("P6AQ scan bank incomplete")
    return bank


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha256(path) != binding["sha256"]:
            raise ValueError("P6AQ parent drift")
        if "required_decision" in binding:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("decision") != binding["required_decision"]:
                raise ValueError("P6AQ parent decision drift")
    bank = _load_bank(root, contract)
    pairs = list(contract["hardware_pairs"])
    candidate = contract["candidate"]
    all_identity, all_shared, all_wrong = [], [], []
    folds, receipts = [], []
    for index, held in enumerate(pairs):
        train_pairs = [row for row in pairs if row is not held]
        train_source = np.concatenate(
            [
                bank[row["source_role"]][slide]
                for row in train_pairs
                for slide in range(1, 6)
            ]
        )
        train_target = np.concatenate(
            [
                bank[row["target_role"]][slide]
                for row in train_pairs
                for slide in range(1, 6)
            ]
        )
        shared_model = _fit(train_source, train_target, candidate)
        wrong_pair = pairs[(index + 1) % len(pairs)]
        wrong_source = np.concatenate(
            [bank[wrong_pair["source_role"]][slide] for slide in range(1, 6)]
        )
        wrong_target = np.concatenate(
            [bank[wrong_pair["target_role"]][slide] for slide in range(1, 6)]
        )
        wrong_model = _fit(wrong_source, wrong_target, candidate)
        source = np.concatenate(
            [bank[held["source_role"]][slide] for slide in range(1, 6)]
        )
        target = np.concatenate(
            [bank[held["target_role"]][slide] for slide in range(1, 6)]
        )
        shared, receipt = _apply(source, shared_model)
        wrong, _ = _apply(source, wrong_model)
        identity_error = np.linalg.norm(source - target, axis=1)
        shared_error = np.linalg.norm(shared - target, axis=1)
        wrong_error = np.linalg.norm(wrong - target, axis=1)
        all_identity.append(identity_error)
        all_shared.append(shared_error)
        all_wrong.append(wrong_error)
        receipts.append(receipt)
        identity_median = float(np.median(identity_error))
        shared_median = float(np.median(shared_error))
        wrong_median = float(np.median(wrong_error))
        folds.append(
            {
                "held_hardware": held["id"],
                "wrong_operator_hardware": wrong_pair["id"],
                "patches": len(source),
                "selected_power": shared_model[0],
                "identity_median_l2": identity_median,
                "shared_median_l2": shared_median,
                "wrong_median_l2": wrong_median,
                "improvement_over_identity_fraction": 1.0
                - shared_median / identity_median,
                "improvement_over_wrong_fraction": 1.0 - shared_median / wrong_median,
                **receipt,
            }
        )
    identity = np.concatenate(all_identity)
    shared = np.concatenate(all_shared)
    wrong = np.concatenate(all_wrong)
    metrics = {
        "hardware_folds": len(folds),
        "patch_predictions": len(shared),
        "identity_median_l2": float(np.median(identity)),
        "shared_median_l2": float(np.median(shared)),
        "cyclic_wrong_median_l2": float(np.median(wrong)),
        "aggregate_median_improvement_over_identity_fraction": 1.0
        - float(np.median(shared)) / float(np.median(identity)),
        "aggregate_median_improvement_over_cyclic_wrong_fraction": 1.0
        - float(np.median(shared)) / float(np.median(wrong)),
        "hardware_wins_over_cyclic_wrong": sum(
            row["improvement_over_wrong_fraction"] > 0 for row in folds
        ),
        "out_of_cube_fraction": 0.0,
        "limited_patch_fraction": float(
            np.mean([row["limited_fraction"] for row in folds])
        ),
        "median_safe_scale": float(np.median([row["median_scale"] for row in folds])),
        "maximum_residual_direction_error": max(
            row["direction_error"] for row in folds
        ),
    }
    gates = contract["gates"]
    checks = {
        "folds": metrics["hardware_folds"] == gates["required_hardware_folds"],
        "support": metrics["patch_predictions"] == gates["required_patch_predictions"],
        "every_hardware_improves_identity": min(
            row["improvement_over_identity_fraction"] for row in folds
        )
        >= gates["minimum_each_hardware_improvement_over_identity_fraction"],
        "identity_gain": metrics["aggregate_median_improvement_over_identity_fraction"]
        >= gates["minimum_aggregate_median_improvement_over_identity_fraction"],
        "wrong_wins": metrics["hardware_wins_over_cyclic_wrong"]
        >= gates["minimum_hardware_wins_over_cyclic_wrong_operator"],
        "wrong_gain": metrics["aggregate_median_improvement_over_cyclic_wrong_fraction"]
        >= gates["minimum_aggregate_median_improvement_over_cyclic_wrong_fraction"],
        "cube": metrics["out_of_cube_fraction"]
        <= gates["maximum_out_of_cube_fraction"],
        "limited": metrics["limited_patch_fraction"]
        <= gates["maximum_limited_patch_fraction"],
        "scale": metrics["median_safe_scale"] >= gates["minimum_median_safe_scale"],
        "direction": metrics["maximum_residual_direction_error"]
        <= gates["maximum_residual_direction_error"],
        "finite": all(np.isfinite(value) for value in metrics.values()),
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "folds": folds,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
