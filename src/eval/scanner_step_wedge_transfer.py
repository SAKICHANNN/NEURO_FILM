"""Fixed-parameter transfer diagnostic for a second scanner step wedge."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from src.eval.scanner_log_oecf_profile import canonical_json, hash_file

SCHEMA = "neuro_film.u6_p6am_barnard_step_wedge_transfer_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6am_barnard_step_wedge_transfer_report.v1"


class ScannerOecfTransferError(RuntimeError):
    """Raised when the fixed transfer diagnostic cannot be executed exactly."""


def load_contract(path: Path, root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ScannerOecfTransferError("unsupported scanner OECF transfer contract")
    source = payload["source"]
    for path_key, hash_key in (
        ("tiff_path", "tiff_sha256"),
        ("density_table_path", "density_table_sha256"),
        ("paper_path", "paper_sha256"),
    ):
        if hash_file(root / source[path_key]) != source[hash_key]:
            raise ScannerOecfTransferError(f"source binding mismatch: {path_key}")
    binding = payload["p6al_binding"]
    for path_key, hash_key in (
        ("profile_contract_path", "profile_contract_sha256"),
        ("evidence_path", "evidence_sha256"),
    ):
        if hash_file(root / binding[path_key]) != binding[hash_key]:
            raise ScannerOecfTransferError(f"P6AL binding mismatch: {path_key}")
    measurement = payload["measurement"]
    boundaries = measurement["step_column_boundaries"]
    densities = measurement["density_values"]
    valid = measurement["paper_stated_valid_indices_zero_based"]
    if (
        len(boundaries) != len(densities) + 1
        or boundaries != sorted(set(boundaries))
        or valid != list(range(2, 15))
        or measurement["pixel_reduction"]
        != "arithmetic mean over all three RGB channels and all retained plateau pixels"
    ):
        raise ScannerOecfTransferError("scanner OECF transfer measurement drift")
    return payload


def _plateau_codes(path: Path, config: dict[str, Any]) -> np.ndarray:
    measurement = config["measurement"]
    with tifffile.TiffFile(path) as document:
        if len(document.pages) != int(measurement["required_page_count"]):
            raise ScannerOecfTransferError("TIFF page count drift")
        page = document.pages[int(measurement["page_index"])]
        if list(page.shape) != measurement["required_shape"]:
            raise ScannerOecfTransferError("TIFF shape drift")
        if page.dtype != np.dtype(np.uint16) or document.byteorder != ">":
            raise ScannerOecfTransferError("TIFF dtype drift")
    image = tifffile.memmap(path, page=int(measurement["page_index"]))
    row_start = int(measurement["row_start_inclusive"])
    row_end = int(measurement["row_end_exclusive"])
    inner_fraction = float(measurement["plateau_inner_fraction"])
    codes = []
    for lo, hi in pairwise(measurement["step_column_boundaries"]):
        margin = round((hi - lo) * (1.0 - inner_fraction) / 2.0)
        region = image[row_start:row_end, lo + margin : hi - margin, :]
        if region.size == 0:
            raise ScannerOecfTransferError("step-wedge plateau collapsed")
        codes.append(float(np.mean(region, dtype=np.float64)))
    result = np.asarray(codes, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ScannerOecfTransferError("non-finite step-wedge code")
    return result


def _density(codes: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    black = float(candidate["black_code"])
    gamma = float(candidate["gamma"])
    shifted = codes - black
    if not math.isfinite(black) or not math.isfinite(gamma):
        raise ScannerOecfTransferError("non-finite fixed candidate")
    if gamma <= 0.0 or not np.all(shifted > 0.0):
        raise ScannerOecfTransferError("fixed candidate left logarithmic domain")
    return gamma * np.log10(65535.0 / shifted)


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    return {
        "rmse": float(math.sqrt(float(np.mean(error * error)))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def evaluate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source = config["source"]
    codes = _plateau_codes(root / source["tiff_path"], config)
    truth = np.asarray(config["measurement"]["density_values"], dtype=np.float64)
    valid = np.asarray(
        config["measurement"]["paper_stated_valid_indices_zero_based"],
        dtype=np.int64,
    )
    candidates = config["fixed_candidates"]
    p6al_prediction = _density(codes, candidates["p6al_exact_profile"])
    local_prediction = _density(codes, candidates["barnard_paper_blue_equation"])
    p6al_metrics = _metrics(p6al_prediction[valid], truth[valid])
    local_metrics = _metrics(local_prediction[valid], truth[valid])
    gates = config["gates"]
    local_pass = (
        local_metrics["rmse"]
        <= float(gates["barnard_paper_valid_density_rmse_maximum"])
        and local_metrics["maximum_absolute_error"]
        <= float(gates["barnard_paper_valid_density_max_abs_error_maximum"])
    )
    p6al_pass = (
        p6al_metrics["rmse"]
        <= float(gates["p6al_transfer_density_rmse_maximum"])
        and p6al_metrics["maximum_absolute_error"]
        <= float(gates["p6al_transfer_density_max_abs_error_maximum"])
    )
    valid_monotone = bool(np.all(np.diff(codes[valid]) < 0.0))
    highest_density_failure = bool(codes[-1] >= codes[-2])
    if not valid_monotone or not highest_density_failure:
        raise ScannerOecfTransferError("fixed source-domain controls failed")
    if local_pass and p6al_pass:
        status = "pass-fixed-parameter-transfer-diagnostic"
        decision = "open_prospective_third_workflow_confirmation"
    elif local_pass:
        status = "pass-local-equation-reject-p6al-parameter-transfer"
        decision = "retain_per_workflow_typed_scanner_oecf_profiles"
    else:
        status = "fail-closed-barnard-source-extraction"
        decision = "close_barnard_step_wedge_transfer_diagnostic"
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "mode": config["mode"],
        "status": status,
        "decision": decision,
        "source": {
            "tiff_sha256": hash_file(root / source["tiff_path"]),
            "density_table_sha256": hash_file(root / source["density_table_path"]),
            "paper_sha256": hash_file(root / source["paper_path"]),
            "plateau_codes": codes.tolist(),
        },
        "valid_density_indices_zero_based": valid.tolist(),
        "p6al_exact_profile": {
            **candidates["p6al_exact_profile"],
            "valid_density_metrics": p6al_metrics,
            "passes_transfer_gate": p6al_pass,
        },
        "barnard_paper_blue_equation": {
            **candidates["barnard_paper_blue_equation"],
            "valid_density_metrics": local_metrics,
            "passes_local_positive_control": local_pass,
        },
        "controls": {
            "valid_range_codes_strictly_decreasing": valid_monotone,
            "highest_density_scanner_reversal_observed": highest_density_failure,
            "parameters_fit_on_10b_161": False,
        },
        "claim_ceiling": config["claim_ceiling"],
    }


def stable_id(report: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(report)).hexdigest()


__all__ = [
    "ScannerOecfTransferError",
    "evaluate",
    "load_contract",
    "stable_id",
]
