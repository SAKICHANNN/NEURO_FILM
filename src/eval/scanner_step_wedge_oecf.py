"""Measured scanner OECF discrimination from exact step-wedge scans."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import tifffile

SCHEMA = "neuro_film.u6_p6aj_uchicago_step_wedge_oecf_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6aj_uchicago_step_wedge_oecf_report.v1"


class StepWedgeOecfError(RuntimeError):
    """Raised when the frozen source or measurement contract is violated."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise StepWedgeOecfError("unsupported step-wedge contract")
    files = payload["acquisition"]["files"]
    if (
        len(files) != 3
        or sum(int(row["bytes"]) for row in files)
        != int(payload["acquisition"]["maximum_total_bytes"])
        or len(payload["density_values"]) != int(payload["measurement"]["step_count"])
        or set(payload["measurement"]["development_step_indices_zero_based"])
        & set(payload["measurement"]["confirmation_step_indices_zero_based"])
    ):
        raise StepWedgeOecfError("step-wedge contract drift")
    for row in files:
        parsed = urlparse(row["url"])
        if parsed.scheme != "https" or parsed.hostname != "knowledge.uchicago.edu":
            raise StepWedgeOecfError("source URL escaped the frozen host")
    return payload


def acquire(config: dict[str, Any], root: Path) -> Path:
    destination = root / config["acquisition"]["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    for row in config["acquisition"]["files"]:
        target = destination / row["name"]
        if target.is_file() and target.stat().st_size == int(row["bytes"]):
            if hash_file(target, "md5") == row["md5"]:
                continue
            raise StepWedgeOecfError("existing source hash mismatch")
        if target.exists():
            raise StepWedgeOecfError("existing source entry is invalid")
        request = Request(row["url"], headers={"User-Agent": "neuro-film-p6aj/1"})
        part = target.with_suffix(target.suffix + ".part")
        with urlopen(request, timeout=60) as response, part.open("xb") as stream:
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != "knowledge.uchicago.edu":
                raise StepWedgeOecfError("source redirected off frozen host")
            written = 0
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > int(row["bytes"]):
                    raise StepWedgeOecfError("source exceeded frozen size")
                stream.write(chunk)
        if written != int(row["bytes"]) or hash_file(part, "md5") != row["md5"]:
            part.unlink(missing_ok=True)
            raise StepWedgeOecfError("source integrity mismatch")
        part.replace(target)
    return destination


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise StepWedgeOecfError("invalid rank-correlation inputs")
    left_ranks = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(
        np.float64
    )
    right_ranks = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(
        np.float64
    )
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _axis_values(
    image: np.ndarray, axis: int, count: int, trim_fraction: float
) -> np.ndarray:
    length = image.shape[axis]
    values = []
    for index in range(count):
        lo = round(index * length / count)
        hi = round((index + 1) * length / count)
        trim = round((hi - lo) * trim_fraction)
        if hi - lo - 2 * trim < 1:
            raise StepWedgeOecfError("step bin collapsed")
        region = (
            image[lo + trim : hi - trim, :]
            if axis == 0
            else image[:, lo + trim : hi - trim]
        )
        values.append(float(np.median(region)) / 65535.0)
    return np.asarray(values, dtype=np.float64)


def _measure_wedge(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    with tifffile.TiffFile(path) as document:
        if len(document.pages) != 1:
            raise StepWedgeOecfError("wedge must be single frame")
        image = document.asarray()
    if image.ndim != 2 or image.dtype != np.uint16:
        raise StepWedgeOecfError("wedge must be grayscale uint16")
    count = int(config["measurement"]["step_count"])
    trim = float(config["measurement"]["trim_fraction"])
    rows = _axis_values(image, 0, count, trim)
    columns = _axis_values(image, 1, count, trim)
    row_range = float(np.ptp(rows))
    column_range = float(np.ptp(columns))
    if row_range >= column_range:
        values, axis, ratio = rows, "rows", row_range / max(column_range, 1e-15)
    else:
        values, axis, ratio = columns, "columns", column_range / max(row_range, 1e-15)
    densities = np.asarray(config["density_values"], dtype=np.float64)
    direct = _rank_correlation(densities, values)
    reverse = _rank_correlation(densities, values[::-1])
    reversed_order = reverse < direct
    if reversed_order:
        values = values[::-1].copy()
    correlation = min(direct, reverse)
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "minimum_code": int(np.min(image)),
        "maximum_code": int(np.max(image)),
        "step_axis": axis,
        "axis_range_ratio": ratio,
        "reversed": reversed_order,
        "density_code_spearman": correlation,
        "normalized_codes": values.tolist(),
    }


def _fit_affine(x: np.ndarray, y: np.ndarray, development: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(development.size), x[development]])
    coefficients = np.linalg.lstsq(design, y[development], rcond=None)[0]
    return coefficients[0] + coefficients[1] * x


def _score(
    prediction: np.ndarray, truth: np.ndarray, confirmation: np.ndarray
) -> dict[str, float]:
    errors = prediction[confirmation] - truth[confirmation]
    return {
        "confirmation_rmse": float(math.sqrt(float(np.mean(errors * errors)))),
        "confirmation_max_abs_error": float(np.max(np.abs(errors))),
    }


def evaluate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source = root / config["acquisition"]["destination"]
    sources = []
    measurements = []
    for row in config["acquisition"]["files"]:
        path = source / row["name"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or hash_file(path, "md5") != row["md5"]
        ):
            raise StepWedgeOecfError("local source integrity mismatch")
        sources.append(
            {
                "name": row["name"],
                "role": row["role"],
                "bytes": path.stat().st_size,
                "md5": row["md5"],
                "sha256": hash_file(path),
            }
        )
        if row["role"].endswith("step_wedge"):
            measured = _measure_wedge(path, config)
            measured["role"] = row["role"]
            measurements.append(measured)

    densities = np.asarray(config["density_values"], dtype=np.float64)
    development = np.asarray(
        config["measurement"]["development_step_indices_zero_based"], dtype=np.int64
    )
    confirmation = np.asarray(
        config["measurement"]["confirmation_step_indices_zero_based"], dtype=np.int64
    )
    gates = config["gates"]
    rows = []
    best_families = []
    for measured in measurements:
        truth = np.asarray(measured["normalized_codes"], dtype=np.float64)
        transmittance = np.power(10.0, -densities)
        predictions = {
            "affine_code_from_transmittance_10_pow_minus_density": _fit_affine(
                transmittance, truth, development
            ),
            "affine_code_from_optical_density": _fit_affine(
                densities, truth, development
            ),
        }
        if not np.all(np.diff(truth[development]) < 0.0):
            raise StepWedgeOecfError("development wedge codes are not monotone")
        predictions["monotone_piecewise_linear_code_from_density"] = np.interp(
            densities, densities[development], truth[development]
        )
        scores = {
            name: _score(prediction, truth, confirmation)
            for name, prediction in predictions.items()
        }
        parametric = [
            "affine_code_from_transmittance_10_pow_minus_density",
            "affine_code_from_optical_density",
        ]
        best = min(parametric, key=lambda name: scores[name]["confirmation_rmse"])
        other = next(name for name in parametric if name != best)
        best_families.append(best)
        rows.append(
            {
                "role": measured["role"],
                "measurement": measured,
                "scores": scores,
                "best_parametric_family": best,
                "best_parametric_rmse_ratio_vs_other": scores[best]["confirmation_rmse"]
                / scores[other]["confirmation_rmse"],
                "best_parametric_max_error_ratio_vs_other": scores[best][
                    "confirmation_max_abs_error"
                ]
                / scores[other]["confirmation_max_abs_error"],
                "monotone_rmse_ratio_vs_best_parametric": scores[
                    "monotone_piecewise_linear_code_from_density"
                ]["confirmation_rmse"]
                / scores[best]["confirmation_rmse"],
            }
        )

    source_gate = all(
        row["measurement"]["axis_range_ratio"]
        >= float(gates["axis_range_ratio_minimum"])
        and abs(row["measurement"]["density_code_spearman"])
        >= float(gates["absolute_step_spearman_minimum"])
        for row in rows
    )
    same_parametric = len(set(best_families)) == 1
    parametric_pass = (
        source_gate
        and same_parametric
        and all(
            row["best_parametric_rmse_ratio_vs_other"]
            <= float(gates["best_parametric_confirmation_rmse_ratio_vs_other_maximum"])
            and row["best_parametric_max_error_ratio_vs_other"]
            <= float(
                gates["best_parametric_confirmation_max_error_ratio_vs_other_maximum"]
            )
            for row in rows
        )
    )
    monotone_pass = source_gate and all(
        row["monotone_rmse_ratio_vs_best_parametric"]
        <= float(
            gates[
                "monotone_calibrator_confirmation_rmse_ratio_vs_best_parametric_maximum"
            ]
        )
        for row in rows
    )
    if parametric_pass:
        status = "pass-parametric-scanner-code-domain"
        decision = "retain_exact_workflow_parametric_scanner_code_domain"
    elif monotone_pass:
        status = "pass-monotone-only-scanner-oecf-required"
        decision = "require_scanner_specific_monotone_oecf"
    else:
        status = "fail-closed-step-wedge-oecf-unidentified"
        decision = "close_exact_step_wedge_oecf_route"
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": status,
        "decision": decision,
        "source_files": sources,
        "wedge_results": rows,
        "gates": {
            "source_gate": source_gate,
            "same_best_parametric_family": same_parametric,
            "parametric_pass": parametric_pass,
            "monotone_only_pass": monotone_pass and not parametric_pass,
        },
        "claim_ceiling": config["claim_ceiling"],
    }
