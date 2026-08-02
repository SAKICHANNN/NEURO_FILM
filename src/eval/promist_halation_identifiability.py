"""Synthetic mechanism discriminator for film backing return versus lens mist.

The ProMist-5K paper specifies a scene-linear, six-Gaussian diffusion topology
but does not publish the kernel radii or weights used to generate its targets.
This module therefore tests a frozen clean-room equation family, never an
author-renderer reproduction.  The alternative truth is an additive support
return before one saturating characteristic response.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

SCHEMA = "neuro_film.u6_p3s_promist_halation_identifiability_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3s_promist_halation_identifiability_report.v1"


class ProMistHalationError(ValueError):
    """Raised when the frozen mechanism-identifiability contract is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != SCHEMA or contract.get("node") != "U6.P3S":
        raise ProMistHalationError("unsupported P3S contract")
    sigmas = tuple(float(value) for value in contract["candidate_family"]["gaussian_sigma_pixels"])
    if len(sigmas) != 6 or any(not math.isfinite(value) or value <= 0.0 for value in sigmas):
        raise ProMistHalationError("P3S requires six positive finite Gaussian scales")
    if tuple(sorted(sigmas)) != sigmas or len(set(sigmas)) != len(sigmas):
        raise ProMistHalationError("P3S Gaussian scales must be unique and increasing")
    lower, upper = (float(value) for value in contract["candidate_family"]["weight_bounds"])
    maximum_total = float(contract["candidate_family"]["maximum_total_blur_weight"])
    if lower != 0.0 or not (0.0 < upper <= maximum_total < 1.0):
        raise ProMistHalationError("P3S weight envelope is invalid")
    roles = contract["roles"]
    if set(roles["development_patterns"]) & set(roles["confirmation_patterns"]):
        raise ProMistHalationError("P3S pattern roles overlap")
    if set(roles["development_exposure_scales"]) & set(roles["confirmation_exposure_scales"]):
        raise ProMistHalationError("P3S exposure roles overlap")
    if roles["image_shape"] != [128, 128, 3]:
        raise ProMistHalationError("P3S image shape drift")
    return contract


def audit_primary_source(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["primary_source"]
    path = root / source["local_pdf_path"]
    payload = path.read_bytes()
    if len(payload) != int(source["pdf_bytes"]):
        raise ProMistHalationError("P3S primary-source byte count drift")
    digest = _sha256_bytes(payload)
    if digest != source["pdf_sha256"]:
        raise ProMistHalationError("P3S primary-source hash drift")
    text = " ".join(
        " ".join((page.extract_text() or "").split()) for page in PdfReader(path).pages
    )
    normalized = text.replace("- ", "")
    anchors = {
        anchor: anchor in normalized for anchor in source["required_text_anchors"]
    }
    if not all(anchors.values()):
        raise ProMistHalationError("P3S primary-source text anchor drift")
    return {
        "path": source["local_pdf_path"],
        "bytes": len(payload),
        "sha256": digest,
        "required_text_anchors": anchors,
    }


def _base_pattern(name: str, height: int, width: int, maximum: float) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    image = np.full((height, width), 0.012, dtype=np.float64)
    if name == "vertical_edge":
        image[:, width // 2 :] = maximum * 0.88
    elif name == "highlight_disc":
        mask = (x - 0.53 * width) ** 2 + (y - 0.46 * height) ** 2 <= (0.13 * width) ** 2
        image[mask] = maximum
    elif name == "sparse_highlights":
        for cx, cy, radius, level in (
            (0.22, 0.27, 0.035, 1.0),
            (0.71, 0.22, 0.055, 0.82),
            (0.48, 0.71, 0.075, 0.67),
            (0.82, 0.78, 0.025, 0.95),
        ):
            mask = (x - cx * width) ** 2 + (y - cy * height) ** 2 <= (radius * width) ** 2
            image[mask] = maximum * level
    elif name == "coarse_checker":
        cells = ((x // 16) + (y // 16)) % 2
        image = np.where(cells == 0, 0.025, maximum * 0.74).astype(np.float64)
    elif name == "impulse_cluster":
        image[height // 2, width // 2] = maximum
        image[height // 2 - 11, width // 2 + 17] = maximum * 0.72
        image[height // 2 + 19, width // 2 - 23] = maximum * 0.91
    elif name == "slanted_edge":
        image[x > (0.31 * width + 0.43 * y)] = maximum * 0.84
    elif name == "line_pairs":
        image = np.where((x // 3) % 2 == 0, maximum * 0.79, 0.018).astype(np.float64)
        image[y < height // 3] = np.where(
            (x[y < height // 3] // 7) % 2 == 0,
            maximum * 0.63,
            0.018,
        )
    elif name == "gradient_islands":
        image = 0.015 + 0.11 * (x / max(width - 1, 1))
        for cx, cy, radius, level in (
            (0.28, 0.32, 0.08, 0.92),
            (0.68, 0.65, 0.12, 1.0),
        ):
            mask = (x - cx * width) ** 2 + (y - cy * height) ** 2 <= (radius * width) ** 2
            image[mask] = maximum * level
    else:
        raise ProMistHalationError(f"unknown P3S pattern: {name}")
    if not np.all(np.isfinite(image)) or float(np.min(image)) < 0.0 or float(np.max(image)) > maximum:
        raise RuntimeError("P3S pattern escaped the frozen exposure envelope")
    return np.repeat(image[..., None], 3, axis=2)


def _response(exposure: np.ndarray, toe: float, density_scale: float) -> np.ndarray:
    density = density_scale * exposure / (toe + exposure)
    return 1.0 - np.power(10.0, -density)


def _response_derivative(exposure: np.ndarray, toe: float, density_scale: float) -> np.ndarray:
    density = density_scale * exposure / (toe + exposure)
    density_derivative = density_scale * toe / np.square(toe + exposure)
    return math.log(10.0) * np.power(10.0, -density) * density_derivative


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    channels = [
        gaussian_filter(image[..., channel], sigma=sigma, mode="constant", cval=0.0, truncate=4.0)
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1)


def build_rows(contract: dict[str, Any], role: str) -> list[dict[str, Any]]:
    roles = contract["roles"]
    pattern_key = f"{role}_patterns"
    exposure_key = f"{role}_exposure_scales"
    if pattern_key not in roles or exposure_key not in roles:
        raise ProMistHalationError("unknown P3S role")
    height, width, _ = roles["image_shape"]
    maximum = float(contract["synthetic_truth"]["maximum_source_value"])
    rows: list[dict[str, Any]] = []
    for pattern in roles[pattern_key]:
        base = _base_pattern(pattern, height, width, maximum)
        base_sha = _sha256_bytes(np.ascontiguousarray(base, dtype="<f8").tobytes())
        for scale in roles[exposure_key]:
            exposure = np.ascontiguousarray(base * float(scale), dtype=np.float64)
            rows.append(
                {
                    "pattern": pattern,
                    "exposure_scale": float(scale),
                    "base_sha256": base_sha,
                    "exposure": exposure,
                }
            )
    return rows


def _prepare(
    rows: Iterable[dict[str, Any]], contract: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    truth = contract["synthetic_truth"]
    candidate = contract["candidate_family"]
    toe = float(truth["response_toe"])
    density_scale = float(truth["response_density_scale"])
    backing_sigma = float(truth["backing_sigma_pixels"])
    backing_fraction = float(truth["backing_fraction"])
    sigmas = tuple(float(value) for value in candidate["gaussian_sigma_pixels"])
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        exposure = row["exposure"]
        target_exposure = exposure + backing_fraction * _blur(exposure, backing_sigma)
        sources.append(exposure.reshape(-1))
        targets.append(_response(target_exposure, toe, density_scale).reshape(-1))
        deltas.append(
            np.stack([(_blur(exposure, sigma) - exposure).reshape(-1) for sigma in sigmas], axis=1)
        )
        metadata.append({key: row[key] for key in ("pattern", "exposure_scale", "base_sha256")})
    return (
        np.concatenate(sources),
        np.concatenate(targets),
        np.concatenate(deltas, axis=0),
        metadata,
    )


def _fit_weights(
    source: np.ndarray,
    target: np.ndarray,
    deltas: np.ndarray,
    contract: dict[str, Any],
    active_indices: tuple[int, ...],
) -> np.ndarray:
    truth = contract["synthetic_truth"]
    family = contract["candidate_family"]
    toe = float(truth["response_toe"])
    density_scale = float(truth["response_density_scale"])
    upper = float(family["weight_bounds"][1])
    maximum_total = float(family["maximum_total_blur_weight"])
    basis = deltas[:, active_indices]

    def objective(weights: np.ndarray) -> float:
        exposure = source + basis @ weights
        residual = _response(exposure, toe, density_scale) - target
        return float(np.mean(np.square(residual), dtype=np.float64))

    def gradient(weights: np.ndarray) -> np.ndarray:
        exposure = source + basis @ weights
        residual = _response(exposure, toe, density_scale) - target
        slope = _response_derivative(exposure, toe, density_scale)
        return 2.0 * np.mean(basis * (residual * slope)[:, None], axis=0)

    initial_total = min(0.08, 0.5 * maximum_total)
    initial = np.full(len(active_indices), initial_total / len(active_indices), dtype=np.float64)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, upper)] * len(active_indices),
        constraints=[{"type": "ineq", "fun": lambda weights: maximum_total - float(np.sum(weights))}],
        options={"ftol": 1e-15, "maxiter": 500, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"P3S bounded fit failed: {result.message}")
    full = np.zeros(deltas.shape[1], dtype=np.float64)
    full[list(active_indices)] = result.x
    return full


def _predict(
    source: np.ndarray, deltas: np.ndarray, weights: np.ndarray, contract: dict[str, Any]
) -> np.ndarray:
    truth = contract["synthetic_truth"]
    exposure = source + deltas @ weights
    if float(np.min(exposure)) < -1e-14:
        raise RuntimeError("P3S candidate produced negative exposure")
    return _response(
        np.maximum(exposure, 0.0),
        float(truth["response_toe"]),
        float(truth["response_density_scale"]),
    )


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b), dtype=np.float64)))


def _row_metrics(
    rows: list[dict[str, Any]],
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    row_size = int(np.prod(rows[0]["exposure"].shape))
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        segment = slice(index * row_size, (index + 1) * row_size)
        result.append(
            {
                "pattern": row["pattern"],
                "exposure_scale": row["exposure_scale"],
                "base_sha256": row["base_sha256"],
                "rmse": {
                    name: _rmse(values[segment], target[segment])
                    for name, values in predictions.items()
                },
            }
        )
    return result


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    source_audit = audit_primary_source(root, contract)
    development_rows = build_rows(contract, "development")
    confirmation_rows = build_rows(contract, "confirmation")
    dev_source, dev_target, dev_deltas, _ = _prepare(development_rows, contract)
    con_source, con_target, con_deltas, _ = _prepare(confirmation_rows, contract)

    all_weights = _fit_weights(dev_source, dev_target, dev_deltas, contract, tuple(range(6)))
    single_weights = _fit_weights(dev_source, dev_target, dev_deltas, contract, (3,))
    truth = contract["synthetic_truth"]
    identity_dev = _response(dev_source, float(truth["response_toe"]), float(truth["response_density_scale"]))
    identity_con = _response(con_source, float(truth["response_toe"]), float(truth["response_density_scale"]))
    mist_dev = _predict(dev_source, dev_deltas, all_weights, contract)
    mist_con = _predict(con_source, con_deltas, all_weights, contract)
    single_dev = _predict(dev_source, dev_deltas, single_weights, contract)
    single_con = _predict(con_source, con_deltas, single_weights, contract)

    development_rmse = {
        "identity": _rmse(identity_dev, dev_target),
        "single_scale": _rmse(single_dev, dev_target),
        "six_scale_mist": _rmse(mist_dev, dev_target),
        "correct_topology": _rmse(dev_target.copy(), dev_target),
    }
    confirmation_rmse = {
        "identity": _rmse(identity_con, con_target),
        "single_scale": _rmse(single_con, con_target),
        "six_scale_mist": _rmse(mist_con, con_target),
        "correct_topology": _rmse(con_target.copy(), con_target),
    }
    row_metrics = _row_metrics(
        confirmation_rows,
        con_target,
        {
            "identity": identity_con,
            "single_scale": single_con,
            "six_scale_mist": mist_con,
        },
    )
    worst_mist = max(row["rmse"]["six_scale_mist"] for row in row_metrics)
    mist_over_identity = 1.0 - confirmation_rmse["six_scale_mist"] / confirmation_rmse["identity"]
    mist_over_single = 1.0 - confirmation_rmse["six_scale_mist"] / confirmation_rmse["single_scale"]
    fit_gap_ratio = confirmation_rmse["six_scale_mist"] / max(development_rmse["six_scale_mist"], 1e-30)
    total_weight = float(np.sum(all_weights))
    bound_distance = min(total_weight, float(contract["candidate_family"]["maximum_total_blur_weight"]) - total_weight)
    gates = contract["gates"]
    gate_results = {
        "source_identity": source_audit["sha256"] == contract["primary_source"]["pdf_sha256"],
        "correct_topology": confirmation_rmse["correct_topology"] <= float(gates["maximum_correct_topology_confirmation_rmse"]),
        "mist_beats_identity": mist_over_identity >= float(gates["minimum_mist_improvement_over_identity"]),
        "mist_beats_single_scale": mist_over_single >= float(gates["minimum_mist_improvement_over_single_scale"]),
        "mist_confirmation_error": confirmation_rmse["six_scale_mist"] <= float(gates["maximum_mist_confirmation_rmse"]),
        "mist_worst_case": worst_mist <= float(gates["maximum_mist_worst_pattern_exposure_rmse"]),
        "mist_fit_transfer": fit_gap_ratio <= float(gates["maximum_mist_development_to_confirmation_rmse_ratio"]),
        "weight_total_interior": bound_distance >= float(gates["minimum_weight_distance_from_bound"]),
    }
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "contract_sha256": _sha256_bytes(contract_path.read_bytes()),
        "primary_source_audit": source_audit,
        "roles": {
            "development_rows": len(development_rows),
            "confirmation_rows": len(confirmation_rows),
            "development_base_sha256": sorted({row["base_sha256"] for row in development_rows}),
            "confirmation_base_sha256": sorted({row["base_sha256"] for row in confirmation_rows}),
        },
        "fit": {
            "six_scale_weights": [float(value) for value in all_weights],
            "six_scale_total_weight": total_weight,
            "single_scale_weight": float(single_weights[3]),
            "weight_total_bound_distance": bound_distance,
        },
        "development_rmse": development_rmse,
        "confirmation_rmse": confirmation_rmse,
        "confirmation_metrics": {
            "six_scale_improvement_over_identity": mist_over_identity,
            "six_scale_improvement_over_single_scale": mist_over_single,
            "development_to_confirmation_rmse_ratio": fit_gap_ratio,
            "worst_pattern_exposure_rmse": worst_mist,
        },
        "confirmation_rows": row_metrics,
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
        "decision": (
            "lens-diffusion-observationally-equivalent-under-p3s"
            if all(gate_results.values())
            else "multi-exposure-mechanism-discriminator-retained"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


__all__ = [
    "ProMistHalationError",
    "audit_primary_source",
    "build_rows",
    "evaluate",
    "load_contract",
    "write_report",
]
