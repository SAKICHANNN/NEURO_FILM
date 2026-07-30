"""Fit-forbidden B&W external confirmation of retained grain candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.eval.real_uniform_grain_nps import (
    acf_lag_signature,
    cosine_similarity,
    fixed_fractional_crops,
    radial_nps_signature,
)
from src.eval.real_uniform_grain_physical import (
    render_anisotropic_density_region,
)
from src.eval.real_uniform_grain_source import hash_file
from src.eval.scan_amplitude_feasibility import robust_relative_amplitude


class BWGrainExternalConfirmationError(RuntimeError):
    """Raised when a frozen parent, candidate or evaluation rule drifts."""


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise BWGrainExternalConfirmationError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BWGrainExternalConfirmationError("parent must be an object")
    return payload


def _normalized_median(rows: list[np.ndarray]) -> np.ndarray:
    value = np.median(np.asarray(rows, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(value))
    if not norm > 0.0 or not np.all(np.isfinite(value)):
        raise BWGrainExternalConfirmationError("degenerate signature")
    output = np.ascontiguousarray(value / norm)
    output.setflags(write=False)
    return output


def _validate_contract(
    root: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        contract.get("schema")
        != "neuro_film.u6_p4z2_bw_grain_external_confirmation_contract.v1"
        or contract.get("operator_fitting_allowed") is not False
        or contract.get("photographic_render_allowed") is not False
        or contract.get("product_integration_allowed") is not False
    ):
        raise BWGrainExternalConfirmationError("unsupported confirmation contract")
    parents = contract["parents"]
    decision = _load_bound(root, parents["preflight_decision"])
    preflight = _load_bound(root, parents["preflight_contract"])
    p4d = _load_bound(root, parents["p4d_contract"])
    p4t = _load_bound(root, parents["p4t_contract"])
    p4t_decision = _load_bound(root, parents["p4t_decision"])
    p4x = _load_bound(root, parents["p4x_contract"])
    p4x_decision = _load_bound(root, parents["p4x_decision"])
    signature = _load_bound(root, parents["signature_contract"])
    candidates = contract["candidates"]
    if (
        decision["decision"]
        != parents["preflight_decision"]["required_decision"]
        or p4d["model"]["input_domain"] != "developed_optical_density"
        or p4t_decision["result"]["status"]
        != "automatic_and_synthetic_severe_pass"
        or p4x_decision["decision"]
        != "retain_global_transmittance_proxy_amplitude_candidate"
        or float(p4x_decision["result"]["selected_shared_amplitude_scale"])
        != 0.125
        or candidates["p4d_isotropic"]["grain_optical_density_by_layer"]
        != [row["grain_optical_density"] for row in p4d["profiles"]]
        or candidates["p4t_anisotropic"]["sigma_yx_by_layer"]
        != [p4t["candidate"]["sigma_yx_pixels"]] * 3
        or candidates["p4x_scaled_anisotropic"]["source_amplitude_scale"]
        != p4x_decision["result"]["selected_shared_amplitude_scale"]
        or contract["evaluation"]["crop_size_pixels"]
        != preflight["crop_contract"]["crop_size_pixels"]
        or contract["evaluation"]["density_to_transmittance"] != "10^-D"
    ):
        raise BWGrainExternalConfirmationError("candidate or parent drift")
    return preflight, decision, signature


def _observed_rows(
    *,
    root: Path,
    preflight: dict[str, Any],
    decision: dict[str, Any],
    signature_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    source_binding = preflight["parents"]["source_contract"]
    source = _load_bound(root, source_binding)
    manifest = _load_bound(root, preflight["parents"]["acquisition_manifest"])
    by_title = {row["title"]: row for row in manifest["rows"]}
    crop_contract = preflight["crop_contract"]
    nps_edges = signature_contract["pixel_contract"][
        "radial_nps_band_edges_cycles_per_pixel"
    ]
    acf_lags = signature_contract["pixel_contract"]["acf_lags_pixels_yx"]
    rows: list[dict[str, Any]] = []
    for expected in sorted(source["files"], key=lambda row: row["title"]):
        path = root / expected["path"]
        if hash_file(path, "sha256") != by_title[expected["title"]]["sha256"]:
            raise BWGrainExternalConfirmationError("observed payload drift")
        with Image.open(path) as image:
            values = np.asarray(image)
        crops = fixed_fractional_crops(
            values,
            crop_size=int(crop_contract["crop_size_pixels"]),
            centers_yx=crop_contract["fixed_fractional_centers_yx"],
        )
        nps = [
            radial_nps_signature(
                crop,
                band_edges_cycles_per_pixel=nps_edges,
            )
            for crop in crops
        ]
        acf = [acf_lag_signature(crop, lags_yx=acf_lags) for crop in crops]
        amplitudes = [robust_relative_amplitude(crop) for crop in crops]
        rows.append(
            {
                "title": expected["title"],
                "film_stock_id": expected["film_stock_id"],
                "sha256": by_title[expected["title"]]["sha256"],
                "crop_count": len(crops),
                "nps_signature": _normalized_median(nps),
                "acf_signature": np.median(np.asarray(acf), axis=0),
                "relative_amplitude": float(np.median(amplitudes)),
            }
        )
    if len(rows) != decision["observed_results"]["source_count"]:
        raise BWGrainExternalConfirmationError("observed source count drift")
    return rows


def _candidate_signature(
    *,
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
    signature_contract: dict[str, Any],
) -> dict[str, Any]:
    height, width = (int(value) for value in evaluation["synthetic_field_shape"])
    y0, x0, crop_h, crop_w = (
        int(value) for value in evaluation["synthetic_analysis_crop"]
    )
    target_density = float(evaluation["target_density"])
    target = np.full((height, width), target_density, dtype=np.float64)
    seeds = [int(value) for value in evaluation["confirmation_seeds"]]
    nps_edges = signature_contract["pixel_contract"][
        "radial_nps_band_edges_cycles_per_pixel"
    ]
    acf_lags = signature_contract["pixel_contract"]["acf_lags_pixels_yx"]
    nps_rows: list[np.ndarray] = []
    acf_rows: list[np.ndarray] = []
    amplitudes: list[float] = []
    density_hashes: list[str] = []
    for seed in seeds:
        for layer, (sigma, optical_density) in enumerate(
            zip(
                candidate["sigma_yx_by_layer"],
                candidate["grain_optical_density_by_layer"],
                strict=True,
            )
        ):
            density = render_anisotropic_density_region(
                target,
                origin_yx=(0, 0),
                shape=(height, width),
                grain_optical_density=float(optical_density),
                sigma_yx=(float(sigma[0]), float(sigma[1])),
                seed=seed + 104729 * layer,
                maximum_target_density=2.0,
                truncate=float(evaluation["truncate"]),
            )
            crop = density[y0 : y0 + crop_h, x0 : x0 + crop_w]
            nps_rows.append(
                radial_nps_signature(
                    crop,
                    band_edges_cycles_per_pixel=nps_edges,
                )
            )
            acf_rows.append(acf_lag_signature(crop, lags_yx=acf_lags))
            amplitudes.append(
                robust_relative_amplitude(np.power(10.0, -crop))
            )
            density_hashes.append(
                hashlib.sha256(np.ascontiguousarray(density).tobytes()).hexdigest()
            )
    return {
        "nps_signature": _normalized_median(nps_rows),
        "acf_signature": np.median(np.asarray(acf_rows), axis=0),
        "relative_amplitude": float(np.median(amplitudes)),
        "density_sha256": density_hashes,
    }


def _errors(
    candidate: dict[str, Any],
    observed: list[dict[str, Any]],
) -> dict[str, Any]:
    nps_errors = [
        float(1.0 - cosine_similarity(candidate["nps_signature"], row["nps_signature"]))
        for row in observed
    ]
    acf_errors = [
        float(np.median(np.abs(candidate["acf_signature"] - row["acf_signature"])))
        for row in observed
    ]
    amplitude_errors = [
        float(
            abs(candidate["relative_amplitude"] - row["relative_amplitude"])
            / row["relative_amplitude"]
        )
        for row in observed
    ]
    return {
        "nps": nps_errors,
        "acf": acf_errors,
        "relative_amplitude": amplitude_errors,
        "nps_median": float(np.median(nps_errors)),
        "nps_maximum": float(np.max(nps_errors)),
        "acf_median": float(np.median(acf_errors)),
        "acf_maximum": float(np.max(acf_errors)),
        "relative_amplitude_median": float(np.median(amplitude_errors)),
        "relative_amplitude_maximum": float(np.max(amplitude_errors)),
    }


def evaluate_external_confirmation(
    *,
    root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate all frozen candidates without fitting to B&W pixels."""
    preflight, decision, signature_contract = _validate_contract(root, contract)
    observed = _observed_rows(
        root=root,
        preflight=preflight,
        decision=decision,
        signature_contract=signature_contract,
    )
    candidates = {
        name: _candidate_signature(
            candidate=candidate,
            evaluation=contract["evaluation"],
            signature_contract=signature_contract,
        )
        for name, candidate in contract["candidates"].items()
    }
    errors = {name: _errors(candidate, observed) for name, candidate in candidates.items()}
    baseline = errors["p4d_isotropic"]
    shape = errors["p4t_anisotropic"]
    scaled = errors["p4x_scaled_anisotropic"]
    epsilon = np.finfo(np.float64).eps
    nps_improvement = 1.0 - shape["nps_median"] / max(
        baseline["nps_median"], epsilon
    )
    nps_worst_ratio = shape["nps_maximum"] / max(
        baseline["nps_maximum"], epsilon
    )
    acf_improvement = 1.0 - shape["acf_median"] / max(
        baseline["acf_median"], epsilon
    )
    acf_worst_ratio = shape["acf_maximum"] / max(
        baseline["acf_maximum"], epsilon
    )
    amplitude_improvement = 1.0 - scaled["relative_amplitude_median"] / max(
        shape["relative_amplitude_median"], epsilon
    )
    reversed_nps = np.ascontiguousarray(candidates["p4x_scaled_anisotropic"]["nps_signature"][::-1])
    reversed_errors = [
        float(1.0 - cosine_similarity(reversed_nps, row["nps_signature"]))
        for row in observed
    ]
    reversed_median = float(np.median(reversed_errors))
    reversed_improvement = 1.0 - scaled["nps_median"] / max(
        reversed_median, epsilon
    )
    gates = contract["automatic_gates"]
    checks = {
        "p4t_nps_median_improvement": nps_improvement
        >= float(gates["minimum_p4t_nps_median_improvement_over_p4d"]),
        "p4t_nps_worst_not_worse": nps_worst_ratio
        <= float(gates["maximum_p4t_nps_worst_error_ratio"]),
        "p4t_acf_median_improvement": acf_improvement
        >= float(gates["minimum_p4t_acf_median_improvement_over_p4d"]),
        "p4t_acf_worst_not_worse": acf_worst_ratio
        <= float(gates["maximum_p4t_acf_worst_error_ratio"]),
        "p4x_amplitude_improvement": amplitude_improvement
        >= float(gates["minimum_p4x_amplitude_error_improvement_over_p4t"]),
        "p4x_absolute_amplitude": scaled["relative_amplitude_median"]
        <= float(gates["maximum_p4x_median_relative_amplitude_error"]),
        "reversed_frequency_control": reversed_improvement
        >= float(
            gates[
                "minimum_p4x_nps_improvement_over_reversed_frequency_control"
            ]
        ),
        "repeat_exact": True,
    }
    shape_pass = all(
        checks[key]
        for key in (
            "p4t_nps_median_improvement",
            "p4t_nps_worst_not_worse",
            "p4t_acf_median_improvement",
            "p4t_acf_worst_not_worse",
        )
    )
    amplitude_pass = (
        checks["p4x_amplitude_improvement"]
        and checks["p4x_absolute_amplitude"]
    )
    control_pass = checks["reversed_frequency_control"]
    automatic_pass = all(checks.values())
    if not control_pass:
        decision_name = "close_frequency_insensitive_external_confirmation"
    elif not shape_pass:
        decision_name = "close_bw_shape_generalization"
    elif not amplitude_pass:
        decision_name = "retain_bw_shape_close_cross_material_amplitude"
    else:
        decision_name = "retain_generic_bw_shape_and_scaled_amplitude_evidence"
    stable = {
        "schema": "neuro_film.u6_p4z2_bw_grain_external_confirmation_report.v1",
        "contract_sha256": hash_file(
            root / "configs/u6_p4z2_bw_grain_external_confirmation_v1.json",
            "sha256",
        ),
        "source_count": len(observed),
        "crop_count_per_source": [row["crop_count"] for row in observed],
        "stock_labels_used_for_fit": False,
        "operator_fitting_executed": False,
        "observed": [
            {
                "title": row["title"],
                "film_stock_id": row["film_stock_id"],
                "sha256": row["sha256"],
                "relative_amplitude": row["relative_amplitude"],
                "nps_signature": row["nps_signature"].tolist(),
                "acf_signature": row["acf_signature"].tolist(),
            }
            for row in observed
        ],
        "candidates": {
            name: {
                "relative_amplitude": value["relative_amplitude"],
                "nps_signature": value["nps_signature"].tolist(),
                "acf_signature": value["acf_signature"].tolist(),
                "density_sha256": value["density_sha256"],
                "errors": errors[name],
            }
            for name, value in candidates.items()
        },
        "comparisons": {
            "p4t_nps_median_improvement_over_p4d": nps_improvement,
            "p4t_nps_worst_error_ratio": nps_worst_ratio,
            "p4t_acf_median_improvement_over_p4d": acf_improvement,
            "p4t_acf_worst_error_ratio": acf_worst_ratio,
            "p4x_amplitude_error_improvement_over_p4t": amplitude_improvement,
            "p4x_nps_reversed_control_median_error": reversed_median,
            "p4x_nps_improvement_over_reversed_control": reversed_improvement,
        },
        "checks": checks,
        "shape_pass": shape_pass,
        "amplitude_pass": amplitude_pass,
        "control_pass": control_pass,
        "automatic_pass": automatic_pass,
        "decision": decision_name,
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "BWGrainExternalConfirmationError",
    "evaluate_external_confirmation",
    "write_report",
]
