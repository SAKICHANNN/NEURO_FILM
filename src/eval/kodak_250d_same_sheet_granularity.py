"""U6.P4AW same-sheet Kodak 250D granularity compatibility test."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls
from scipy.signal import fftconvolve
from scipy.stats import spearmanr

from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = (
    "neuro_film.u6_p4aw_kodak_250d_same_sheet_granularity_compatibility_contract.v1"
)
REPORT_SCHEMA = (
    "neuro_film.u6_p4aw_kodak_250d_same_sheet_granularity_compatibility_report.v1"
)
CHANNELS = ("blue", "green", "red")


class SameSheetGranularityError(RuntimeError):
    """Raised when a frozen P4AW input or contract drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SameSheetGranularityError("P4AW paths must be repository-relative")
    return root / relative


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    grid = model.get("spatial_energy_grid", {})
    split = payload.get("split", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or model.get("aperture_diameter_micrometres") != 48.0
        or grid
        != {
            "primary_pitch_micrometres": 0.25,
            "confirmation_pitch_micrometres": 0.125,
            "half_extent_micrometres": 64.0,
            "channel_energy_normalization": (
                "divide by geometric mean across blue, green and red"
            ),
        }
        or model.get("channel_order") != list(CHANNELS)
        or model.get("wrong_channel_cycle")
        != {"blue": "green", "green": "red", "red": "blue"}
        or model.get("post_result_parameter_or_family_change_allowed")
        or split.get("development")
        != "eligible sorted index modulo 3 is 0 or 2"
        or split.get("confirmation")
        != "eligible sorted index modulo 3 is 1"
        or split.get("minimum_development_rows_per_channel") != 10
        or split.get("minimum_confirmation_rows_per_channel") != 5
        or gates.get("maximum_relative_channel_energy_change_between_grid_pitches")
        != 0.002
        or gates.get("minimum_improvement_vs_channel_constant") != 0.1
        or gates.get("minimum_improvement_vs_density_only") != 0.05
        or gates.get("minimum_improvement_vs_characteristic_slope_without_p5j")
        != 0.005
        or gates.get("minimum_improvement_vs_wrong_channel_p5j") != 0.005
        or gates.get("maximum_confirmation_median_log_sigma_error") != 0.3
        or gates.get("maximum_confirmation_p95_log_sigma_error") != 0.7
        or gates.get("maximum_worst_channel_median_log_sigma_error") != 0.4
        or gates.get("minimum_median_channel_spearman") != 0.35
        or gates.get("minimum_channel_spearman") != 0.0
        or not gates.get("require_positive_shared_amplitude")
        or not gates.get("two_byte_identical_reports")
    ):
        raise SameSheetGranularityError("P4AW frozen contract drift")
    return payload


def _load_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> tuple[Path, dict[str, Any]]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise SameSheetGranularityError(f"P4AW parent integrity mismatch: {stem}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_parents(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], ManufacturerCharacteristicPrior, dict[str, Any], dict[str, str]]:
    parents = contract["parents"]
    identities: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for stem in (
        "p4av_contract",
        "p4av_trace",
        "p4av_report",
        "p2q_contract",
        "p2q_decision",
        "p2q_bundle",
        "p5j_contract",
        "p5j_decision",
    ):
        path, payload = _load_parent(parents, root, stem)
        payloads[stem] = payload
        identities[stem] = _hash_file(path)

    p4av = payloads["p4av_report"]
    p2q_decision = payloads["p2q_decision"]
    p5j = payloads["p5j_decision"]
    if (
        p4av.get("source_pass") is not True
        or p4av.get("stable_evidence_id") != parents["p4av_stable_evidence_id"]
        or p2q_decision.get("prior_identity") != parents["p2q_prior_identity"]
        or p2q_decision.get("bundle_sha256") != parents["p2q_bundle_sha256"]
        or p5j.get("automatic_pass") is not True
        or p5j.get("bundle_id") != parents["p5j_bundle_id"]
        or p5j.get("bundle_sha256") != parents["p5j_bundle_sha256"]
    ):
        raise SameSheetGranularityError("P4AW parent decision mismatch")
    prior = ManufacturerCharacteristicPrior.from_dict(payloads["p2q_bundle"]["prior"])
    if prior.identity() != parents["p2q_prior_identity"]:
        raise SameSheetGranularityError("P4AW characteristic prior identity mismatch")
    return p4av, prior, p5j, identities


def _sample_psf(
    coordinates: np.ndarray,
    components: Sequence[Sequence[float]],
) -> np.ndarray:
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    result = np.zeros_like(xx, dtype=np.float64)
    centre = len(coordinates) // 2
    for weight_value, sigma_value in components:
        weight = float(weight_value)
        sigma = float(sigma_value)
        if not math.isfinite(weight) or weight <= 0.0 or sigma < 0.0:
            raise SameSheetGranularityError("P4AW invalid positive PSF component")
        if sigma == 0.0:
            result[centre, centre] += weight
        else:
            gaussian = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
            gaussian /= np.sum(gaussian)
            result += weight * gaussian
    if not math.isclose(float(np.sum(result)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SameSheetGranularityError("P4AW PSF weights do not normalize")
    return result


def _normalized_channel_energies(
    p5j: Mapping[str, Any], *, pitch: float, half_extent: float, diameter: float
) -> dict[str, float]:
    size = round((2.0 * half_extent) / pitch) + 1
    if size % 2 == 0:
        size += 1
    coordinates = (np.arange(size, dtype=np.float64) - size // 2) * pitch
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    aperture = ((xx * xx + yy * yy) <= (diameter * 0.5) ** 2).astype(np.float64)
    aperture /= np.sum(aperture)
    raw: dict[str, float] = {}
    for channel in CHANNELS:
        psf = _sample_psf(coordinates, p5j["selected_components"][channel])
        combined = fftconvolve(aperture, psf, mode="same")
        raw[channel] = float(np.sum(combined * combined))
    geometric_mean = math.exp(sum(math.log(raw[channel]) for channel in CHANNELS) / 3.0)
    return {channel: raw[channel] / geometric_mean for channel in CHANNELS}


def _piecewise_slope(curve: Any, exposure: float) -> float:
    knots = curve.log_exposure_knots
    index = int(np.searchsorted(knots, exposure, side="right") - 1)
    index = min(max(index, 0), len(knots) - 2)
    return float(
        (curve.density_knots[index + 1] - curve.density_knots[index])
        / (knots[index + 1] - knots[index])
    )


def _build_rows(
    p4av: Mapping[str, Any], prior: ManufacturerCharacteristicPrior
) -> list[dict[str, Any]]:
    curves = {curve.layer: curve for curve in prior.curves}
    rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        curve = curves[channel]
        source = p4av["channels"][channel]
        eligible: list[dict[str, Any]] = []
        for exposure_value, sigma_value in zip(
            source["log_relative_exposure"], source["sigma_d"], strict=True
        ):
            exposure = float(exposure_value)
            sigma = float(sigma_value)
            if curve.domain[0] <= exposure <= curve.domain[1]:
                density = float(curve.apply(np.asarray([exposure]))[0])
                eligible.append(
                    {
                        "channel": channel,
                        "log_exposure": exposure,
                        "sigma_d": sigma,
                        "variance": sigma * sigma,
                        "density": density,
                        "density_above_minimum": max(0.0, density - curve.density_bounds[0]),
                        "characteristic_slope": _piecewise_slope(curve, exposure),
                    }
                )
        eligible.sort(key=lambda item: item["log_exposure"])
        for index, row in enumerate(eligible):
            row["role"] = "confirmation" if index % 3 == 1 else "development"
            row["eligible_index"] = index
            rows.append(row)
    return rows


def _design_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    channel_energies: Mapping[str, float],
) -> np.ndarray:
    matrix = np.zeros((len(rows), 4), dtype=np.float64)
    for row_index, row in enumerate(rows):
        channel = str(row["channel"])
        matrix[row_index, CHANNELS.index(channel)] = 1.0
        if family == "density_only":
            shape = float(row["density_above_minimum"])
        else:
            slope = float(row["characteristic_slope"])
            shape = slope * slope / (10.0 ** float(row["log_exposure"]))
            shape *= float(channel_energies[channel])
        matrix[row_index, 3] = shape
    return matrix


def _fit_and_predict(
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
    *,
    family: str,
    channel_energies: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    design = _design_matrix(development, family=family, channel_energies=channel_energies)
    target = np.asarray([float(row["variance"]) for row in development], dtype=np.float64)
    parameters, residual_norm = nnls(design, target)
    predicted_variance = _design_matrix(
        confirmation, family=family, channel_energies=channel_energies
    ) @ parameters
    predicted_sigma = np.sqrt(np.maximum(predicted_variance, np.finfo(np.float64).tiny))
    return predicted_sigma, {
        "channel_floor_variance": {
            channel: float(parameters[index]) for index, channel in enumerate(CHANNELS)
        },
        "shared_amplitude": float(parameters[3]),
        "development_residual_norm": float(residual_norm),
    }


def _constant_predict(
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    floor = {
        channel: float(
            np.mean(
                [
                    float(row["variance"])
                    for row in development
                    if row["channel"] == channel
                ]
            )
        )
        for channel in CHANNELS
    }
    return (
        np.asarray([math.sqrt(floor[str(row["channel"])]) for row in confirmation]),
        {"channel_floor_variance": floor, "shared_amplitude": 0.0},
    )


def _score(
    confirmation: Sequence[Mapping[str, Any]], predicted: np.ndarray
) -> dict[str, Any]:
    observed = np.asarray([float(row["sigma_d"]) for row in confirmation], dtype=np.float64)
    error = np.abs(np.log(predicted / observed))
    per_channel: dict[str, Any] = {}
    correlations: list[float] = []
    for channel in CHANNELS:
        indices = [index for index, row in enumerate(confirmation) if row["channel"] == channel]
        channel_error = error[indices]
        channel_observed = observed[indices]
        channel_predicted = predicted[indices]
        if np.unique(channel_observed).size < 2 or np.unique(channel_predicted).size < 2:
            correlation = -1.0
        else:
            correlation = float(
                spearmanr(channel_observed, channel_predicted).statistic
            )
            if not math.isfinite(correlation):
                correlation = -1.0
        correlations.append(correlation)
        per_channel[channel] = {
            "row_count": len(indices),
            "median_log_sigma_error": float(np.median(channel_error)),
            "p95_log_sigma_error": float(np.percentile(channel_error, 95.0)),
            "spearman": correlation,
        }
    return {
        "median_log_sigma_error": float(np.median(error)),
        "p95_log_sigma_error": float(np.percentile(error, 95.0)),
        "maximum_log_sigma_error": float(np.max(error)),
        "median_channel_spearman": float(np.median(correlations)),
        "minimum_channel_spearman": float(np.min(correlations)),
        "per_channel": per_channel,
        "predicted_sigma_d": predicted.tolist(),
        "point_log_sigma_error": error.tolist(),
    }


def evaluate_compatibility(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    p4av, prior, p5j, parent_sha = _validate_parents(contract, root)
    grid = contract["model"]["spatial_energy_grid"]
    diameter = float(contract["model"]["aperture_diameter_micrometres"])
    primary_energy = _normalized_channel_energies(
        p5j,
        pitch=float(grid["primary_pitch_micrometres"]),
        half_extent=float(grid["half_extent_micrometres"]),
        diameter=diameter,
    )
    confirmation_energy = _normalized_channel_energies(
        p5j,
        pitch=float(grid["confirmation_pitch_micrometres"]),
        half_extent=float(grid["half_extent_micrometres"]),
        diameter=diameter,
    )
    energy_change = {
        channel: abs(primary_energy[channel] / confirmation_energy[channel] - 1.0)
        for channel in CHANNELS
    }

    rows = _build_rows(p4av, prior)
    development = [row for row in rows if row["role"] == "development"]
    confirmation = [row for row in rows if row["role"] == "confirmation"]
    minimum_development = int(contract["split"]["minimum_development_rows_per_channel"])
    minimum_confirmation = int(contract["split"]["minimum_confirmation_rows_per_channel"])
    split_counts = {
        channel: {
            "development": sum(row["channel"] == channel for row in development),
            "confirmation": sum(row["channel"] == channel for row in confirmation),
        }
        for channel in CHANNELS
    }
    if any(
        counts["development"] < minimum_development
        or counts["confirmation"] < minimum_confirmation
        for counts in split_counts.values()
    ):
        raise SameSheetGranularityError("P4AW eligible split is undersized")

    wrong_cycle = contract["model"]["wrong_channel_cycle"]
    wrong_energy = {
        channel: primary_energy[str(wrong_cycle[channel])] for channel in CHANNELS
    }
    unit_energy = {channel: 1.0 for channel in CHANNELS}
    predictions: dict[str, np.ndarray] = {}
    parameters: dict[str, Any] = {}
    predictions["channel_constant"], parameters["channel_constant"] = _constant_predict(
        development, confirmation
    )
    for name, family, energy in (
        ("density_only", "density_only", unit_energy),
        ("characteristic_slope_without_p5j", "slope", unit_energy),
        ("wrong_channel_p5j", "slope", wrong_energy),
        ("selected", "slope", primary_energy),
    ):
        predictions[name], parameters[name] = _fit_and_predict(
            development, confirmation, family=family, channel_energies=energy
        )
    scores = {name: _score(confirmation, value) for name, value in predictions.items()}
    selected_median = float(scores["selected"]["median_log_sigma_error"])
    improvement = {
        name: (float(scores[name]["median_log_sigma_error"]) - selected_median)
        / float(scores[name]["median_log_sigma_error"])
        for name in (
            "channel_constant",
            "density_only",
            "characteristic_slope_without_p5j",
            "wrong_channel_p5j",
        )
    }
    worst_channel_median = max(
        float(item["median_log_sigma_error"])
        for item in scores["selected"]["per_channel"].values()
    )
    gates = contract["gates"]
    gate_results = {
        "parent_identity": True,
        "split_size": True,
        "spatial_energy_convergence": max(energy_change.values())
        <= float(gates["maximum_relative_channel_energy_change_between_grid_pitches"]),
        "improvement_vs_channel_constant": improvement["channel_constant"]
        >= float(gates["minimum_improvement_vs_channel_constant"]),
        "improvement_vs_density_only": improvement["density_only"]
        >= float(gates["minimum_improvement_vs_density_only"]),
        "improvement_vs_characteristic_slope_without_p5j": improvement[
            "characteristic_slope_without_p5j"
        ]
        >= float(gates["minimum_improvement_vs_characteristic_slope_without_p5j"]),
        "improvement_vs_wrong_channel_p5j": improvement["wrong_channel_p5j"]
        >= float(gates["minimum_improvement_vs_wrong_channel_p5j"]),
        "confirmation_median_error": selected_median
        <= float(gates["maximum_confirmation_median_log_sigma_error"]),
        "confirmation_p95_error": float(scores["selected"]["p95_log_sigma_error"])
        <= float(gates["maximum_confirmation_p95_log_sigma_error"]),
        "worst_channel_median_error": worst_channel_median
        <= float(gates["maximum_worst_channel_median_log_sigma_error"]),
        "median_channel_spearman": float(scores["selected"]["median_channel_spearman"])
        >= float(gates["minimum_median_channel_spearman"]),
        "minimum_channel_spearman": float(scores["selected"]["minimum_channel_spearman"])
        >= float(gates["minimum_channel_spearman"]),
        "positive_shared_amplitude": float(parameters["selected"]["shared_amplitude"])
        > 0.0,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "parent_sha256": parent_sha,
        "p2q_prior_identity": prior.identity(),
        "p5j_bundle_id": p5j["bundle_id"],
        "split_counts": split_counts,
        "channel_energy_primary": primary_energy,
        "channel_energy_confirmation": confirmation_energy,
        "channel_energy_relative_change": energy_change,
        "parameters": parameters,
        "scores": scores,
        "control_improvement": improvement,
        "worst_channel_median_log_sigma_error": worst_channel_median,
        "confirmation_rows": confirmation,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_effective_same_sheet_joint_family"
            if automatic_pass
            else "close_effective_same_sheet_joint_family_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "SameSheetGranularityError",
    "evaluate_compatibility",
    "load_contract",
    "write_report",
]
