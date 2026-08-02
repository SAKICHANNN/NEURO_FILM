"""U6.P4BN finite-window radial NPS recovery evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)
from src.film_physics.measured_nps_field import (
    physical_frequency_grids,
    physical_periodogram,
    synthesize_measured_nps_field_pair,
    target_nps_grids,
)

SCHEMA = "neuro_film.u6_p4bn_finite_window_nps_recovery_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bn_finite_window_nps_recovery_report.v1"


class FiniteWindowNPSRecoveryError(RuntimeError):
    """Raised when frozen P4BN evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FiniteWindowNPSRecoveryError("P4BN paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    estimator = payload.get("estimator", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or estimator.get("parent_field_shape") != [512, 512]
        or estimator.get("sample_pitch_millimetres") != 0.001
        or estimator.get("field_kind") != "aperture_observed"
        or estimator.get("seeds") != [2608022301, 2608022302, 2608022303]
        or estimator.get("crop_sizes") != [128, 256]
        or estimator.get("windows") != ["rectangular", "hann"]
        or estimator.get("mean_removal")
        != "window_weighted_mean_before_window_application"
        or estimator.get("post_score_window_or_bin_tuning_allowed") is not False
        or estimator.get("render_allowed") is not False
        or evaluation.get("required_profile_count") != 5
        or evaluation.get("required_observations_per_profile_and_crop_size") != 12
        or evaluation.get("maximum_periodic_control_log10_rmse") != 1e-10
        or evaluation.get("maximum_256_hann_median_log10_rmse") != 0.25
        or evaluation.get("maximum_256_hann_worst_log10_rmse") != 0.4
        or evaluation.get("maximum_128_hann_median_log10_rmse") != 0.35
        or evaluation.get("maximum_128_hann_worst_log10_rmse") != 0.55
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise FiniteWindowNPSRecoveryError("P4BN frozen contract drift")
    return payload


def _load_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise FiniteWindowNPSRecoveryError(f"P4BN parent mismatch: {stem}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_window_periodogram(
    field: np.ndarray,
    sample_pitch_millimetres: float,
    window_name: str,
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("finite-window NPS input must be a finite 2D field")
    if window_name == "rectangular":
        window = np.ones(values.shape, dtype=np.float64)
    elif window_name == "hann":
        window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    else:
        raise ValueError("unsupported finite-window NPS window")
    weight_sum = float(np.sum(window, dtype=np.float64))
    energy = float(np.sum(np.square(window), dtype=np.float64))
    if weight_sum <= 0.0 or energy <= 0.0:
        raise ValueError("finite-window NPS window has no support")
    weighted_mean = float(np.sum(values * window, dtype=np.float64) / weight_sum)
    transformed = np.fft.fft2((values - weighted_mean) * window)
    return (
        np.square(np.abs(transformed))
        * sample_pitch_millimetres**2
        / energy
    )


def radial_bin_means(
    spectrum: np.ndarray,
    sample_pitch_millimetres: float,
    bin_edges_lines_per_mm: np.ndarray,
) -> np.ndarray:
    values = np.asarray(spectrum, dtype=np.float64)
    edges = np.asarray(bin_edges_lines_per_mm, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or edges.ndim != 1
        or edges.size < 2
        or not np.all(np.diff(edges) > 0.0)
    ):
        raise ValueError("invalid radial NPS bin request")
    _, _, radius = physical_frequency_grids(values.shape, sample_pitch_millimetres)
    means = []
    for lower, upper in pairwise(edges):
        selected = (radius >= lower) & (radius < upper)
        if not np.any(selected):
            raise ValueError("radial NPS bin has no frequency samples")
        means.append(float(np.mean(values[selected], dtype=np.float64)))
    return np.asarray(means, dtype=np.float64)


def _log10_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    if np.any(actual <= 0.0) or np.any(expected <= 0.0):
        return float("inf")
    return float(
        np.sqrt(
            np.mean(np.square(np.log10(actual) - np.log10(expected)), dtype=np.float64)
        )
    )


def _crop_origins(parent_size: int, crop_size: int) -> tuple[tuple[int, int], ...]:
    final = parent_size - crop_size
    return ((0, 0), (0, final), (final, 0), (final, final))


def evaluate_finite_window_nps_recovery(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    _load_parent(parents, root, "p4bm_contract")
    decision = _load_parent(parents, root, "p4bm_decision")
    parent_report = _load_parent(parents, root, "p4bm_report")
    bundle = _load_parent(parents, root, "p4bl_bundle")
    if (
        decision.get("automatic_pass") is not True
        or decision.get("decision")
        != "retain_offline_measured_nps_field_synthesizer"
        or parent_report.get("stable_evidence_id")
        != parents["p4bm_stable_evidence_id"]
        or bundle.get("bundle_id") != parents["p4bl_bundle_id"]
        or bundle.get("render_allowed") is not False
    ):
        raise FiniteWindowNPSRecoveryError("P4BN parent decision mismatch")

    profiles = [
        HistoricalBWNoiseSpectrumProfile.from_dict(payload)
        for payload in bundle["profiles"]
    ]
    estimator = contract["estimator"]
    parent_shape = tuple(int(value) for value in estimator["parent_field_shape"])
    pitch = float(estimator["sample_pitch_millimetres"])
    edges = np.asarray(estimator["radial_bin_edges_lines_per_mm"], dtype=np.float64)
    fields: dict[tuple[str, int], np.ndarray] = {}
    periodic_errors: list[float] = []
    for profile in profiles:
        _, exact_observed, supported = target_nps_grids(profile, parent_shape, pitch)
        for seed in estimator["seeds"]:
            pair = synthesize_measured_nps_field_pair(
                profile,
                shape=parent_shape,
                sample_pitch_millimetres=pitch,
                seed=int(seed),
            )
            fields[(profile.identity(), int(seed))] = pair.aperture_observed
            actual = physical_periodogram(pair.aperture_observed, pitch)
            selected = supported & (exact_observed > 0.0)
            periodic_errors.append(
                _log10_rmse(actual[selected], exact_observed[selected])
            )

    condition_rows: list[dict[str, Any]] = []
    condition_metrics: dict[tuple[int, str], dict[str, float]] = {}
    all_finite_positive = True
    all_repeat_exact = True
    for crop_size in estimator["crop_sizes"]:
        target_curves = {}
        for profile in profiles:
            _, target, _ = target_nps_grids(
                profile, (int(crop_size), int(crop_size)), pitch
            )
            target_curves[profile.identity()] = radial_bin_means(target, pitch, edges)
        for window_name in estimator["windows"]:
            errors = []
            correct = 0
            for profile in profiles:
                accumulated = np.zeros((int(crop_size), int(crop_size)), dtype=np.float64)
                observations = 0
                for seed in estimator["seeds"]:
                    field = fields[(profile.identity(), int(seed))]
                    for origin_y, origin_x in _crop_origins(
                        parent_shape[0], int(crop_size)
                    ):
                        crop = field[
                            origin_y : origin_y + int(crop_size),
                            origin_x : origin_x + int(crop_size),
                        ]
                        accumulated += finite_window_periodogram(
                            crop, pitch, str(window_name)
                        )
                        observations += 1
                estimate = radial_bin_means(
                    accumulated / observations, pitch, edges
                )
                expected = target_curves[profile.identity()]
                error = _log10_rmse(estimate, expected)
                distances = {
                    candidate_id: _log10_rmse(estimate, curve)
                    for candidate_id, curve in target_curves.items()
                }
                predicted = min(distances, key=distances.__getitem__)
                is_correct = predicted == profile.identity()
                correct += int(is_correct)
                errors.append(error)
                all_finite_positive &= bool(
                    np.all(np.isfinite(estimate)) and np.all(estimate > 0.0)
                )
                repeated = radial_bin_means(
                    accumulated / observations, pitch, edges
                )
                all_repeat_exact &= bool(np.array_equal(estimate, repeated))
                condition_rows.append(
                    {
                        "crop_size": int(crop_size),
                        "window": str(window_name),
                        "profile_id": profile.identity(),
                        "material_id": profile.material_id,
                        "diffuse_density": profile.diffuse_density,
                        "observation_count": observations,
                        "log10_rmse": error,
                        "predicted_profile_id": predicted,
                        "profile_top1_correct": is_correct,
                    }
                )
            condition_metrics[(int(crop_size), str(window_name))] = {
                "median_log10_rmse": float(np.median(errors)),
                "worst_log10_rmse": max(errors),
                "profile_top1_rate": correct / len(profiles),
            }

    metrics_128 = condition_metrics[(128, "hann")]
    metrics_256 = condition_metrics[(256, "hann")]
    evaluation = contract["evaluation"]
    gates = {
        "profile_count": len(profiles) == evaluation["required_profile_count"],
        "observation_count": all(
            row["observation_count"]
            == evaluation["required_observations_per_profile_and_crop_size"]
            for row in condition_rows
        ),
        "periodic_control": max(periodic_errors)
        <= evaluation["maximum_periodic_control_log10_rmse"],
        "hann_256_median": metrics_256["median_log10_rmse"]
        <= evaluation["maximum_256_hann_median_log10_rmse"],
        "hann_256_worst": metrics_256["worst_log10_rmse"]
        <= evaluation["maximum_256_hann_worst_log10_rmse"],
        "hann_256_top1": metrics_256["profile_top1_rate"]
        >= evaluation["minimum_256_hann_profile_top1_rate"],
        "hann_128_median": metrics_128["median_log10_rmse"]
        <= evaluation["maximum_128_hann_median_log10_rmse"],
        "hann_128_worst": metrics_128["worst_log10_rmse"]
        <= evaluation["maximum_128_hann_worst_log10_rmse"],
        "hann_128_top1": metrics_128["profile_top1_rate"]
        >= evaluation["minimum_128_hann_profile_top1_rate"],
        "finite_positive_radial_estimates": all_finite_positive,
        "repeat_exact": all_repeat_exact,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "parent_bundle_id": parents["p4bl_bundle_id"],
        "profile_count": len(profiles),
        "maximum_periodic_control_log10_rmse": max(periodic_errors),
        "condition_metrics": {
            f"crop_{size}_{window}": values
            for (size, window), values in condition_metrics.items()
        },
        "rows": condition_rows,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_hann_finite_window_nps_measurement_bridge"
            if automatic_pass
            else "close_direct_finite_window_nps_recovery"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_payload = dict(report)
    stable_payload.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(stable_payload)
    ).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
