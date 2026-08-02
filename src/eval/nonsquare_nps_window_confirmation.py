"""U6.P4BO fresh-seed non-square window confirmation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.finite_window_nps_recovery import (
    finite_window_periodogram,
    radial_bin_means,
)
from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)
from src.film_physics.measured_nps_field import (
    synthesize_measured_nps_field_pair,
    target_nps_grids,
)

SCHEMA = "neuro_film.u6_p4bo_nonsquare_nps_window_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bo_nonsquare_nps_window_confirmation_report.v1"


class NonsquareNPSWindowConfirmationError(RuntimeError):
    """Raised when frozen P4BO evidence or semantics drift."""


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
        raise NonsquareNPSWindowConfirmationError("P4BO paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    confirmation = payload.get("confirmation", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or confirmation.get("parent_field_shape") != [512, 512]
        or confirmation.get("sample_pitch_millimetres") != 0.001
        or confirmation.get("fresh_seeds")
        != [2608022401, 2608022402, 2608022403]
        or confirmation.get("window_shapes") != [[160, 224], [224, 160]]
        or confirmation.get("candidate_window") != "rectangular"
        or confirmation.get("incumbent_window") != "hann"
        or confirmation.get("post_score_shape_seed_window_or_bin_tuning_allowed")
        is not False
        or confirmation.get("render_allowed") is not False
        or evaluation.get("required_profile_count") != 5
        or evaluation.get("required_shape_count") != 2
        or evaluation.get("minimum_each_shape_relative_median_improvement") != 0.2
        or evaluation.get("maximum_rectangular_median_log10_rmse") != 0.05
        or evaluation.get("maximum_rectangular_worst_log10_rmse") != 0.08
        or evaluation.get("minimum_rectangular_profile_top1_rate") != 1.0
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise NonsquareNPSWindowConfirmationError("P4BO frozen contract drift")
    return payload


def _load_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise NonsquareNPSWindowConfirmationError(f"P4BO parent mismatch: {stem}")
    return json.loads(path.read_text(encoding="utf-8"))


def _log10_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    if np.any(actual <= 0.0) or np.any(expected <= 0.0):
        return float("inf")
    return float(
        np.sqrt(
            np.mean(np.square(np.log10(actual) - np.log10(expected)), dtype=np.float64)
        )
    )


def _origins(
    parent_shape: tuple[int, int], crop_shape: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    final_y = parent_shape[0] - crop_shape[0]
    final_x = parent_shape[1] - crop_shape[1]
    return ((0, 0), (0, final_x), (final_y, 0), (final_y, final_x))


def evaluate_nonsquare_nps_window_confirmation(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    _load_parent(parents, root, "p4bn_contract")
    decision = _load_parent(parents, root, "p4bn_decision")
    parent_report = _load_parent(parents, root, "p4bn_report")
    bundle = _load_parent(parents, root, "p4bl_bundle")
    if (
        decision.get("automatic_pass") is not True
        or decision.get("decision")
        != "retain_hann_finite_window_nps_measurement_bridge"
        or decision.get("rectangular_diagnostic_outperformed_hann") is not True
        or parent_report.get("stable_evidence_id")
        != parents["p4bn_stable_evidence_id"]
        or bundle.get("bundle_id") != parents["p4bl_bundle_id"]
    ):
        raise NonsquareNPSWindowConfirmationError("P4BO parent decision mismatch")

    profiles = [
        HistoricalBWNoiseSpectrumProfile.from_dict(payload)
        for payload in bundle["profiles"]
    ]
    confirmation = contract["confirmation"]
    parent_shape = tuple(int(value) for value in confirmation["parent_field_shape"])
    pitch = float(confirmation["sample_pitch_millimetres"])
    edges = np.asarray(
        confirmation["radial_bin_edges_lines_per_mm"], dtype=np.float64
    )
    fields: dict[tuple[str, int], np.ndarray] = {}
    for profile in profiles:
        for seed in confirmation["fresh_seeds"]:
            fields[(profile.identity(), int(seed))] = (
                synthesize_measured_nps_field_pair(
                    profile,
                    shape=parent_shape,
                    sample_pitch_millimetres=pitch,
                    seed=int(seed),
                ).aperture_observed
            )

    rows: list[dict[str, Any]] = []
    shape_metrics: dict[str, dict[str, float]] = {}
    all_candidate_not_worse = True
    all_finite_positive = True
    all_repeat_exact = True
    for shape_values in confirmation["window_shapes"]:
        shape = tuple(int(value) for value in shape_values)
        target_curves = {}
        for profile in profiles:
            _, target, _ = target_nps_grids(profile, shape, pitch)
            target_curves[profile.identity()] = radial_bin_means(target, pitch, edges)
        candidate_errors: list[float] = []
        incumbent_errors: list[float] = []
        candidate_correct = 0
        for profile in profiles:
            estimates = {}
            averaged_periodograms = {}
            for window in (
                confirmation["candidate_window"],
                confirmation["incumbent_window"],
            ):
                accumulated = np.zeros(shape, dtype=np.float64)
                observations = 0
                for seed in confirmation["fresh_seeds"]:
                    field = fields[(profile.identity(), int(seed))]
                    for origin_y, origin_x in _origins(parent_shape, shape):
                        crop = field[
                            origin_y : origin_y + shape[0],
                            origin_x : origin_x + shape[1],
                        ]
                        accumulated += finite_window_periodogram(
                            crop, pitch, str(window)
                        )
                        observations += 1
                averaged = accumulated / observations
                averaged_periodograms[str(window)] = averaged
                estimates[str(window)] = radial_bin_means(averaged, pitch, edges)
            expected = target_curves[profile.identity()]
            candidate = estimates[str(confirmation["candidate_window"])]
            incumbent = estimates[str(confirmation["incumbent_window"])]
            candidate_error = _log10_rmse(candidate, expected)
            incumbent_error = _log10_rmse(incumbent, expected)
            distances = {
                profile_id: _log10_rmse(candidate, curve)
                for profile_id, curve in target_curves.items()
            }
            predicted = min(distances, key=distances.__getitem__)
            correct = predicted == profile.identity()
            candidate_correct += int(correct)
            candidate_errors.append(candidate_error)
            incumbent_errors.append(incumbent_error)
            all_candidate_not_worse &= candidate_error <= incumbent_error
            all_finite_positive &= bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate > 0.0)
                and np.all(np.isfinite(incumbent))
                and np.all(incumbent > 0.0)
            )
            repeated = radial_bin_means(
                averaged_periodograms[str(confirmation["candidate_window"])],
                pitch,
                edges,
            )
            all_repeat_exact &= bool(np.array_equal(candidate, repeated))
            rows.append(
                {
                    "window_shape": list(shape),
                    "profile_id": profile.identity(),
                    "material_id": profile.material_id,
                    "diffuse_density": profile.diffuse_density,
                    "observation_count": observations,
                    "rectangular_log10_rmse": candidate_error,
                    "hann_log10_rmse": incumbent_error,
                    "rectangular_relative_improvement": (
                        (incumbent_error - candidate_error) / incumbent_error
                    ),
                    "rectangular_profile_top1_correct": correct,
                }
            )
        candidate_median = float(np.median(candidate_errors))
        incumbent_median = float(np.median(incumbent_errors))
        shape_metrics[f"{shape[0]}x{shape[1]}"] = {
            "rectangular_median_log10_rmse": candidate_median,
            "rectangular_worst_log10_rmse": max(candidate_errors),
            "hann_median_log10_rmse": incumbent_median,
            "hann_worst_log10_rmse": max(incumbent_errors),
            "rectangular_relative_median_improvement": (
                (incumbent_median - candidate_median) / incumbent_median
            ),
            "rectangular_profile_top1_rate": candidate_correct / len(profiles),
        }

    evaluation = contract["evaluation"]
    metrics = list(shape_metrics.values())
    gates = {
        "profile_count": len(profiles) == evaluation["required_profile_count"],
        "shape_count": len(metrics) == evaluation["required_shape_count"],
        "observation_count": all(
            row["observation_count"]
            == evaluation["required_observations_per_profile_and_shape"]
            for row in rows
        ),
        "rectangular_median_beats_hann": sum(
            item["rectangular_median_log10_rmse"]
            < item["hann_median_log10_rmse"]
            for item in metrics
        )
        >= evaluation["minimum_shapes_where_rectangular_median_beats_hann"],
        "relative_median_improvement": all(
            item["rectangular_relative_median_improvement"]
            >= evaluation["minimum_each_shape_relative_median_improvement"]
            for item in metrics
        ),
        "rectangular_median_error": all(
            item["rectangular_median_log10_rmse"]
            <= evaluation["maximum_rectangular_median_log10_rmse"]
            for item in metrics
        ),
        "rectangular_worst_error": all(
            item["rectangular_worst_log10_rmse"]
            <= evaluation["maximum_rectangular_worst_log10_rmse"]
            for item in metrics
        ),
        "rectangular_top1": all(
            item["rectangular_profile_top1_rate"]
            >= evaluation["minimum_rectangular_profile_top1_rate"]
            for item in metrics
        ),
        "every_profile_not_worse": all_candidate_not_worse,
        "finite_positive_radial_estimates": all_finite_positive,
        "repeat_exact": all_repeat_exact,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "parent_bundle_id": parents["p4bl_bundle_id"],
        "profile_count": len(profiles),
        "shape_metrics": shape_metrics,
        "rows": rows,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "promote_rectangular_synthetic_nps_measurement_bridge"
            if automatic_pass
            else "retain_hann_synthetic_nps_measurement_bridge"
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
