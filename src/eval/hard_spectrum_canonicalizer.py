"""U5.R2H0C2 hard measured-spectrum canonicalizer evaluator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from src.eval.cave_conditional_variability import (
    RepresentativePopulation,
    array_sha256,
    audit_cave_snapshot,
    build_overlap_witness,
    load_representatives,
)
from src.eval.velvia_datasheet_witness import (
    delta_e76,
    reconstruct_reflectances,
    render_witness,
)


class HardCanonicalizerError(ValueError):
    """Raised when H0C2 frozen state is invalid."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nearest_other_scene(population: RepresentativePopulation) -> tuple[np.ndarray, np.ndarray]:
    nearest = np.full(len(population.scene), -1, dtype=np.int64)
    distance = np.full(len(population.scene), np.inf, dtype=np.float64)
    for scene in np.unique(population.scene):
        query = np.flatnonzero(population.scene == scene)
        bank = np.flatnonzero(population.scene != scene)
        tree = cKDTree(population.lab[bank])
        values, local = tree.query(population.lab[query], k=1)
        nearest[query] = bank[np.asarray(local, dtype=np.int64)]
        distance[query] = np.asarray(values, dtype=np.float64)
    if np.any(nearest < 0) or np.any(population.scene == population.scene[nearest]):
        raise HardCanonicalizerError("nearest-neighbour leakage across scene boundary")
    return nearest, distance


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    mass = weights[order]
    cumulative = np.cumsum(mass)
    target = quantile * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(ordered) - 1)
    return float(ordered[index])


def policy_scene_bootstrap(
    scenes: np.ndarray,
    smooth_error: np.ndarray,
    hard_error: np.ndarray,
    tie_tolerance: float,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    unique = np.unique(scenes)
    rng = np.random.default_rng(seed)
    win_rates = []
    reductions = []
    for _ in range(repeats):
        draw = rng.choice(unique, size=len(unique), replace=True)
        counts = Counter(str(value) for value in draw)
        weights = np.asarray([counts[str(scene)] for scene in scenes], dtype=np.float64)
        selected = weights > 0
        mass = float(np.sum(weights[selected]))
        wins = hard_error[selected] < smooth_error[selected] - tie_tolerance
        win_rates.append(float(np.sum(weights[selected] * wins) / mass))
        smooth_median = _weighted_quantile(
            smooth_error[selected], weights[selected], 0.5
        )
        hard_median = _weighted_quantile(hard_error[selected], weights[selected], 0.5)
        reductions.append((smooth_median - hard_median) / max(smooth_median, 1e-12))
    return {
        "win_rate": {
            "lower": float(np.percentile(win_rates, 2.5)),
            "upper": float(np.percentile(win_rates, 97.5)),
        },
        "median_relative_error_reduction": {
            "lower": float(np.percentile(reductions, 2.5)),
            "upper": float(np.percentile(reductions, 97.5)),
        },
    }


def evaluate_threshold(
    threshold: float,
    population: RepresentativePopulation,
    nearest_distance: np.ndarray,
    smooth_error: np.ndarray,
    hard_error: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    selected = nearest_distance <= threshold
    count = int(np.sum(selected))
    if count == 0:
        raise HardCanonicalizerError("threshold selects no query")
    scenes = population.scene[selected]
    scene_counts = Counter(str(value) for value in scenes)
    tolerance = float(config["tie_tolerance_delta_e76"])
    wins = hard_error[selected] < smooth_error[selected] - tolerance
    losses = hard_error[selected] > smooth_error[selected] + tolerance
    ties = ~(wins | losses)
    smooth_summary = _summary(smooth_error[selected])
    hard_summary = _summary(hard_error[selected])
    reduction = (smooth_summary["median"] - hard_summary["median"]) / max(
        smooth_summary["median"], 1e-12
    )
    policy_error = np.where(selected, hard_error, smooth_error)
    policy_summary = _summary(policy_error)
    bootstrap = policy_scene_bootstrap(
        scenes,
        smooth_error[selected],
        hard_error[selected],
        tolerance,
        int(config["gates"]["scene_bootstrap_repeats"]),
        int(config["seed"]),
    )
    gates = config["gates"]
    checks = {
        "selected_queries": count >= int(gates["selected_queries_min"]),
        "coverage": count / len(selected) >= float(gates["coverage_min"]),
        "selected_scenes": len(scene_counts) >= int(gates["selected_scenes_min"]),
        "scene_share": max(scene_counts.values()) / count
        <= float(gates["max_selected_scene_share"]),
        "win_rate": float(np.mean(wins)) >= float(gates["selected_win_rate_min"]),
        "win_rate_bootstrap": bootstrap["win_rate"]["lower"]
        >= float(gates["selected_win_rate_bootstrap_95_lcb_min"]),
        "median_reduction": reduction
        >= float(gates["selected_median_relative_error_reduction_min"]),
        "median_reduction_bootstrap": bootstrap["median_relative_error_reduction"][
            "lower"
        ]
        >= float(
            gates["selected_median_relative_error_reduction_bootstrap_95_lcb_min"]
        ),
        "hard_median": hard_summary["median"]
        <= float(gates["selected_hard_error_delta_e76_median_max"]),
        "hard_p95": hard_summary["p95"]
        <= float(gates["selected_hard_error_delta_e76_p95_max"]),
        "full_p95_noninferiority": policy_summary["p95"]
        <= _summary(smooth_error)["p95"]
        + float(gates["fallback_policy_p95_vs_smooth_max_increase"]),
    }
    return (
        {
            "threshold_delta_e76": threshold,
            "selected_queries": count,
            "coverage": float(count / len(selected)),
            "selected_scenes": len(scene_counts),
            "max_selected_scene_share": float(max(scene_counts.values()) / count),
            "scene_counts": dict(sorted(scene_counts.items())),
            "win_rate": float(np.mean(wins)),
            "tie_rate": float(np.mean(ties)),
            "loss_rate": float(np.mean(losses)),
            "smooth_selected_error_delta_e76": smooth_summary,
            "hard_selected_error_delta_e76": hard_summary,
            "selected_median_relative_error_reduction": float(reduction),
            "fallback_policy_error_delta_e76": policy_summary,
            "fallback_policy_p95_change_vs_smooth": float(
                policy_summary["p95"] - _summary(smooth_error)["p95"]
            ),
            "scene_bootstrap_95_interval": bootstrap,
            "checks": checks,
            "eligible": all(checks.values()),
        },
        policy_error,
    )


def evaluate_hard_canonicalizer(
    root: Path,
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    h0a_config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
    snapshot_root: Path,
    tail_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    audit, scenes = audit_cave_snapshot(snapshot_root, tail_path, parent_config)
    curves, context = build_overlap_witness(root, h0a_config, curve_data, parent_config)
    population = load_representatives(scenes, context, parent_config)
    nearest, nearest_distance = nearest_other_scene(population)

    target = render_witness(population.spectra, context, curves, "D65")
    smooth_spectra = reconstruct_reflectances(
        population.xyz, context.xyz_from_reflectance_d65
    )
    smooth = render_witness(smooth_spectra, context, curves, "D65")
    hard = render_witness(population.spectra[nearest], context, curves, "D65")
    identity_error = delta_e76(population.xyz, target["xyz"])
    smooth_error = delta_e76(smooth["xyz"], target["xyz"])
    hard_error = delta_e76(hard["xyz"], target["xyz"])

    policies = []
    policy_errors: dict[str, np.ndarray] = {}
    for threshold in config["thresholds_delta_e76"]:
        policy, errors = evaluate_threshold(
            float(threshold),
            population,
            nearest_distance,
            smooth_error,
            hard_error,
            config,
        )
        policies.append(policy)
        policy_errors[str(threshold)] = errors
    eligible = [policy for policy in policies if policy["eligible"]]
    selected = max(
        eligible,
        key=lambda policy: (policy["coverage"], -policy["threshold_delta_e76"]),
        default=None,
    )
    decision = (
        "hard_empirical_canonicalizer_candidate"
        if selected is not None
        else "hard_retrieval_rejected"
    )
    all_rgb = np.concatenate(
        [target["linear_srgb"], smooth["linear_srgb"], hard["linear_srgb"]], axis=0
    )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "selected_threshold_delta_e76": (
            selected["threshold_delta_e76"] if selected is not None else None
        ),
        "source_audit": audit,
        "population": {
            "queries": int(len(population.scene)),
            "scenes": int(len(np.unique(population.scene))),
            "wavelengths": int(population.spectra.shape[1]),
        },
        "metrics": {
            "nearest_other_scene_delta_e76": _summary(nearest_distance),
            "known_target_effect_delta_e76": _summary(identity_error),
            "identity_error_delta_e76": _summary(identity_error),
            "smooth_error_delta_e76": _summary(smooth_error),
            "raw_hard_error_delta_e76": _summary(hard_error),
            "raw_linear_srgb_min": float(np.min(all_rgb)),
            "raw_linear_srgb_max": float(np.max(all_rgb)),
            "raw_linear_srgb_out_of_gamut_fraction": float(
                np.mean((all_rgb < 0.0) | (all_rgb > 1.0))
            ),
        },
        "policies": policies,
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "scene": population.scene,
        "cell_row": population.cell_row,
        "cell_column": population.cell_column,
        "nearest_index": nearest,
        "nearest_distance_delta_e76": nearest_distance,
        "identity_error_delta_e76": identity_error,
        "smooth_error_delta_e76": smooth_error,
        "hard_error_delta_e76": hard_error,
        "target_output_xyz": target["xyz"],
        "smooth_output_xyz": smooth["xyz"],
        "hard_output_xyz": hard["xyz"],
    }
    for threshold, errors in policy_errors.items():
        arrays[f"policy_error_delta_e76_{threshold}"] = errors
    return report, arrays


def result_hashes(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "report": canonical_sha256(report),
        "arrays": {name: array_sha256(value) for name, value in sorted(arrays.items())},
    }
