"""Source-only hard retrieval over a frozen bank of explicit FiveK operators."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_casebank_oracle import _even_samples, _fit_one, _rmse
from src.roll2film.filmset_case_retrieval import content_descriptor
from src.roll2film.triangular_logit_transport import TriangularLogitTransport


class FiveKSourceHardRetrievalError(ValueError):
    """Raised when source-only selector evidence violates the frozen protocol."""


def _rgb(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[-1] != 3
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise FiveKSourceHardRetrievalError("invalid source RGB")
    return array


def _cell_ids(height: int, width: int, grid: int) -> np.ndarray:
    yy = np.minimum(np.arange(height) * grid // height, grid - 1)
    xx = np.minimum(np.arange(width) * grid // width, grid - 1)
    return (yy[:, None] * grid + xx[None, :]).ravel()


def source_descriptor(
    rgb: np.ndarray,
    family: str,
    spec: Mapping[str, Any],
) -> np.ndarray:
    """Extract one deterministic source-only descriptor."""

    image = _rgb(rgb)
    grid = int(spec["spatial_grid"])
    if family in {"global_photometric", "spatial_photometric"}:
        # The mature FilmSet helper validates a population with at least two
        # images. Duplicate this one source only at the extraction boundary;
        # each returned row is image-local and therefore unchanged.
        samples = np.repeat(image.reshape(1, -1, 3), 2, axis=0)
        result = content_descriptor(
            samples,
            _cell_ids(image.shape[0], image.shape[1], grid),
            include_spatial_cells=family == "spatial_photometric",
        )[0]
    elif family == "tone_layout":
        luma = (
            0.2126 * image[..., 0]
            + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2]
        )
        features: list[float] = list(
            np.quantile(
                luma,
                np.linspace(0.0, 1.0, int(spec["luma_quantiles"])),
            )
        )
        y_edges = np.linspace(0, luma.shape[0], grid + 1, dtype=int)
        x_edges = np.linspace(0, luma.shape[1], grid + 1, dtype=int)
        for y in range(grid):
            for x in range(grid):
                block = luma[
                    y_edges[y] : y_edges[y + 1],
                    x_edges[x] : x_edges[x + 1],
                ]
                features.extend((float(np.mean(block)), float(np.std(block))))
        dx = np.diff(luma, axis=1)[:-1]
        dy = np.diff(luma, axis=0)[:, :-1]
        magnitude = np.sqrt(dx * dx + dy * dy) / np.sqrt(2.0)
        features.extend(
            np.quantile(
                magnitude,
                np.linspace(
                    0.0, 1.0, int(spec["gradient_magnitude_quantiles"])
                ),
            ).tolist()
        )
        result = np.asarray(features, dtype=np.float64)
    else:
        raise FiveKSourceHardRetrievalError("unknown descriptor family")
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise FiveKSourceHardRetrievalError("non-finite source descriptor")
    return result


def _standardized_distances(
    bank: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(bank, axis=0)
    scale = np.std(bank, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    bank_z = (bank - mean) / scale
    query_z = (query - mean) / scale
    distances = np.sqrt(
        np.sum((query_z[:, None, :] - bank_z[None, :, :]) ** 2, axis=-1)
    )
    return distances, bank_z, query_z


def _source_only_threshold(
    bank: np.ndarray,
    groups: np.ndarray,
    quantile: float,
) -> float:
    distances, _, _ = _standardized_distances(bank, bank)
    mask = groups[:, None] == groups[None, :]
    distances[mask] = np.inf
    nearest = np.min(distances, axis=1)
    if not np.all(np.isfinite(nearest)):
        raise FiveKSourceHardRetrievalError("insufficient cross-group support")
    return float(np.quantile(nearest, quantile))


def _operators(report: Mapping[str, Any]) -> list[TriangularLogitTransport]:
    return [
        TriangularLogitTransport(
            np.asarray(item["parameters"], dtype=np.float64),
            dose=float(item["dose"]),
        )
        for item in report["case_bank"]
    ]


def prepare_development_evidence(
    rows: Sequence[Mapping[str, Any]],
    oracle_report: Mapping[str, Any],
    operator_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Cache target-dependent errors once; selectors consume sources only."""

    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in ordered]
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    report_ids = [str(item["pair_id"]) for item in oracle_report["case_bank"]]
    if ids != report_ids:
        raise FiveKSourceHardRetrievalError("case-bank identity drift")
    operators = _operators(oracle_report)
    samples = int(evaluation["samples_per_image"])
    error_matrix = np.empty((len(ordered), len(ordered)), dtype=np.float64)
    for query_index, row in enumerate(ordered):
        source = _even_samples(_rgb(row["source"]), samples)
        target = _even_samples(_rgb(row["target"]), samples)
        for case_index, operator in enumerate(operators):
            error_matrix[query_index, case_index] = _rmse(
                operator.apply(source), target
            )

    global_errors = np.empty(len(ordered), dtype=np.float64)
    oracle_errors = np.empty(len(ordered), dtype=np.float64)
    random_errors = np.empty(len(ordered), dtype=np.float64)
    random_seed = int(evaluation["random_case_seed"])
    for held_group in sorted(set(groups.tolist())):
        train = np.flatnonzero(groups != held_group)
        held = np.flatnonzero(groups == held_group)
        pooled_source = []
        pooled_target = []
        for index in train:
            pooled_source.append(
                _even_samples(
                    _rgb(ordered[index]["source"]),
                    int(operator_config["pooled_samples_per_image"]),
                )
            )
            pooled_target.append(
                _even_samples(
                    _rgb(ordered[index]["target"]),
                    int(operator_config["pooled_samples_per_image"]),
                )
            )
        _, pooled, _ = _fit_one(
            np.concatenate(pooled_source)[:, None, :],
            np.concatenate(pooled_target)[:, None, :],
            {
                **operator_config,
                "fit_samples_per_image": sum(map(len, pooled_source)),
            },
        )
        for index in held:
            source = _even_samples(_rgb(ordered[index]["source"]), samples)
            target = _even_samples(_rgb(ordered[index]["target"]), samples)
            global_errors[index] = _rmse(pooled.apply(source), target)
            oracle_errors[index] = float(np.min(error_matrix[index, train]))
            digest = hashlib.sha256(
                f"{random_seed}\0{ids[index]}".encode("utf-8")
            ).digest()
            random_index = train[int.from_bytes(digest[:8], "big") % len(train)]
            random_errors[index] = error_matrix[index, random_index]
    return {
        "ids": ids,
        "groups": groups,
        "operators": operators,
        "error_matrix": error_matrix,
        "global_errors": global_errors,
        "oracle_errors": oracle_errors,
        "random_errors": random_errors,
    }


def _group_bootstrap(
    baseline: np.ndarray,
    selected: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> list[float]:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        base = float(np.mean(baseline[chosen]))
        values.append(
            (base - float(np.mean(selected[chosen]))) / max(base, 1.0e-12)
        )
    return values


def _metrics(
    *,
    baseline: np.ndarray,
    selected: np.ndarray,
    oracle: np.ndarray,
    random: np.ndarray,
    groups: np.ndarray,
    selected_ids: list[str],
    fallback: np.ndarray,
    gates: Mapping[str, Any],
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    bootstrap = _group_bootstrap(
        baseline,
        selected,
        groups,
        seed=bootstrap_seed,
        repetitions=bootstrap_repetitions,
    )
    counts = Counter(
        case_id for case_id, is_fallback in zip(selected_ids, fallback) if not is_fallback
    )
    base_mean = float(np.mean(baseline))
    selected_mean = float(np.mean(selected))
    oracle_mean = float(np.mean(oracle))
    metrics = {
        "mean_improvement_over_global": (base_mean - selected_mean)
        / max(base_mean, 1.0e-12),
        "win_fraction_over_global": float(np.mean(selected < baseline)),
        "p95_ratio_to_global": float(np.quantile(selected, 0.95))
        / max(float(np.quantile(baseline, 0.95)), 1.0e-12),
        "worst_ratio_to_global": float(np.max(selected))
        / max(float(np.max(baseline)), 1.0e-12),
        "oracle_gap_closure": (base_mean - selected_mean)
        / max(base_mean - oracle_mean, 1.0e-12),
        "mean_improvement_over_random_case": (
            float(np.mean(random)) - selected_mean
        )
        / max(float(np.mean(random)), 1.0e-12),
        "win_fraction_over_random_case": float(np.mean(selected < random)),
        "group_bootstrap_improvement_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "fallback_fraction": float(np.mean(fallback)),
        "distinct_selected_cases": len(counts),
        "maximum_selected_case_share": (
            max(counts.values()) / len(selected) if counts else 0.0
        ),
    }
    passed = {
        "mean": metrics["mean_improvement_over_global"]
        >= gates["minimum_mean_improvement_over_global"],
        "wins": metrics["win_fraction_over_global"]
        >= gates["minimum_win_fraction_over_global"],
        "p95": metrics["p95_ratio_to_global"]
        <= gates["maximum_p95_ratio_to_global"],
        "worst": metrics["worst_ratio_to_global"]
        <= gates["maximum_worst_ratio_to_global"],
        "oracle_gap": metrics["oracle_gap_closure"]
        >= gates["minimum_oracle_gap_closure"],
        "random_mean": metrics["mean_improvement_over_random_case"]
        >= gates["minimum_mean_improvement_over_random_case"],
        "random_wins": metrics["win_fraction_over_random_case"]
        >= gates["minimum_win_fraction_over_random_case"],
        "bootstrap": metrics["group_bootstrap_improvement_ci95"][0]
        > gates["minimum_bootstrap_lower_improvement"],
        "fallback": metrics["fallback_fraction"]
        <= gates["maximum_fallback_fraction"],
        "support": metrics["distinct_selected_cases"]
        >= gates["minimum_distinct_selected_cases"],
        "concentration": metrics["maximum_selected_case_share"]
        <= gates["maximum_selected_case_share"],
    }
    return {
        "metrics": metrics,
        "gates": passed,
        "automatic_pass": all(passed.values()),
        "selected_case_counts": dict(sorted(counts.items())),
    }


def evaluate_development_family(
    *,
    rows: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    family: str,
    descriptor_spec: Mapping[str, Any],
    selector_spec: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = prepared["ids"]
    groups = np.asarray(prepared["groups"], dtype=object)
    features = np.stack(
        [source_descriptor(row["source"], family, descriptor_spec) for row in ordered]
    )
    selected_errors = np.empty(len(ordered), dtype=np.float64)
    fallback = np.zeros(len(ordered), dtype=bool)
    selected_ids = [""] * len(ordered)
    distances_out = np.empty(len(ordered), dtype=np.float64)
    thresholds_out = np.empty(len(ordered), dtype=np.float64)
    for held_group in sorted(set(groups.tolist())):
        train = np.flatnonzero(groups != held_group)
        held = np.flatnonzero(groups == held_group)
        bank = features[train]
        bank_groups = groups[train]
        distances, _, _ = _standardized_distances(bank, features[held])
        nearest_local = np.argmin(distances, axis=1)
        nearest_distances = distances[np.arange(len(held)), nearest_local]
        threshold = _source_only_threshold(
            bank,
            bank_groups,
            float(selector_spec["ood_distance_quantile"]),
        )
        for local_index, query_index in enumerate(held):
            case_index = int(train[nearest_local[local_index]])
            selected_ids[query_index] = ids[case_index]
            distances_out[query_index] = nearest_distances[local_index]
            thresholds_out[query_index] = threshold
            fallback[query_index] = nearest_distances[local_index] > threshold
            selected_errors[query_index] = (
                prepared["global_errors"][query_index]
                if fallback[query_index]
                else prepared["error_matrix"][query_index, case_index]
            )
    result = _metrics(
        baseline=np.asarray(prepared["global_errors"]),
        selected=selected_errors,
        oracle=np.asarray(prepared["oracle_errors"]),
        random=np.asarray(prepared["random_errors"]),
        groups=groups,
        selected_ids=selected_ids,
        fallback=fallback,
        gates=gates,
        bootstrap_seed=int(selector_spec["bootstrap_seed"]),
        bootstrap_repetitions=int(selector_spec["bootstrap_repetitions"]),
    )
    result["rows"] = [
        {
            "pair_id": ids[index],
            "group": str(groups[index]),
            "selected_case_id": selected_ids[index],
            "distance": float(distances_out[index]),
            "threshold": float(thresholds_out[index]),
            "fallback": bool(fallback[index]),
            "global_rmse": float(prepared["global_errors"][index]),
            "selected_rmse": float(selected_errors[index]),
            "oracle_rmse": float(prepared["oracle_errors"][index]),
        }
        for index in range(len(ordered))
    ]
    return result


def evaluate_confirmation_family(
    *,
    development_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    oracle_report: Mapping[str, Any],
    family: str,
    descriptor_spec: Mapping[str, Any],
    selector_spec: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    development = sorted(development_rows, key=lambda row: str(row["pair_id"]))
    confirmation = sorted(
        confirmation_rows, key=lambda row: str(row["pair_id"])
    )
    development_features = np.stack(
        [source_descriptor(row["source"], family, descriptor_spec) for row in development]
    )
    confirmation_features = np.stack(
        [source_descriptor(row["source"], family, descriptor_spec) for row in confirmation]
    )
    development_groups = np.asarray(
        [str(row["group"]) for row in development], dtype=object
    )
    distances, _, _ = _standardized_distances(
        development_features, confirmation_features
    )
    nearest = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(confirmation)), nearest]
    threshold = _source_only_threshold(
        development_features,
        development_groups,
        float(selector_spec["ood_distance_quantile"]),
    )
    fallback = nearest_distances > threshold
    operators = _operators(oracle_report)
    pooled = TriangularLogitTransport(
        np.asarray(oracle_report["pooled_operator"]["parameters"], dtype=np.float64),
        dose=float(oracle_report["pooled_operator"]["dose"]),
    )
    oracle_rows = {str(row["pair_id"]): row for row in oracle_report["rows"]}
    selected_errors = []
    baseline_errors = []
    oracle_errors = []
    random_errors = []
    selected_ids = []
    output_rows = []
    samples = int(selector_spec["samples_per_confirmation_image"])
    for query_index, row in enumerate(confirmation):
        pair_id = str(row["pair_id"])
        if pair_id not in oracle_rows:
            raise FiveKSourceHardRetrievalError("confirmation identity drift")
        source = _even_samples(_rgb(row["source"]), samples)
        target = _even_samples(_rgb(row["target"]), samples)
        case_index = int(nearest[query_index])
        selected_id = str(development[case_index]["pair_id"])
        selected_error = _rmse(
            pooled.apply(source) if fallback[query_index] else operators[case_index].apply(source),
            target,
        )
        evidence = oracle_rows[pair_id]
        baseline_errors.append(float(evidence["global_rmse"]))
        oracle_errors.append(float(evidence["case_oracle_rmse"]))
        random_errors.append(float(evidence["random_case_rmse"]))
        selected_errors.append(selected_error)
        selected_ids.append(selected_id)
        output_rows.append(
            {
                "pair_id": pair_id,
                "group": str(row["group"]),
                "selected_case_id": selected_id,
                "distance": float(nearest_distances[query_index]),
                "threshold": threshold,
                "fallback": bool(fallback[query_index]),
                "global_rmse": float(evidence["global_rmse"]),
                "selected_rmse": selected_error,
                "oracle_rmse": float(evidence["case_oracle_rmse"]),
            }
        )
    result = _metrics(
        baseline=np.asarray(baseline_errors),
        selected=np.asarray(selected_errors),
        oracle=np.asarray(oracle_errors),
        random=np.asarray(random_errors),
        groups=np.asarray([str(row["group"]) for row in confirmation], dtype=object),
        selected_ids=selected_ids,
        fallback=fallback,
        gates=gates,
        bootstrap_seed=int(selector_spec["bootstrap_seed"]),
        bootstrap_repetitions=int(selector_spec["bootstrap_repetitions"]),
    )
    result["rows"] = output_rows
    return result


__all__ = [
    "FiveKSourceHardRetrievalError",
    "evaluate_confirmation_family",
    "evaluate_development_family",
    "prepare_development_evidence",
    "source_descriptor",
]
