"""Group-held evaluator for already-rendered deterministic look candidates.

This module never renders or predicts RGB.  It extracts bounded, interpretable
statistics from a source and the explicit candidate images that already exist,
then asks whether those statistics can rank the sealed blind winner while
holding out whole sources, experiments and source cohorts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.srgb_transfer import encoded_srgb_to_linear


SCHEMA = "neuro_film.u5_r2bn9_historical_blind_candidate_evaluator_report.v1"


class HistoricalBlindCandidateEvaluatorError(RuntimeError):
    """Raised when candidate pixels, labels or frozen evaluation drift."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_exact_json(root: Path, binding: Mapping[str, Any]) -> Any:
    path = root / str(binding["path"])
    actual = _sha256_file(path)
    if actual != binding["sha256"]:
        raise HistoricalBlindCandidateEvaluatorError(
            f"identity drift: {binding['path']}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_rgb(path: Path, expected_sha256: str, maximum_side: int) -> np.ndarray:
    if _sha256_file(path) != expected_sha256:
        raise HistoricalBlindCandidateEvaluatorError(f"pixel drift: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise HistoricalBlindCandidateEvaluatorError(f"expected RGB image: {path}")
    if decoded.dtype == np.uint8:
        scale = np.float32(255.0)
    elif decoded.dtype == np.uint16:
        scale = np.float32(65535.0)
    else:
        raise HistoricalBlindCandidateEvaluatorError(
            f"unsupported candidate dtype {decoded.dtype}: {path}"
        )
    rgb = decoded[..., ::-1].astype(np.float32) / scale
    height, width = rgb.shape[:2]
    factor = min(1.0, float(maximum_side) / max(height, width))
    if factor < 1.0:
        rgb = cv2.resize(
            rgb,
            (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
            interpolation=cv2.INTER_AREA,
        )
    minimum = float(np.min(rgb))
    maximum = float(np.max(rgb))
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise HistoricalBlindCandidateEvaluatorError(
            f"resampled statistic input outside encoded RGB tolerance: {path}"
        )
    # OpenCV's float32 area reducer may overshoot a legal endpoint by one ULP.
    # This projection is confined to the 192px statistics view; source files,
    # rendered candidates and severe-boundary evidence remain untouched.
    rgb = np.clip(rgb, np.float32(0.0), np.float32(1.0))
    return np.asarray(rgb, dtype=np.float32)


_STAT_NAMES = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "luma_q01",
    "luma_q10",
    "luma_q25",
    "luma_q50",
    "luma_q75",
    "luma_q90",
    "luma_q99",
    "luma_mean",
    "luma_std",
    "chroma_q50",
    "chroma_q90",
    "chroma_mean",
    "chroma_std",
    "ab_mean_a",
    "ab_mean_b",
    "gradient_mean",
    "gradient_p90",
)


def _image_stats(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = np.asarray(
        encoded_srgb_to_linear(np.asarray(encoded, dtype=np.float32)),
        dtype=np.float32,
    )
    lab = linear_rgb_to_lab(linear, working_space="linear_srgb")
    luma = lab[..., 0].astype(np.float64) / 100.0
    chroma = np.linalg.norm(lab[..., 1:3].astype(np.float64), axis=-1) / 100.0
    dx = np.diff(luma, axis=1, append=luma[:, -1:])
    dy = np.diff(luma, axis=0, append=luma[-1:, :])
    gradient = np.sqrt(dx * dx + dy * dy)
    rgb64 = encoded.astype(np.float64)
    values = np.concatenate(
        [
            np.mean(rgb64, axis=(0, 1)),
            np.std(rgb64, axis=(0, 1)),
            np.quantile(luma, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]),
            np.asarray([np.mean(luma), np.std(luma)]),
            np.quantile(chroma, [0.50, 0.90]),
            np.asarray([np.mean(chroma), np.std(chroma)]),
            np.mean(lab[..., 1:3].astype(np.float64), axis=(0, 1)) / 100.0,
            np.asarray([np.mean(gradient), np.quantile(gradient, 0.90)]),
        ]
    )
    if len(values) != len(_STAT_NAMES) or not np.isfinite(values).all():
        raise HistoricalBlindCandidateEvaluatorError("non-finite image descriptor")
    return values, lab.astype(np.float32), gradient


def _utility_features(
    source: np.ndarray, output: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], int]:
    source_stats, source_lab, source_gradient = _image_stats(source)
    output_stats, output_lab, output_gradient = _image_stats(output)
    delta_stats = output_stats - source_stats
    epsilon = 1.0 / 65535.0
    source_boundary_fraction = float(
        np.mean(np.any((source <= epsilon) | (source >= 1.0 - epsilon), axis=-1))
    )
    output_boundary_fraction = float(
        np.mean(np.any((output <= epsilon) | (output >= 1.0 - epsilon), axis=-1))
    )
    # Historical sources and renders are not guaranteed to share pixel geometry:
    # BH1, for example, compares a bounded display preview with the full RAW
    # WorkingImage render.  Keep this evaluator distributional and never invent
    # pixel correspondence through resizing or orientation guessing.
    lab_mean_delta = np.mean(output_lab.astype(np.float64), axis=(0, 1)) - np.mean(
        source_lab.astype(np.float64), axis=(0, 1)
    )
    source_gradient_mean = float(np.mean(source_gradient))
    output_gradient_mean = float(np.mean(output_gradient))
    if source_gradient_mean <= 1e-8 and output_gradient_mean <= 1e-8:
        gradient_ratio_delta = 0.0
    else:
        gradient_ratio_delta = (
            output_gradient_mean / max(source_gradient_mean, 1e-8) - 1.0
        )
    change = np.asarray(
        [
            np.linalg.norm(lab_mean_delta) / 100.0,
            np.mean(np.abs(delta_stats[[6, 7, 8, 9, 10, 11, 12]])),
            np.mean(np.abs(delta_stats[[15, 16, 17, 18]])),
            output_boundary_fraction - source_boundary_fraction,
            output_boundary_fraction,
            gradient_ratio_delta,
        ],
        dtype=np.float64,
    )
    change_names = (
        "delta_lab_mean_distance",
        "delta_luma_quantile_l1",
        "delta_chroma_summary_l1",
        "delta_boundary_fraction",
        "output_boundary_fraction",
        "delta_gradient_ratio",
    )
    base = np.concatenate([output_stats, delta_stats, change])
    base_names = tuple(f"output_{name}" for name in _STAT_NAMES) + tuple(
        f"delta_{name}" for name in _STAT_NAMES
    ) + change_names
    source_indices = (7, 9, 11, 14, 17, 21)
    source_names = tuple(_STAT_NAMES[index] for index in source_indices)
    signal = np.asarray(
        [
            delta_stats[9],
            delta_stats[14],
            delta_stats[17],
            delta_stats[19],
            change[0],
            change[5],
        ],
        dtype=np.float64,
    )
    signal_names = (
        "delta_luma_q50",
        "delta_luma_std",
        "delta_chroma_mean",
        "delta_ab_mean_a",
        "delta_lab_mean_distance",
        "delta_gradient_ratio",
    )
    interactions = np.outer(source_stats[list(source_indices)], signal).reshape(-1)
    interaction_names = tuple(
        f"interaction_{source_name}__{signal_name}"
        for source_name in source_names
        for signal_name in signal_names
    )
    features = np.concatenate([base, interactions])
    names = base_names + interaction_names
    if len(features) != len(names) or not np.isfinite(features).all():
        raise HistoricalBlindCandidateEvaluatorError("invalid utility feature vector")
    return features, names, len(base)


def _canonical_arm(arm: str, aliases: Mapping[str, str]) -> str:
    return str(aliases.get(arm, arm))


def _build_asset_inventory(
    *, root: Path, config: Mapping[str, Any]
) -> tuple[
    dict[str, tuple[Path, str]],
    dict[tuple[str, str, str], tuple[Path, str]],
    list[dict[str, str]],
]:
    aliases = {str(k): str(v) for k, v in config.get("arm_aliases", {}).items()}
    sources: dict[str, tuple[Path, str]] = {}
    outputs: dict[tuple[str, str, str], tuple[Path, str]] = {}
    bindings: list[dict[str, str]] = []

    def add_source(source_id: str, path: Path, sha256: str) -> None:
        value = (path, sha256)
        previous = sources.get(source_id)
        if previous is not None and previous != value:
            raise HistoricalBlindCandidateEvaluatorError(
                f"source identity conflict: {source_id}"
            )
        sources[source_id] = value

    def add_output(
        experiment_id: str,
        source_id: str,
        arm: str,
        path: Path,
        sha256: str,
    ) -> None:
        key = (experiment_id, source_id, _canonical_arm(arm, aliases))
        value = (path, sha256)
        previous = outputs.get(key)
        if previous is not None and previous != value:
            raise HistoricalBlindCandidateEvaluatorError(
                f"candidate identity conflict: {key}"
            )
        outputs[key] = value

    for provider in config["asset_providers"]:
        experiment_ids = [str(value) for value in provider["experiment_ids"]]
        if not experiment_ids:
            raise HistoricalBlindCandidateEvaluatorError("empty asset namespace")
        report_binding = provider["report"]
        report = _load_exact_json(root, report_binding)
        bindings.append(
            {"path": report_binding["path"], "sha256": report_binding["sha256"]}
        )
        report_path = root / report_binding["path"]
        output_root = root / provider.get("output_root", str(report_path.parent.relative_to(root)))
        source_rows: dict[str, Mapping[str, Any]] = {}
        if provider["source_adapter"] == "report_source_path":
            source_rows = {str(row["source_id"]): row for row in report["rows"]}
            for source_id, row in source_rows.items():
                add_source(source_id, root / row["source_path"], row["source_sha256"])
        elif provider["source_adapter"] == "manifest":
            manifest_binding = provider["manifest"]
            manifest = _load_exact_json(root, manifest_binding)
            bindings.append(
                {"path": manifest_binding["path"], "sha256": manifest_binding["sha256"]}
            )
            source_rows = {str(row["id"]): row for row in manifest}
            for source_id, row in source_rows.items():
                add_source(source_id, root / row["decoded_path"], row["decoded_sha256"])
        else:
            raise HistoricalBlindCandidateEvaluatorError("unknown source adapter")

        adapter = provider["output_adapter"]
        if adapter == "nested_arms":
            for row in report["rows"]:
                source_id = str(row["source_id"])
                for arm, artifact in row["arms"].items():
                    for experiment_id in experiment_ids:
                        add_output(
                            experiment_id,
                            source_id,
                            str(arm),
                            output_root / artifact["output"],
                            artifact["output_sha256"],
                        )
        elif adapter == "row_arm":
            for row in report[provider.get("rows_key", "rows")]:
                for experiment_id in experiment_ids:
                    add_output(
                        experiment_id,
                        str(row["source_id"]),
                        str(row["arm_id"]),
                        output_root / row["output"],
                        row["output_sha256"],
                    )
        elif adapter == "fixed_arm_rows":
            arm = str(provider["fixed_arm_id"])
            for row in report["rows"]:
                for experiment_id in experiment_ids:
                    add_output(
                        experiment_id,
                        str(row["source_id"]),
                        arm,
                        output_root / row["output"],
                        row["output_sha256"],
                    )
        else:
            raise HistoricalBlindCandidateEvaluatorError("unknown output adapter")
    return sources, outputs, bindings


def _pair_training_rows(
    units: Sequence[Mapping[str, Any]],
    features: Mapping[tuple[str, str, str], np.ndarray],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for index in indices:
        unit = units[int(index)]
        winner = str(unit["majority_arm"])
        arms = list(unit["arms"])
        losers = [arm for arm in arms if arm != winner]
        for loser in losers:
            first, second = sorted((winner, loser))
            x_rows.append(
                features[(unit["experiment_id"], unit["source_id"], first)]
                - features[(unit["experiment_id"], unit["source_id"], second)]
            )
            labels.append(int(winner == first))
            weights.append(1.0 / len(losers))
    if not x_rows or len(set(labels)) != 2:
        raise HistoricalBlindCandidateEvaluatorError("training fold lacks both labels")
    return np.stack(x_rows), np.asarray(labels), np.asarray(weights)


def _fit_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    regularization_c: float,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler(with_mean=False)
    scaled = scaler.fit_transform(x, sample_weight=weights)
    model = LogisticRegression(
        C=float(regularization_c),
        class_weight="balanced",
        fit_intercept=False,
        max_iter=2000,
        random_state=int(seed),
        solver="liblinear",
    )
    model.fit(scaled, y, sample_weight=weights)
    return scaler, model


def _predict_unit(
    unit: Mapping[str, Any],
    features: Mapping[tuple[str, str, str], np.ndarray],
    scaler: StandardScaler,
    model: LogisticRegression,
) -> tuple[str, dict[str, float]]:
    scores = {
        arm: float(
            model.decision_function(
                scaler.transform(
                    features[(unit["experiment_id"], unit["source_id"], arm)][None, :]
                )
            )[0]
        )
        for arm in unit["arms"]
    }
    selected = min(scores, key=lambda arm: (-scores[arm], arm))
    return selected, scores


def _cross_validate(
    *,
    units: Sequence[Mapping[str, Any]],
    features: Mapping[tuple[str, str, str], np.ndarray],
    groups: Sequence[str],
    feature_slice: slice,
    regularization_c: float,
    seed: int,
) -> dict[str, Any]:
    sliced = {key: value[feature_slice] for key, value in features.items()}
    predictions: list[dict[str, Any]] = []
    group_values = sorted(set(groups))
    for fold_index, held_group in enumerate(group_values):
        train = np.asarray([i for i, group in enumerate(groups) if group != held_group])
        held = [i for i, group in enumerate(groups) if group == held_group]
        x, y, weights = _pair_training_rows(units, sliced, train)
        scaler, model = _fit_model(
            x,
            y,
            weights,
            regularization_c=regularization_c,
            seed=seed + fold_index,
        )
        for index in held:
            unit = units[index]
            selected, scores = _predict_unit(unit, sliced, scaler, model)
            predictions.append(
                {
                    "unit_index": index,
                    "held_group": held_group,
                    "selected_arm": selected,
                    "winner_arm": unit["majority_arm"],
                    "correct": selected == unit["majority_arm"],
                    "scores": scores,
                }
            )
    predictions.sort(key=lambda row: row["unit_index"])
    return {
        "accuracy": float(np.mean([row["correct"] for row in predictions])),
        "correct_units": int(sum(row["correct"] for row in predictions)),
        "unit_count": len(predictions),
        "predictions": predictions,
    }


def _experiment_global_baseline(
    units: Sequence[Mapping[str, Any]], groups: Sequence[str]
) -> dict[str, Any]:
    predictions = []
    for index, (unit, held_source) in enumerate(zip(units, groups, strict=True)):
        train = [
            other
            for other, source in zip(units, groups, strict=True)
            if source != held_source and other["experiment_id"] == unit["experiment_id"]
        ]
        counts = Counter(other["majority_arm"] for other in train)
        selected = min(counts, key=lambda arm: (-counts[arm], arm))
        predictions.append(
            {
                "unit_index": index,
                "selected_arm": selected,
                "winner_arm": unit["majority_arm"],
                "correct": selected == unit["majority_arm"],
            }
        )
    return {
        "accuracy": float(np.mean([row["correct"] for row in predictions])),
        "correct_units": int(sum(row["correct"] for row in predictions)),
        "unit_count": len(predictions),
        "predictions": predictions,
    }


def _bootstrap_gain_lower(
    full: Mapping[str, Any],
    baseline: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> float:
    by_source: dict[str, list[float]] = defaultdict(list)
    for full_row, base_row, unit in zip(
        full["predictions"], baseline["predictions"], units, strict=True
    ):
        by_source[str(unit["source_id"])].append(
            float(full_row["correct"]) - float(base_row["correct"])
        )
    source_gains = np.asarray(
        [np.mean(values) for _, values in sorted(by_source.items())], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(source_gains), size=(resamples, len(source_gains)))
    gains = np.mean(source_gains[samples], axis=1)
    return float(np.quantile(gains, 0.025))


def _permutation_p_value(
    *,
    units: Sequence[Mapping[str, Any]],
    features: Mapping[tuple[str, str, str], np.ndarray],
    groups: Sequence[str],
    feature_slice: slice,
    observed_gain: float,
    regularization_c: float,
    seed: int,
    permutations: int,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    indices_by_experiment: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        indices_by_experiment[str(unit["experiment_id"])].append(index)
    gains = []
    for permutation in range(permutations):
        shuffled = [dict(unit) for unit in units]
        for indices in indices_by_experiment.values():
            winners = [shuffled[index]["majority_arm"] for index in indices]
            rng.shuffle(winners)
            for index, winner in zip(indices, winners, strict=True):
                shuffled[index]["majority_arm"] = winner
        full = _cross_validate(
            units=shuffled,
            features=features,
            groups=groups,
            feature_slice=feature_slice,
            regularization_c=regularization_c,
            seed=seed + 1000 + permutation,
        )
        baseline = _experiment_global_baseline(shuffled, groups)
        gains.append(float(full["accuracy"] - baseline["accuracy"]))
    p_value = (1 + sum(value >= observed_gain for value in gains)) / (
        permutations + 1
    )
    return float(p_value), gains


def run_evaluator(*, root: Path, config_path: Path) -> dict[str, Any]:
    """Run the frozen candidate-evaluator pilot and return exact evidence."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "neuro_film.u5_r2bn9_historical_blind_candidate_evaluator.v1":
        raise HistoricalBlindCandidateEvaluatorError("BN9 contract drift")
    feasibility = _load_exact_json(root, config["feasibility_report"])
    if not feasibility.get("pass"):
        raise HistoricalBlindCandidateEvaluatorError("BN8 did not open BN9")
    aliases = {str(k): str(v) for k, v in config.get("arm_aliases", {}).items()}
    units = []
    for row in feasibility["effective_units"]:
        if row["majority_arm"] is None:
            continue
        units.append(
            {
                **row,
                "majority_arm": _canonical_arm(str(row["majority_arm"]), aliases),
                "arms": sorted({_canonical_arm(str(arm), aliases) for arm in row["arms"]}),
            }
        )
    sources, outputs, asset_bindings = _build_asset_inventory(root=root, config=config)
    maximum_side = int(config["features"]["maximum_side"])
    cache: dict[tuple[Path, str], np.ndarray] = {}

    def decode(binding: tuple[Path, str]) -> np.ndarray:
        if binding not in cache:
            cache[binding] = _decode_rgb(binding[0], binding[1], maximum_side)
        return cache[binding]

    features: dict[tuple[str, str, str], np.ndarray] = {}
    feature_names: tuple[str, ...] | None = None
    base_dimension: int | None = None
    pixel_identities = []
    for unit in units:
        experiment_id = str(unit["experiment_id"])
        source_id = str(unit["source_id"])
        if source_id not in sources:
            raise HistoricalBlindCandidateEvaluatorError(f"missing source asset: {source_id}")
        source = decode(sources[source_id])
        for arm in unit["arms"]:
            key = (experiment_id, source_id, arm)
            if key in features:
                continue
            if key not in outputs:
                raise HistoricalBlindCandidateEvaluatorError(f"missing candidate asset: {key}")
            output = decode(outputs[key])
            vector, names, base_dim = _utility_features(source, output)
            features[key] = vector
            if feature_names is None:
                feature_names, base_dimension = names, base_dim
            elif names != feature_names or base_dim != base_dimension:
                raise HistoricalBlindCandidateEvaluatorError("feature schema drift")
            pixel_identities.append(
                {
                    "source_id": source_id,
                    "experiment_id": experiment_id,
                    "arm": arm,
                    "source_sha256": sources[source_id][1],
                    "output_sha256": outputs[key][1],
                }
            )
    assert feature_names is not None and base_dimension is not None
    expected_dimension = int(config["features"]["expected_dimension"])
    expected_base = int(config["features"]["expected_base_dimension"])
    if len(feature_names) != expected_dimension or base_dimension != expected_base:
        raise HistoricalBlindCandidateEvaluatorError("frozen feature dimension drift")
    feature_identity = hashlib.sha256()
    for key in sorted(features):
        feature_identity.update("\0".join(key).encode("utf-8"))
        feature_identity.update(np.asarray(features[key], dtype="<f8").tobytes())

    source_groups = [str(unit["source_id"]) for unit in units]
    experiment_groups = [str(unit["experiment_id"]) for unit in units]
    cohort_by_experiment = {
        experiment_id: f"cohort_{index:02d}"
        for index, cohort in enumerate(feasibility["source_cohorts"])
        for experiment_id in cohort["experiment_ids"]
    }
    cohort_groups = [cohort_by_experiment[value] for value in experiment_groups]
    c_value = float(config["model"]["regularization_c"])
    seed = int(config["model"]["seed"])
    full_slice = slice(0, len(feature_names))
    base_slice = slice(0, base_dimension)
    source_cv = _cross_validate(
        units=units,
        features=features,
        groups=source_groups,
        feature_slice=full_slice,
        regularization_c=c_value,
        seed=seed,
    )
    base_feature_cv = _cross_validate(
        units=units,
        features=features,
        groups=source_groups,
        feature_slice=base_slice,
        regularization_c=c_value,
        seed=seed,
    )
    experiment_cv = _cross_validate(
        units=units,
        features=features,
        groups=experiment_groups,
        feature_slice=full_slice,
        regularization_c=c_value,
        seed=seed,
    )
    cohort_cv = _cross_validate(
        units=units,
        features=features,
        groups=cohort_groups,
        feature_slice=full_slice,
        regularization_c=c_value,
        seed=seed,
    )
    baseline = _experiment_global_baseline(units, source_groups)
    source_gain = float(source_cv["accuracy"] - baseline["accuracy"])
    interaction_gain = float(source_cv["accuracy"] - base_feature_cv["accuracy"])
    chance = float(np.mean([1.0 / len(unit["arms"]) for unit in units]))
    lower = _bootstrap_gain_lower(
        source_cv,
        baseline,
        units,
        seed=int(config["controls"]["bootstrap_seed"]),
        resamples=int(config["controls"]["bootstrap_resamples"]),
    )
    p_value, permutation_gains = _permutation_p_value(
        units=units,
        features=features,
        groups=source_groups,
        feature_slice=full_slice,
        observed_gain=source_gain,
        regularization_c=c_value,
        seed=int(config["controls"]["permutation_seed"]),
        permutations=int(config["controls"]["permutations"]),
    )
    metrics = {
        "unit_count": len(units),
        "source_count": len(set(source_groups)),
        "experiment_count": len(set(experiment_groups)),
        "cohort_count": len(set(cohort_groups)),
        "candidate_asset_count": len(features),
        "feature_dimension": len(feature_names),
        "base_feature_dimension": base_dimension,
        "leave_one_source_out_accuracy": source_cv["accuracy"],
        "experiment_global_baseline_accuracy": baseline["accuracy"],
        "leave_one_source_out_gain": source_gain,
        "base_feature_only_accuracy": base_feature_cv["accuracy"],
        "source_interaction_gain": interaction_gain,
        "leave_one_experiment_out_accuracy": experiment_cv["accuracy"],
        "leave_one_cohort_out_accuracy": cohort_cv["accuracy"],
        "mean_random_choice_accuracy": chance,
        "leave_one_experiment_gain_over_chance": experiment_cv["accuracy"] - chance,
        "leave_one_cohort_gain_over_chance": cohort_cv["accuracy"] - chance,
        "source_bootstrap_gain_lower_95": lower,
        "grouped_permutation_p_value": p_value,
        "confirmed_severe_artifact_count": feasibility["measurements"][
            "confirmed_severe_artifact_count"
        ],
    }
    thresholds = config["gates"]
    gates = {
        "minimum_leave_one_source_out_gain": source_gain
        >= thresholds["minimum_leave_one_source_out_gain"],
        "minimum_source_interaction_gain": interaction_gain
        >= thresholds["minimum_source_interaction_gain"],
        "minimum_leave_one_experiment_gain_over_chance": metrics[
            "leave_one_experiment_gain_over_chance"
        ]
        >= thresholds["minimum_leave_one_experiment_gain_over_chance"],
        "minimum_leave_one_cohort_gain_over_chance": metrics[
            "leave_one_cohort_gain_over_chance"
        ]
        >= thresholds["minimum_leave_one_cohort_gain_over_chance"],
        "minimum_source_bootstrap_gain_lower_95": lower
        >= thresholds["minimum_source_bootstrap_gain_lower_95"],
        "maximum_grouped_permutation_p_value": p_value
        <= thresholds["maximum_grouped_permutation_p_value"],
        "maximum_confirmed_severe_artifact_count": metrics[
            "confirmed_severe_artifact_count"
        ]
        <= thresholds["maximum_confirmed_severe_artifact_count"],
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": (
            "candidate_evaluator_pass_open_fresh_confirmation"
            if passed
            else "candidate_evaluator_fail_close_learning_route"
        ),
        "pass": passed,
        "inputs": {
            "config_sha256": _sha256_file(config_path),
            "feasibility_report_sha256": config["feasibility_report"]["sha256"],
            "asset_bindings": asset_bindings,
            "feature_identity_sha256": feature_identity.hexdigest(),
            "pixel_identity_count": len(pixel_identities),
        },
        "features": {
            "names": list(feature_names),
            "candidate_identity_or_experiment_id_used": False,
            "final_rgb_prediction": False,
            "source_interactions_start": base_dimension,
        },
        "metrics": metrics,
        "thresholds": thresholds,
        "gates": gates,
        "source_cv": source_cv,
        "base_feature_cv": base_feature_cv,
        "experiment_cv": experiment_cv,
        "cohort_cv": cohort_cv,
        "experiment_global_baseline": baseline,
        "permutation_gain_summary": {
            "count": len(permutation_gains),
            "median": float(np.median(permutation_gains)),
            "p95": float(np.quantile(permutation_gains, 0.95)),
            "maximum": float(np.max(permutation_gains)),
        },
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(report)
    return report


__all__ = [
    "HistoricalBlindCandidateEvaluatorError",
    "SCHEMA",
    "_utility_features",
    "run_evaluator",
]
