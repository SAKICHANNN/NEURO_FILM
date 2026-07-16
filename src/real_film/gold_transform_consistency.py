"""Whole-roll paired explicit-operator evaluation for BlueNeg Gold100 proxies."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab


class GoldTransformConsistencyError(ValueError):
    """Raised when the frozen RF1.4B1 evaluator contract is violated."""


class ColorOperator(Protocol):
    def apply(self, rgb: np.ndarray) -> np.ndarray: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PairedFrameSamples:
    frame_id: str
    roll_id: str
    source: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        source = _pixels(self.source)
        target = _pixels(self.target)
        if source.shape != target.shape:
            raise GoldTransformConsistencyError("paired frame sample shapes differ")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)


@dataclass(frozen=True)
class IdentityOperator:
    working_space: str

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return _pixels(rgb).copy()

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "identity", "working_space": self.working_space}


@dataclass(frozen=True)
class BoundedAffineOperator:
    matrix: np.ndarray
    bias: np.ndarray
    working_space: str
    identity_shrinkage: float = 0.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if matrix.shape != (3, 3) or bias.shape != (3,):
            raise GoldTransformConsistencyError("affine parameter shape mismatch")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
            raise GoldTransformConsistencyError("affine parameters must be finite")
        if float(np.linalg.det(matrix)) <= 1e-10:
            raise GoldTransformConsistencyError("affine matrix must have positive determinant")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bias", bias)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return _pixels(rgb) @ self.matrix.T + self.bias

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "bounded_affine",
            "working_space": self.working_space,
            "matrix": self.matrix.tolist(),
            "bias": self.bias.tolist(),
            "determinant": float(np.linalg.det(self.matrix)),
            "identity_shrinkage": float(self.identity_shrinkage),
        }


@dataclass(frozen=True)
class SepLUT17Operator:
    x_knots: np.ndarray
    y_knots: np.ndarray
    affine: BoundedAffineOperator

    def __post_init__(self) -> None:
        x_knots = np.asarray(self.x_knots, dtype=np.float64)
        y_knots = np.asarray(self.y_knots, dtype=np.float64)
        if x_knots.ndim != 1 or y_knots.shape != (3, len(x_knots)):
            raise GoldTransformConsistencyError("SepLUT knot shape mismatch")
        if np.any(np.diff(x_knots) <= 0.0) or np.any(np.diff(y_knots, axis=1) <= 0.0):
            raise GoldTransformConsistencyError("SepLUT knots must be strictly monotone")
        if np.min(y_knots) < 0.0 or np.max(y_knots) > 1.0:
            raise GoldTransformConsistencyError("SepLUT outputs must stay in [0, 1]")
        object.__setattr__(self, "x_knots", x_knots)
        object.__setattr__(self, "y_knots", y_knots)

    @property
    def working_space(self) -> str:
        return self.affine.working_space

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _pixels(rgb)
        mapped = np.stack(
            [np.interp(values[:, channel], self.x_knots, self.y_knots[channel]) for channel in range(3)],
            axis=-1,
        )
        return self.affine.apply(mapped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "seplut17_monotone_plus_bounded_ridge_3x3",
            "working_space": self.working_space,
            "x_knots": self.x_knots.tolist(),
            "y_knots": self.y_knots.tolist(),
            "monotone": bool(np.all(np.diff(self.y_knots, axis=1) > 0.0)),
            "affine": self.affine.to_dict(),
        }


def operator_from_dict(record: Mapping[str, Any]) -> ColorOperator:
    """Reconstruct a frozen explicit operator without refitting pixels."""
    kind = str(record["kind"])
    if kind == "identity":
        return IdentityOperator(str(record["working_space"]))
    if kind == "bounded_affine":
        return BoundedAffineOperator(
            np.asarray(record["matrix"], dtype=np.float64),
            np.asarray(record["bias"], dtype=np.float64),
            str(record["working_space"]),
            float(record.get("identity_shrinkage", 0.0)),
        )
    if kind == "seplut17_monotone_plus_bounded_ridge_3x3":
        affine = operator_from_dict(record["affine"])
        if not isinstance(affine, BoundedAffineOperator):
            raise GoldTransformConsistencyError("SepLUT affine payload is not affine")
        return SepLUT17Operator(
            np.asarray(record["x_knots"], dtype=np.float64),
            np.asarray(record["y_knots"], dtype=np.float64),
            affine,
        )
    raise GoldTransformConsistencyError(f"unknown operator kind: {kind}")


def load_paired_frame_samples(
    *,
    download_root: Path,
    pair_records: Sequence[Mapping[str, Any]],
    maximum_pixels_per_frame: int,
) -> list[PairedFrameSamples]:
    """Load exact official bbox/proxy pixels on a deterministic uniform grid."""
    if maximum_pixels_per_frame < 64:
        raise GoldTransformConsistencyError("sample budget must be at least 64 pixels")
    frames: list[PairedFrameSamples] = []
    seen: set[str] = set()
    for row in sorted(pair_records, key=lambda item: str(item["frame_id"])):
        frame_id = str(row["frame_id"])
        if frame_id in seen:
            raise GoldTransformConsistencyError(f"duplicate pair frame: {frame_id}")
        seen.add(frame_id)
        preview_path = download_root / Path(*Path(str(row["preview_path"])).parts)
        proxy_path = download_root / Path(*Path(str(row["proxy_path"])).parts)
        x0, y0, x1, y1 = (int(value) for value in row["bbox"])
        with Image.open(preview_path) as image:
            source_image = ImageOps.exif_transpose(image).convert("RGB").crop((x0, y0, x1, y1))
            source = np.asarray(source_image, dtype=np.float64) / 255.0
        with Image.open(proxy_path) as image:
            target = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float64) / 255.0
        if source.shape != target.shape:
            raise GoldTransformConsistencyError(f"official crop/proxy shape mismatch: {frame_id}")
        height, width = source.shape[:2]
        rows_count = min(height, max(1, int(np.floor(np.sqrt(maximum_pixels_per_frame * height / width)))))
        columns_count = min(width, max(1, maximum_pixels_per_frame // rows_count))
        y_indices = np.linspace(0, height - 1, rows_count, dtype=np.int64)
        x_indices = np.linspace(0, width - 1, columns_count, dtype=np.int64)
        yy, xx = np.meshgrid(y_indices, x_indices, indexing="ij")
        frames.append(
            PairedFrameSamples(
                frame_id=frame_id,
                roll_id=str(row["roll_id"]),
                source=source[yy, xx].reshape(-1, 3),
                target=target[yy, xx].reshape(-1, 3),
            )
        )
    return frames


def balanced_training_arrays(
    frames: Sequence[PairedFrameSamples],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel arrays with equal roll and equal within-roll frame mass."""
    if not frames:
        raise GoldTransformConsistencyError("training frames are empty")
    roll_frame_counts = Counter(frame.roll_id for frame in frames)
    roll_count = len(roll_frame_counts)
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for frame in frames:
        pixel_count = len(frame.source)
        mass = 1.0 / (roll_count * roll_frame_counts[frame.roll_id] * pixel_count)
        sources.append(frame.source)
        targets.append(frame.target)
        weights.append(np.full(pixel_count, mass, dtype=np.float64))
    weight = np.concatenate(weights)
    weight /= weight.sum()
    return np.concatenate(sources), np.concatenate(targets), weight


def fit_per_channel_affine(
    frames: Sequence[PairedFrameSamples], bounds: Mapping[str, Any], working_space: str
) -> BoundedAffineOperator:
    source, target, weights = balanced_training_arrays(frames)
    gains = np.empty(3, dtype=np.float64)
    biases = np.empty(3, dtype=np.float64)
    root_weight = np.sqrt(weights)
    for channel in range(3):
        design = np.stack([source[:, channel], np.ones(len(source))], axis=1)
        parameters = np.linalg.lstsq(design * root_weight[:, None], target[:, channel] * root_weight, rcond=None)[0]
        gains[channel] = np.clip(parameters[0], *map(float, bounds["channel_gain"]))
        biases[channel] = np.clip(parameters[1], *map(float, bounds["channel_bias"]))
    return BoundedAffineOperator(np.diag(gains), biases, working_space)


def fit_matrix_affine(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    bounds: Mapping[str, Any],
    working_space: str,
) -> BoundedAffineOperator:
    source = _pixels(source)
    target = _pixels(target)
    weights = np.asarray(weights, dtype=np.float64)
    if target.shape != source.shape or weights.shape != (len(source),):
        raise GoldTransformConsistencyError("weighted affine inputs have incompatible shapes")
    design = np.concatenate([source, np.ones((len(source), 1))], axis=1)
    root_weight = np.sqrt(weights / weights.sum())
    weighted = design * root_weight[:, None]
    ridge = float(bounds["ridge"])
    regularizer = np.diag([ridge, ridge, ridge, 0.0])
    parameters = np.linalg.solve(weighted.T @ weighted + regularizer, weighted.T @ (target * root_weight[:, None]))
    matrix = np.clip(parameters[:3].T, *map(float, bounds["matrix_coefficient"]))
    bias = np.clip(parameters[3], *map(float, bounds["channel_bias"]))
    matrix, shrinkage = _positive_orientation(matrix)
    return BoundedAffineOperator(matrix, bias, working_space, shrinkage)


def fit_full_affine(
    frames: Sequence[PairedFrameSamples], bounds: Mapping[str, Any], working_space: str
) -> BoundedAffineOperator:
    source, target, weights = balanced_training_arrays(frames)
    return fit_matrix_affine(source, target, weights, bounds, working_space)


def fit_seplut17(
    frames: Sequence[PairedFrameSamples], bounds: Mapping[str, Any], working_space: str
) -> SepLUT17Operator:
    source, target, weights = balanced_training_arrays(frames)
    knot_count = int(bounds["lut_knots"])
    if knot_count != 17:
        raise GoldTransformConsistencyError("RF1.4B1 requires exactly 17 LUT knots")
    x_knots = np.linspace(0.0, 1.0, knot_count)
    y_knots = np.stack(
        [_fit_monotone_channel(source[:, channel], target[:, channel], weights, knot_count) for channel in range(3)]
    )
    mapped = np.stack(
        [np.interp(source[:, channel], x_knots, y_knots[channel]) for channel in range(3)],
        axis=-1,
    )
    affine = fit_matrix_affine(mapped, target, weights, bounds, working_space)
    return SepLUT17Operator(x_knots, y_knots, affine)


def evaluate_operator(
    operator: ColorOperator, frames: Sequence[PairedFrameSamples]
) -> dict[str, Any]:
    per_frame: list[dict[str, Any]] = []
    for frame in frames:
        raw = operator.apply(frame.source)
        if not np.all(np.isfinite(raw)):
            raise GoldTransformConsistencyError("operator produced non-finite output")
        output = np.clip(raw, 0.0, 1.0)
        target_lab = rgb2lab(frame.target.reshape(-1, 1, 3)).reshape(-1, 3)
        output_lab = rgb2lab(output.reshape(-1, 1, 3)).reshape(-1, 3)
        source_lab = rgb2lab(frame.source.reshape(-1, 1, 3)).reshape(-1, 3)
        target_delta = np.linalg.norm(output_lab - target_lab, axis=1)
        style_delta = np.linalg.norm(output_lab - source_lab, axis=1)
        per_frame.append(
            {
                "frame_id": frame.frame_id,
                "roll_id": frame.roll_id,
                "mean_delta_e76_to_target": float(target_delta.mean()),
                "p95_delta_e76_to_target": float(np.quantile(target_delta, 0.95)),
                "rgb_mae_to_target": float(np.mean(np.abs(output - frame.target))),
                "raw_output_clip_fraction": float(np.mean((raw < 0.0) | (raw > 1.0))),
                "median_delta_e76_from_identity": float(np.median(style_delta)),
            }
        )
    keys = (
        "mean_delta_e76_to_target",
        "p95_delta_e76_to_target",
        "rgb_mae_to_target",
        "raw_output_clip_fraction",
        "median_delta_e76_from_identity",
    )
    return {
        "frames": len(per_frame),
        **{key: float(np.mean([row[key] for row in per_frame])) for key in keys},
        "per_frame": per_frame,
    }


def run_whole_roll_evaluation(
    frames: Sequence[PairedFrameSamples], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute all frozen LOO folds, wrong-roll controls, bootstrap, and gates."""
    rolls = sorted({frame.roll_id for frame in frames})
    if len(rolls) < 2:
        raise GoldTransformConsistencyError("whole-roll evaluation needs at least two rolls")
    working_space = str(config["working_space"])
    bounds = config["operator_bounds"]
    fold_results: list[dict[str, Any]] = []
    for held_out in rolls:
        train = [frame for frame in frames if frame.roll_id != held_out]
        test = [frame for frame in frames if frame.roll_id == held_out]
        operators: dict[str, ColorOperator] = {
            "identity": IdentityOperator(working_space),
            "bounded_per_channel_affine": fit_per_channel_affine(train, bounds, working_space),
            "bounded_ridge_3x3_affine": fit_full_affine(train, bounds, working_space),
            "seplut17_monotone_plus_bounded_ridge_3x3": fit_seplut17(train, bounds, working_space),
        }
        metrics = {name: evaluate_operator(operator, test) for name, operator in operators.items()}
        wrong_roll_errors: dict[str, float] = {}
        for training_roll in sorted({frame.roll_id for frame in train}):
            wrong_operator = fit_seplut17(
                [frame for frame in train if frame.roll_id == training_roll], bounds, working_space
            )
            wrong_roll_errors[training_roll] = float(
                evaluate_operator(wrong_operator, test)["mean_delta_e76_to_target"]
            )
        fold_results.append(
            {
                "held_out_roll": held_out,
                "test_frames": len(test),
                "operators": {name: operator.to_dict() for name, operator in operators.items()},
                "metrics": metrics,
                "wrong_roll_seplut_mean_delta_e76": wrong_roll_errors,
                "wrong_roll_median_delta_e76": float(np.median(list(wrong_roll_errors.values()))),
            }
        )
    primary_name = "seplut17_monotone_plus_bounded_ridge_3x3"
    baseline_name = "bounded_per_channel_affine"
    primary = np.asarray([row["metrics"][primary_name]["mean_delta_e76_to_target"] for row in fold_results])
    baseline = np.asarray([row["metrics"][baseline_name]["mean_delta_e76_to_target"] for row in fold_results])
    wrong = np.asarray([row["wrong_roll_median_delta_e76"] for row in fold_results])
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    resamples = int(config["evaluation"]["cluster_bootstrap_resamples"])
    indices = rng.integers(0, len(rolls), size=(resamples, len(rolls)))
    bootstrap = (baseline - primary)[indices].mean(axis=1)
    bootstrap_summary = {
        "resamples": resamples,
        "absolute_improvement_mean": float(np.mean(baseline - primary)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
    }
    aggregate: dict[str, Any] = {}
    for name in ("identity", baseline_name, "bounded_ridge_3x3_affine", primary_name):
        aggregate[name] = {
            key: float(np.mean([row["metrics"][name][key] for row in fold_results]))
            for key in (
                "mean_delta_e76_to_target",
                "p95_delta_e76_to_target",
                "rgb_mae_to_target",
                "raw_output_clip_fraction",
                "median_delta_e76_from_identity",
            )
        }
    gates = config["gates"]
    relative_improvement = float((baseline.mean() - primary.mean()) / max(baseline.mean(), 1e-12))
    improved_rolls = int(np.sum(primary < baseline))
    wrong_rolls_beaten = int(np.sum(primary < wrong))
    style_strength = float(np.median([
        row["metrics"][primary_name]["median_delta_e76_from_identity"] for row in fold_results
    ]))
    clipping_increase = float(
        aggregate[primary_name]["raw_output_clip_fraction"]
        - aggregate[baseline_name]["raw_output_clip_fraction"]
    )
    all_monotone = all(bool(row["operators"][primary_name]["monotone"]) for row in fold_results)
    all_positive = all(
        float(row["operators"][name]["affine"]["determinant"] if name == primary_name else row["operators"][name]["determinant"]) > 0.0
        for row in fold_results
        for name in ("bounded_per_channel_affine", "bounded_ridge_3x3_affine", primary_name)
    )
    checks = {
        "relative_improvement_over_simple_affine": relative_improvement
        >= float(gates["minimum_primary_relative_improvement_over_simple_affine"]),
        "rolls_improved_over_simple_affine": improved_rolls
        >= int(gates["minimum_rolls_improved_over_simple_affine"]),
        "cluster_bootstrap_lower_improvement": bootstrap_summary["ci95_low"]
        > float(gates["minimum_cluster_bootstrap_lower_improvement"]),
        "rolls_beating_median_wrong_roll": wrong_rolls_beaten
        >= int(gates["minimum_rolls_beating_median_wrong_roll_operator"]),
        "style_strength": style_strength
        >= float(gates["minimum_median_render_delta_e76_from_identity"]),
        "clipping": clipping_increase <= float(gates["maximum_output_clip_fraction_increase"]),
        "finite_outputs": True,
        "monotone_lut": all_monotone,
        "positive_matrix_determinant": all_positive,
    }
    metric_passed = all(checks.values())
    if not checks["relative_improvement_over_simple_affine"] or not checks["rolls_improved_over_simple_affine"]:
        decision = "simple_affine_ties_or_wins"
    elif not checks["cluster_bootstrap_lower_improvement"]:
        decision = "cluster_interval_crosses_zero"
    elif not checks["rolls_beating_median_wrong_roll"]:
        decision = "wrong_roll_ties_or_wins"
    elif not checks["style_strength"]:
        decision = "insufficient_style_strength"
    elif not checks["clipping"] or not checks["finite_outputs"] or not checks["monotone_lut"] or not checks["positive_matrix_determinant"]:
        decision = "operator_safety_gate_failed"
    else:
        decision = "metric_pass_visual_adjudication_required"
    return {
        "rolls": rolls,
        "frames": len(frames),
        "fold_results": fold_results,
        "aggregate_metrics": aggregate,
        "primary_relative_improvement_over_simple_affine": relative_improvement,
        "rolls_improved_over_simple_affine": improved_rolls,
        "rolls_beating_median_wrong_roll": wrong_rolls_beaten,
        "median_render_delta_e76_from_identity": style_strength,
        "output_clip_fraction_increase": clipping_increase,
        "cluster_bootstrap": bootstrap_summary,
        "gate_checks": checks,
        "metric_passed": metric_passed,
        "decision": decision,
    }


def _fit_monotone_channel(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray, knot_count: int
) -> np.ndarray:
    bins = np.clip(np.rint(source * (knot_count - 1)).astype(np.int64), 0, knot_count - 1)
    sums = np.bincount(bins, weights=weights * target, minlength=knot_count)
    masses = np.bincount(bins, weights=weights, minlength=knot_count)
    observed = masses > 0.0
    if int(observed.sum()) < 2:
        raise GoldTransformConsistencyError("insufficient occupied LUT bins")
    centers = np.arange(knot_count, dtype=np.float64)
    values = np.interp(centers, centers[observed], sums[observed] / masses[observed])
    iso_weights = np.where(observed, masses, np.min(masses[observed]) * 1e-3)
    values = _weighted_pava(values, iso_weights)
    return _strict_monotone_bounded(values)


def _weighted_pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    level: list[float] = []
    mass: list[float] = []
    start: list[int] = []
    end: list[int] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        level.append(float(value))
        mass.append(float(weight))
        start.append(index)
        end.append(index + 1)
        while len(level) >= 2 and level[-2] > level[-1]:
            combined = mass[-2] + mass[-1]
            level[-2] = (level[-2] * mass[-2] + level[-1] * mass[-1]) / combined
            mass[-2] = combined
            end[-2] = end[-1]
            level.pop(); mass.pop(); start.pop(); end.pop()
    result = np.empty(len(values), dtype=np.float64)
    for value, left, right in zip(level, start, end, strict=True):
        result[left:right] = value
    return result


def _strict_monotone_bounded(values: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    values = np.asarray(values, dtype=np.float64)
    index = np.arange(len(values), dtype=np.float64)
    ceiling = 1.0 - epsilon * (len(values) - 1)
    base = np.clip(values - epsilon * index, 0.0, ceiling)
    return np.maximum.accumulate(base) + epsilon * index


def _positive_orientation(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    candidate = np.asarray(matrix, dtype=np.float64)
    if float(np.linalg.det(candidate)) > 1e-10:
        return candidate, 0.0
    for step in range(1, 101):
        alpha = step / 100.0
        blended = (1.0 - alpha) * candidate + alpha * np.eye(3)
        if float(np.linalg.det(blended)) > 1e-10:
            return blended, alpha
    raise GoldTransformConsistencyError("could not restore positive affine orientation")


def _pixels(values: np.ndarray) -> np.ndarray:
    pixels = np.asarray(values, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 3 or len(pixels) < 4:
        raise GoldTransformConsistencyError("RGB pixels must have shape (N>=4, 3)")
    if not np.all(np.isfinite(pixels)):
        raise GoldTransformConsistencyError("RGB pixels must be finite")
    return pixels
