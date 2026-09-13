from math import comb

import numpy as np
from scipy.special import logit

from src.eval.fable_reference_photometry import quantize8


def _rgb(values):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 3 or values.shape[-1] != 3 or not values.size
            or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
        raise ValueError("expected finite nonempty (H, W, 3) encoded RGB in [0, 1]")
    return values


def _logit_statistics(values, clipping):
    pixels = _rgb(values).reshape(-1, 3)
    quartiles = np.quantile(logit(np.clip(pixels, clipping, 1 - clipping)), [.25, .5, .75], axis=0)
    return quartiles[1], quartiles[2] - quartiles[0]


def analytic_prior(canonical_bases: list[np.ndarray], logit_clip: float = 1 / (2 * 255)) -> dict:
    if not 0 < logit_clip < .5 or not np.isfinite(logit_clip) or not len(canonical_bases):
        raise ValueError("nonempty donor bases and valid logit clip required")
    values = [_logit_statistics(base, logit_clip) for base in canonical_bases]
    median = np.stack([v[0] for v in values]).mean(0)
    iqr = np.stack([v[1] for v in values]).mean(0)
    denominator = float(iqr @ iqr)
    return {"mean_median": median, "mean_iqr": iqr, "denominator": denominator,
            "degenerate": denominator == 0., "logit_clip": float(logit_clip),
            "donor_count": len(canonical_bases)}


def analytic_predict(reference: np.ndarray, prior: dict, slope_limit: float, offset_limit: float) -> np.ndarray:
    if not np.isfinite(slope_limit) or slope_limit <= 1 or not np.isfinite(offset_limit) or offset_limit <= 0:
        raise ValueError("invalid transform bounds")
    median, iqr = _logit_statistics(reference, prior["logit_clip"])
    if prior["denominator"] == 0:
        return np.zeros(4, dtype=np.float64)
    slope = np.clip(np.asarray(prior["mean_iqr"]) @ iqr / prior["denominator"],
                    1 / slope_limit, slope_limit)
    offset = np.clip(median - slope * np.asarray(prior["mean_median"]), -offset_limit, offset_limit)
    return np.clip(np.r_[np.log(slope) / np.log(slope_limit), offset / offset_limit], -1., 1.)


def _counts(count, total, eligible):
    return {"count": int(count), "total": int(total), "source_eligible": int(eligible),
            "fraction": float(count / total) if total else None,
            "fraction_of_eligible": float(count / eligible) if eligible else None}


def detail_loss(source: np.ndarray, prediction: np.ndarray) -> dict:
    source, prediction = quantize8(_rgb(source)), quantize8(_rgb(prediction))
    if source.shape != prediction.shape:
        raise ValueError("source and prediction shapes must match")
    interior = (source > 0) & (source < 1)
    endpoint = (prediction == 0) | (prediction == 1)
    saturated = _counts(np.count_nonzero(interior & endpoint), source.size, np.count_nonzero(interior))
    adjacency = {}
    for name, axis in (("horizontal", 1), ("vertical", 0)):
        unequal = np.diff(source, axis=axis) != 0
        collapsed = unequal & (np.diff(prediction, axis=axis) == 0)
        adjacency[name] = _counts(np.count_nonzero(collapsed), unequal.size, np.count_nonzero(unequal))
    adjacency["combined"] = _counts(
        sum(v["count"] for v in adjacency.values()),
        sum(v["total"] for v in adjacency.values()),
        sum(v["source_eligible"] for v in adjacency.values()))
    return {"newly_saturated": saturated, "adjacency": adjacency}


def image_histograms(rgb_codes: np.ndarray) -> dict:
    codes = np.asarray(rgb_codes)
    if codes.dtype != np.uint8 or codes.ndim != 3 or codes.shape[-1] != 3 or not codes.size:
        raise ValueError("expected nonempty uint8 (H, W, 3) RGB codes")
    counts = np.empty((3, 256), dtype=np.int64)
    adjacency = np.empty((2, 3, 256, 256), dtype=np.int64)
    for channel in range(3):
        plane = codes[..., channel].astype(np.int64)
        counts[channel] = np.bincount(plane.ravel(), minlength=256)
        for axis, (before, after) in enumerate(((plane[:, :-1], plane[:, 1:]),
                                               (plane[:-1], plane[1:]))):
            adjacency[axis, channel] = np.bincount((256 * before + after).ravel(), minlength=65536).reshape(256, 256)
    return {"counts": counts, "adjacency": adjacency, "axes": ["horizontal", "vertical"],
            "shape": codes.shape}


def histogram_errors(prediction_table: np.ndarray, target_table: np.ndarray, counts: np.ndarray) -> dict:
    prediction, target = np.asarray(prediction_table, dtype=np.float64), np.asarray(target_table, dtype=np.float64)
    counts = np.asarray(counts)
    if (prediction.shape != (256, 3) or target.shape != (256, 3) or counts.shape != (3, 256)
            or not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0):
        raise ValueError("expected (256, 3) tables and nonnegative nonempty (3, 256) counts")
    quantize8(prediction)
    quantize8(target)
    float_errors = (prediction - target) ** 2
    code_errors = (np.floor(255 * prediction + .5) - np.floor(255 * target + .5)) ** 2
    return {"float_mse": float(np.sum(counts.T * float_errors) / counts.sum()),
            "code_mse": float(np.sum(counts.T * code_errors) / counts.sum())}


def histogram_detail_loss(prediction_table: np.ndarray, histograms: dict) -> dict:
    prediction = np.asarray(prediction_table, dtype=np.float64)
    if prediction.shape != (256, 3):
        raise ValueError("expected (256, 3) prediction table")
    quantize8(prediction)
    codes = np.floor(255 * prediction + .5)
    counts = np.asarray(histograms["counts"])
    pairs = np.asarray(histograms["adjacency"])
    if counts.shape != (3, 256) or pairs.shape != (2, 3, 256, 256):
        raise ValueError("invalid histogram shapes")
    interior = (np.arange(256) > 0) & (np.arange(256) < 255)
    endpoint = (codes == 0) | (codes == 255)
    saturated = _counts(np.sum(counts.T * (interior[:, None] & endpoint)),
                        np.sum(counts), np.sum(counts[:, interior]))
    unequal = ~np.eye(256, dtype=bool)
    collapsed = np.stack([(codes[:, c, None] == codes[None, :, c]) & unequal for c in range(3)])
    adjacency = {}
    for axis, name in enumerate(("horizontal", "vertical")):
        adjacency[name] = _counts(np.sum(pairs[axis] * collapsed), np.sum(pairs[axis]),
                                  np.sum(pairs[axis] * unequal))
    adjacency["combined"] = _counts(
        sum(v["count"] for v in adjacency.values()),
        sum(v["total"] for v in adjacency.values()),
        sum(v["source_eligible"] for v in adjacency.values()))
    return {"newly_saturated": saturated, "adjacency": adjacency}


def exact_sign_test(improvements: np.ndarray, alpha: float = .05) -> dict:
    values = np.asarray(improvements, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError("finite block improvements and alpha in (0, 1) required")
    positive, nontied = int(np.count_nonzero(values > 0)), int(np.count_nonzero(values != 0))
    probability = sum(comb(nontied, k) for k in range(positive, nontied + 1)) / (2 ** nontied)
    mean = float(values.mean()) if len(values) else 0.
    return {"positive": positive, "nontied": nontied, "ties": len(values) - nontied,
            "pvalue": probability, "mean": mean, "alpha": float(alpha),
            "passed": bool(nontied and probability <= alpha and mean > 0)}


def parameter_errors(prediction: np.ndarray, targets: np.ndarray) -> dict:
    prediction, targets = np.asarray(prediction, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if (prediction.shape != targets.shape or prediction.shape[-1:] != (4,)
            or not prediction.size or not np.isfinite(prediction).all() or not np.isfinite(targets).all()):
        raise ValueError("finite aligned (..., 4) parameters required")
    differences = prediction - targets
    squared = differences ** 2
    return {"per_case_mse": squared.mean(-1), "mean_mse": float(squared.mean()),
            "coordinate_mse": squared.reshape(-1, 4).mean(0),
            "signed_bias": differences.reshape(-1, 4).mean(0)}


def donor_error_decomposition(predictions: np.ndarray, targets: np.ndarray, donor_axis: int = 0) -> dict:
    predictions = np.moveaxis(np.asarray(predictions, dtype=np.float64), donor_axis, 0)
    targets = np.asarray(targets, dtype=np.float64)
    if (predictions.ndim < 2 or not predictions.size or predictions.shape[1:] != targets.shape
            or targets.shape[-1:] != (4,) or not np.isfinite(predictions).all()
            or not np.isfinite(targets).all()):
        raise ValueError("predictions must have one donor axis in addition to target (..., 4)")
    mean_prediction = predictions.mean(0)
    mean_error = ((mean_prediction - targets) ** 2).mean(-1)
    donor_variance = ((predictions - mean_prediction) ** 2).mean(axis=(0, -1))
    total_error = ((predictions - targets) ** 2).mean(axis=(0, -1))
    return {"mean_prediction": mean_prediction, "mean_prediction_error": mean_error,
            "between_donor_variance": donor_variance, "donor_mean_error": total_error}
