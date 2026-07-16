"""Shortcut-first, physical-roll-held-out stock-preview identifiability audit."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


class StockIdentifiabilityError(ValueError):
    """Raised when the frozen RF1.4A contract cannot be evaluated safely."""


QUANTILES = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])


def evaluate_structural_support(
    rows: Sequence[Mapping[str, Any]], stage_zero: Mapping[str, Any]
) -> dict[str, Any]:
    """Select the largest comparable stock clique without decoding pixels."""
    previews = [row for row in rows if row.get("lane") == "negative_preview"]
    by_stock: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in previews:
        by_stock[str(row["film_stock_id"])].append(row)
    minimum_rolls = int(stage_zero["minimum_rolls_per_stock"])
    minimum_each_roll = int(stage_zero["minimum_frames_each_roll"])
    minimum_cells = int(stage_zero["minimum_supported_content_cells_per_stock"])
    minimum_cell_frames = int(stage_zero["minimum_frames_per_supported_content_cell"])
    minimum_shared = int(stage_zero["minimum_shared_supported_content_cells_per_stock_pair"])
    support: dict[str, Any] = {}
    eligible: list[str] = []
    supported_cells: dict[str, set[str]] = {}
    for stock, stock_rows in sorted(by_stock.items()):
        roll_counts = Counter(str(row["roll_id"]) for row in stock_rows)
        cell_counts = Counter(str(row["content_cell"]) for row in stock_rows)
        cells = {cell for cell, count in cell_counts.items() if count >= minimum_cell_frames}
        reasons: list[str] = []
        if len(roll_counts) < minimum_rolls:
            reasons.append("too_few_rolls")
        if any(count < minimum_each_roll for count in roll_counts.values()):
            reasons.append("under_supported_roll")
        if len(cells) < minimum_cells:
            reasons.append("too_few_supported_content_cells")
        passed = not reasons
        if passed:
            eligible.append(stock)
        supported_cells[stock] = cells
        support[stock] = {
            "passed": passed,
            "failure_reasons": reasons,
            "roll_frame_counts": dict(sorted(roll_counts.items())),
            "supported_content_cells": sorted(cells),
            "content_cell_counts": dict(sorted(cell_counts.items())),
        }
    adjacency = {stock: set() for stock in eligible}
    shared_cells: dict[str, list[str]] = {}
    for left, right in itertools.combinations(eligible, 2):
        shared = sorted(supported_cells[left] & supported_cells[right])
        shared_cells[f"{left}|{right}"] = shared
        if len(shared) >= minimum_shared:
            adjacency[left].add(right)
            adjacency[right].add(left)
    largest: tuple[str, ...] = ()
    for size in range(1, len(eligible) + 1):
        for candidate in itertools.combinations(eligible, size):
            if all(right in adjacency[left] for left, right in itertools.combinations(candidate, 2)):
                largest = candidate
    minimum_clique = int(stage_zero["minimum_stocks_in_comparable_clique"])
    return {
        "passed": len(largest) >= minimum_clique,
        "eligible_stocks": eligible,
        "largest_comparable_clique": list(largest),
        "shared_supported_content_cells": shared_cells,
        "support_by_stock": support,
    }


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    if not 0.0 < fraction <= 1.0:
        raise StockIdentifiabilityError(f"invalid crop fraction: {fraction}")
    if fraction == 1.0:
        return image
    width, height = image.size
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def _load_rgb(path: Path, crop_fraction: float) -> np.ndarray:
    with Image.open(path) as source:
        image = _center_crop(ImageOps.exif_transpose(source).convert("RGB"), crop_fraction)
        image.thumbnail((256, 256), Image.Resampling.BOX)
        return np.asarray(image, dtype=np.float64) / 255.0


def _rgb_distribution(array: np.ndarray, *, standardized: bool = False) -> np.ndarray:
    pixels = array.reshape(-1, 3)
    if standardized:
        mean = pixels.mean(axis=0)
        scale = np.maximum(pixels.std(axis=0), 1e-6)
        pixels = (pixels - mean) / scale
        return np.concatenate([np.quantile(pixels[:, channel], QUANTILES) for channel in range(3)])
    return np.concatenate(
        [
            *[np.quantile(pixels[:, channel], QUANTILES) for channel in range(3)],
            pixels.mean(axis=0),
            pixels.std(axis=0),
        ]
    )


def _luma_distribution(array: np.ndarray) -> np.ndarray:
    luma = array @ np.asarray([0.2126, 0.7152, 0.0722])
    return np.concatenate([np.quantile(luma, QUANTILES), [luma.mean(), luma.std()]])


def _border_features(array: np.ndarray) -> np.ndarray:
    luma = array @ np.asarray([0.2126, 0.7152, 0.0722])
    height, width = luma.shape
    band = max(1, int(round(min(height, width) * 0.05)))
    mask = np.zeros_like(luma, dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    border = luma[mask]
    center = luma[~mask]
    return np.asarray(
        [
            border.mean(),
            border.std(),
            np.mean(border < 0.05),
            np.mean(border > 0.95),
            border.mean() - (center.mean() if center.size else border.mean()),
        ],
        dtype=np.float64,
    )


def extract_descriptor(
    path: Path, descriptor: str, row: Mapping[str, Any], categories: Mapping[str, Sequence[str]]
) -> np.ndarray:
    """Extract one frozen descriptor from a preview."""
    fractions = {"full": 1.0, "center80": 0.8, "center60": 0.6}
    if descriptor.startswith("rgb_quantiles_"):
        fraction = fractions[descriptor.removeprefix("rgb_quantiles_")]
        return _rgb_distribution(_load_rgb(path, fraction))
    if descriptor == "luma_quantiles_center80":
        return _luma_distribution(_load_rgb(path, 0.8))
    if descriptor == "per_channel_standardized_rgb_quantiles_center80":
        return _rgb_distribution(_load_rgb(path, 0.8), standardized=True)
    if descriptor == "rgb_mean_std_center80":
        pixels = _load_rgb(path, 0.8).reshape(-1, 3)
        return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])
    if descriptor != "metadata_border_date_content":
        raise StockIdentifiabilityError(f"unknown descriptor: {descriptor}")
    array = _load_rgb(path, 1.0)
    frame_id = str(row["frame_id"])
    date = frame_id[:8]
    if len(date) != 8 or not date.isdigit():
        raise StockIdentifiabilityError(f"frame id lacks YYYYMMDD prefix: {frame_id}")
    width = float(row["width"])
    height = float(row["height"])
    numeric = np.asarray(
        [float(date[:4]), float(date[4:6]), float(date[6:8]), width, height, width / height],
        dtype=np.float64,
    )
    categorical: list[float] = []
    for key in ("partition", "content_cell"):
        value = str(row[key])
        categorical.extend(float(value == category) for category in categories[key])
    return np.concatenate([numeric, _border_features(array), categorical])


def roll_balanced_loo(
    features: np.ndarray,
    roll_ids: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Predict every held-out physical roll from roll-balanced stock centroids."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(roll_ids) or len(matrix) != len(labels):
        raise StockIdentifiabilityError("feature/roll/label shape mismatch")
    roll_to_indices: dict[str, list[int]] = defaultdict(list)
    roll_to_label: dict[str, str] = {}
    for index, (roll, label) in enumerate(zip(roll_ids, labels, strict=True)):
        roll = str(roll)
        label = str(label)
        if roll in roll_to_label and roll_to_label[roll] != label:
            raise StockIdentifiabilityError(f"roll has multiple stock labels: {roll}")
        roll_to_label[roll] = label
        roll_to_indices[roll].append(index)
    predictions: list[dict[str, Any]] = []
    for held_out in sorted(roll_to_indices):
        train_rolls = [roll for roll in sorted(roll_to_indices) if roll != held_out]
        train_centroids = np.stack([matrix[roll_to_indices[roll]].mean(axis=0) for roll in train_rolls])
        mean = train_centroids.mean(axis=0)
        scale = np.maximum(train_centroids.std(axis=0), 1e-8)
        standardized = (train_centroids - mean) / scale
        stock_centroids: dict[str, np.ndarray] = {}
        for stock in sorted(set(roll_to_label.values())):
            positions = [index for index, roll in enumerate(train_rolls) if roll_to_label[roll] == stock]
            if not positions:
                raise StockIdentifiabilityError(
                    f"held-out fold leaves no training roll for stock {stock}"
                )
            stock_centroids[stock] = standardized[positions].mean(axis=0)
        test = (matrix[roll_to_indices[held_out]].mean(axis=0) - mean) / scale
        distances = {stock: float(np.linalg.norm(test - centroid)) for stock, centroid in stock_centroids.items()}
        predicted = min(distances, key=lambda stock: (distances[stock], stock))
        predictions.append(
            {
                "roll_id": held_out,
                "true_stock": roll_to_label[held_out],
                "predicted_stock": predicted,
                "correct": predicted == roll_to_label[held_out],
                "distances": dict(sorted(distances.items())),
            }
        )
    stocks = sorted(set(roll_to_label.values()))
    confusion = {
        true: {predicted: 0 for predicted in stocks}
        for true in stocks
    }
    for row in predictions:
        confusion[row["true_stock"]][row["predicted_stock"]] += 1
    recalls = {
        stock: float(confusion[stock][stock] / max(sum(confusion[stock].values()), 1))
        for stock in stocks
    }
    return {
        "rolls": len(predictions),
        "accuracy": float(np.mean([row["correct"] for row in predictions])),
        "per_stock_roll_recall": recalls,
        "confusion": confusion,
        "predictions": predictions,
    }


def permutation_test(
    features: np.ndarray,
    roll_ids: Sequence[str],
    labels: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Count-preserving roll-label permutation test for the frozen classifier."""
    observed = roll_balanced_loo(features, roll_ids, labels)["accuracy"]
    unique_rolls = sorted(set(str(roll) for roll in roll_ids))
    label_by_roll = {str(roll): str(label) for roll, label in zip(roll_ids, labels, strict=True)}
    original = np.asarray([label_by_roll[roll] for roll in unique_rolls], dtype=object)
    rng = np.random.default_rng(seed)
    null_scores: list[float] = []
    for _ in range(permutations):
        shuffled = original[rng.permutation(len(original))]
        shuffled_by_roll = dict(zip(unique_rolls, shuffled, strict=True))
        shuffled_labels = [shuffled_by_roll[str(roll)] for roll in roll_ids]
        null_scores.append(roll_balanced_loo(features, roll_ids, shuffled_labels)["accuracy"])
    null = np.asarray(null_scores, dtype=np.float64)
    return {
        "observed_accuracy": float(observed),
        "permutations": permutations,
        "p_value_greater_equal": float((1 + np.sum(null >= observed)) / (permutations + 1)),
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_max": float(null.max()),
    }
