"""Connected-source, group-held-out stock identifiability diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog


class ConnectedStockIdentifiabilityError(ValueError):
    """Raised when the frozen connected-design contract cannot be evaluated."""


QUANTILES = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_connected_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load only frozen stock/source cells and verify every local pixel hash."""
    aliases = {str(key): str(value) for key, value in config.get("author_group_aliases", {}).items()}
    rows: list[dict[str, Any]] = []
    for cell in config["cells"]:
        manifest_path = root / str(cell["manifest"])
        if _sha256(manifest_path) != str(cell["manifest_sha256"]):
            raise ConnectedStockIdentifiabilityError(f"manifest hash drift: {cell['cell_id']}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_root = root / str(cell["pixel_root"])
        stock_id = str(cell["film_stock_id"])
        cell_rows = [row for row in manifest["rows"] if str(row["film_stock_id"]) == stock_id]
        if "included_page_ids" in cell:
            included_page_ids = [int(page_id) for page_id in cell["included_page_ids"]]
            if len(included_page_ids) != len(set(included_page_ids)) or any(page_id <= 0 for page_id in included_page_ids):
                raise ConnectedStockIdentifiabilityError(f"invalid included page ids: {cell['cell_id']}")
            rows_by_page_id = {int(row["page_id"]): row for row in cell_rows}
            if len(rows_by_page_id) != len(cell_rows) or any(page_id not in rows_by_page_id for page_id in included_page_ids):
                raise ConnectedStockIdentifiabilityError(f"included page id drift: {cell['cell_id']}")
            cell_rows = [rows_by_page_id[page_id] for page_id in included_page_ids]
        if len(cell_rows) != int(cell["expected_files"]):
            raise ConnectedStockIdentifiabilityError(f"cell count drift: {cell['cell_id']}")
        for source in cell_rows:
            path = source_root / str(source["local_path"])
            if not path.is_file() or path.stat().st_size != int(source["bytes"]) or _sha256(path) != str(source["sha256"]):
                raise ConnectedStockIdentifiabilityError(f"pixel drift: {path}")
            raw_group = str(source["normalized_author_group"])
            canonical = aliases.get(raw_group, raw_group)
            rows.append({
                "cell_id": str(cell["cell_id"]),
                "source_id": str(cell["source_id"]),
                "film_stock_id": stock_id,
                "author_group": canonical,
                "group_id": f"{cell['source_id']}::{canonical}",
                "path": path,
                "bytes": int(source["bytes"]),
                "width": int(source["width"]),
                "height": int(source["height"]),
                "source_record": dict(source),
            })
    expected_total = sum(int(cell["expected_files"]) for cell in config["cells"])
    if len(rows) != expected_total:
        raise ConnectedStockIdentifiabilityError("connected row total drift")
    return rows


def _center_crop(image: Image.Image, fraction: float = 0.8) -> Image.Image:
    width, height = image.size
    crop_width, crop_height = max(1, round(width * fraction)), max(1, round(height * fraction))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def extract_descriptors(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract frozen color, content and source descriptors in one decode."""
    with Image.open(Path(row["path"])) as source:
        rgb_image = ImageOps.exif_transpose(source).convert("RGB")
        full_small = rgb_image.copy()
        full_small.thumbnail((256, 256), Image.Resampling.BOX)
        center = _center_crop(rgb_image, 0.8)
        center.thumbnail((256, 256), Image.Resampling.BOX)
        rgb = np.asarray(center, dtype=np.float64) / 255.0
        full = np.asarray(full_small, dtype=np.float64) / 255.0
        low = np.asarray(center.resize((4, 4), Image.Resampling.BOX), dtype=np.float64).reshape(-1) / 255.0
        gray_image = center.convert("L").resize((64, 64), Image.Resampling.BOX)
        gray = np.asarray(gray_image, dtype=np.float64) / 255.0
    pixels = rgb.reshape(-1, 3)
    rgb_distribution = np.concatenate([
        *[np.quantile(pixels[:, channel], QUANTILES) for channel in range(3)],
        pixels.mean(axis=0), pixels.std(axis=0),
    ])
    standardized = (pixels - pixels.mean(axis=0)) / np.maximum(pixels.std(axis=0), 1e-6)
    standardized_distribution = np.concatenate([
        np.quantile(standardized[:, channel], QUANTILES) for channel in range(3)
    ])
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722])
    luma_distribution = np.concatenate([np.quantile(luma, QUANTILES), [luma.mean(), luma.std()]])
    hog_descriptor = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")
    full_luma = full @ np.asarray([0.2126, 0.7152, 0.0722])
    band = max(1, round(min(full_luma.shape) * 0.05))
    border_mask = np.zeros_like(full_luma, dtype=bool)
    border_mask[:band, :] = border_mask[-band:, :] = True
    border_mask[:, :band] = border_mask[:, -band:] = True
    border = full_luma[border_mask]
    width, height, byte_count = float(row["width"]), float(row["height"]), float(row["bytes"])
    geometry = np.asarray([
        np.log1p(width), np.log1p(height), width / height,
        np.log1p(byte_count), byte_count / max(width * height, 1.0),
        border.mean(), border.std(), np.mean(border <= 1 / 255), np.mean(border >= 254 / 255),
    ])
    return {
        "rgb_distribution": rgb_distribution,
        "luma_distribution": luma_distribution,
        "hog_grayscale": np.asarray(hog_descriptor, dtype=np.float64),
        "low_frequency_rgb_4x4": low,
        "standardized_rgb_distribution": standardized_distribution,
        "geometry_border_source": geometry,
    }


def group_loo_centroid(features: np.ndarray, groups: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Hold out whole authors and classify equal-weight author-by-label units."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(groups) or len(matrix) != len(labels):
        raise ConnectedStockIdentifiabilityError("feature/group/label shape mismatch")
    unit_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (group, label) in enumerate(zip(groups, labels, strict=True)):
        group, label = str(group), str(label)
        unit_indices[(group, label)].append(index)
    unique_groups = sorted({str(group) for group in groups})
    unique_labels = sorted({str(label) for label in labels})
    if any(sum(unit_label == label for _, unit_label in unit_indices) < 2 for label in unique_labels):
        raise ConnectedStockIdentifiabilityError("each class needs at least two author groups")
    predictions: list[dict[str, Any]] = []
    for held_out in unique_groups:
        training_units = sorted(unit for unit in unit_indices if unit[0] != held_out)
        centroids = np.stack([matrix[unit_indices[unit]].mean(axis=0) for unit in training_units])
        mean, scale = centroids.mean(axis=0), np.maximum(centroids.std(axis=0), 1e-8)
        standardized = (centroids - mean) / scale
        label_centroids = {
            label: standardized[[unit[1] == label for unit in training_units]].mean(axis=0)
            for label in unique_labels
        }
        for unit in sorted(unit for unit in unit_indices if unit[0] == held_out):
            test = (matrix[unit_indices[unit]].mean(axis=0) - mean) / scale
            distances = {label: float(np.linalg.norm(test - center)) for label, center in label_centroids.items()}
            predicted = min(distances, key=lambda label: (distances[label], label))
            predictions.append({
                "group_id": f"{held_out}::{unit[1]}",
                "held_out_author_group": held_out,
                "true_label": unit[1],
                "predicted_label": predicted,
                "correct": predicted == unit[1],
                "distances": dict(sorted(distances.items())),
            })
    recalls = {
        label: float(np.mean([row["correct"] for row in predictions if row["true_label"] == label]))
        for label in unique_labels
    }
    return {
        "held_out_author_groups": len(unique_groups),
        "author_label_units": len(predictions),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_class_group_recall": recalls,
        "predictions": predictions,
    }


def group_label_permutation_test(
    features: np.ndarray,
    groups: Sequence[str],
    labels: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    observed = group_loo_centroid(features, groups, labels)["balanced_accuracy"]
    unique_units = sorted({(str(group), str(label)) for group, label in zip(groups, labels, strict=True)})
    units_by_author: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for unit in unique_units:
        units_by_author[unit[0]].append(unit)
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(permutations):
        mapping: dict[tuple[str, str], str] = {}
        singleton_units: list[tuple[str, str]] = []
        for author in sorted(units_by_author):
            author_units = sorted(units_by_author[author])
            if len(author_units) == 1:
                singleton_units.extend(author_units)
                continue
            author_labels = np.asarray([unit[1] for unit in author_units], dtype=object)
            shuffled_author_labels = author_labels[rng.permutation(len(author_labels))]
            mapping.update(dict(zip(author_units, shuffled_author_labels, strict=True)))
        singleton_labels = np.asarray([unit[1] for unit in singleton_units], dtype=object)
        shuffled_singleton_labels = singleton_labels[rng.permutation(len(singleton_labels))]
        mapping.update(dict(zip(singleton_units, shuffled_singleton_labels, strict=True)))
        permuted = [mapping[(str(group), str(label))] for group, label in zip(groups, labels, strict=True)]
        null.append(group_loo_centroid(features, groups, permuted)["balanced_accuracy"])
    values = np.asarray(null)
    return {
        "observed_balanced_accuracy": float(observed),
        "permutations": permutations,
        "p_value_greater_equal": float((1 + np.sum(values >= observed)) / (permutations + 1)),
        "null_mean": float(values.mean()),
        "null_q95": float(np.quantile(values, 0.95)),
    }


def paired_group_bootstrap_delta(
    primary: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Stratified group bootstrap of balanced-accuracy differences."""
    left = {row["group_id"]: row for row in primary["predictions"]}
    right = {row["group_id"]: row for row in control["predictions"]}
    if set(left) != set(right):
        raise ConnectedStockIdentifiabilityError("paired predictions use different groups")
    by_label: dict[str, list[str]] = defaultdict(list)
    for group, row in left.items():
        if right[group]["true_label"] != row["true_label"]:
            raise ConnectedStockIdentifiabilityError("paired prediction labels differ")
        by_label[str(row["true_label"])].append(group)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        class_deltas: list[float] = []
        for groups_for_label in by_label.values():
            sampled = rng.choice(groups_for_label, size=len(groups_for_label), replace=True)
            class_deltas.append(float(np.mean([left[group]["correct"] - right[group]["correct"] for group in sampled])))
        deltas.append(float(np.mean(class_deltas)))
    observed = float(primary["balanced_accuracy"] - control["balanced_accuracy"])
    return {
        "observed_delta": observed,
        "iterations": iterations,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def run_connected_audit(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    descriptors_by_row = [extract_descriptors(row) for row in rows]
    descriptor_names = list(config["descriptors"])
    contrasts: dict[str, Any] = {}
    for contrast in config["contrasts"]:
        indices = [
            index for index, row in enumerate(rows)
            if row[str(contrast["filter_field"])] in set(contrast["filter_values"])
            and row[str(contrast["label_field"])] in set(contrast["label_values"])
        ]
        selected = [rows[index] for index in indices]
        groups = [str(row["group_id"]) for row in selected]
        labels = [str(row[str(contrast["label_field"])]) for row in selected]
        descriptor_results: dict[str, Any] = {}
        for descriptor in descriptor_names:
            matrix = np.stack([descriptors_by_row[index][descriptor] for index in indices])
            result = group_loo_centroid(matrix, groups, labels)
            result["permutation"] = group_label_permutation_test(
                matrix, groups, labels,
                permutations=int(config["statistics"]["permutations"]),
                seed=int(config["statistics"]["permutation_seed"]),
            )
            descriptor_results[descriptor] = result
        primary_name = str(config["gates"]["primary_descriptor"])
        controls = list(config["gates"]["nuisance_control_descriptors"])
        best_control = max(controls, key=lambda name: (descriptor_results[name]["balanced_accuracy"], name))
        delta = paired_group_bootstrap_delta(
            descriptor_results[primary_name], descriptor_results[best_control],
            iterations=int(config["statistics"]["bootstrap_iterations"]),
            seed=int(config["statistics"]["bootstrap_seed"]),
        )
        checks = {
            "minimum_groups_per_label": all(
                count >= int(config["gates"]["minimum_groups_per_label"])
                for count in Counter(label for _, label in set(zip(groups, labels, strict=True))).values()
            ),
            "primary_accuracy": descriptor_results[primary_name]["balanced_accuracy"] >= float(config["gates"]["minimum_primary_balanced_accuracy"]),
            "primary_permutation": descriptor_results[primary_name]["permutation"]["p_value_greater_equal"] <= float(config["gates"]["maximum_primary_permutation_p"]),
            "delta_vs_nuisance": delta["observed_delta"] >= float(config["gates"]["minimum_delta_vs_nuisance"]),
            "delta_ci_above_zero": delta["ci95_low"] > 0.0,
            "hog_content_ceiling": descriptor_results["hog_grayscale"]["balanced_accuracy"] <= float(config["gates"]["maximum_hog_control_accuracy"]),
        }
        contrasts[str(contrast["contrast_id"])] = {
            "rows": len(selected),
            "author_groups": len(set(groups)),
            "author_label_unit_counts": dict(sorted(Counter(label for _, label in set(zip(groups, labels, strict=True))).items())),
            "descriptor_results": descriptor_results,
            "best_nuisance_control": best_control,
            "primary_minus_best_nuisance": delta,
            "checks": checks,
            "stock_signal_gate_passed": all(checks.values()) if contrast["kind"] == "stock" else False,
            "kind": str(contrast["kind"]),
        }
    stock_contrasts = [row for row in contrasts.values() if row["kind"] == "stock"]
    return {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "rows": len(rows),
        "cells": {
            cell: {"files": count, "groups": len({row["group_id"] for row in rows if row["cell_id"] == cell})}
            for cell, count in sorted(Counter(row["cell_id"] for row in rows).items())
        },
        "contrasts": contrasts,
        "all_stock_edges_passed": bool(stock_contrasts) and all(row["stock_signal_gate_passed"] for row in stock_contrasts),
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
