"""Explicit neutral-control hypotheses for generated operator recovery."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .lab_statistics import extract_lab_statistics
from .synthetic_recovery import (
    operator_from_manifest,
    palette_cloud,
    palette_pairs,
)


@dataclass(frozen=True)
class NeutralBankItem:
    item_id: str
    palette: str
    seed: int
    rgb: np.ndarray
    lab_descriptor: np.ndarray
    normalized_quantiles: np.ndarray
    rgb_sha256: str


@dataclass(frozen=True)
class CanonicalObservation:
    operator_id: str
    family: str
    split: str
    query_palette: str
    reference_palette: str
    parameters: np.ndarray
    signatures: dict[str, np.ndarray]
    selections: dict[str, str]


def normalized_rgb_quantiles(
    rgb: np.ndarray,
    *,
    quantile_count: int,
    minimum_iqr: float,
) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must be finite [0, 1] data with shape (N, 3)")
    if quantile_count < 3 or minimum_iqr <= 0.0:
        raise ValueError("quantile_count and minimum_iqr must be positive")
    levels = np.linspace(0.0, 1.0, quantile_count)
    features = []
    for channel in range(3):
        quantiles = np.quantile(values[:, channel], levels)
        median = np.quantile(values[:, channel], 0.5)
        iqr = max(
            float(
                np.quantile(values[:, channel], 0.75)
                - np.quantile(values[:, channel], 0.25)
            ),
            minimum_iqr,
        )
        features.append((quantiles - median) / iqr)
    return np.concatenate(features)


def build_neutral_bank(config: dict[str, Any]) -> list[NeutralBankItem]:
    specification = config["neutral_bank"]
    items: list[NeutralBankItem] = []
    for palette_index, palette in enumerate(specification["palette_families"]):
        for replica in range(int(specification["clouds_per_palette"])):
            seed = int(specification["seed"]) + palette_index * 1000 + replica
            rgb = palette_cloud(
                str(palette),
                int(specification["pixels_per_cloud"]),
                seed,
            )
            item_id = f"{palette}-{replica:02d}"
            items.append(
                NeutralBankItem(
                    item_id=item_id,
                    palette=str(palette),
                    seed=seed,
                    rgb=rgb,
                    lab_descriptor=extract_lab_statistics(rgb).vector(),
                    normalized_quantiles=normalized_rgb_quantiles(
                        rgb,
                        quantile_count=int(specification["rgb_quantiles"]),
                        minimum_iqr=float(specification["minimum_iqr"]),
                    ),
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                )
            )
    return items


def neutral_bank_manifest(items: list[NeutralBankItem]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "palette": item.palette,
            "seed": item.seed,
            "rgb_sha256": item.rgb_sha256,
            "pixels": len(item.rgb),
        }
        for item in items
    ]


def neutral_bank_manifest_sha256(items: list[NeutralBankItem]) -> str:
    encoded = (
        json.dumps(
            neutral_bank_manifest(items),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nearest_index(
    query: np.ndarray,
    bank_features: np.ndarray,
    eligible: np.ndarray | None = None,
) -> int:
    if eligible is None:
        eligible = np.ones(len(bank_features), dtype=bool)
    indices = np.flatnonzero(eligible)
    if len(indices) == 0:
        raise ValueError("nearest-neighbour candidate set is empty")
    distances = np.linalg.norm(bank_features[indices] - query, axis=1)
    return int(indices[int(np.argmin(distances))])


def _gaussian_neutral_estimate(
    transformed: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    source_mean = transformed.mean(axis=0)
    source_std = np.maximum(transformed.std(axis=0), 1e-8)
    target_mean = template.mean(axis=0)
    target_std = np.maximum(template.std(axis=0), 1e-8)
    normalized = (transformed - source_mean) / source_std
    return np.clip(normalized * target_std + target_mean, 0.0, 1.0)


def build_canonical_observations(
    parent_config: dict[str, Any],
    parent_manifest: list[dict[str, Any]],
    config: dict[str, Any],
    bank: list[NeutralBankItem],
) -> list[CanonicalObservation]:
    parent_palettes = parent_config["palettes"]
    pixel_count = int(parent_palettes["pixels_per_cloud"])
    seed = int(parent_config["seed"])
    names_by_split = {
        "fit": list(parent_palettes["development"]),
        "validation": list(parent_palettes["development"]),
        "confirmation": list(parent_palettes["confirmation"]),
        "stress": list(parent_palettes["stress"]),
    }
    palette_names = sorted({name for names in names_by_split.values() for name in names})
    clouds = {name: palette_cloud(name, pixel_count, seed) for name in palette_names}
    raw_descriptors = {
        name: extract_lab_statistics(cloud).vector() for name, cloud in clouds.items()
    }
    bank_lab = np.stack([item.lab_descriptor for item in bank])
    bank_lab_scale = np.maximum(bank_lab.std(axis=0), 1e-8)
    bank_lab_center = bank_lab.mean(axis=0)
    standardized_bank_lab = (bank_lab - bank_lab_center) / bank_lab_scale
    bank_quantiles = np.stack([item.normalized_quantiles for item in bank])
    balanced_template = next(item.rgb for item in bank if item.palette == "balanced")
    quantile_count = int(config["neutral_bank"]["rgb_quantiles"])
    minimum_iqr = float(config["neutral_bank"]["minimum_iqr"])

    observations: list[CanonicalObservation] = []
    for row in parent_manifest:
        operator = operator_from_manifest(row)
        split = str(row["split"])
        pairs = palette_pairs(
            names_by_split[split],
            operator_index=int(row["family_index"]),
            observations=2,
        )
        for query_name, reference_name in pairs:
            query = clouds[query_name]
            reference = clouds[reference_name]
            transformed_reference = operator.apply(reference)
            target_descriptor = extract_lab_statistics(transformed_reference).vector()
            target_quantiles = normalized_rgb_quantiles(
                transformed_reference,
                quantile_count=quantile_count,
                minimum_iqr=minimum_iqr,
            )
            raw_lab_index = _nearest_index(
                (target_descriptor - bank_lab_center) / bank_lab_scale,
                standardized_bank_lab,
            )
            quantile_index = _nearest_index(target_quantiles, bank_quantiles)
            same_palette = np.asarray(
                [item.palette == reference_name for item in bank],
                dtype=bool,
            )
            palette_oracle_index = _nearest_index(
                target_quantiles,
                bank_quantiles,
                same_palette,
            )
            different_palette = ~same_palette
            raw_leaveout_index = _nearest_index(
                (target_descriptor - bank_lab_center) / bank_lab_scale,
                standardized_bank_lab,
                different_palette,
            )
            quantile_leaveout_index = _nearest_index(
                target_quantiles,
                bank_quantiles,
                different_palette,
            )
            gaussian_neutral = _gaussian_neutral_estimate(
                transformed_reference,
                balanced_template,
            )
            estimated = {
                "query_as_neutral": raw_descriptors[query_name],
                "nearest_raw_lab": bank[raw_lab_index].lab_descriptor,
                "nearest_basic_normalized_quantiles": bank[
                    quantile_index
                ].lab_descriptor,
                "fixed_gaussian_neutralization": extract_lab_statistics(
                    gaussian_neutral
                ).vector(),
                "palette_oracle_nearest": bank[
                    palette_oracle_index
                ].lab_descriptor,
                "exact_raw_reference_oracle": raw_descriptors[reference_name],
                "nearest_raw_lab_leave_true_palette_out": bank[
                    raw_leaveout_index
                ].lab_descriptor,
                "nearest_basic_normalized_quantiles_leave_true_palette_out": bank[
                    quantile_leaveout_index
                ].lab_descriptor,
            }
            signatures = {
                name: target_descriptor - neutral_descriptor
                for name, neutral_descriptor in estimated.items()
            }
            selections = {
                "nearest_raw_lab": bank[raw_lab_index].item_id,
                "nearest_basic_normalized_quantiles": bank[quantile_index].item_id,
                "palette_oracle_nearest": bank[palette_oracle_index].item_id,
                "nearest_raw_lab_leave_true_palette_out": bank[
                    raw_leaveout_index
                ].item_id,
                "nearest_basic_normalized_quantiles_leave_true_palette_out": bank[
                    quantile_leaveout_index
                ].item_id,
            }
            observations.append(
                CanonicalObservation(
                    operator_id=operator.operator_id,
                    family=operator.family,
                    split=split,
                    query_palette=query_name,
                    reference_palette=reference_name,
                    parameters=operator.parameters,
                    signatures=signatures,
                    selections=selections,
                )
            )
    return observations


def canonical_rows_for_split(
    observations: list[CanonicalObservation],
    split: str,
) -> list[CanonicalObservation]:
    rows = [row for row in observations if row.split == split]
    if not rows:
        raise ValueError(f"no canonical observations for split {split!r}")
    return rows


def signature_matrix(
    observations: list[CanonicalObservation],
    canonicalizer: str,
) -> np.ndarray:
    try:
        return np.stack([row.signatures[canonicalizer] for row in observations])
    except KeyError as error:
        raise ValueError(f"unknown canonicalizer: {canonicalizer!r}") from error


def target_matrix(observations: list[CanonicalObservation]) -> np.ndarray:
    return np.stack([row.parameters for row in observations])


def retrieval_diagnostics(
    observations: list[CanonicalObservation],
    bank: list[NeutralBankItem],
) -> dict[str, Any]:
    palette_by_id = {item.item_id: item.palette for item in bank}
    result: dict[str, Any] = {}
    for method in (
        "nearest_raw_lab",
        "nearest_basic_normalized_quantiles",
        "palette_oracle_nearest",
        "nearest_raw_lab_leave_true_palette_out",
        "nearest_basic_normalized_quantiles_leave_true_palette_out",
    ):
        selections = [row.selections[method] for row in observations]
        counts = {item_id: selections.count(item_id) for item_id in sorted(set(selections))}
        result[method] = {
            "same_palette_fraction": float(
                np.mean(
                    [
                        palette_by_id[item_id] == row.reference_palette
                        for row, item_id in zip(observations, selections)
                    ]
                )
            ),
            "maximum_item_share": float(max(counts.values()) / len(selections)),
            "unique_items": len(counts),
        }
    return result
