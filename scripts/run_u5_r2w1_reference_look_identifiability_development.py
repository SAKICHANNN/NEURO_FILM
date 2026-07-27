#!/usr/bin/env python
"""Run frozen U5.R2W1D generated reference-look identifiability development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2s1_histogram_case_retrieval_development import (  # noqa: E402
    _evaluate_method,
    _grid,
)
from scripts.run_u5_r2s2_content_palette_nuisance_development import (  # noqa: E402
    _generator_kwargs,
    _make_styles,
)
from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
)
from src.roll2film.hierarchical_colour_coupling import (  # noqa: E402
    fit_paired_cube_diffeomorphic_flow,
)
from src.roll2film.histogram_case_retrieval import (  # noqa: E402
    generate_synthetic_palette,
    sample_palette,
)
from src.roll2film.palette_score_flow import (  # noqa: E402
    DiagonalGaussianMixturePalette,
)
from src.roll2film.reference_look_identifiability import (  # noqa: E402
    aggregate_reference_descriptors,
    build_direction_strength_bank,
    build_reference_look_bank,
    factorized_reference_descriptor,
    fit_reference_descriptor_ridge,
    raw_rgb_histogram_descriptor,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_activation(
    config: dict[str, Any],
    s4_decision: dict[str, Any],
    u1_decision: dict[str, Any] | None,
) -> None:
    if config["status"] != (
        "implemented_execution_queued_behind_active_s4_u1_leaves"
    ):
        raise ValueError("unexpected W1 contract status")
    gate = config["activation_gate"]
    s4_branch = str(s4_decision.get("decision_branch"))
    if s4_branch not in gate["s4_valid_branches"]:
        raise RuntimeError("W1 activation rejected: invalid S4 branch")
    if (
        gate["s4_requires_repeat_report_sha256_equal"]
        and s4_decision.get("repeat_report_sha256_equal") is not True
    ):
        raise RuntimeError("W1 activation requires an exact S4 repeat")
    if s4_branch in gate["u1_required_on_s4_branches"]:
        if u1_decision is None:
            raise RuntimeError("W1 activation requires the U1 decision")
        if str(u1_decision.get("decision_branch")) not in gate["u1_valid_branches"]:
            raise RuntimeError("W1 activation rejected: invalid U1 branch")
        if (
            gate["u1_requires_repeat_report_sha256_equal"]
            and u1_decision.get("repeat_report_sha256_equal") is not True
        ):
            raise RuntimeError("W1 activation requires an exact U1 repeat")


def _validate_experiment_partition(config: dict[str, Any]) -> None:
    look = config["look_generator"]
    content = config["content_generator"]
    nuisance = config["reference_nuisance"]
    direction_count = int(look["development_base_direction_count"])
    fit_directions = [int(x) for x in look["development_fit_direction_ids"]]
    unseen_directions = [
        int(x) for x in look["development_unseen_direction_ids"]
    ]
    if (
        set(fit_directions).intersection(unseen_directions)
        or sorted(fit_directions + unseen_directions)
        != list(range(direction_count))
    ):
        raise ValueError("fit/unseen direction partition is invalid")
    if int(look["development_operator_instance_count"]) != direction_count * len(
        look["strengths"]
    ):
        raise ValueError("development operator-instance count is invalid")
    group_count = int(
        content["independent_content_groups_per_operator_instance"]
    )
    fit_groups = [int(x) for x in content["fit_and_bank_content_group_ids"]]
    query_groups = [int(x) for x in content["heldout_query_content_group_ids"]]
    if (
        set(fit_groups).intersection(query_groups)
        or sorted(fit_groups + query_groups) != list(range(group_count))
    ):
        raise ValueError("fit/query content-group partition is invalid")
    centres = np.asarray(
        content["content_group_palette_centres"], dtype=np.float64
    )
    if centres.shape != (group_count, 3) or not np.all(np.isfinite(centres)):
        raise ValueError("content-group palette centres are invalid")
    anchors = look["owner_anchor_strength_path"]
    if set(anchors) != {"53", "55", "56", "required_interpretation"}:
        raise ValueError("owner strength-path keys are invalid")
    if any(float(anchors[key]) <= 0.0 for key in ("53", "55", "56")):
        raise ValueError("owner strength-path values must be positive")
    family_specs = look["base_direction_generators"]
    if (
        {str(item["id"]) for item in family_specs}
        != {"palette_score_velocity_grid", "smooth_random_velocity_grid"}
        or sum(int(item["direction_count"]) for item in family_specs)
        != direction_count
    ):
        raise ValueError("development operator-family specification is invalid")
    development_seeds = {
        *(int(item["seed"]) for item in family_specs),
        int(content["fit_content_seed"]),
        int(content["heldout_content_seed"]),
        int(content["identity_reference_fit_content_seed"]),
        int(content["identity_reference_heldout_content_seed"]),
        int(content["owner_strength_fixture_content_seed"]),
        int(nuisance["fit_nuisance_seed"]),
        int(nuisance["heldout_nuisance_seed"]),
        int(nuisance["identity_fit_nuisance_seed"]),
        int(nuisance["identity_heldout_nuisance_seed"]),
        int(nuisance["owner_strength_fixture_nuisance_seed"]),
        int(config["paired_upper_bound_optimization"]["seed"]),
    }
    reserved_seeds = {
        int(look["reserved_confirmation_direction_seed"]),
        int(content["reserved_confirmation_content_seed"]),
        int(nuisance["reserved_confirmation_seed"]),
    }
    if len(development_seeds) != 13 or len(reserved_seeds) != 3:
        raise ValueError("development or reserved seeds are not unique")
    if development_seeds.intersection(reserved_seeds):
        raise ValueError("development and confirmation seeds overlap")


def _smooth_random_velocity_grids(
    *,
    count: int,
    seed: int,
    axis_size: int,
    smoothing_passes: int,
    minimum_norm: float,
    maximum_norm: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grids = []
    for _ in range(count):
        grid = rng.normal(size=(axis_size, axis_size, axis_size, 3))
        for _ in range(smoothing_passes):
            for axis in range(3):
                before = np.take(grid, [0], axis=axis)
                after = np.take(grid, [-1], axis=axis)
                padded = np.concatenate((before, grid, after), axis=axis)
                left = np.take(
                    padded, np.arange(0, axis_size), axis=axis
                )
                centre = np.take(
                    padded, np.arange(1, axis_size + 1), axis=axis
                )
                right = np.take(
                    padded, np.arange(2, axis_size + 2), axis=axis
                )
                grid = 0.25 * left + 0.5 * centre + 0.25 * right
        grid -= np.mean(grid, axis=(0, 1, 2), keepdims=True)
        maximum = float(np.max(np.linalg.norm(grid, axis=-1)))
        target = float(rng.uniform(minimum_norm, maximum_norm))
        grids.append(target * grid / max(maximum, 1e-15))
    return np.stack(grids)


def _look_instances(
    config: dict[str, Any],
) -> tuple[
    np.ndarray,
    list[DiagonalGaussianMixturePalette],
    list[dict[str, Any]],
]:
    look = config["look_generator"]
    specs = {str(item["id"]): item for item in look["base_direction_generators"]}
    palette_spec = specs["palette_score_velocity_grid"]
    palettes, palette_grids = _make_styles(
        count=int(palette_spec["direction_count"]),
        seed=int(palette_spec["seed"]),
        generator_kwargs=_generator_kwargs(
            look["base_direction_generator_parameters"]
        ),
        velocity_grid_axis_size=int(look["velocity_grid_axis_size"]),
        coefficient_vector_norm_cap=float(look["coefficient_vector_norm_cap"]),
    )
    random_spec = specs["smooth_random_velocity_grid"]
    random_grids = _smooth_random_velocity_grids(
        count=int(random_spec["direction_count"]),
        seed=int(random_spec["seed"]),
        axis_size=int(look["velocity_grid_axis_size"]),
        smoothing_passes=int(random_spec["smoothing_passes"]),
        minimum_norm=float(random_spec["minimum_maximum_vector_norm"]),
        maximum_norm=float(random_spec["maximum_maximum_vector_norm"]),
    )
    if len(palette_grids) != len(random_grids):
        raise ValueError("v1 requires equally interleaved operator families")
    base_grids, base_palettes, families = [], [], []
    for index in range(len(palette_grids)):
        base_grids.extend((palette_grids[index], random_grids[index]))
        base_palettes.extend((palettes[index], palettes[index]))
        families.extend(
            ("palette_score_velocity_grid", "smooth_random_velocity_grid")
        )
    grids, repeated_palettes, metadata = [], [], []
    for direction, (palette, base_grid, family) in enumerate(
        zip(base_palettes, base_grids, families, strict=True)
    ):
        for strength_index, strength in enumerate(look["strengths"]):
            value = float(strength)
            grids.append(value * base_grid)
            repeated_palettes.append(palette)
            metadata.append(
                {
                    "direction_id": direction,
                    "strength_index": strength_index,
                    "strength": value,
                    "look_id": f"d{direction:02d}-s{value:.2f}",
                    "operator_family": family,
                }
            )
    return np.stack(grids), repeated_palettes, metadata


def _content_palette(
    rng: np.random.Generator,
    centre: np.ndarray,
    config: dict[str, Any],
) -> DiagonalGaussianMixturePalette:
    counts = tuple(int(value) for value in config["mixture_component_counts"])
    count = counts[int(rng.integers(0, len(counts)))]
    means = np.clip(
        centre[None, :]
        + rng.normal(
            scale=float(config["component_mean_jitter_standard_deviation"]),
            size=(count, 3),
        ),
        float(config["mean_minimum"]),
        float(config["mean_maximum"]),
    )
    return DiagonalGaussianMixturePalette(
        weights=rng.dirichlet(
            np.full(count, float(config["weight_dirichlet_alpha"]))
        ),
        means=means,
        standard_deviations=rng.uniform(
            float(config["standard_deviation_minimum"]),
            float(config["standard_deviation_maximum"]),
            size=(count, 3),
        ),
    )


def _mobius_gain(values: np.ndarray, gains: np.ndarray) -> np.ndarray:
    denominator = 1.0 + (gains[None, :] - 1.0) * values
    return gains[None, :] * values / denominator


def _apply_reference_nuisance(
    rgb: np.ndarray,
    *,
    family_id: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(rgb, dtype=np.float64)
    if family_id == 0:
        return values.copy(), {"family_id": 0, "family": "identity"}
    if family_id == 1:
        ev = float(
            rng.uniform(
                -float(config["maximum_absolute_exposure_ev"]),
                float(config["maximum_absolute_exposure_ev"]),
            )
        )
        output = _mobius_gain(values, np.full(3, 2.0**ev))
        return output, {"family_id": 1, "family": "exposure", "ev": ev}
    if family_id == 2:
        limit = float(config["maximum_white_balance_log_gain"])
        log_gains = rng.uniform(-limit, limit, size=3)
        log_gains -= np.mean(log_gains)
        output = _mobius_gain(values, np.exp(log_gains))
        return output, {
            "family_id": 2,
            "family": "white_balance",
            "log_gains": log_gains.tolist(),
        }
    if family_id == 3:
        limit = float(config["maximum_basic_tone_deviation"])
        exponent = float(rng.uniform(1.0 - limit, 1.0 + limit))
        return values**exponent, {
            "family_id": 3,
            "family": "monotone_basic_tone",
            "exponent": exponent,
        }
    raise ValueError("unsupported nuisance family")


def _make_observations(
    grids: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    identity: bool = False,
) -> list[dict[str, Any]]:
    content = config["content_generator"]
    nuisance = config["reference_nuisance"]
    centres = np.asarray(
        content["content_group_palette_centres"], dtype=np.float64
    )
    group_count = int(
        content["independent_content_groups_per_operator_instance"]
    )
    if centres.shape != (group_count, 3):
        raise ValueError("content centres do not match the group count")
    fit_groups = set(int(x) for x in content["fit_and_bank_content_group_ids"])
    fit_rng = np.random.default_rng(
        int(
            content[
                "identity_reference_fit_content_seed"
                if identity
                else "fit_content_seed"
            ]
        )
    )
    heldout_rng = np.random.default_rng(
        int(
            content[
                "identity_reference_heldout_content_seed"
                if identity
                else "heldout_content_seed"
            ]
        )
    )
    fit_nuisance_rng = np.random.default_rng(
        int(
            nuisance[
                "identity_fit_nuisance_seed"
                if identity
                else "fit_nuisance_seed"
            ]
        )
    )
    heldout_nuisance_rng = np.random.default_rng(
        int(
            nuisance[
                "identity_heldout_nuisance_seed"
                if identity
                else "heldout_nuisance_seed"
            ]
        )
    )
    rows = []
    for instance_index, (grid, item) in enumerate(
        zip(grids, metadata, strict=True)
    ):
        operator = CubeDiffeomorphicColourFlow(
            grid,
            integration_steps=int(config["look_generator"]["integration_steps"]),
        )
        for group_id, centre in enumerate(centres):
            rng = fit_rng if group_id in fit_groups else heldout_rng
            source_scenes = []
            for _ in range(int(content["scenes_per_group"])):
                palette = _content_palette(rng, centre, content)
                source_scenes.append(
                    sample_palette(
                        palette,
                        sample_count=int(content["samples_per_scene"]),
                        rng=rng,
                    )
                )
            source = np.concatenate(source_scenes)
            family_id = (instance_index + group_id) % len(nuisance["families"])
            prelook, nuisance_record = _apply_reference_nuisance(
                source,
                family_id=family_id,
                rng=(
                    fit_nuisance_rng
                    if group_id in fit_groups
                    else heldout_nuisance_rng
                ),
                config=nuisance,
            )
            styled = operator.apply(prelook)
            row = dict(item)
            row.update(
                {
                    "instance_index": instance_index,
                    "group_id": group_id,
                    "nuisance": nuisance_record,
                    "target_grid": np.asarray(grid, dtype=np.float64).copy(),
                    "factorized_descriptor": factorized_reference_descriptor(
                        styled
                    ),
                    "raw_histogram_descriptor": raw_rgb_histogram_descriptor(
                        styled,
                        bins_per_channel=int(
                            config["descriptor"]["raw_histogram_bins_per_channel"]
                        ),
                    ),
                    "prelook": prelook,
                    "styled": styled,
                }
            )
            rows.append(row)
    return rows


def _make_owner_strength_fixture(
    base_grid: np.ndarray,
    *,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate same-input 53/55/56 references on one operator direction."""

    content = config["content_generator"]
    nuisance = config["reference_nuisance"]
    anchors = config["look_generator"]["owner_anchor_strength_path"]
    centres = np.asarray(
        content["content_group_palette_centres"], dtype=np.float64
    )
    groups = [
        int(x) for x in content["heldout_query_content_group_ids"]
    ]
    content_rng = np.random.default_rng(
        int(content["owner_strength_fixture_content_seed"])
    )
    nuisance_rng = np.random.default_rng(
        int(nuisance["owner_strength_fixture_nuisance_seed"])
    )
    rows = []
    for group_id in groups:
        source_scenes = []
        for _ in range(int(content["scenes_per_group"])):
            palette = _content_palette(content_rng, centres[group_id], content)
            source_scenes.append(
                sample_palette(
                    palette,
                    sample_count=int(content["samples_per_scene"]),
                    rng=content_rng,
                )
            )
        source = np.concatenate(source_scenes)
        family_id = group_id % len(nuisance["families"])
        prelook, nuisance_record = _apply_reference_nuisance(
            source,
            family_id=family_id,
            rng=nuisance_rng,
            config=nuisance,
        )
        for anchor_id in ("53", "55", "56"):
            strength = float(anchors[anchor_id])
            target_grid = strength * np.asarray(base_grid, dtype=np.float64)
            styled = CubeDiffeomorphicColourFlow(
                target_grid,
                integration_steps=int(
                    config["look_generator"]["integration_steps"]
                ),
            ).apply(prelook)
            rows.append(
                {
                    "direction_id": 0,
                    "strength_index": -1,
                    "strength": strength,
                    "look_id": f"anchor-{anchor_id}",
                    "anchor_id": anchor_id,
                    "instance_index": 0,
                    "group_id": group_id,
                    "nuisance": nuisance_record,
                    "target_grid": target_grid,
                    "factorized_descriptor": factorized_reference_descriptor(
                        styled
                    ),
                    "raw_histogram_descriptor": raw_rgb_histogram_descriptor(
                        styled,
                        bins_per_channel=int(
                            config["descriptor"]["raw_histogram_bins_per_channel"]
                        ),
                    ),
                    "prelook": prelook,
                    "styled": styled,
                }
            )
    return rows


def _rows(
    observations: list[dict[str, Any]],
    *,
    directions: set[int],
    groups: set[int],
) -> list[dict[str, Any]]:
    return [
        row
        for row in observations
        if int(row["direction_id"]) in directions
        and int(row["group_id"]) in groups
    ]


def _stack_descriptors(
    rows: list[dict[str, Any]], *, key: str
) -> np.ndarray:
    return np.stack([np.asarray(row[key], dtype=np.float64) for row in rows])


def _stack_targets(rows: list[dict[str, Any]], grids: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                row.get("target_grid", grids[int(row["instance_index"])]),
                dtype=np.float64,
            )
            for row in rows
        ]
    )


def _aggregate_by_look(
    rows: list[dict[str, Any]],
    *,
    descriptor_key: str,
    grids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["look_id"]), []).append(row)
    descriptors, targets, look_ids = [], [], []
    for look_id in sorted(grouped):
        group = grouped[look_id]
        descriptors.append(
            aggregate_reference_descriptors(
                _stack_descriptors(group, key=descriptor_key)
            )
        )
        targets.append(
            np.asarray(
                group[0].get(
                    "target_grid", grids[int(group[0]["instance_index"])]
                ),
                dtype=np.float64,
            )
        )
        look_ids.append(look_id)
    return np.stack(descriptors), np.stack(targets), look_ids


def _build_direction_strength_retriever(
    bank_rows: list[dict[str, Any]],
    identity_rows: list[dict[str, Any]],
    *,
    fit_directions: set[int],
    fit_groups: set[int],
    strengths: list[float],
) -> Any:
    identity_descriptor = aggregate_reference_descriptors(
        _stack_descriptors(
            [
                row
                for row in identity_rows
                if int(row["group_id"]) in fit_groups
            ],
            key="factorized_descriptor",
        )
    )
    ordered_strengths = np.asarray(
        [0.0, *sorted(float(x) for x in strengths)], dtype=np.float64
    )
    prototypes, base_grids, direction_ids = [], [], []
    for direction in sorted(fit_directions):
        path = [identity_descriptor]
        direction_rows = [
            row
            for row in bank_rows
            if int(row["direction_id"]) == direction
        ]
        for strength in ordered_strengths[1:]:
            rows = [
                row
                for row in direction_rows
                if float(row["strength"]) == strength
            ]
            path.append(
                aggregate_reference_descriptors(
                    _stack_descriptors(rows, key="factorized_descriptor")
                )
            )
        strongest = max(direction_rows, key=lambda row: float(row["strength"]))
        base_grids.append(
            np.asarray(strongest["target_grid"], dtype=np.float64)
            / float(strongest["strength"])
        )
        prototypes.append(np.stack(path))
        direction_ids.append(f"d{direction:02d}")
    return build_direction_strength_bank(
        np.stack(prototypes),
        ordered_strengths,
        np.stack(base_grids),
        direction_ids=direction_ids,
    )


def _operator_output_errors(
    predicted: np.ndarray,
    oracle: np.ndarray,
    *,
    points: np.ndarray,
    integration_steps: int,
) -> np.ndarray:
    values = []
    for left, right in zip(predicted, oracle, strict=True):
        left_output = CubeDiffeomorphicColourFlow(
            left, integration_steps=integration_steps
        ).apply(points)
        right_output = CubeDiffeomorphicColourFlow(
            right, integration_steps=integration_steps
        ).apply(points)
        values.append(float(np.sqrt(np.mean((left_output - right_output) ** 2))))
    return np.asarray(values, dtype=np.float64)


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _select_ridge_alpha(
    observations: list[dict[str, Any]],
    grids: np.ndarray,
    *,
    config: dict[str, Any],
    points: np.ndarray,
    multi_reference: bool,
) -> tuple[float, dict[str, float]]:
    content = config["content_generator"]
    fit_direction_ids = set(
        int(x) for x in config["look_generator"]["development_fit_direction_ids"]
    )
    fit_group_ids = set(int(x) for x in content["fit_and_bank_content_group_ids"])
    scores = {}
    for alpha in config["descriptor"]["ridge_alpha_candidates"]:
        fold_errors = []
        for fold in range(4):
            if multi_reference:
                train_rows = [
                    row
                    for row in observations
                    if int(row["direction_id"]) in fit_direction_ids
                    and int(row["group_id"]) in fit_group_ids
                    and int(row["direction_id"]) % 4 != fold
                ]
                validation_rows = [
                    row
                    for row in observations
                    if int(row["direction_id"]) in fit_direction_ids
                    and int(row["group_id"]) in fit_group_ids
                    and int(row["direction_id"]) % 4 == fold
                ]
                train, train_targets, _ = _aggregate_by_look(
                    train_rows,
                    descriptor_key="factorized_descriptor",
                    grids=grids,
                )
                validation, validation_targets, _ = _aggregate_by_look(
                    validation_rows,
                    descriptor_key="factorized_descriptor",
                    grids=grids,
                )
            else:
                train_rows = [
                    row
                    for row in observations
                    if int(row["direction_id"]) in fit_direction_ids
                    and int(row["group_id"]) in fit_group_ids
                    and int(row["direction_id"]) % 4 != fold
                    and int(row["group_id"]) != fold
                ]
                validation_rows = [
                    row
                    for row in observations
                    if int(row["direction_id"]) in fit_direction_ids
                    and int(row["group_id"]) == fold
                    and int(row["direction_id"]) % 4 == fold
                ]
                train = _stack_descriptors(
                    train_rows, key="factorized_descriptor"
                )
                train_targets = _stack_targets(train_rows, grids)
                validation = _stack_descriptors(
                    validation_rows, key="factorized_descriptor"
                )
                validation_targets = _stack_targets(validation_rows, grids)
            model = fit_reference_descriptor_ridge(
                train,
                train_targets,
                alpha=float(alpha),
                maximum_vector_norm=np.nextafter(
                    float(
                        config["look_generator"][
                            "coefficient_vector_norm_cap"
                        ]
                    ),
                    0.0,
                ),
            )
            prediction = model.predict(
                validation
            )
            fold_errors.extend(
                _operator_output_errors(
                    prediction,
                    validation_targets,
                    points=points,
                    integration_steps=int(
                        config["look_generator"]["integration_steps"]
                    ),
                )
            )
        scores[str(alpha)] = float(np.median(fold_errors))
    selected = min(
        (float(alpha) for alpha in config["descriptor"]["ridge_alpha_candidates"]),
        key=lambda alpha: (scores[str(alpha)], alpha),
    )
    return selected, scores


def _predict_methods(
    query_rows: list[dict[str, Any]],
    *,
    grids: np.ndarray,
    factorized_bank: Any,
    direction_strength_bank: Any,
    raw_bank: Any,
    ridge: Any,
    global_mean: np.ndarray,
    multi_reference: bool,
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, tuple[str, ...]],
    dict[str, np.ndarray],
]:
    if multi_reference:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in query_rows:
            grouped.setdefault(str(row["look_id"]), []).append(row)
        collapsed = [grouped[key][0] for key in sorted(grouped)]
        factorized = np.stack(
            [
                aggregate_reference_descriptors(
                    _stack_descriptors(
                        grouped[str(row["look_id"])],
                        key="factorized_descriptor",
                    )
                )
                for row in collapsed
            ]
        )
        raw = np.stack(
            [
                aggregate_reference_descriptors(
                    _stack_descriptors(
                        grouped[str(row["look_id"])],
                        key="raw_histogram_descriptor",
                    )
                )
                for row in collapsed
            ]
        )
        effective_rows = collapsed
    else:
        factorized = _stack_descriptors(
            query_rows, key="factorized_descriptor"
        )
        raw = _stack_descriptors(query_rows, key="raw_histogram_descriptor")
        effective_rows = query_rows
    factor_grids, factor_ids, _ = factorized_bank.hard_retrieve(factorized)
    direction_grids, direction_ids, direction_strengths, _ = (
        direction_strength_bank.retrieve(factorized)
    )
    raw_grids, raw_ids, _ = raw_bank.hard_retrieve(raw)
    count = len(effective_rows)
    zero = np.zeros_like(global_mean)
    methods = {
        "identity": np.repeat(zero[None, ...], count, axis=0),
        "global_mean_operator": np.repeat(global_mean[None, ...], count, axis=0),
        "raw_rgb_histogram_nearest": raw_grids,
        "factorized_descriptor_hard_retrieval": factor_grids,
        "factorized_direction_strength_retrieval": direction_grids,
        "factorized_descriptor_ridge_operator": ridge.predict(factorized),
    }
    return (
        methods,
        effective_rows,
        {
            "raw_rgb_histogram_nearest": raw_ids,
            "factorized_descriptor_hard_retrieval": factor_ids,
            "factorized_direction_strength_retrieval": direction_ids,
        },
        {
            "factorized_direction_strength_retrieval": direction_strengths,
        },
    )


def _same_look_replicate_error(
    rows: list[dict[str, Any]],
    predicted: np.ndarray,
    *,
    points: np.ndarray,
    integration_steps: int,
) -> float:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["look_id"]), []).append(index)
    errors = []
    for indices in grouped.values():
        outputs = [
            CubeDiffeomorphicColourFlow(
                predicted[index], integration_steps=integration_steps
            ).apply(points)
            for index in indices
        ]
        for left in range(len(outputs)):
            for right in range(left + 1, len(outputs)):
                errors.append(
                    float(
                        np.sqrt(np.mean((outputs[left] - outputs[right]) ** 2))
                    )
                )
    return float(np.median(errors))


def _spearman(values: np.ndarray, target: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if np.std(values) <= 1e-15:
        return 0.0
    def rank_average(array: np.ndarray) -> np.ndarray:
        order = np.argsort(array, kind="stable")
        ranks = np.empty(len(array), dtype=np.float64)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and array[order[end]] == array[order[cursor]]:
                end += 1
            ranks[order[cursor:end]] = 0.5 * (cursor + end - 1)
            cursor = end
        return ranks

    value_rank = rank_average(values)
    target_rank = rank_average(target)
    return float(np.corrcoef(value_rank, target_rank)[0, 1])


def _strength_metrics(
    rows: list[dict[str, Any]],
    predicted: np.ndarray,
    *,
    points: np.ndarray,
    integration_steps: int,
) -> dict[str, float]:
    grouped: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(int(row["direction_id"]), []).append(index)
    correlations = []
    for indices in grouped.values():
        by_strength: dict[float, list[float]] = {}
        for index in indices:
            output = CubeDiffeomorphicColourFlow(
                predicted[index], integration_steps=integration_steps
            ).apply(points)
            magnitude = float(np.sqrt(np.mean((output - points) ** 2)))
            by_strength.setdefault(float(rows[index]["strength"]), []).append(
                magnitude
            )
        strengths = np.array(sorted(by_strength), dtype=np.float64)
        magnitudes = np.array(
            [np.mean(by_strength[value]) for value in strengths],
            dtype=np.float64,
        )
        correlations.append(_spearman(magnitudes, strengths))
    return {
        "median_spearman": float(np.median(correlations)),
        "minimum_spearman": float(np.min(correlations)),
    }


def _balanced_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
) -> float:
    classes = np.unique(np.concatenate((train_y, test_y)))
    mean = np.mean(train_x, axis=0)
    scale = np.maximum(np.std(train_x, axis=0), 1e-8)
    left = (train_x - mean) / scale
    right = (test_x - mean) / scale
    one_hot = np.equal(train_y[:, None], classes[None, :]).astype(np.float64)
    gram = left.T @ left + float(alpha) * np.eye(left.shape[1])
    coefficients = np.linalg.solve(gram, left.T @ one_hot)
    prediction = classes[np.argmax(right @ coefficients, axis=1)]
    recalls = [
        np.mean(prediction[test_y == value] == value)
        for value in classes
        if np.any(test_y == value)
    ]
    return float(np.mean(recalls))


def _probe_metrics(
    fit_rows: list[dict[str, Any]],
    fit_predictions: np.ndarray,
    test_rows: list[dict[str, Any]],
    test_predictions: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    def residuals(
        rows: list[dict[str, Any]], predictions: np.ndarray
    ) -> np.ndarray:
        result = np.empty_like(predictions)
        grouped: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row["look_id"]), []).append(index)
        for indices in grouped.values():
            centre = np.mean(predictions[indices], axis=0)
            result[indices] = predictions[indices] - centre
        return result.reshape(len(result), -1)

    fit_x = residuals(fit_rows, fit_predictions)
    test_x = residuals(test_rows, test_predictions)
    fit_content = np.array([int(row["group_id"]) for row in fit_rows])
    test_content = np.array([int(row["group_id"]) for row in test_rows])
    fit_nuisance = np.array(
        [int(row["nuisance"]["family_id"]) for row in fit_rows]
    )
    test_nuisance = np.array(
        [int(row["nuisance"]["family_id"]) for row in test_rows]
    )
    return {
        "content_group_balanced_accuracy": _balanced_accuracy(
            fit_x, fit_content, test_x, test_content, alpha=alpha
        ),
        "content_group_chance": 1.0 / len(np.unique(test_content)),
        "nuisance_family_balanced_accuracy": _balanced_accuracy(
            fit_x, fit_nuisance, test_x, test_nuisance, alpha=alpha
        ),
        "nuisance_family_chance": 1.0 / len(np.unique(test_nuisance)),
    }


def _owner_strength_fixture_metrics(
    rows: list[dict[str, Any]],
    predicted: np.ndarray,
    *,
    points: np.ndarray,
    integration_steps: int,
) -> dict[str, float]:
    strengths = np.asarray(
        [float(row["strength"]) for row in rows], dtype=np.float64
    )
    magnitudes = []
    directions = []
    for grid in predicted:
        output = CubeDiffeomorphicColourFlow(
            grid, integration_steps=integration_steps
        ).apply(points)
        magnitudes.append(float(np.sqrt(np.mean((output - points) ** 2))))
        flat = np.asarray(grid, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(flat)
        directions.append(flat / max(norm, 1e-15))
    cosines = []
    for left in range(len(directions)):
        for right in range(left + 1, len(directions)):
            cosines.append(float(np.dot(directions[left], directions[right])))
    return {
        "strength_order_spearman": _spearman(
            np.asarray(magnitudes, dtype=np.float64), strengths
        ),
        "minimum_pairwise_direction_cosine": float(np.min(cosines)),
        "median_pairwise_direction_cosine": float(np.median(cosines)),
    }


def _evaluate_predictions(
    predictions: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    *,
    grids: np.ndarray,
    palettes: list[DiagonalGaussianMixturePalette],
    config: dict[str, Any],
    points: np.ndarray,
    retrieved_ids: dict[str, tuple[str, ...]],
    retrieved_strengths: dict[str, np.ndarray],
    multi_reference: bool,
) -> dict[str, Any]:
    oracle = _stack_targets(rows, grids)
    row_palettes = [palettes[int(row["instance_index"])] for row in rows]
    reports = {}
    for method_id, predicted in predictions.items():
        metrics = _evaluate_method(
            grids=predicted,
            oracle_grids=oracle,
            palettes=row_palettes,
            points=points,
            integration_steps=int(config["look_generator"]["integration_steps"]),
            permutation_grid=predicted[0],
        )
        family_metrics = {}
        for family in sorted(
            {str(row["operator_family"]) for row in rows}
        ):
            indices = [
                index
                for index, row in enumerate(rows)
                if str(row["operator_family"]) == family
            ]
            family_metrics[family] = _summary(
                _operator_output_errors(
                    predicted[indices],
                    oracle[indices],
                    points=points,
                    integration_steps=int(
                        config["look_generator"]["integration_steps"]
                    ),
                )
            )
        metrics["operator_family_output_rmse"] = family_metrics
        if not multi_reference:
            metrics["same_look_replicate_operator_rmse_median"] = (
                _same_look_replicate_error(
                    rows,
                    predicted,
                    points=points,
                    integration_steps=int(
                        config["look_generator"]["integration_steps"]
                    ),
                )
            )
            metrics["strength"] = _strength_metrics(
                rows,
                predicted,
                points=points,
                integration_steps=int(
                    config["look_generator"]["integration_steps"]
                ),
            )
        if method_id in retrieved_ids:
            direction_strength = (
                method_id == "factorized_direction_strength_retrieval"
            )
            expected = [
                (
                    f"d{int(row['direction_id']):02d}"
                    if direction_strength
                    else str(row["look_id"])
                )
                for row in rows
            ]
            actual = list(retrieved_ids[method_id])
            accuracy_key = (
                "seen_direction_retrieval_accuracy"
                if direction_strength
                else "seen_look_retrieval_accuracy"
            )
            family_key = f"{accuracy_key}_by_operator_family"
            metrics[accuracy_key] = float(
                np.mean(np.equal(expected, actual))
            )
            metrics[family_key] = {
                family: float(
                    np.mean(
                        [
                            expected[index] == actual[index]
                            for index, row in enumerate(rows)
                            if str(row["operator_family"]) == family
                        ]
                    )
                )
                for family in sorted(
                    {str(row["operator_family"]) for row in rows}
                )
            }
            if method_id in retrieved_strengths:
                expected_strength = np.asarray(
                    [float(row["strength"]) for row in rows],
                    dtype=np.float64,
                )
                errors = np.abs(
                    retrieved_strengths[method_id] - expected_strength
                )
                metrics["seen_strength_absolute_error"] = _summary(errors)
        reports[method_id] = metrics
    return reports


def _identity_false_positive(
    identity_rows: list[dict[str, Any]],
    *,
    factorized_bank: Any,
    direction_strength_bank: Any,
    raw_bank: Any,
    ridge: Any,
    global_mean: np.ndarray,
    points: np.ndarray,
    integration_steps: int,
    multi_reference: bool,
) -> dict[str, float]:
    methods, rows, _, _ = _predict_methods(
        identity_rows,
        grids=np.zeros((1, *global_mean.shape), dtype=np.float64),
        factorized_bank=factorized_bank,
        direction_strength_bank=direction_strength_bank,
        raw_bank=raw_bank,
        ridge=ridge,
        global_mean=global_mean,
        multi_reference=multi_reference,
    )
    result = {}
    for method_id, predicted in methods.items():
        errors = []
        for grid in predicted:
            output = CubeDiffeomorphicColourFlow(
                grid, integration_steps=integration_steps
            ).apply(points)
            errors.append(float(np.sqrt(np.mean((output - points) ** 2))))
        result[method_id] = float(np.median(errors))
    return result


def _paired_upper_bound(
    observations: list[dict[str, Any]],
    grids: np.ndarray,
    *,
    config: dict[str, Any],
    points: np.ndarray,
) -> dict[str, Any]:
    fit = config["paired_upper_bound_optimization"]
    unseen = set(
        int(x)
        for x in config["look_generator"]["development_unseen_direction_ids"]
    )
    rows = [
        row
        for row in observations
        if int(row["direction_id"]) in unseen and int(row["group_id"]) == 4
    ]
    device = (
        "cuda"
        if fit["device"] == "cuda_if_available_else_cpu"
        and torch.cuda.is_available()
        else "cpu"
    )
    predicted, traces = [], []
    for index, row in enumerate(rows):
        operator, trace = fit_paired_cube_diffeomorphic_flow(
            row["prelook"],
            row["styled"],
            axis_size=int(config["look_generator"]["velocity_grid_axis_size"]),
            integration_steps=int(config["look_generator"]["integration_steps"]),
            coefficient_vector_norm_cap=float(
                config["look_generator"]["coefficient_vector_norm_cap"]
            ),
            steps=int(fit["steps"]),
            learning_rate=float(fit["learning_rate"]),
            coefficient_l2=float(fit["coefficient_l2"]),
            velocity_smoothness_l2=float(fit["velocity_smoothness_l2"]),
            gradient_clip_norm=float(fit["gradient_clip_norm"]),
            seed=int(fit["seed"]) + index,
            device=device,
            deterministic_algorithms=bool(fit["deterministic_algorithms"]),
            optimization_dtype=str(fit["dtype"]),
        )
        predicted.append(operator.velocity_grid)
        traces.append(trace)
    errors = _operator_output_errors(
        np.stack(predicted),
        _stack_targets(rows, grids),
        points=points,
        integration_steps=int(config["look_generator"]["integration_steps"]),
    )
    return {
        "device": device,
        "operator_output_rmse_median": float(np.median(errors)),
        "operator_output_rmse_p90": float(np.quantile(errors, 0.9)),
        "initial_pair_mse_median": float(
            np.median([trace["initial_pair_mse"] for trace in traces])
        ),
        "final_pair_mse_median": float(
            np.median([trace["final_pair_mse"] for trace in traces])
        ),
        "sample_count": len(rows),
    }


def _practical_gate(
    metrics: dict[str, Any],
    *,
    method_id: str,
    baseline_identity: dict[str, Any],
    baseline_global: dict[str, Any],
    identity_error: float,
    probes: dict[str, float] | None,
    owner_fixture: dict[str, float],
    gates: dict[str, Any],
    require_retrieval: bool,
) -> dict[str, bool]:
    error = float(metrics["operator_output_rmse"]["median"])
    results = {
        "operator_median": error
        <= float(gates["maximum_oracle_operator_output_rmse_median"]),
        "operator_p90": float(metrics["operator_output_rmse"]["p90"])
        <= float(gates["maximum_oracle_operator_output_rmse_p90"]),
        "operator_families": all(
            family["median"]
            <= float(gates["maximum_oracle_operator_output_rmse_median"])
            and family["p90"]
            <= float(gates["maximum_oracle_operator_output_rmse_p90"])
            for family in metrics["operator_family_output_rmse"].values()
        ),
        "replicate": float(
            metrics.get("same_look_replicate_operator_rmse_median", 0.0)
        )
        <= float(gates["maximum_same_look_replicate_operator_rmse_median"]),
        "beats_identity": 1.0
        - error
        / max(float(baseline_identity["operator_output_rmse"]["median"]), 1e-30)
        >= float(gates["minimum_operator_error_improvement_over_identity_fraction"]),
        "beats_global": 1.0
        - error
        / max(float(baseline_global["operator_output_rmse"]["median"]), 1e-30)
        >= float(
            gates["minimum_operator_error_improvement_over_global_mean_fraction"]
        ),
        "identity_false_positive": identity_error
        <= float(gates["maximum_identity_reference_grid_rmse"]),
        "strength": float(metrics.get("strength", {}).get("median_spearman", 1.0))
        >= float(gates["minimum_strength_order_spearman"]),
        "owner_strength_order": owner_fixture["strength_order_spearman"]
        >= float(gates["minimum_strength_order_spearman"]),
        "owner_same_direction": owner_fixture[
            "minimum_pairwise_direction_cosine"
        ]
        >= float(gates["minimum_53_55_56_direction_cosine"]),
        "range": metrics["structure"]["minimum_output"]
        >= float(gates["minimum_output"])
        and metrics["structure"]["maximum_output"]
        <= float(gates["maximum_output"]),
        "jacobian": metrics["structure"]["minimum_jacobian_determinant"]
        > float(gates["minimum_jacobian_determinant_exclusive"]),
        "norm": metrics["structure"]["maximum_jacobian_spectral_norm"]
        <= float(gates["maximum_jacobian_spectral_norm"]),
        "inverse": metrics["structure"]["maximum_inverse_error"]
        <= float(gates["maximum_inverse_error"]),
        "replay": metrics["structure"]["maximum_replay_error"]
        <= float(gates["maximum_replay_error"]),
        "coefficient": metrics["structure"]["maximum_coefficient_vector_norm"]
        <= float(gates["maximum_coefficient_vector_norm"]),
    }
    if require_retrieval:
        direction_strength = (
            method_id == "factorized_direction_strength_retrieval"
        )
        accuracy_key = (
            "seen_direction_retrieval_accuracy"
            if direction_strength
            else "seen_look_retrieval_accuracy"
        )
        family_key = f"{accuracy_key}_by_operator_family"
        results["retrieval"] = float(metrics[accuracy_key]) >= float(
            gates["minimum_seen_look_retrieval_accuracy"]
        )
        results["retrieval_families"] = all(
            value >= float(gates["minimum_seen_look_retrieval_accuracy"])
            for value in metrics[family_key].values()
        )
        if direction_strength:
            results["strength_absolute_error"] = float(
                metrics["seen_strength_absolute_error"]["median"]
            ) <= float(gates["maximum_seen_strength_absolute_error"])
    if probes is not None:
        margin = float(gates["maximum_content_or_nuisance_probe_above_chance"])
        results["content_probe"] = (
            probes["content_group_balanced_accuracy"]
            <= probes["content_group_chance"] + margin
        )
        results["nuisance_probe"] = (
            probes["nuisance_family_balanced_accuracy"]
            <= probes["nuisance_family_chance"] + margin
        )
    results["all_except_repeat"] = bool(all(results.values()))
    results["method_id_matches"] = method_id in {
        "factorized_descriptor_hard_retrieval",
        "factorized_direction_strength_retrieval",
        "factorized_descriptor_ridge_operator",
    }
    return results


def run_development(
    config: dict[str, Any],
    s4_decision: dict[str, Any],
    u1_decision: dict[str, Any] | None,
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    _validate_activation(config, s4_decision, u1_decision)
    _validate_experiment_partition(config)
    grids, palettes, metadata = _look_instances(config)
    observations = _make_observations(grids, metadata, config=config)
    identity_grid = np.zeros_like(grids[:1])
    identity_metadata = [
        {
            "direction_id": -1,
            "strength_index": 0,
            "strength": 0.0,
            "look_id": "identity",
        }
    ]
    identity_rows = _make_observations(
        identity_grid, identity_metadata, config=config, identity=True
    )
    fit_directions = set(
        int(x) for x in config["look_generator"]["development_fit_direction_ids"]
    )
    unseen_directions = set(
        int(x)
        for x in config["look_generator"]["development_unseen_direction_ids"]
    )
    fit_groups = set(
        int(x)
        for x in config["content_generator"]["fit_and_bank_content_group_ids"]
    )
    query_groups = set(
        int(x)
        for x in config["content_generator"]["heldout_query_content_group_ids"]
    )
    bank_rows = _rows(
        observations, directions=fit_directions, groups=fit_groups
    )
    bank_rows += [
        row for row in identity_rows if int(row["group_id"]) in fit_groups
    ]
    factor_features, factor_targets, look_ids = _aggregate_by_look(
        bank_rows,
        descriptor_key="factorized_descriptor",
        grids=grids,
    )
    raw_features, raw_targets, raw_look_ids = _aggregate_by_look(
        bank_rows,
        descriptor_key="raw_histogram_descriptor",
        grids=grids,
    )
    factorized_bank = build_reference_look_bank(
        factor_features, factor_targets, look_ids=look_ids
    )
    raw_bank = build_reference_look_bank(
        raw_features, raw_targets, look_ids=raw_look_ids
    )
    direction_strength_bank = _build_direction_strength_retriever(
        bank_rows,
        identity_rows,
        fit_directions=fit_directions,
        fit_groups=fit_groups,
        strengths=[float(x) for x in config["look_generator"]["strengths"]],
    )
    points = _grid(int(config["evaluation"]["uniform_grid_axis_size"]))
    selected_alpha_single, alpha_scores_single = _select_ridge_alpha(
        observations,
        grids,
        config=config,
        points=points,
        multi_reference=False,
    )
    selected_alpha_multi, alpha_scores_multi = _select_ridge_alpha(
        observations,
        grids,
        config=config,
        points=points,
        multi_reference=True,
    )
    ridge_single = fit_reference_descriptor_ridge(
        _stack_descriptors(bank_rows, key="factorized_descriptor"),
        _stack_targets(bank_rows, grids),
        alpha=selected_alpha_single,
        maximum_vector_norm=np.nextafter(
            float(
                config["look_generator"]["coefficient_vector_norm_cap"]
            ),
            0.0,
        ),
    )
    ridge_multi = fit_reference_descriptor_ridge(
        factor_features,
        factor_targets,
        alpha=selected_alpha_multi,
        maximum_vector_norm=np.nextafter(
            float(
                config["look_generator"]["coefficient_vector_norm_cap"]
            ),
            0.0,
        ),
    )
    global_mean = np.mean(_stack_targets(bank_rows, grids), axis=0)

    owner_fixture_rows = _make_owner_strength_fixture(
        grids[0] / float(metadata[0]["strength"]),
        config=config,
    )
    owner_predictions, owner_effective_rows, _, _ = _predict_methods(
        owner_fixture_rows,
        grids=grids,
        factorized_bank=factorized_bank,
        direction_strength_bank=direction_strength_bank,
        raw_bank=raw_bank,
        ridge=ridge_multi,
        global_mean=global_mean,
        multi_reference=True,
    )
    owner_fixture_metrics = {
        method_id: _owner_strength_fixture_metrics(
            owner_effective_rows,
            prediction,
            points=points,
            integration_steps=int(
                config["look_generator"]["integration_steps"]
            ),
        )
        for method_id, prediction in owner_predictions.items()
    }

    regimes: dict[str, Any] = {}
    predictions_by_regime: dict[str, dict[str, np.ndarray]] = {}
    rows_by_regime: dict[str, list[dict[str, Any]]] = {}
    ids_by_regime: dict[str, dict[str, tuple[str, ...]]] = {}
    strengths_by_regime: dict[str, dict[str, np.ndarray]] = {}
    for regime_id, directions, multi in (
        ("seen_single_output_only", fit_directions, False),
        ("seen_four_output_only", fit_directions, True),
        ("unseen_single_output_only", unseen_directions, False),
        ("unseen_four_output_only", unseen_directions, True),
    ):
        query = _rows(observations, directions=directions, groups=query_groups)
        (
            predictions,
            effective_rows,
            retrieved_ids,
            retrieved_strengths,
        ) = _predict_methods(
            query,
            grids=grids,
            factorized_bank=factorized_bank,
            direction_strength_bank=direction_strength_bank,
            raw_bank=raw_bank,
            ridge=ridge_multi if multi else ridge_single,
            global_mean=global_mean,
            multi_reference=multi,
        )
        predictions_by_regime[regime_id] = predictions
        rows_by_regime[regime_id] = effective_rows
        ids_by_regime[regime_id] = retrieved_ids
        strengths_by_regime[regime_id] = retrieved_strengths
        regimes[regime_id] = _evaluate_predictions(
            predictions,
            effective_rows,
            grids=grids,
            palettes=palettes,
            config=config,
            points=points,
            retrieved_ids=retrieved_ids,
            retrieved_strengths=retrieved_strengths,
            multi_reference=multi,
        )

    identity_heldout = [
        row for row in identity_rows if int(row["group_id"]) in query_groups
    ]
    identity_errors_single = _identity_false_positive(
        identity_heldout,
        factorized_bank=factorized_bank,
        direction_strength_bank=direction_strength_bank,
        raw_bank=raw_bank,
        ridge=ridge_single,
        global_mean=global_mean,
        points=points,
        integration_steps=int(config["look_generator"]["integration_steps"]),
        multi_reference=False,
    )
    identity_errors_multi = _identity_false_positive(
        identity_heldout,
        factorized_bank=factorized_bank,
        direction_strength_bank=direction_strength_bank,
        raw_bank=raw_bank,
        ridge=ridge_multi,
        global_mean=global_mean,
        points=points,
        integration_steps=int(config["look_generator"]["integration_steps"]),
        multi_reference=True,
    )

    fit_probe_rows = rows_by_regime["seen_single_output_only"]
    unseen_probe_rows = rows_by_regime["unseen_single_output_only"]
    probe_metrics = {}
    for method_id in (
        "raw_rgb_histogram_nearest",
        "factorized_descriptor_hard_retrieval",
        "factorized_direction_strength_retrieval",
        "factorized_descriptor_ridge_operator",
    ):
        probe_metrics[method_id] = _probe_metrics(
            fit_probe_rows,
            predictions_by_regime["seen_single_output_only"][method_id],
            unseen_probe_rows,
            predictions_by_regime["unseen_single_output_only"][method_id],
            alpha=float(config["evaluation"]["probe_ridge_alpha"]),
        )

    paired = _paired_upper_bound(
        observations, grids, config=config, points=points
    )
    gates = config["gates"]
    seen_single = regimes["seen_single_output_only"]
    seen_four = regimes["seen_four_output_only"]
    unseen_single = regimes["unseen_single_output_only"]
    unseen_four = regimes["unseen_four_output_only"]
    gate_results = {
        "seen_single_retrieval": _practical_gate(
            seen_single["factorized_direction_strength_retrieval"],
            method_id="factorized_direction_strength_retrieval",
            baseline_identity=seen_single["identity"],
            baseline_global=seen_single["global_mean_operator"],
            identity_error=identity_errors_single[
                "factorized_direction_strength_retrieval"
            ],
            probes=probe_metrics["factorized_direction_strength_retrieval"],
            owner_fixture=owner_fixture_metrics[
                "factorized_direction_strength_retrieval"
            ],
            gates=gates,
            require_retrieval=True,
        ),
        "seen_four_retrieval": _practical_gate(
            seen_four["factorized_direction_strength_retrieval"],
            method_id="factorized_direction_strength_retrieval",
            baseline_identity=seen_four["identity"],
            baseline_global=seen_four["global_mean_operator"],
            identity_error=identity_errors_multi[
                "factorized_direction_strength_retrieval"
            ],
            probes=probe_metrics["factorized_direction_strength_retrieval"],
            owner_fixture=owner_fixture_metrics[
                "factorized_direction_strength_retrieval"
            ],
            gates=gates,
            require_retrieval=True,
        ),
        "unseen_single_regression": _practical_gate(
            unseen_single["factorized_descriptor_ridge_operator"],
            method_id="factorized_descriptor_ridge_operator",
            baseline_identity=unseen_single["identity"],
            baseline_global=unseen_single["global_mean_operator"],
            identity_error=identity_errors_single[
                "factorized_descriptor_ridge_operator"
            ],
            probes=probe_metrics["factorized_descriptor_ridge_operator"],
            owner_fixture=owner_fixture_metrics[
                "factorized_descriptor_ridge_operator"
            ],
            gates=gates,
            require_retrieval=False,
        ),
        "unseen_four_regression": _practical_gate(
            unseen_four["factorized_descriptor_ridge_operator"],
            method_id="factorized_descriptor_ridge_operator",
            baseline_identity=unseen_four["identity"],
            baseline_global=unseen_four["global_mean_operator"],
            identity_error=identity_errors_multi[
                "factorized_descriptor_ridge_operator"
            ],
            probes=probe_metrics["factorized_descriptor_ridge_operator"],
            owner_fixture=owner_fixture_metrics[
                "factorized_descriptor_ridge_operator"
            ],
            gates=gates,
            require_retrieval=False,
        ),
    }
    paired_pass = bool(
        paired["operator_output_rmse_median"]
        <= float(gates["maximum_oracle_operator_output_rmse_median"])
        and paired["operator_output_rmse_p90"]
        <= float(gates["maximum_oracle_operator_output_rmse_p90"])
    )
    primary_gates = tuple(gate_results.values())

    def passes_without(
        result: dict[str, bool], excluded: set[str]
    ) -> bool:
        return all(
            value
            for key, value in result.items()
            if key not in excluded
            and key not in {"all_except_repeat", "method_id_matches"}
        )

    structure_keys = {"range", "jacobian", "norm", "inverse", "replay", "coefficient"}
    probe_keys = {"content_probe", "nuisance_probe"}
    structure_failure = any(
        passes_without(result, structure_keys)
        and not all(result[key] for key in structure_keys)
        for result in primary_gates
    )
    probe_failure = any(
        passes_without(result, probe_keys)
        and not all(result[key] for key in probe_keys)
        for result in primary_gates
    )
    if gate_results["unseen_single_regression"]["all_except_repeat"]:
        branch = "pending_repeat_unseen_regression_passes"
    elif gate_results["unseen_four_regression"]["all_except_repeat"]:
        branch = "pending_repeat_single_fails_multi_passes"
    elif gate_results["seen_single_retrieval"]["all_except_repeat"]:
        branch = "pending_repeat_seen_retrieval_only_passes"
    elif gate_results["seen_four_retrieval"]["all_except_repeat"]:
        branch = "pending_repeat_single_fails_multi_passes"
    elif structure_failure:
        branch = "structure_or_repeat_fails"
    elif probe_failure:
        branch = "content_or_nuisance_control_fails"
    elif paired_pass:
        branch = "paired_upper_bound_only_passes"
    else:
        branch = "all_practical_methods_fail"

    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "used_seeds": sorted(
            {
                *(
                    int(item["seed"])
                    for item in config["look_generator"][
                        "base_direction_generators"
                    ]
                ),
                int(config["content_generator"]["fit_content_seed"]),
                int(config["content_generator"]["heldout_content_seed"]),
                int(
                    config["content_generator"][
                        "identity_reference_fit_content_seed"
                    ]
                ),
                int(
                    config["content_generator"][
                        "identity_reference_heldout_content_seed"
                    ]
                ),
                int(
                    config["content_generator"][
                        "owner_strength_fixture_content_seed"
                    ]
                ),
                int(config["reference_nuisance"]["fit_nuisance_seed"]),
                int(config["reference_nuisance"]["heldout_nuisance_seed"]),
                int(
                    config["reference_nuisance"][
                        "identity_fit_nuisance_seed"
                    ]
                ),
                int(
                    config["reference_nuisance"][
                        "identity_heldout_nuisance_seed"
                    ]
                ),
                int(
                    config["reference_nuisance"][
                        "owner_strength_fixture_nuisance_seed"
                    ]
                ),
                int(config["paired_upper_bound_optimization"]["seed"]),
            }
        ),
        "reserved_confirmation_seeds_accessed": False,
        "selected_ridge_alpha": {
            "single_reference": selected_alpha_single,
            "four_reference": selected_alpha_multi,
        },
        "ridge_alpha_cv_operator_rmse_median": {
            "single_reference": alpha_scores_single,
            "four_reference": alpha_scores_multi,
        },
        "regimes": regimes,
        "identity_reference_false_positive_output_rmse_median": {
            "single_reference": identity_errors_single,
            "four_reference": identity_errors_multi,
        },
        "content_nuisance_probes": probe_metrics,
        "owner_53_55_56_strength_path": owner_fixture_metrics,
        "paired_reference_upper_bound": paired,
        "gate_results": gate_results,
        "paired_upper_bound_pass": paired_pass,
        "decision_branch_before_repeat": branch,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2w1_reference_look_identifiability_development_v1.json",
    )
    parser.add_argument("--s4-decision", type=Path, default=None)
    parser.add_argument("--u1-decision", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    gate = config["activation_gate"]
    s4_path = args.s4_decision or ROOT / gate["s4_decision_path"]
    if not s4_path.is_file():
        raise SystemExit(f"W1 S4 activation evidence is absent: {s4_path}")
    s4_decision = json.loads(s4_path.read_text(encoding="utf-8"))
    u1_path = args.u1_decision or ROOT / gate["u1_decision_path"]
    u1_decision = (
        json.loads(u1_path.read_text(encoding="utf-8"))
        if u1_path.is_file()
        else None
    )
    software_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = run_development(
        config,
        s4_decision,
        u1_decision,
        config_sha256=_sha256(config_bytes),
        software_commit=software_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
