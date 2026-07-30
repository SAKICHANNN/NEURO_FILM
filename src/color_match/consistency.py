"""Cross-image context-invariance evidence for album-consistent matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.color_engine import linear_rgb_to_lab
from src.preprocess import SourceProfile, WorkingImage

from .contracts import ReferenceLookRecipe, ReferenceMatchContractError
from .render import render_reference_look


@dataclass(frozen=True)
class ContextInvariancePolicy:
    """Maximum allowed shared-colour drift across unrelated surroundings."""

    max_delta_e76_median: float = 0.50
    max_delta_e76_p95: float = 1.00
    max_delta_e76: float = 3.00


@dataclass(frozen=True)
class ContextInvarianceMetrics:
    """One recipe's shared-patch drift between two image contexts."""

    passed: bool
    reasons: tuple[str, ...]
    delta_e76_median: float
    delta_e76_p95: float
    delta_e76_maximum: float


@dataclass(frozen=True)
class ContextInvarianceBatchMetrics:
    """Tail aggregation across independently fitted reference recipes."""

    samples: tuple[ContextInvarianceMetrics, ...]
    passed_recipe_count: int
    failed_recipe_count: int
    maximum_delta_e76_median: float
    maximum_delta_e76_p95: float
    maximum_delta_e76: float

    @property
    def passed(self) -> bool:
        return bool(self.samples) and all(
            sample.passed for sample in self.samples
        )


def validate_context_invariance_policy(
    policy: ContextInvariancePolicy,
) -> None:
    if not isinstance(policy, ContextInvariancePolicy):
        raise ReferenceMatchContractError(
            "context invariance policy must be ContextInvariancePolicy"
        )
    values = (
        policy.max_delta_e76_median,
        policy.max_delta_e76_p95,
        policy.max_delta_e76,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ReferenceMatchContractError(
            "context invariance limits must be finite and non-negative"
        )
    if not (
        policy.max_delta_e76_median
        <= policy.max_delta_e76_p95
        <= policy.max_delta_e76
    ):
        raise ReferenceMatchContractError(
            "context invariance limits must be ordered median <= p95 <= max"
        )


def _working(pixels: np.ndarray, name: str) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile(
            "synthetic_context_invariance_probe_v1",
            "shared colours in distinct global contexts",
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path(f"synthetic/{name}.exr"),
    )


def make_context_invariance_probes(
) -> tuple[WorkingImage, WorkingImage, tuple[slice, slice]]:
    """Return two images with one exactly shared colour-chart region."""

    height, width = 32, 48
    dark_cool = np.empty((height, width, 3), dtype=np.float32)
    bright_warm = np.empty_like(dark_cool)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    dark_cool[:] = np.asarray([0.025, 0.045, 0.095], dtype=np.float32)
    dark_cool += x * np.asarray([0.05, 0.08, 0.12], dtype=np.float32)
    dark_cool += y * np.asarray([0.03, 0.05, 0.08], dtype=np.float32)
    bright_warm[:] = np.asarray([0.62, 0.46, 0.27], dtype=np.float32)
    bright_warm += x * np.asarray([0.20, 0.13, 0.06], dtype=np.float32)
    bright_warm += y * np.asarray([0.10, 0.08, 0.04], dtype=np.float32)

    rows = slice(8, 24)
    columns = slice(12, 36)
    shared_height = rows.stop - rows.start
    shared_width = columns.stop - columns.start
    chart_x = np.linspace(
        0.08,
        0.88,
        shared_width,
        dtype=np.float32,
    )[None, :]
    chart_y = np.linspace(
        0.10,
        0.82,
        shared_height,
        dtype=np.float32,
    )[:, None]
    shared = np.stack(
        [
            np.broadcast_to(chart_x, (shared_height, shared_width)),
            np.broadcast_to(chart_y, (shared_height, shared_width)),
            np.clip(0.65 * chart_x + 0.35 * chart_y, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32)
    dark_cool[rows, columns] = shared
    bright_warm[rows, columns] = shared
    return (
        _working(dark_cool, "context_dark_cool_v1"),
        _working(bright_warm, "context_bright_warm_v1"),
        (rows, columns),
    )


def evaluate_recipe_context_invariance(
    recipe: ReferenceLookRecipe,
    *,
    policy: ContextInvariancePolicy | None = None,
) -> ContextInvarianceMetrics:
    """Measure shared-colour drift caused only by image surroundings."""

    resolved = policy or ContextInvariancePolicy()
    validate_context_invariance_policy(resolved)
    first, second, shared_region = make_context_invariance_probes()
    first_output = render_reference_look(recipe, first, source_index=0).image
    second_output = render_reference_look(recipe, second, source_index=1).image
    return evaluate_context_invariance_outputs(
        first_output,
        second_output,
        shared_region=shared_region,
        policy=resolved,
    )


def evaluate_context_invariance_outputs(
    first_output: WorkingImage,
    second_output: WorkingImage,
    *,
    shared_region: tuple[slice, slice] | None = None,
    policy: ContextInvariancePolicy | None = None,
) -> ContextInvarianceMetrics:
    """Evaluate two rendered probes, allowing future renderer adapters."""

    resolved = policy or ContextInvariancePolicy()
    validate_context_invariance_policy(resolved)
    if not isinstance(first_output, WorkingImage) or not isinstance(
        second_output,
        WorkingImage,
    ):
        raise ReferenceMatchContractError(
            "context invariance outputs must be WorkingImage values"
        )
    if (
        first_output.pixels.shape != (32, 48, 3)
        or second_output.pixels.shape != first_output.pixels.shape
        or first_output.working_space != "linear_srgb"
        or second_output.working_space != "linear_srgb"
        or first_output.transfer_state != "display_linear"
        or second_output.transfer_state != "display_linear"
    ):
        raise ReferenceMatchContractError(
            "context invariance outputs must be 32x48 display-linear "
            "linear_srgb"
        )
    region = shared_region or (slice(8, 24), slice(12, 36))
    first_lab = linear_rgb_to_lab(
        first_output.pixels[region],
        working_space="linear_srgb",
    )
    second_lab = linear_rgb_to_lab(
        second_output.pixels[region],
        working_space="linear_srgb",
    )
    delta = np.linalg.norm(first_lab - second_lab, axis=-1)
    median = float(np.median(delta))
    p95 = float(np.percentile(delta, 95))
    maximum = float(np.max(delta))
    reasons: list[str] = []
    if median > resolved.max_delta_e76_median:
        reasons.append("shared-colour-median-drift")
    if p95 > resolved.max_delta_e76_p95:
        reasons.append("shared-colour-p95-drift")
    if maximum > resolved.max_delta_e76:
        reasons.append("shared-colour-maximum-drift")
    return ContextInvarianceMetrics(
        passed=not reasons,
        reasons=tuple(reasons),
        delta_e76_median=median,
        delta_e76_p95=p95,
        delta_e76_maximum=maximum,
    )


def evaluate_recipe_batch_context_invariance(
    recipes: Iterable[ReferenceLookRecipe],
    *,
    policy: ContextInvariancePolicy | None = None,
) -> ContextInvarianceBatchMetrics:
    """Evaluate context invariance across all independent references."""

    if isinstance(recipes, (str, bytes, ReferenceLookRecipe)):
        raise ReferenceMatchContractError(
            "recipes must be an iterable of ReferenceLookRecipe values"
        )
    try:
        recipe_items = tuple(recipes)
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "recipes must be an iterable of ReferenceLookRecipe values"
        ) from exc
    if not recipe_items or any(
        not isinstance(recipe, ReferenceLookRecipe) for recipe in recipe_items
    ):
        raise ReferenceMatchContractError(
            "recipes must contain at least one ReferenceLookRecipe"
        )
    samples = tuple(
        evaluate_recipe_context_invariance(recipe, policy=policy)
        for recipe in recipe_items
    )
    failed = sum(not sample.passed for sample in samples)
    return ContextInvarianceBatchMetrics(
        samples=samples,
        passed_recipe_count=len(samples) - failed,
        failed_recipe_count=failed,
        maximum_delta_e76_median=max(
            sample.delta_e76_median for sample in samples
        ),
        maximum_delta_e76_p95=max(
            sample.delta_e76_p95 for sample in samples
        ),
        maximum_delta_e76=max(
            sample.delta_e76_maximum for sample in samples
        ),
    )


__all__ = [
    "ContextInvarianceBatchMetrics",
    "ContextInvarianceMetrics",
    "ContextInvariancePolicy",
    "evaluate_context_invariance_outputs",
    "evaluate_recipe_batch_context_invariance",
    "evaluate_recipe_context_invariance",
    "make_context_invariance_probes",
    "validate_context_invariance_policy",
]
