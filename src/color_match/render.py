"""Deterministic application of one reference-look recipe to one or N images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.color_engine import (
    apply_safe_lab_transform,
    compress_chroma_to_working_gamut,
    compress_source_to_working_gamut,
    in_working_gamut,
    lab_to_linear_rgb,
    linear_rgb_to_lab,
    safe_lab_context_from_lab,
)
from src.preprocess.types import WorkingImage

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .contracts import (
    SUPPORTED_WORKING_SPACES,
    ReferenceLookRecipe,
    ReferenceMatchContractError,
    validate_recipe,
)

_REFERENCE_RENDER_ROW_CHUNK = 128


def _working_image_batch(
    sources: Iterable[WorkingImage],
) -> tuple[WorkingImage, ...]:
    if isinstance(sources, (WorkingImage, np.ndarray, str, bytes)):
        raise ReferenceMatchContractError(
            "sources must be an iterable of WorkingImage"
        )
    batch: list[WorkingImage] = []
    try:
        for source in sources:
            if len(batch) >= MAX_REFERENCE_MATCH_BATCH_SOURCES:
                raise ReferenceMatchContractError(
                    "reference match supports at most "
                    f"{MAX_REFERENCE_MATCH_BATCH_SOURCES} sources"
                )
            batch.append(source)
    except ReferenceMatchContractError:
        raise
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "sources must be an iterable of WorkingImage"
        ) from exc
    if not batch:
        raise ReferenceMatchContractError("sources batch must not be empty")
    return tuple(batch)


@dataclass(frozen=True)
class ReferenceMatchDiagnostics:
    """Small deterministic evidence record for one rendered source."""

    recipe_id: str
    claim_ceiling: str
    source_index: int
    source_working_space: str
    source_shape: tuple[int, int, int]
    gamut_adjusted_fraction: float
    output_min: float
    output_max: float


@dataclass(frozen=True)
class ReferenceMatchResult:
    """Rendered WorkingImage plus the recipe and safety diagnostics used."""

    image: WorkingImage
    diagnostics: ReferenceMatchDiagnostics


def _validate_source(source: WorkingImage) -> None:
    if not isinstance(source, WorkingImage):
        raise ReferenceMatchContractError("source must be WorkingImage")
    if source.transfer_state != "display_linear":
        raise ReferenceMatchContractError(
            "reference matching currently requires display-linear SDR sources"
        )
    if source.working_space not in SUPPORTED_WORKING_SPACES:
        raise ReferenceMatchContractError("source working space is unsupported")
    source_lab = linear_rgb_to_lab(source.pixels, working_space=source.working_space)
    if not in_working_gamut(
        source_lab,
        working_space=source.working_space,
        tolerance=2e-6,
    ).all():
        raise ReferenceMatchContractError(
            "source pixels are outside the declared working gamut"
        )


def _styled_lab(
    recipe: ReferenceLookRecipe,
    source_lab: np.ndarray,
) -> np.ndarray:
    policy = recipe.policy
    context = safe_lab_context_from_lab(source_lab)
    return apply_safe_lab_transform(
        source_lab,
        source_context=context,
        destination_mean=np.asarray(recipe.destination_lab_mean, dtype=np.float32),
        destination_std=np.asarray(recipe.destination_lab_std, dtype=np.float32),
        style="reference_look",
        strength=policy.strength,
        luma_strength=policy.luma_strength,
        tone_rolloff=policy.tone_rolloff,
        shadow_floor_l=policy.shadow_floor_l,
        highlight_ceiling_l=policy.highlight_ceiling_l,
        preserve_luma_detail_strength=policy.preserve_luma_detail_strength,
        chroma_curve_strength=policy.chroma_curve_strength,
        neutral_protect=policy.neutral_protect,
        skin_protect=policy.skin_protect,
        max_chroma_gain=policy.max_chroma_gain,
        max_chroma_boost=policy.max_chroma_boost,
        max_chroma_absolute=policy.max_chroma_absolute,
    )


def _gamut_safe_lab(
    recipe: ReferenceLookRecipe,
    source_lab: np.ndarray,
    styled_lab: np.ndarray,
    *,
    working_space: str,
) -> np.ndarray:
    output = np.empty_like(styled_lab, dtype=np.float32)
    for y0 in range(0, styled_lab.shape[0], _REFERENCE_RENDER_ROW_CHUNK):
        y1 = min(y0 + _REFERENCE_RENDER_ROW_CHUNK, styled_lab.shape[0])
        rows = slice(y0, y1)
        if recipe.policy.gamut_mode == "source":
            output[rows] = compress_source_to_working_gamut(
                source_lab[rows],
                styled_lab[rows],
                working_space=working_space,
                iterations=recipe.policy.gamut_iterations,
            )
        else:
            output[rows] = compress_chroma_to_working_gamut(
                styled_lab[rows],
                working_space=working_space,
                iterations=recipe.policy.gamut_iterations,
            )
    return output


def _lab_to_linear_rgb_rows(
    lab: np.ndarray,
    *,
    working_space: str,
) -> np.ndarray:
    output = np.empty_like(lab, dtype=np.float32)
    for y0 in range(0, lab.shape[0], _REFERENCE_RENDER_ROW_CHUNK):
        y1 = min(y0 + _REFERENCE_RENDER_ROW_CHUNK, lab.shape[0])
        output[y0:y1] = lab_to_linear_rgb(
            lab[y0:y1],
            working_space=working_space,
        )
    return output


def _in_working_gamut_rows(
    lab: np.ndarray,
    *,
    working_space: str,
    tolerance: float,
) -> bool:
    for y0 in range(0, lab.shape[0], _REFERENCE_RENDER_ROW_CHUNK):
        y1 = min(y0 + _REFERENCE_RENDER_ROW_CHUNK, lab.shape[0])
        if not in_working_gamut(
            lab[y0:y1],
            working_space=working_space,
            tolerance=tolerance,
        ).all():
            return False
    return True


def render_reference_look(
    recipe: ReferenceLookRecipe,
    source: WorkingImage,
    *,
    source_index: int = 0,
) -> ReferenceMatchResult:
    """Apply a frozen reference recipe without mutating source or recipe."""

    validate_recipe(recipe)
    _validate_source(source)
    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
        raise ReferenceMatchContractError("source_index must be a non-negative integer")

    source_lab = linear_rgb_to_lab(source.pixels, working_space=source.working_space)
    styled_lab = _styled_lab(recipe, source_lab)
    output_lab = _gamut_safe_lab(
        recipe,
        source_lab,
        styled_lab,
        working_space=source.working_space,
    )
    output_pixels = _lab_to_linear_rgb_rows(
        output_lab,
        working_space=source.working_space,
    )
    if not np.isfinite(output_pixels).all():
        raise ReferenceMatchContractError("reference-look output is non-finite")
    if not _in_working_gamut_rows(
        output_lab,
        working_space=source.working_space,
        tolerance=2e-6,
    ):
        raise ReferenceMatchContractError(
            "reference-look output violates the declared working gamut"
        )

    adjusted = np.any(np.abs(output_lab - styled_lab) > 1e-6, axis=-1)
    output = WorkingImage(
        pixels=np.asarray(output_pixels, dtype=np.float32),
        working_space=source.working_space,
        transfer_state=source.transfer_state,
        source_transfer_state=source.source_transfer_state,
        source_profile=source.source_profile,
        hdr_metadata=dict(source.hdr_metadata),
        orientation_applied=source.orientation_applied,
        alpha_policy=source.alpha_policy,
        bit_depth_in=source.bit_depth_in,
        source_path=source.source_path,
        warnings=list(source.warnings),
    )
    diagnostics = ReferenceMatchDiagnostics(
        recipe_id=recipe.recipe_id,
        claim_ceiling=recipe.claim_ceiling,
        source_index=source_index,
        source_working_space=source.working_space,
        source_shape=tuple(int(value) for value in source.pixels.shape),
        gamut_adjusted_fraction=float(np.mean(adjusted, dtype=np.float64)),
        output_min=float(np.min(output_pixels)),
        output_max=float(np.max(output_pixels)),
    )
    return ReferenceMatchResult(image=output, diagnostics=diagnostics)


def render_reference_batch(
    recipe: ReferenceLookRecipe,
    sources: Iterable[WorkingImage],
) -> tuple[ReferenceMatchResult, ...]:
    """Apply exactly one frozen recipe to a non-empty ordered source batch."""

    validate_recipe(recipe)
    batch = _working_image_batch(sources)
    return tuple(
        render_reference_look(recipe, source, source_index=index)
        for index, source in enumerate(batch)
    )


__all__ = [
    "ReferenceMatchDiagnostics",
    "ReferenceMatchResult",
    "render_reference_batch",
    "render_reference_look",
]
