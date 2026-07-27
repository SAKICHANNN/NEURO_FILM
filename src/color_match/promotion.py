"""Fail-closed photographic safety and promotion adjudication.

Automated probes may reject an algorithm or make it eligible for visual
review. They never establish photographic preference on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from src.color_engine import linear_rgb_to_lab
from src.preprocess import SourceProfile, WorkingImage

from .contracts import ReferenceLookRecipe, ReferenceMatchContractError
from .consistency import ContextInvarianceBatchMetrics
from .evaluation import KnownOperatorBatchMetrics
from .render import render_reference_look

_FLOAT32_GAMUT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PhotographicSafetyPolicy:
    """Conservative structural-colour limits for a deterministic probe."""

    max_neutral_chroma_p95: float = 18.0
    max_tone_reversal_fraction: float = 0.0
    max_tone_plateau_fraction: float = 0.20
    max_new_boundary_fraction: float = 0.05
    max_semantic_hue_rotation_p95_degrees: float = 75.0
    boundary_epsilon: float = 1.0 / 65535.0


@dataclass(frozen=True)
class PhotographicSafetyMetrics:
    """Measured structural colour behaviour on the frozen probe."""

    passed: bool
    reasons: tuple[str, ...]
    neutral_chroma_p95: float
    tone_reversal_fraction: float
    tone_largest_reversal_delta_l: float
    tone_plateau_fraction: float
    new_boundary_fraction: float
    semantic_hue_rotation_p95_degrees: float


@dataclass(frozen=True)
class PhotographicSafetyBatchMetrics:
    """Tail aggregation across independently fitted reference recipes."""

    samples: tuple[PhotographicSafetyMetrics, ...]
    passed_recipe_count: int
    failed_recipe_count: int
    maximum_neutral_chroma_p95: float
    maximum_tone_reversal_fraction: float
    largest_tone_reversal_delta_l: float
    maximum_tone_plateau_fraction: float
    maximum_new_boundary_fraction: float
    maximum_semantic_hue_rotation_p95_degrees: float

    @property
    def passed(self) -> bool:
        return bool(self.samples) and all(
            sample.passed for sample in self.samples
        )


@dataclass(frozen=True)
class BlindAestheticReview:
    """Independent visual evidence required after automated gates."""

    reviewed_sample_count: int
    preferred_sample_count: int
    severe_artifact_count: int
    blinded: bool

    @property
    def preference_rate(self) -> float:
        if self.reviewed_sample_count <= 0:
            return 0.0
        return self.preferred_sample_count / self.reviewed_sample_count


@dataclass(frozen=True)
class PromotionPolicy:
    """Frozen minimum evidence for replacing the product baseline."""

    minimum_known_operator_samples: int = 12
    minimum_improvement_rate: float = 0.75
    minimum_median_improvement_fraction: float = 0.10
    minimum_worst_improvement_fraction: float = -0.10
    maximum_new_boundary_fraction: float = 0.05
    minimum_blind_review_samples: int = 12
    minimum_blind_preference_rate: float = 0.50


@dataclass(frozen=True)
class PromotionDecision:
    """Final machine-readable decision with no metric-to-aesthetic shortcut."""

    status: Literal["rejected", "eligible-for-visual-review", "promoted"]
    reasons: tuple[str, ...]


def _validate_fraction(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ReferenceMatchContractError(
            f"{name} must be finite and within [0, 1]"
        )
    return float(value)


def validate_photographic_safety_policy(
    policy: PhotographicSafetyPolicy,
) -> None:
    if not isinstance(policy, PhotographicSafetyPolicy):
        raise ReferenceMatchContractError(
            "photographic safety policy must be PhotographicSafetyPolicy"
        )
    for name in (
        "max_tone_reversal_fraction",
        "max_tone_plateau_fraction",
        "max_new_boundary_fraction",
    ):
        _validate_fraction(name, getattr(policy, name))
    for name in (
        "max_neutral_chroma_p95",
        "max_semantic_hue_rotation_p95_degrees",
    ):
        value = getattr(policy, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ReferenceMatchContractError(
                f"{name} must be finite and non-negative"
            )
    epsilon = policy.boundary_epsilon
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not np.isfinite(float(epsilon))
        or not 0.0 <= float(epsilon) < 0.5
    ):
        raise ReferenceMatchContractError(
            "boundary_epsilon must be finite and within [0, 0.5)"
        )


def validate_promotion_policy(policy: PromotionPolicy) -> None:
    if not isinstance(policy, PromotionPolicy):
        raise ReferenceMatchContractError(
            "promotion policy must be PromotionPolicy"
        )
    for name in (
        "minimum_known_operator_samples",
        "minimum_blind_review_samples",
    ):
        value = getattr(policy, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ReferenceMatchContractError(
                f"{name} must be a positive integer"
            )
    for name in (
        "minimum_improvement_rate",
        "maximum_new_boundary_fraction",
        "minimum_blind_preference_rate",
    ):
        _validate_fraction(name, getattr(policy, name))
    for name in (
        "minimum_median_improvement_fraction",
        "minimum_worst_improvement_fraction",
    ):
        value = getattr(policy, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
        ):
            raise ReferenceMatchContractError(f"{name} must be finite")


def make_photographic_probe() -> WorkingImage:
    """Return the frozen v1 neutral/tone/semantic-colour probe."""

    width = 257
    ramp = np.linspace(0.01, 0.99, width, dtype=np.float32)
    pixels = np.empty((8, width, 3), dtype=np.float32)
    pixels[:4] = ramp[None, :, None]

    shade = np.linspace(0.45, 1.15, width, dtype=np.float32)[:, None]
    semantic_bases = np.asarray(
        [
            [0.58, 0.28, 0.17],
            [0.16, 0.38, 0.72],
            [0.16, 0.42, 0.13],
        ],
        dtype=np.float32,
    )
    pixels[4:7] = np.clip(
        semantic_bases[:, None, :] * shade[None, :, :],
        0.01,
        0.98,
    )
    highlights = np.linspace(0.72, 0.99, width, dtype=np.float32)
    pixels[7] = np.stack(
        [highlights, highlights * 0.995, highlights * 0.99],
        axis=-1,
    )
    return WorkingImage(
        pixels=pixels,
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile(
            "synthetic_reference_match_probe_v1",
            "deterministic structural-colour probe",
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("synthetic/reference_match_probe_v1.exr"),
    )


def _validate_probe_pair(source: WorkingImage, candidate: WorkingImage) -> None:
    if not isinstance(source, WorkingImage) or not isinstance(
        candidate, WorkingImage
    ):
        raise ReferenceMatchContractError(
            "photographic probe inputs must be WorkingImage values"
        )
    if source.pixels.shape != (8, 257, 3):
        raise ReferenceMatchContractError(
            "photographic source probe must have frozen shape (8, 257, 3)"
        )
    if candidate.pixels.shape != source.pixels.shape:
        raise ReferenceMatchContractError(
            "photographic candidate probe shape must match source"
        )
    for image in (source, candidate):
        if (
            image.working_space != "linear_srgb"
            or image.transfer_state != "display_linear"
        ):
            raise ReferenceMatchContractError(
                "photographic probes must be display-linear linear_srgb"
            )
        if (
            not np.isfinite(image.pixels).all()
            or float(np.min(image.pixels)) < -_FLOAT32_GAMUT_TOLERANCE
            or float(np.max(image.pixels))
            > 1.0 + _FLOAT32_GAMUT_TOLERANCE
        ):
            raise ReferenceMatchContractError(
                "photographic probes must be finite and within float32 gamut "
                "tolerance"
            )


def _hue_rotation_degrees(
    source_lab: np.ndarray,
    candidate_lab: np.ndarray,
) -> np.ndarray:
    source_hue = np.arctan2(source_lab[..., 2], source_lab[..., 1])
    candidate_hue = np.arctan2(
        candidate_lab[..., 2],
        candidate_lab[..., 1],
    )
    delta = np.angle(np.exp(1j * (candidate_hue - source_hue)))
    return np.abs(np.degrees(delta))


def evaluate_photographic_probe(
    source: WorkingImage,
    candidate: WorkingImage,
    *,
    policy: PhotographicSafetyPolicy | None = None,
) -> PhotographicSafetyMetrics:
    """Measure severe structural-colour failures on the frozen v1 probe."""

    resolved = policy or PhotographicSafetyPolicy()
    validate_photographic_safety_policy(resolved)
    _validate_probe_pair(source, candidate)
    source_lab = linear_rgb_to_lab(source.pixels, working_space="linear_srgb")
    candidate_lab = linear_rgb_to_lab(
        candidate.pixels,
        working_space="linear_srgb",
    )

    neutral = candidate_lab[:4]
    neutral_chroma = np.hypot(neutral[..., 1], neutral[..., 2])
    neutral_chroma_p95 = float(np.percentile(neutral_chroma, 95))

    candidate_l = np.mean(candidate_lab[:4, :, 0], axis=0)
    tone_delta = np.diff(candidate_l)
    reversal = tone_delta < -0.1
    plateau = np.abs(tone_delta) <= 1e-4
    reversal_fraction = float(np.mean(reversal, dtype=np.float64))
    largest_reversal = float(min(0.0, np.min(tone_delta)))
    plateau_fraction = float(np.mean(plateau, dtype=np.float64))

    epsilon = float(resolved.boundary_epsilon)
    source_boundary = np.any(
        (source.pixels <= epsilon) | (source.pixels >= 1.0 - epsilon),
        axis=-1,
    )
    candidate_boundary = np.any(
        (candidate.pixels <= epsilon)
        | (candidate.pixels >= 1.0 - epsilon),
        axis=-1,
    )
    new_boundary_fraction = float(
        np.mean(candidate_boundary & ~source_boundary, dtype=np.float64)
    )

    semantic_hue = _hue_rotation_degrees(
        source_lab[4:7],
        candidate_lab[4:7],
    )
    semantic_hue_p95 = float(np.percentile(semantic_hue, 95))

    reasons: list[str] = []
    if neutral_chroma_p95 > resolved.max_neutral_chroma_p95:
        reasons.append("neutral-axis-chroma")
    if reversal_fraction > resolved.max_tone_reversal_fraction:
        reasons.append("tone-reversal")
    if plateau_fraction > resolved.max_tone_plateau_fraction:
        reasons.append("tone-plateau")
    if new_boundary_fraction > resolved.max_new_boundary_fraction:
        reasons.append("new-boundary-fraction")
    if (
        semantic_hue_p95
        > resolved.max_semantic_hue_rotation_p95_degrees
    ):
        reasons.append("semantic-hue-rotation")
    return PhotographicSafetyMetrics(
        passed=not reasons,
        reasons=tuple(reasons),
        neutral_chroma_p95=neutral_chroma_p95,
        tone_reversal_fraction=reversal_fraction,
        tone_largest_reversal_delta_l=largest_reversal,
        tone_plateau_fraction=plateau_fraction,
        new_boundary_fraction=new_boundary_fraction,
        semantic_hue_rotation_p95_degrees=semantic_hue_p95,
    )


def evaluate_recipe_photographic_safety(
    recipe: ReferenceLookRecipe,
    *,
    policy: PhotographicSafetyPolicy | None = None,
) -> PhotographicSafetyMetrics:
    """Render the frozen probe through one recipe and evaluate its output."""

    probe = make_photographic_probe()
    candidate = render_reference_look(recipe, probe).image
    return evaluate_photographic_probe(probe, candidate, policy=policy)


def evaluate_recipe_batch_photographic_safety(
    recipes: Iterable[ReferenceLookRecipe],
    *,
    policy: PhotographicSafetyPolicy | None = None,
) -> PhotographicSafetyBatchMetrics:
    """Evaluate all fitted reference recipes and retain their worst tail."""

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
        evaluate_recipe_photographic_safety(recipe, policy=policy)
        for recipe in recipe_items
    )
    failed = sum(not sample.passed for sample in samples)
    return PhotographicSafetyBatchMetrics(
        samples=samples,
        passed_recipe_count=len(samples) - failed,
        failed_recipe_count=failed,
        maximum_neutral_chroma_p95=max(
            sample.neutral_chroma_p95 for sample in samples
        ),
        maximum_tone_reversal_fraction=max(
            sample.tone_reversal_fraction for sample in samples
        ),
        largest_tone_reversal_delta_l=min(
            sample.tone_largest_reversal_delta_l for sample in samples
        ),
        maximum_tone_plateau_fraction=max(
            sample.tone_plateau_fraction for sample in samples
        ),
        maximum_new_boundary_fraction=max(
            sample.new_boundary_fraction for sample in samples
        ),
        maximum_semantic_hue_rotation_p95_degrees=max(
            sample.semantic_hue_rotation_p95_degrees for sample in samples
        ),
    )


def _validate_safety_metrics(metrics: PhotographicSafetyMetrics) -> None:
    if metrics.passed != (not metrics.reasons):
        raise ReferenceMatchContractError(
            "photographic safety passed state must agree with reasons"
        )
    for name in (
        "neutral_chroma_p95",
        "tone_reversal_fraction",
        "tone_largest_reversal_delta_l",
        "tone_plateau_fraction",
        "new_boundary_fraction",
        "semantic_hue_rotation_p95_degrees",
    ):
        if not np.isfinite(float(getattr(metrics, name))):
            raise ReferenceMatchContractError(
                f"photographic safety {name} must be finite"
            )


def adjudicate_promotion(
    known_operator: KnownOperatorBatchMetrics,
    photographic_safety: PhotographicSafetyBatchMetrics,
    context_invariance: ContextInvarianceBatchMetrics,
    *,
    visual_review: BlindAestheticReview | None = None,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Reject, send to visual review, or promote with independent evidence."""

    resolved = policy or PromotionPolicy()
    validate_promotion_policy(resolved)
    if not isinstance(known_operator, KnownOperatorBatchMetrics):
        raise ReferenceMatchContractError(
            "known_operator must be KnownOperatorBatchMetrics"
        )
    if not isinstance(
        photographic_safety,
        PhotographicSafetyBatchMetrics,
    ):
        raise ReferenceMatchContractError(
            "photographic_safety must be PhotographicSafetyBatchMetrics"
        )
    if not photographic_safety.samples:
        raise ReferenceMatchContractError(
            "photographic safety batch must not be empty"
        )
    for sample in photographic_safety.samples:
        _validate_safety_metrics(sample)
    failed_safety_count = sum(
        not sample.passed for sample in photographic_safety.samples
    )
    if (
        photographic_safety.failed_recipe_count != failed_safety_count
        or photographic_safety.passed_recipe_count
        != len(photographic_safety.samples) - failed_safety_count
    ):
        raise ReferenceMatchContractError(
            "photographic safety batch counts must agree with samples"
        )
    if not isinstance(context_invariance, ContextInvarianceBatchMetrics):
        raise ReferenceMatchContractError(
            "context_invariance must be ContextInvarianceBatchMetrics"
        )
    if not context_invariance.samples:
        raise ReferenceMatchContractError(
            "context invariance batch must not be empty"
        )
    failed_context_count = 0
    for sample in context_invariance.samples:
        if sample.passed != (not sample.reasons):
            raise ReferenceMatchContractError(
                "context invariance passed state must agree with reasons"
            )
        if not np.isfinite(
            np.asarray(
                [
                    sample.delta_e76_median,
                    sample.delta_e76_p95,
                    sample.delta_e76_maximum,
                ],
                dtype=np.float64,
            )
        ).all():
            raise ReferenceMatchContractError(
                "context invariance metrics must be finite"
            )
        failed_context_count += int(not sample.passed)
    if (
        context_invariance.failed_recipe_count != failed_context_count
        or context_invariance.passed_recipe_count
        != len(context_invariance.samples) - failed_context_count
    ):
        raise ReferenceMatchContractError(
            "context invariance batch counts must agree with samples"
        )

    reasons: list[str] = []
    sample_count = len(known_operator.samples)
    if sample_count < resolved.minimum_known_operator_samples:
        reasons.append("insufficient-known-operator-samples")
    improvements = np.asarray(
        [row.median_improvement_fraction for row in known_operator.samples],
        dtype=np.float64,
    )
    boundaries = np.asarray(
        [row.candidate_new_boundary_fraction for row in known_operator.samples],
        dtype=np.float64,
    )
    if (
        not np.isfinite(improvements).all()
        or not np.isfinite(boundaries).all()
        or np.any((boundaries < 0.0) | (boundaries > 1.0))
    ):
        raise ReferenceMatchContractError(
            "known-operator promotion metrics must be finite and valid"
        )
    improvement_rate = (
        float(np.mean(improvements > 0.0, dtype=np.float64))
        if sample_count
        else 0.0
    )
    median_improvement = (
        float(np.median(improvements)) if sample_count else float("-inf")
    )
    worst_improvement = (
        float(np.min(improvements)) if sample_count else float("-inf")
    )
    maximum_boundary = (
        float(np.max(boundaries)) if sample_count else float("inf")
    )
    if improvement_rate < resolved.minimum_improvement_rate:
        reasons.append("known-operator-improvement-rate")
    if (
        median_improvement
        < resolved.minimum_median_improvement_fraction
    ):
        reasons.append("known-operator-median")
    if (
        worst_improvement
        < resolved.minimum_worst_improvement_fraction
    ):
        reasons.append("known-operator-tail")
    if (
        maximum_boundary
        > resolved.maximum_new_boundary_fraction
    ):
        reasons.append("known-operator-new-boundary")
    if not photographic_safety.passed:
        unique_safety_reasons = sorted(
            {
                reason
                for sample in photographic_safety.samples
                for reason in sample.reasons
            }
        )
        reasons.extend(
            f"photographic-safety:{reason}"
            for reason in unique_safety_reasons
        )
    if not context_invariance.passed:
        unique_context_reasons = sorted(
            {
                reason
                for sample in context_invariance.samples
                for reason in sample.reasons
            }
        )
        reasons.extend(
            f"context-invariance:{reason}"
            for reason in unique_context_reasons
        )
    if reasons:
        return PromotionDecision("rejected", tuple(reasons))

    if visual_review is None:
        return PromotionDecision(
            "eligible-for-visual-review",
            ("blind-aesthetic-review-required",),
        )
    if not isinstance(visual_review, BlindAestheticReview):
        raise ReferenceMatchContractError(
            "visual_review must be BlindAestheticReview or None"
        )
    review_reasons: list[str] = []
    if not visual_review.blinded:
        review_reasons.append("visual-review-not-blinded")
    if (
        visual_review.reviewed_sample_count
        < resolved.minimum_blind_review_samples
    ):
        review_reasons.append("insufficient-visual-review-samples")
    if (
        visual_review.reviewed_sample_count < 0
        or visual_review.preferred_sample_count < 0
        or visual_review.preferred_sample_count
        > visual_review.reviewed_sample_count
        or visual_review.severe_artifact_count < 0
    ):
        review_reasons.append("invalid-visual-review-counts")
    if visual_review.severe_artifact_count != 0:
        review_reasons.append("visual-severe-artifact-veto")
    if (
        visual_review.preference_rate
        <= resolved.minimum_blind_preference_rate
    ):
        review_reasons.append("visual-preference-rate")
    if review_reasons:
        return PromotionDecision("rejected", tuple(review_reasons))
    return PromotionDecision("promoted", ())


__all__ = [
    "BlindAestheticReview",
    "PhotographicSafetyBatchMetrics",
    "PhotographicSafetyMetrics",
    "PhotographicSafetyPolicy",
    "PromotionDecision",
    "PromotionPolicy",
    "adjudicate_promotion",
    "evaluate_photographic_probe",
    "evaluate_recipe_batch_photographic_safety",
    "evaluate_recipe_photographic_safety",
    "make_photographic_probe",
    "validate_photographic_safety_policy",
    "validate_promotion_policy",
]
