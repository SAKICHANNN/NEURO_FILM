"""Deterministic tail-risk guard for reference-match product rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.preprocess import WorkingImage

from .contracts import (
    ReferenceLookRecipe,
    ReferenceMatchContractError,
)
from .render import (
    ReferenceMatchDiagnostics,
    _working_image_batch,
    render_reference_look,
)


REFERENCE_RENDER_GUARD_POLICY_ID = "reference-render-guard.v2"


@dataclass(frozen=True)
class ReferenceRenderGuardPolicy:
    """Frozen product thresholds; identity is the only safe fallback."""

    max_gamut_adjusted_fraction: float = 0.25
    max_new_boundary_fraction: float = 0.05
    boundary_epsilon: float = 1.0 / 65535.0
    allow_research_baseline: bool = False


@dataclass(frozen=True)
class ReferenceSafetyDecision:
    """Acceptance or identity-fallback evidence for one candidate."""

    policy_id: str
    accepted: bool
    action: str
    reasons: tuple[str, ...]
    candidate_gamut_adjusted_fraction: float
    candidate_new_boundary_fraction: float
    research_baseline_override: bool


@dataclass(frozen=True)
class GuardedReferenceMatchResult:
    """Delivered image plus candidate diagnostics and guard decision."""

    image: WorkingImage
    candidate_diagnostics: ReferenceMatchDiagnostics
    safety: ReferenceSafetyDecision


def validate_guard_policy(policy: ReferenceRenderGuardPolicy) -> None:
    if not isinstance(policy, ReferenceRenderGuardPolicy):
        raise ReferenceMatchContractError(
            "guard policy must be ReferenceRenderGuardPolicy"
        )
    for label, value in (
        (
            "max_gamut_adjusted_fraction",
            policy.max_gamut_adjusted_fraction,
        ),
        ("max_new_boundary_fraction", policy.max_new_boundary_fraction),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > 1.0
        ):
            raise ReferenceMatchContractError(
                f"guard policy {label} must be finite and within [0, 1]"
            )
    epsilon = policy.boundary_epsilon
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not np.isfinite(float(epsilon))
        or float(epsilon) < 0.0
        or float(epsilon) >= 0.5
    ):
        raise ReferenceMatchContractError(
            "guard policy boundary_epsilon must be finite and within [0, 0.5)"
        )
    if not isinstance(policy.allow_research_baseline, bool):
        raise ReferenceMatchContractError(
            "guard policy allow_research_baseline must be boolean"
        )


def _clone_source(source: WorkingImage) -> WorkingImage:
    return WorkingImage(
        pixels=np.array(source.pixels, dtype=np.float32, copy=True),
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


def _new_boundary_fraction(
    source: WorkingImage,
    candidate: WorkingImage,
    *,
    epsilon: float,
) -> float:
    source_boundary = np.any(
        (source.pixels <= epsilon) | (source.pixels >= 1.0 - epsilon),
        axis=-1,
    )
    candidate_boundary = np.any(
        (candidate.pixels <= epsilon) | (candidate.pixels >= 1.0 - epsilon),
        axis=-1,
    )
    return float(
        np.mean(candidate_boundary & ~source_boundary, dtype=np.float64)
    )


def render_reference_look_guarded(
    recipe: ReferenceLookRecipe,
    source: WorkingImage,
    *,
    source_index: int = 0,
    policy: ReferenceRenderGuardPolicy | None = None,
) -> GuardedReferenceMatchResult:
    """Render a candidate, then either deliver it or return source identity."""

    resolved = policy or ReferenceRenderGuardPolicy()
    validate_guard_policy(resolved)
    candidate = render_reference_look(
        recipe,
        source,
        source_index=source_index,
    )
    new_boundary = _new_boundary_fraction(
        source,
        candidate.image,
        epsilon=float(resolved.boundary_epsilon),
    )
    reasons: list[str] = []
    if not resolved.allow_research_baseline:
        reasons.append("algorithm-not-promoted")
    if (
        candidate.diagnostics.gamut_adjusted_fraction
        > resolved.max_gamut_adjusted_fraction
    ):
        reasons.append("gamut-adjusted-fraction")
    if new_boundary > resolved.max_new_boundary_fraction:
        reasons.append("new-boundary-fraction")
    accepted = not reasons
    decision = ReferenceSafetyDecision(
        policy_id=REFERENCE_RENDER_GUARD_POLICY_ID,
        accepted=accepted,
        action="applied" if accepted else "identity-fallback",
        reasons=tuple(reasons),
        candidate_gamut_adjusted_fraction=(
            candidate.diagnostics.gamut_adjusted_fraction
        ),
        candidate_new_boundary_fraction=new_boundary,
        research_baseline_override=resolved.allow_research_baseline,
    )
    diagnostics = candidate.diagnostics
    if accepted:
        return GuardedReferenceMatchResult(
            image=candidate.image,
            candidate_diagnostics=diagnostics,
            safety=decision,
        )
    # The rejected candidate pixels are not part of the identity fallback.
    # Release them before cloning the source so three full-resolution images
    # do not coexist during fallback construction.
    del candidate
    return GuardedReferenceMatchResult(
        image=_clone_source(source),
        candidate_diagnostics=diagnostics,
        safety=decision,
    )


def render_reference_batch_guarded(
    recipe: ReferenceLookRecipe,
    sources: Iterable[WorkingImage],
    *,
    policy: ReferenceRenderGuardPolicy | None = None,
) -> tuple[GuardedReferenceMatchResult, ...]:
    batch = _working_image_batch(sources)
    return tuple(
        render_reference_look_guarded(
            recipe,
            source,
            source_index=index,
            policy=policy,
        )
        for index, source in enumerate(batch)
    )


__all__ = [
    "REFERENCE_RENDER_GUARD_POLICY_ID",
    "GuardedReferenceMatchResult",
    "ReferenceRenderGuardPolicy",
    "ReferenceSafetyDecision",
    "render_reference_batch_guarded",
    "render_reference_look_guarded",
    "validate_guard_policy",
]
