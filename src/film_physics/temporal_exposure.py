"""Deterministic bounded exposure-domain trajectories for film video."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.film_physics.structure_compiler import counter_normal_region


class TemporalExposureDomainError(ValueError):
    """Raised when a temporal exposure request violates the typed contract."""


@dataclass(frozen=True)
class TemporalExposureProfile:
    theta: float
    stationary_sigma_stops: float
    maximum_absolute_stops: float
    seed: int

    def __post_init__(self) -> None:
        values = (
            self.theta,
            self.stationary_sigma_stops,
            self.maximum_absolute_stops,
        )
        if any(not math.isfinite(value) for value in values):
            raise TemporalExposureDomainError("exposure profile must be finite")
        if not 0.0 < self.theta <= 1.0:
            raise TemporalExposureDomainError("mean reversion must be in (0, 1]")
        if self.stationary_sigma_stops < 0.0:
            raise TemporalExposureDomainError("stationary scale must be nonnegative")
        if self.maximum_absolute_stops <= 0.0:
            raise TemporalExposureDomainError("exposure bound must be positive")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise TemporalExposureDomainError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True)
class TemporalExposureState:
    frame_index: int
    offset_stops: float


@dataclass(frozen=True)
class TemporalExposureSegment:
    frame_indices: np.ndarray
    offset_stops: np.ndarray
    multiplier: np.ndarray
    clipped: np.ndarray


def initial_temporal_exposure_state() -> TemporalExposureState:
    return TemporalExposureState(frame_index=0, offset_stops=0.0)


def _validate_state(
    profile: TemporalExposureProfile,
    state: TemporalExposureState,
    total_frame_count: int,
) -> None:
    if not isinstance(total_frame_count, int) or total_frame_count < 1:
        raise TemporalExposureDomainError("total frame count must be positive")
    if (
        not isinstance(state.frame_index, int)
        or not 0 <= state.frame_index < total_frame_count
    ):
        raise TemporalExposureDomainError("state frame index is outside the sequence")
    if not math.isfinite(state.offset_stops):
        raise TemporalExposureDomainError("state offset must be finite")
    if abs(state.offset_stops) > profile.maximum_absolute_stops:
        raise TemporalExposureDomainError("state offset is outside the profile bound")


def temporal_exposure_innovations(
    profile: TemporalExposureProfile,
    *,
    total_frame_count: int,
    start_transition: int,
    transition_count: int,
) -> np.ndarray:
    """Return counter-addressed innovations for absolute transition indices."""
    if not isinstance(start_transition, int) or not isinstance(transition_count, int):
        raise TemporalExposureDomainError("transition coordinates must be integers")
    if start_transition < 0 or transition_count < 0:
        raise TemporalExposureDomainError("transition coordinates must be nonnegative")
    if total_frame_count < 1:
        raise TemporalExposureDomainError("total frame count must be positive")
    total_transitions = total_frame_count - 1
    if start_transition + transition_count > total_transitions:
        raise TemporalExposureDomainError("requested transitions exceed the sequence")
    if transition_count == 0:
        return np.empty(0, dtype=np.float64)
    return counter_normal_region(
        (total_transitions, 1),
        origin_yx=(start_transition, 0),
        shape=(transition_count, 1),
        seed=profile.seed,
    )[:, 0]


def advance_temporal_exposure(
    profile: TemporalExposureProfile,
    *,
    total_frame_count: int,
    state: TemporalExposureState,
    transition_count: int,
) -> tuple[TemporalExposureSegment, TemporalExposureState]:
    """Advance a mean-reverting log2 exposure state with exact handoff."""
    _validate_state(profile, state, total_frame_count)
    innovations = temporal_exposure_innovations(
        profile,
        total_frame_count=total_frame_count,
        start_transition=state.frame_index,
        transition_count=transition_count,
    )
    if state.frame_index + transition_count >= total_frame_count:
        raise TemporalExposureDomainError("advanced state would exceed the sequence")
    if transition_count == 0:
        empty = np.empty(0, dtype=np.float64)
        return (
            TemporalExposureSegment(
                frame_indices=np.empty(0, dtype=np.int64),
                offset_stops=empty,
                multiplier=empty.copy(),
                clipped=np.empty(0, dtype=np.bool_),
            ),
            state,
        )
    offsets = np.empty(transition_count, dtype=np.float64)
    clipped = np.zeros(transition_count, dtype=np.bool_)
    value = state.offset_stops
    innovation_scale = profile.stationary_sigma_stops * math.sqrt(
        2.0 * profile.theta
    )
    for index, innovation in enumerate(innovations):
        proposed = value - profile.theta * value + innovation_scale * innovation
        bounded = float(
            np.clip(
                proposed,
                -profile.maximum_absolute_stops,
                profile.maximum_absolute_stops,
            )
        )
        clipped[index] = bounded != proposed
        offsets[index] = bounded
        value = bounded
    frames = np.arange(
        state.frame_index + 1,
        state.frame_index + transition_count + 1,
        dtype=np.int64,
    )
    segment = TemporalExposureSegment(
        frame_indices=frames,
        offset_stops=offsets,
        multiplier=np.exp2(offsets),
        clipped=clipped,
    )
    return segment, TemporalExposureState(int(frames[-1]), value)


def generate_temporal_exposure(
    profile: TemporalExposureProfile, frame_count: int
) -> TemporalExposureSegment:
    """Generate frame-zero identity followed by a complete exposure trajectory."""
    if not isinstance(frame_count, int) or frame_count < 1:
        raise TemporalExposureDomainError("frame count must be positive")
    tail, _ = advance_temporal_exposure(
        profile,
        total_frame_count=frame_count,
        state=initial_temporal_exposure_state(),
        transition_count=frame_count - 1,
    )
    return TemporalExposureSegment(
        frame_indices=np.concatenate((np.array([0], dtype=np.int64), tail.frame_indices)),
        offset_stops=np.concatenate((np.array([0.0]), tail.offset_stops)),
        multiplier=np.concatenate((np.array([1.0]), tail.multiplier)),
        clipped=np.concatenate((np.array([False]), tail.clipped)),
    )


def apply_temporal_exposure(
    layer_exposure: np.ndarray, offset_stops: float
) -> np.ndarray:
    """Apply one finite exposure offset before development or density response."""
    exposure = np.asarray(layer_exposure)
    if exposure.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TemporalExposureDomainError("layer exposure must use float32 or float64")
    if not np.all(np.isfinite(exposure)) or np.any(exposure < 0.0):
        raise TemporalExposureDomainError("layer exposure must be finite and nonnegative")
    if not math.isfinite(offset_stops):
        raise TemporalExposureDomainError("exposure offset must be finite")
    if offset_stops == 0.0:
        return exposure.copy()
    multiplier = math.exp2(offset_stops)
    with np.errstate(over="ignore", invalid="ignore"):
        result = exposure * multiplier
    if not np.all(np.isfinite(result)):
        raise TemporalExposureDomainError("adjusted layer exposure is not finite")
    return result
