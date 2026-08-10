"""Deterministic bounded gate-weave trajectories for temporal film simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.film_physics.structure_compiler import counter_normal_region


class GateWeaveDomainError(ValueError):
    """Raised when a gate-weave profile or trajectory request is invalid."""


@dataclass(frozen=True)
class GateWeaveProfile:
    theta_x: float
    sigma_x_pixels: float
    theta_y_initial: float
    theta_y_average: float
    kappa: float
    theta_y_sigma: float
    theta_y_minimum: float
    theta_y_maximum: float
    sigma_y_pixels: float
    padding_x_pixels: int
    padding_y_pixels: int
    seed: int

    def __post_init__(self) -> None:
        values = (
            self.theta_x,
            self.sigma_x_pixels,
            self.theta_y_initial,
            self.theta_y_average,
            self.kappa,
            self.theta_y_sigma,
            self.theta_y_minimum,
            self.theta_y_maximum,
            self.sigma_y_pixels,
        )
        if any(not math.isfinite(value) for value in values):
            raise GateWeaveDomainError("gate-weave parameters must be finite")
        if not 0.0 < self.theta_x <= 1.0:
            raise GateWeaveDomainError("horizontal mean reversion must be in (0, 1]")
        if not 0.0 < self.kappa <= 1.0:
            raise GateWeaveDomainError("vertical-rate mean reversion must be in (0, 1]")
        if self.sigma_x_pixels < 0.0 or self.sigma_y_pixels < 0.0:
            raise GateWeaveDomainError("coordinate scales must be nonnegative")
        if self.theta_y_sigma < 0.0:
            raise GateWeaveDomainError("vertical-rate scale must be nonnegative")
        if not (
            0.0
            < self.theta_y_minimum
            <= self.theta_y_initial
            <= self.theta_y_maximum
            <= 1.0
        ):
            raise GateWeaveDomainError("initial vertical rate is outside its bounds")
        if not self.theta_y_minimum <= self.theta_y_average <= self.theta_y_maximum:
            raise GateWeaveDomainError("average vertical rate is outside its bounds")
        if (
            not isinstance(self.padding_x_pixels, int)
            or not isinstance(self.padding_y_pixels, int)
            or self.padding_x_pixels < 0
            or self.padding_y_pixels < 0
        ):
            raise GateWeaveDomainError("padding must be nonnegative integer pixels")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise GateWeaveDomainError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True)
class GateWeaveState:
    """State at one absolute frame index, suitable for exact handoff."""

    frame_index: int
    x_pixels: float
    y_pixels: float
    theta_y: float


@dataclass(frozen=True)
class GateWeaveSegment:
    """The states reached by a requested number of transitions."""

    frame_indices: np.ndarray
    x_pixels: np.ndarray
    y_pixels: np.ndarray
    theta_y: np.ndarray
    x_integer_pixels: np.ndarray
    y_integer_pixels: np.ndarray
    coordinate_clipped: np.ndarray


def initial_gate_weave_state(profile: GateWeaveProfile) -> GateWeaveState:
    return GateWeaveState(0, 0.0, 0.0, profile.theta_y_initial)


def _validate_state(
    profile: GateWeaveProfile, state: GateWeaveState, total_frame_count: int
) -> None:
    if (
        not isinstance(state.frame_index, int)
        or not 0 <= state.frame_index < total_frame_count
    ):
        raise GateWeaveDomainError("state frame index is outside the sequence")
    values = (state.x_pixels, state.y_pixels, state.theta_y)
    if any(not math.isfinite(value) for value in values):
        raise GateWeaveDomainError("state must be finite")
    if abs(state.x_pixels) > profile.padding_x_pixels:
        raise GateWeaveDomainError("horizontal state is outside padding")
    if abs(state.y_pixels) > profile.padding_y_pixels:
        raise GateWeaveDomainError("vertical state is outside padding")
    if not profile.theta_y_minimum <= state.theta_y <= profile.theta_y_maximum:
        raise GateWeaveDomainError("vertical-rate state is outside bounds")


def gate_weave_innovations(
    profile: GateWeaveProfile,
    *,
    total_frame_count: int,
    start_transition: int,
    transition_count: int,
) -> np.ndarray:
    """Return coordinate-stable innovations for absolute transition indices."""
    if not isinstance(total_frame_count, int) or total_frame_count < 1:
        raise GateWeaveDomainError("total frame count must be positive")
    if not isinstance(start_transition, int) or not isinstance(transition_count, int):
        raise GateWeaveDomainError("transition coordinates must be integers")
    if transition_count < 0 or start_transition < 0:
        raise GateWeaveDomainError("transition coordinates must be nonnegative")
    total_transitions = total_frame_count - 1
    if start_transition + transition_count > total_transitions:
        raise GateWeaveDomainError("requested transitions exceed the sequence")
    if transition_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    with np.errstate(over="ignore"):
        return counter_normal_region(
            (total_transitions, 3),
            origin_yx=(start_transition, 0),
            shape=(transition_count, 3),
            seed=profile.seed,
        )


def advance_gate_weave(
    profile: GateWeaveProfile,
    *,
    total_frame_count: int,
    state: GateWeaveState,
    transition_count: int,
) -> tuple[GateWeaveSegment, GateWeaveState]:
    """Advance from ``state`` with exact absolute-frame counter semantics."""
    _validate_state(profile, state, total_frame_count)
    innovations = gate_weave_innovations(
        profile,
        total_frame_count=total_frame_count,
        start_transition=state.frame_index,
        transition_count=transition_count,
    )
    if state.frame_index + transition_count >= total_frame_count:
        raise GateWeaveDomainError("advanced state would exceed the sequence")
    if transition_count == 0:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int32)
        empty_bool = np.empty(0, dtype=np.bool_)
        return (
            GateWeaveSegment(
                frame_indices=np.empty(0, dtype=np.int64),
                x_pixels=empty_float,
                y_pixels=empty_float.copy(),
                theta_y=empty_float.copy(),
                x_integer_pixels=empty_int,
                y_integer_pixels=empty_int.copy(),
                coordinate_clipped=empty_bool,
            ),
            state,
        )
    x_values = np.empty(transition_count, dtype=np.float64)
    y_values = np.empty(transition_count, dtype=np.float64)
    theta_values = np.empty(transition_count, dtype=np.float64)
    clipped = np.zeros(transition_count, dtype=np.bool_)
    x = state.x_pixels
    y = state.y_pixels
    theta_y = state.theta_y
    for index, (epsilon_x, epsilon_theta, epsilon_y) in enumerate(innovations):
        x_next = (
            x
            - profile.theta_x * x
            + profile.sigma_x_pixels * math.sqrt(2.0 * profile.theta_x) * epsilon_x
        )
        y_next = (
            y
            - theta_y * y
            + profile.sigma_y_pixels * math.sqrt(2.0 * theta_y) * epsilon_y
        )
        theta_next = (
            theta_y
            + profile.kappa * (profile.theta_y_average - theta_y)
            + profile.theta_y_sigma * math.sqrt(2.0 * profile.kappa) * epsilon_theta
        )
        bounded_x = float(
            np.clip(x_next, -profile.padding_x_pixels, profile.padding_x_pixels)
        )
        bounded_y = float(
            np.clip(y_next, -profile.padding_y_pixels, profile.padding_y_pixels)
        )
        bounded_theta = float(
            np.clip(theta_next, profile.theta_y_minimum, profile.theta_y_maximum)
        )
        clipped[index] = bounded_x != x_next or bounded_y != y_next
        x, y, theta_y = bounded_x, bounded_y, bounded_theta
        x_values[index] = x
        y_values[index] = y
        theta_values[index] = theta_y
    frame_indices = np.arange(
        state.frame_index + 1,
        state.frame_index + transition_count + 1,
        dtype=np.int64,
    )
    segment = GateWeaveSegment(
        frame_indices=frame_indices,
        x_pixels=x_values,
        y_pixels=y_values,
        theta_y=theta_values,
        x_integer_pixels=np.rint(x_values).astype(np.int32),
        y_integer_pixels=np.rint(y_values).astype(np.int32),
        coordinate_clipped=clipped,
    )
    next_state = GateWeaveState(
        state.frame_index + transition_count,
        x,
        y,
        theta_y,
    )
    return segment, next_state


def generate_gate_weave(
    profile: GateWeaveProfile, frame_count: int
) -> GateWeaveSegment:
    """Generate frame-zero identity followed by the complete bounded trajectory."""
    if not isinstance(frame_count, int) or frame_count < 1:
        raise GateWeaveDomainError("frame count must be positive")
    initial = initial_gate_weave_state(profile)
    tail, _ = advance_gate_weave(
        profile,
        total_frame_count=frame_count,
        state=initial,
        transition_count=frame_count - 1,
    )
    return GateWeaveSegment(
        frame_indices=np.concatenate(
            (np.array([0], dtype=np.int64), tail.frame_indices)
        ),
        x_pixels=np.concatenate((np.array([0.0]), tail.x_pixels)),
        y_pixels=np.concatenate((np.array([0.0]), tail.y_pixels)),
        theta_y=np.concatenate((np.array([profile.theta_y_initial]), tail.theta_y)),
        x_integer_pixels=np.concatenate(
            (np.array([0], dtype=np.int32), tail.x_integer_pixels)
        ),
        y_integer_pixels=np.concatenate(
            (np.array([0], dtype=np.int32), tail.y_integer_pixels)
        ),
        coordinate_clipped=np.concatenate((np.array([False]), tail.coordinate_clipped)),
    )
