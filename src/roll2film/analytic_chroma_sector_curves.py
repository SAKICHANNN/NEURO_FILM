"""Clean-room analytic colour-sector operators with bounded monotone curves."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import torch

from .cube_diffeomorphic_flow import _validate_rgb


ANALYTIC_CHROMA_SECTOR_CURVE_SCHEMA = (
    "roll2film.analytic_chroma_sector_curve.v1"
)
GLOBAL_BERNSTEIN_CURVE_SCHEMA = "roll2film.global_bernstein_curve.v1"

_LINEAR_SRGB_TO_XYZ_D65 = np.asarray(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)
_D65_WHITE = np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)
_LAB_EPSILON = 0.008856
_LAB_LINEAR_SCALE = 7.787
_LAB_OFFSET = 16.0 / 116.0
_SECTOR_CENTRES_DEGREES = (0.0, 72.0, 144.0, 216.0, 288.0)
_BERNSTEIN_DEGREE = 6
_CONTROL_COUNT = _BERNSTEIN_DEGREE + 1
_BINOMIAL = np.asarray(
    [comb(_BERNSTEIN_DEGREE, index) for index in range(_CONTROL_COUNT)],
    dtype=np.float64,
)


def _validate_controls(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    minimum_increment: float,
    label: str,
) -> np.ndarray:
    controls = np.asarray(values, dtype=np.float64)
    if (
        controls.shape != expected_shape
        or not np.all(np.isfinite(controls))
        or not np.all(controls[..., 0] == 0.0)
        or not np.all(controls[..., -1] == 1.0)
        or np.any(np.diff(controls, axis=-1) < minimum_increment - 1e-12)
    ):
        raise ValueError(
            f"{label} must have shape {expected_shape}, exact endpoints, "
            "and the required positive increments"
        )
    result = controls.copy()
    result.setflags(write=False)
    return result


def _lab_ab_numpy(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = rgb @ _LINEAR_SRGB_TO_XYZ_D65.T
    relative = xyz / _D65_WHITE
    transformed = np.where(
        relative > _LAB_EPSILON,
        np.cbrt(relative),
        _LAB_LINEAR_SCALE * relative + _LAB_OFFSET,
    )
    return (
        500.0 * (transformed[..., 0] - transformed[..., 1]),
        200.0 * (transformed[..., 1] - transformed[..., 2]),
    )


def _analytic_partition_numpy(
    rgb: np.ndarray,
    *,
    sector_centres_degrees: tuple[float, ...],
    direction_softening_lab: float,
    direction_concentration: float,
    neutral_chroma_half_activation: float,
) -> np.ndarray:
    red, green, blue = np.moveaxis(rgb, -1, 0)
    opponent_u = (2.0 * red - green - blue) / np.sqrt(6.0)
    opponent_v = (green - blue) / np.sqrt(2.0)
    chroma_squared = opponent_u**2 + opponent_v**2
    half = neutral_chroma_half_activation
    chromatic = chroma_squared / (chroma_squared + half * half)

    lab_a, lab_b = _lab_ab_numpy(rgb)
    centres = np.deg2rad(np.asarray(sector_centres_degrees, dtype=np.float64))
    denominator = np.sqrt(
        lab_a**2 + lab_b**2 + direction_softening_lab**2
    )
    scores = direction_concentration * (
        lab_a[..., None] * np.cos(centres)
        + lab_b[..., None] * np.sin(centres)
    ) / denominator[..., None]
    scores -= np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return np.concatenate(
        (
            (1.0 - chromatic)[..., None],
            chromatic[..., None] * probabilities,
        ),
        axis=-1,
    )


def _bernstein_basis_numpy(values: np.ndarray) -> np.ndarray:
    powers = np.arange(_CONTROL_COUNT, dtype=np.int64)
    return (
        _BINOMIAL
        * values[..., None] ** powers
        * (1.0 - values[..., None]) ** (_BERNSTEIN_DEGREE - powers)
    )


def _evaluate_controls_numpy(
    rgb: np.ndarray,
    controls: np.ndarray,
) -> np.ndarray:
    basis = _bernstein_basis_numpy(rgb)
    return np.sum(basis * controls, axis=-1)


@dataclass(frozen=True)
class AnalyticChromaSectorCurveOperator:
    """One achromatic and five analytic chromatic monotone-curve branches."""

    achromatic_control_values: np.ndarray
    chromatic_control_values: np.ndarray
    strength: float = 1.0
    sector_centres_degrees: tuple[float, ...] = _SECTOR_CENTRES_DEGREES
    direction_softening_lab: float = 4.0
    direction_concentration: float = 4.0
    neutral_chroma_half_activation: float = 0.08
    minimum_control_increment: float = 0.04
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        centres = tuple(float(value) for value in self.sector_centres_degrees)
        if (
            centres != _SECTOR_CENTRES_DEGREES
            or not np.isfinite(self.strength)
            or self.strength < 0.0
            or self.strength > 1.0
            or self.direction_softening_lab <= 0.0
            or self.direction_concentration <= 0.0
            or self.neutral_chroma_half_activation <= 0.0
            or self.minimum_control_increment <= 0.0
            or self.working_space != "linear_srgb_d65"
        ):
            raise ValueError("unsupported analytic chroma-sector contract")
        achromatic = _validate_controls(
            self.achromatic_control_values,
            expected_shape=(_CONTROL_COUNT,),
            minimum_increment=self.minimum_control_increment,
            label="achromatic_control_values",
        )
        chromatic = _validate_controls(
            self.chromatic_control_values,
            expected_shape=(len(centres), 3, _CONTROL_COUNT),
            minimum_increment=self.minimum_control_increment,
            label="chromatic_control_values",
        )
        object.__setattr__(self, "achromatic_control_values", achromatic)
        object.__setattr__(self, "chromatic_control_values", chromatic)
        object.__setattr__(self, "sector_centres_degrees", centres)

    @classmethod
    def identity(cls, *, strength: float = 1.0) -> "AnalyticChromaSectorCurveOperator":
        identity = np.linspace(0.0, 1.0, _CONTROL_COUNT, dtype=np.float64)
        return cls(
            achromatic_control_values=identity,
            chromatic_control_values=np.broadcast_to(
                identity, (len(_SECTOR_CENTRES_DEGREES), 3, _CONTROL_COUNT)
            ).copy(),
            strength=strength,
        )

    def partition(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        return _analytic_partition_numpy(
            values,
            sector_centres_degrees=self.sector_centres_degrees,
            direction_softening_lab=self.direction_softening_lab,
            direction_concentration=self.direction_concentration,
            neutral_chroma_half_activation=self.neutral_chroma_half_activation,
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        rows = values.reshape(-1, 3)
        weights = self.partition(rows)
        achromatic = _evaluate_controls_numpy(
            rows,
            np.broadcast_to(self.achromatic_control_values, rows.shape + (7,)),
        )
        branches = [achromatic]
        for index in range(len(self.sector_centres_degrees)):
            branches.append(
                _evaluate_controls_numpy(
                    rows,
                    np.broadcast_to(
                        self.chromatic_control_values[index],
                        rows.shape + (7,),
                    ),
                )
            )
        full = np.sum(
            weights[..., None] * np.stack(branches, axis=1),
            axis=1,
        )
        result = (1.0 - self.strength) * rows + self.strength * full
        # Every scalar curve fixes both endpoints analytically. Canonicalize
        # those exact faces after floating-point Bernstein accumulation.
        result = np.where(rows == 0.0, 0.0, result)
        result = np.where(rows == 1.0, 1.0, result)
        if (
            not np.all(np.isfinite(result))
            or np.any(result < -1e-12)
            or np.any(result > 1.0 + 1e-12)
        ):
            raise RuntimeError("analytic curve operator escaped the RGB cube")
        return result.reshape(shape)

    def inverse(
        self,
        rgb: np.ndarray,
        *,
        maximum_iterations: int = 30,
        convergence_tolerance: float = 1e-10,
    ) -> np.ndarray:
        target = _validate_rgb(rgb)
        estimate = target.copy()
        for _ in range(maximum_iterations):
            residual = self.apply(estimate) - target
            if float(np.max(np.abs(residual))) <= convergence_tolerance:
                return estimate
            estimate = np.clip(estimate - residual, 0.0, 1.0)
        if float(np.max(np.abs(self.apply(estimate) - target))) > convergence_tolerance:
            raise RuntimeError("analytic curve inverse did not converge")
        return estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYTIC_CHROMA_SECTOR_CURVE_SCHEMA,
            "working_space": self.working_space,
            "strength": self.strength,
            "sector_centres_degrees": list(self.sector_centres_degrees),
            "direction_softening_lab": self.direction_softening_lab,
            "direction_concentration": self.direction_concentration,
            "neutral_chroma_half_activation": self.neutral_chroma_half_activation,
            "minimum_control_increment": self.minimum_control_increment,
            "achromatic_control_values": self.achromatic_control_values.tolist(),
            "chromatic_control_values": self.chromatic_control_values.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnalyticChromaSectorCurveOperator":
        if payload.get("schema") != ANALYTIC_CHROMA_SECTOR_CURVE_SCHEMA:
            raise ValueError("unsupported analytic chroma-sector schema")
        return cls(
            achromatic_control_values=np.asarray(
                payload["achromatic_control_values"], dtype=np.float64
            ),
            chromatic_control_values=np.asarray(
                payload["chromatic_control_values"], dtype=np.float64
            ),
            strength=float(payload["strength"]),
            sector_centres_degrees=tuple(payload["sector_centres_degrees"]),
            direction_softening_lab=float(payload["direction_softening_lab"]),
            direction_concentration=float(payload["direction_concentration"]),
            neutral_chroma_half_activation=float(
                payload["neutral_chroma_half_activation"]
            ),
            minimum_control_increment=float(payload["minimum_control_increment"]),
            working_space=str(payload["working_space"]),
        )


@dataclass(frozen=True)
class GlobalBernsteinCurveOperator:
    """Three endpoint-fixed monotone Bernstein curves without colour routing."""

    control_values: np.ndarray
    minimum_control_increment: float = 0.04

    def __post_init__(self) -> None:
        controls = _validate_controls(
            self.control_values,
            expected_shape=(3, _CONTROL_COUNT),
            minimum_increment=self.minimum_control_increment,
            label="control_values",
        )
        object.__setattr__(self, "control_values", controls)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        result = _evaluate_controls_numpy(
            values,
            np.broadcast_to(self.control_values, values.shape + (7,)),
        )
        result = np.where(values == 0.0, 0.0, result)
        return np.where(values == 1.0, 1.0, result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_BERNSTEIN_CURVE_SCHEMA,
            "minimum_control_increment": self.minimum_control_increment,
            "control_values": self.control_values.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalBernsteinCurveOperator":
        if payload.get("schema") != GLOBAL_BERNSTEIN_CURVE_SCHEMA:
            raise ValueError("unsupported global Bernstein curve schema")
        return cls(
            control_values=np.asarray(payload["control_values"], dtype=np.float64),
            minimum_control_increment=float(payload["minimum_control_increment"]),
        )


def _lab_ab_torch(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.as_tensor(
        _LINEAR_SRGB_TO_XYZ_D65, dtype=rgb.dtype, device=rgb.device
    )
    white = torch.as_tensor(_D65_WHITE, dtype=rgb.dtype, device=rgb.device)
    relative = (rgb @ matrix.T) / white
    transformed = torch.where(
        relative > _LAB_EPSILON,
        torch.pow(relative, 1.0 / 3.0),
        _LAB_LINEAR_SCALE * relative + _LAB_OFFSET,
    )
    return (
        500.0 * (transformed[:, 0] - transformed[:, 1]),
        200.0 * (transformed[:, 1] - transformed[:, 2]),
    )


def _partition_torch(
    rgb: torch.Tensor,
    *,
    direction_softening_lab: float,
    direction_concentration: float,
    neutral_chroma_half_activation: float,
) -> torch.Tensor:
    red, green, blue = rgb.unbind(dim=1)
    opponent_u = (2.0 * red - green - blue) / np.sqrt(6.0)
    opponent_v = (green - blue) / np.sqrt(2.0)
    chroma_squared = opponent_u**2 + opponent_v**2
    half = neutral_chroma_half_activation
    chromatic = chroma_squared / (chroma_squared + half * half)
    lab_a, lab_b = _lab_ab_torch(rgb)
    centres = torch.deg2rad(
        torch.tensor(
            _SECTOR_CENTRES_DEGREES, dtype=rgb.dtype, device=rgb.device
        )
    )
    denominator = torch.sqrt(
        lab_a**2 + lab_b**2 + direction_softening_lab**2
    )
    scores = direction_concentration * (
        lab_a[:, None] * torch.cos(centres)
        + lab_b[:, None] * torch.sin(centres)
    ) / denominator[:, None]
    probabilities = torch.softmax(scores, dim=1)
    return torch.cat(
        ((1.0 - chromatic)[:, None], chromatic[:, None] * probabilities),
        dim=1,
    )


def _controls_from_logits_torch(
    logits: torch.Tensor,
    *,
    minimum_increment: float,
) -> torch.Tensor:
    interval_count = logits.shape[-1]
    remainder = 1.0 - interval_count * minimum_increment
    if remainder <= 0.0:
        raise ValueError("minimum increment is infeasible")
    increments = minimum_increment + remainder * torch.softmax(logits, dim=-1)
    zeros = torch.zeros(
        logits.shape[:-1] + (1,), dtype=logits.dtype, device=logits.device
    )
    return torch.cat((zeros, torch.cumsum(increments, dim=-1)), dim=-1)


def _bernstein_basis_torch(values: torch.Tensor) -> torch.Tensor:
    powers = torch.arange(
        _CONTROL_COUNT, dtype=values.dtype, device=values.device
    )
    binomial = torch.as_tensor(_BINOMIAL, dtype=values.dtype, device=values.device)
    return (
        binomial
        * values[..., None] ** powers
        * (1.0 - values[..., None]) ** (_BERNSTEIN_DEGREE - powers)
    )


def _apply_candidate_torch(
    rgb: torch.Tensor,
    achromatic_controls: torch.Tensor,
    chromatic_controls: torch.Tensor,
    *,
    direction_softening_lab: float,
    direction_concentration: float,
    neutral_chroma_half_activation: float,
) -> torch.Tensor:
    basis = _bernstein_basis_torch(rgb)
    achromatic = torch.sum(basis * achromatic_controls, dim=-1)
    chromatic = torch.sum(
        basis[:, None, :, :] * chromatic_controls[None, :, :, :],
        dim=-1,
    )
    branches = torch.cat((achromatic[:, None, :], chromatic), dim=1)
    weights = _partition_torch(
        rgb,
        direction_softening_lab=direction_softening_lab,
        direction_concentration=direction_concentration,
        neutral_chroma_half_activation=neutral_chroma_half_activation,
    )
    return torch.sum(weights[:, :, None] * branches, dim=1)


def _restart_logits(
    shape: tuple[int, ...],
    *,
    restart: int,
    seed: int,
    standard_deviation: float,
) -> torch.Tensor:
    if restart == 0:
        return torch.zeros(shape, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + restart)
    return torch.randn(shape, generator=generator, dtype=torch.float64) * standard_deviation


def fit_analytic_chroma_sector_curve_operator(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    minimum_control_increment: float,
    direction_softening_lab: float,
    direction_concentration: float,
    neutral_chroma_half_activation: float,
    seed: int,
    restarts: int,
    restart_standard_deviation: float,
    steps: int,
    learning_rate: float,
    identity_regularization: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> tuple[AnalyticChromaSectorCurveOperator, dict[str, Any]]:
    """Fit only the bounded curve parameters of the fixed analytic partition."""

    values = _validate_rgb(rgb).reshape(-1, 3)
    targets = _validate_rgb(target).reshape(-1, 3)
    if targets.shape != values.shape or restarts < 1 or steps < 1:
        raise ValueError("invalid analytic curve fit inputs")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(thread_count)
    source = torch.from_numpy(values.copy())
    target_tensor = torch.from_numpy(targets.copy())
    identity = torch.linspace(0.0, 1.0, _CONTROL_COUNT, dtype=torch.float64)
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    histories: list[dict[str, float | int]] = []
    for restart in range(restarts):
        achromatic_logits = torch.nn.Parameter(
            _restart_logits(
                (6,),
                restart=restart,
                seed=seed,
                standard_deviation=restart_standard_deviation,
            )
        )
        chromatic_logits = torch.nn.Parameter(
            _restart_logits(
                (5, 3, 6),
                restart=restart,
                seed=seed + 1000,
                standard_deviation=restart_standard_deviation,
            )
        )
        optimizer = torch.optim.Adam(
            [achromatic_logits, chromatic_logits], lr=learning_rate
        )
        objective = float("inf")
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            achromatic_controls = _controls_from_logits_torch(
                achromatic_logits,
                minimum_increment=minimum_control_increment,
            )
            chromatic_controls = _controls_from_logits_torch(
                chromatic_logits,
                minimum_increment=minimum_control_increment,
            )
            prediction = _apply_candidate_torch(
                source,
                achromatic_controls,
                chromatic_controls,
                direction_softening_lab=direction_softening_lab,
                direction_concentration=direction_concentration,
                neutral_chroma_half_activation=neutral_chroma_half_activation,
            )
            regularization = torch.mean((achromatic_controls - identity) ** 2)
            regularization = regularization + torch.mean(
                (chromatic_controls - identity) ** 2
            )
            loss = torch.mean((prediction - target_tensor) ** 2)
            loss = loss + identity_regularization * regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [achromatic_logits, chromatic_logits], gradient_clip_norm
            )
            optimizer.step()
            objective = float(loss.detach())
        achromatic_array = achromatic_controls.detach().cpu().numpy()
        chromatic_array = chromatic_controls.detach().cpu().numpy()
        achromatic_array[0], achromatic_array[-1] = 0.0, 1.0
        chromatic_array[..., 0], chromatic_array[..., -1] = 0.0, 1.0
        histories.append({"restart": restart, "final_objective": objective})
        if best is None or (objective, restart) < (best[0], best[1]):
            best = (
                objective,
                restart,
                achromatic_array.copy(),
                chromatic_array.copy(),
            )
    assert best is not None
    operator = AnalyticChromaSectorCurveOperator(
        achromatic_control_values=best[2],
        chromatic_control_values=best[3],
        direction_softening_lab=direction_softening_lab,
        direction_concentration=direction_concentration,
        neutral_chroma_half_activation=neutral_chroma_half_activation,
        minimum_control_increment=minimum_control_increment,
    )
    return operator, {
        "selected_restart": best[1],
        "objective": best[0],
        "restarts": histories,
    }


def fit_global_bernstein_curve_operator(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    minimum_control_increment: float,
    seed: int,
    restarts: int,
    restart_standard_deviation: float,
    steps: int,
    learning_rate: float,
    identity_regularization: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> tuple[GlobalBernsteinCurveOperator, dict[str, Any]]:
    """Fit the frozen global-curve capacity control."""

    values = _validate_rgb(rgb).reshape(-1, 3)
    targets = _validate_rgb(target).reshape(-1, 3)
    if targets.shape != values.shape or restarts < 1 or steps < 1:
        raise ValueError("invalid global curve fit inputs")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(thread_count)
    source = torch.from_numpy(values.copy())
    target_tensor = torch.from_numpy(targets.copy())
    basis = _bernstein_basis_torch(source)
    identity = torch.linspace(0.0, 1.0, _CONTROL_COUNT, dtype=torch.float64)
    best: tuple[float, int, np.ndarray] | None = None
    histories: list[dict[str, float | int]] = []
    for restart in range(restarts):
        logits = torch.nn.Parameter(
            _restart_logits(
                (3, 6),
                restart=restart,
                seed=seed,
                standard_deviation=restart_standard_deviation,
            )
        )
        optimizer = torch.optim.Adam([logits], lr=learning_rate)
        objective = float("inf")
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            controls = _controls_from_logits_torch(
                logits, minimum_increment=minimum_control_increment
            )
            prediction = torch.sum(basis * controls, dim=-1)
            loss = torch.mean((prediction - target_tensor) ** 2)
            loss = loss + identity_regularization * torch.mean(
                (controls - identity) ** 2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], gradient_clip_norm)
            optimizer.step()
            objective = float(loss.detach())
        array = controls.detach().cpu().numpy()
        array[:, 0], array[:, -1] = 0.0, 1.0
        histories.append({"restart": restart, "final_objective": objective})
        if best is None or (objective, restart) < (best[0], best[1]):
            best = (objective, restart, array.copy())
    assert best is not None
    return (
        GlobalBernsteinCurveOperator(
            best[2], minimum_control_increment=minimum_control_increment
        ),
        {
            "selected_restart": best[1],
            "objective": best[0],
            "restarts": histories,
        },
    )


__all__ = [
    "ANALYTIC_CHROMA_SECTOR_CURVE_SCHEMA",
    "GLOBAL_BERNSTEIN_CURVE_SCHEMA",
    "AnalyticChromaSectorCurveOperator",
    "GlobalBernsteinCurveOperator",
    "fit_analytic_chroma_sector_curve_operator",
    "fit_global_bernstein_curve_operator",
]
