"""Deterministic classical colour-transfer baselines for CT5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from skimage.color import lab2rgb, rgb2lab

from .operators import AffineColorOperator, _validate_rgb
from .splines import AffineMonotoneSplineOperator, RationalQuadraticSpline


SLICED_SCHEMA = "roll2film.sliced_transport.v1"
LAB_STATS_SCHEMA = "roll2film.lab_mean_std.v1"
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def fit_basic_adjustment_family(
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
) -> dict[str, AffineColorOperator]:
    """Fit exposure/WB/contrast/saturation adversaries from unpaired pixels."""
    source = _pixels(source_pixels)
    target = _pixels(target_pixels)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_luma = source @ LUMA
    target_luma = target @ LUMA
    exposure = float(target_luma.mean() / max(source_luma.mean(), 1e-8))
    exposure_operator = AffineColorOperator(np.eye(3) * exposure, np.zeros(3))

    channel_ratio = target_mean / np.maximum(source_mean, 1e-8)
    channel_ratio /= np.exp(np.mean(np.log(np.maximum(channel_ratio, 1e-8))))
    wb_operator = AffineColorOperator(np.diag(channel_ratio), np.zeros(3))

    contrast = float(target_luma.std() / max(source_luma.std(), 1e-8))
    contrast = max(contrast, 1e-4)
    pivot = float(np.median(source_luma))
    contrast_operator = AffineColorOperator(
        np.eye(3) * contrast,
        np.full(3, pivot * (1.0 - contrast)),
    )

    source_chroma = source - source_luma[:, None]
    target_chroma = target - target_luma[:, None]
    saturation = float(
        np.sqrt(np.mean(target_chroma**2))
        / max(float(np.sqrt(np.mean(source_chroma**2))), 1e-8)
    )
    saturation_operator = AffineColorOperator(_saturation_matrix(saturation), np.zeros(3))

    joint = _fit_joint_basic(source, target, pivot)
    return {
        "exposure_only": exposure_operator,
        "white_balance_only": wb_operator,
        "contrast_only": contrast_operator,
        "saturation_only": saturation_operator,
        "wb_contrast_saturation": joint,
    }


def fit_per_channel_quantile_operator(
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.001, 0.01, 0.04, 0.12, 0.3, 0.5, 0.7, 0.88, 0.96, 0.99, 0.999),
) -> AffineMonotoneSplineOperator:
    source = _pixels(source_pixels)
    target = _pixels(target_pixels)
    probability = _validate_quantiles(quantiles)
    splines = []
    for channel in range(3):
        x = _strictly_increasing(np.quantile(source[:, channel], probability))
        y = _strictly_increasing(np.quantile(target[:, channel], probability))
        splines.append(RationalQuadraticSpline.from_knots(x, y))
    return AffineMonotoneSplineOperator(
        AffineColorOperator.identity(),
        tuple(splines),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class LabMeanStdOperator:
    source_mean: np.ndarray
    source_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    working_space: str = "linear_srgb"

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (self.source_mean, self.source_std, self.target_mean, self.target_std)
        )
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Lab statistics must be finite three-vectors")
        if np.any(arrays[1] <= 0.0) or np.any(arrays[3] <= 0.0):
            raise ValueError("Lab standard deviations must be positive")
        for name, value in zip(
            ("source_mean", "source_std", "target_mean", "target_std"), arrays
        ):
            object.__setattr__(self, name, value)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        encoded = _linear_to_srgb(values)
        lab = rgb2lab(encoded.reshape(-1, 1, 3)).reshape(values.shape)
        transferred = (lab - self.source_mean) / self.source_std * self.target_std + self.target_mean
        rendered = lab2rgb(transferred.reshape(-1, 1, 3)).reshape(values.shape)
        return _srgb_to_linear(rendered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LAB_STATS_SCHEMA,
            "working_space": self.working_space,
            "source_mean": self.source_mean.tolist(),
            "source_std": self.source_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "implicit_gamut_clip": True,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabMeanStdOperator":
        if payload.get("schema") != LAB_STATS_SCHEMA:
            raise ValueError(f"unsupported Lab stats schema: {payload.get('schema')!r}")
        return cls(
            payload["source_mean"],
            payload["source_std"],
            payload["target_mean"],
            payload["target_std"],
            str(payload["working_space"]),
        )


def fit_lab_mean_std_operator(
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
) -> LabMeanStdOperator:
    source = _pixels(source_pixels)
    target = _pixels(target_pixels)
    source_lab = rgb2lab(_linear_to_srgb(source).reshape(-1, 1, 3)).reshape(-1, 3)
    target_lab = rgb2lab(_linear_to_srgb(target).reshape(-1, 1, 3)).reshape(-1, 3)
    return LabMeanStdOperator(
        source_lab.mean(axis=0),
        np.maximum(source_lab.std(axis=0), 1e-6),
        target_lab.mean(axis=0),
        np.maximum(target_lab.std(axis=0), 1e-6),
    )


@dataclass(frozen=True)
class SlicedTransportStep:
    direction: np.ndarray
    spline: RationalQuadraticSpline

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("sliced transport direction must be a finite RGB vector")
        norm = float(np.linalg.norm(direction))
        if abs(norm - 1.0) > 1e-10:
            raise ValueError("sliced transport direction must have unit norm")
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class SlicedTransportOperator:
    steps: tuple[SlicedTransportStep, ...]
    working_space: str = "linear_srgb"

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("sliced transport requires at least one step")

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb).copy()
        for step in self.steps:
            projection = values @ step.direction
            mapped = step.spline.apply(projection)
            values += (mapped - projection)[..., None] * step.direction
        return values

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb).copy()
        for step in reversed(self.steps):
            projection = values @ step.direction
            mapped = step.spline.inverse(projection)
            values += (mapped - projection)[..., None] * step.direction
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SLICED_SCHEMA,
            "working_space": self.working_space,
            "steps": [
                {"direction": step.direction.tolist(), "spline": step.spline.to_dict()}
                for step in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SlicedTransportOperator":
        if payload.get("schema") != SLICED_SCHEMA:
            raise ValueError(f"unsupported sliced schema: {payload.get('schema')!r}")
        steps = tuple(
            SlicedTransportStep(
                np.asarray(item["direction"], dtype=np.float64),
                RationalQuadraticSpline.from_dict(item["spline"]),
            )
            for item in payload["steps"]
        )
        return cls(steps, str(payload["working_space"]))


def fit_sliced_transport_operator(
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
    *,
    iterations: int = 8,
    seed: int = 2026071505,
    quantiles: tuple[float, ...] = (0.001, 0.01, 0.04, 0.12, 0.3, 0.5, 0.7, 0.88, 0.96, 0.99, 0.999),
) -> SlicedTransportOperator:
    source = _pixels(source_pixels).copy()
    target = _pixels(target_pixels)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    probability = _validate_quantiles(quantiles)
    rng = np.random.default_rng(seed)
    steps: list[SlicedTransportStep] = []
    for _ in range(iterations):
        directions, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        for channel in range(3):
            direction = directions[:, channel]
            source_projection = source @ direction
            target_projection = target @ direction
            x = _strictly_increasing(np.quantile(source_projection, probability))
            y = _strictly_increasing(np.quantile(target_projection, probability))
            spline = RationalQuadraticSpline.from_knots(x, y)
            mapped = spline.apply(source_projection)
            source += (mapped - source_projection)[:, None] * direction
            steps.append(SlicedTransportStep(direction, spline))
    return SlicedTransportOperator(tuple(steps))


def _fit_joint_basic(source: np.ndarray, target: np.ndarray, pivot: float) -> AffineColorOperator:
    source_summary = _summary_vector(source)
    target_summary = _summary_vector(target)
    scale = np.maximum(np.abs(source_summary) + np.abs(target_summary), 0.05)

    def residual(parameters: np.ndarray) -> np.ndarray:
        operator = _joint_operator(parameters, pivot)
        return (_summary_vector(operator.apply(source)) - target_summary) / scale

    result = least_squares(
        residual,
        np.zeros(5, dtype=np.float64),
        bounds=(np.array([-0.7, -0.3, -0.3, -0.7, -0.7]), np.array([0.7, 0.3, 0.3, 0.7, 0.7])),
        max_nfev=100,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    return _joint_operator(result.x, pivot)


def _joint_operator(parameters: np.ndarray, pivot: float) -> AffineColorOperator:
    exposure = float(np.exp(parameters[0]))
    wb_log = np.array([parameters[1], parameters[2], -parameters[1] - parameters[2]])
    diagonal = np.diag(exposure * np.exp(wb_log))
    contrast = float(np.exp(parameters[3]))
    saturation = float(np.exp(parameters[4]))
    contrast_matrix = contrast * diagonal
    contrast_bias = np.full(3, pivot * (1.0 - contrast))
    saturation_matrix = _saturation_matrix(saturation)
    return AffineColorOperator(
        saturation_matrix @ contrast_matrix,
        saturation_matrix @ contrast_bias,
    )


def _saturation_matrix(saturation: float) -> np.ndarray:
    if not np.isfinite(saturation) or saturation <= 0.0:
        raise ValueError("saturation must be finite and positive")
    return saturation * np.eye(3) + (1.0 - saturation) * np.ones((3, 1)) @ LUMA[None, :]


def _summary_vector(pixels: np.ndarray) -> np.ndarray:
    luma = pixels @ LUMA
    chroma = pixels - luma[:, None]
    covariance = np.cov(pixels, rowvar=False)
    return np.concatenate(
        (
            pixels.mean(axis=0),
            covariance[np.triu_indices(3)],
            np.quantile(luma, [0.05, 0.25, 0.5, 0.75, 0.95]),
            np.array([np.sqrt(np.mean(chroma**2)), np.std(chroma)]),
        )
    )


def _validate_quantiles(values: tuple[float, ...]) -> np.ndarray:
    quantiles = np.asarray(values, dtype=np.float64)
    if (
        quantiles.ndim != 1
        or len(quantiles) < 4
        or quantiles[0] <= 0.0
        or quantiles[-1] >= 1.0
        or np.any(np.diff(quantiles) <= 0.0)
    ):
        raise ValueError("quantiles must be strictly increasing inside (0, 1)")
    return quantiles


def _strictly_increasing(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for index in range(1, len(result)):
        result[index] = max(result[index], result[index - 1] + epsilon)
    return result


def _pixels(values: np.ndarray) -> np.ndarray:
    pixels = np.asarray(values, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("pixels must have shape (N, 3)")
    if len(pixels) < 16 or not np.all(np.isfinite(pixels)):
        raise ValueError("pixels must contain at least sixteen finite RGB samples")
    return pixels


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(np.maximum(values, 0.0), 1.0 / 2.4) - 0.055,
    )


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    )
