"""Known-truth pseudo-roll simulator for the CT1/E0 identifiability gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import AffineColorOperator
from .splines import AffineMonotoneSplineOperator, RationalQuadraticSpline


BASE_MEAN = np.array([0.46, 0.49, 0.43], dtype=np.float64)
BASE_COVARIANCE = np.array(
    [[0.020, 0.008, 0.005], [0.008, 0.018, 0.007], [0.005, 0.007, 0.017]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PseudoRollConfig:
    frames: int
    pixels_per_frame: int = 128
    exposure_sigma: float = 0.0
    white_balance_sigma: float = 0.0
    scene_mean_sigma: float = 0.0
    sensor_noise_sigma: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError("frames must be positive")
        if self.pixels_per_frame < 16:
            raise ValueError("pixels_per_frame must be at least 16")
        for name in (
            "exposure_sigma",
            "white_balance_sigma",
            "scene_mean_sigma",
            "sensor_noise_sigma",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PseudoRoll:
    target_frames: tuple[np.ndarray, ...]
    source_frames: tuple[np.ndarray, ...]
    operator: AffineColorOperator | AffineMonotoneSplineOperator
    exposure_gains: tuple[float, ...]
    white_balance_gains: tuple[np.ndarray, ...]
    config: PseudoRollConfig

    @property
    def target_pixels(self) -> np.ndarray:
        return np.concatenate(self.target_frames, axis=0)

    @property
    def source_pixels(self) -> np.ndarray:
        return np.concatenate(self.source_frames, axis=0)


def default_truth_operator() -> AffineColorOperator:
    """Return a stable non-identity SPD transform recoverable from moments."""
    rotation = np.array(
        [[0.80, -0.48, 0.36], [0.60, 0.64, -0.48], [0.00, 0.60, 0.80]],
        dtype=np.float64,
    )
    scales = np.diag([1.16, 0.91, 1.07])
    matrix = rotation @ scales @ rotation.T
    return AffineColorOperator(matrix=matrix, bias=np.array([0.018, -0.012, 0.010]))


def alternate_truth_operator() -> AffineColorOperator:
    rotation = np.array(
        [[0.72, -0.64, 0.267], [0.69, 0.69, -0.219], [-0.044, 0.338, 0.940]],
        dtype=np.float64,
    )
    q, _ = np.linalg.qr(rotation)
    matrix = q @ np.diag([0.88, 1.14, 1.03]) @ q.T
    return AffineColorOperator(matrix=matrix, bias=np.array([-0.012, 0.016, -0.006]))


def scanner_truth_operator() -> AffineColorOperator:
    """Return a small scanner/profile transform for confounding diagnostics."""
    matrix = np.array(
        [[1.035, -0.018, 0.006], [0.012, 0.972, 0.011], [-0.008, 0.021, 1.026]],
        dtype=np.float64,
    )
    return AffineColorOperator(matrix=matrix, bias=np.array([0.006, -0.004, 0.008]))


def default_l2_truth_operator() -> AffineMonotoneSplineOperator:
    """Return a smooth, globally invertible affine-plus-spline truth."""
    x = np.array([-0.25, 0.0, 0.18, 0.45, 0.75, 1.0, 1.35], dtype=np.float64)
    y_by_channel = (
        np.array([-0.24, 0.0, 0.15, 0.47, 0.82, 1.04, 1.37]),
        np.array([-0.27, -0.01, 0.17, 0.44, 0.72, 0.98, 1.32]),
        np.array([-0.23, 0.01, 0.20, 0.49, 0.78, 1.02, 1.36]),
    )
    splines = tuple(RationalQuadraticSpline.from_knots(x, y) for y in y_by_channel)
    return AffineMonotoneSplineOperator(default_truth_operator(), splines)  # type: ignore[arg-type]


def alternate_l2_truth_operator() -> AffineMonotoneSplineOperator:
    """Return a distinct smooth L2 truth for mixed-operator controls."""
    x = np.array([-0.25, 0.0, 0.18, 0.45, 0.75, 1.0, 1.35], dtype=np.float64)
    y_by_channel = (
        np.array([-0.28, -0.01, 0.19, 0.43, 0.69, 0.95, 1.29]),
        np.array([-0.22, 0.01, 0.14, 0.48, 0.84, 1.06, 1.39]),
        np.array([-0.26, 0.00, 0.16, 0.42, 0.70, 0.97, 1.31]),
    )
    splines = tuple(RationalQuadraticSpline.from_knots(x, y) for y in y_by_channel)
    return AffineMonotoneSplineOperator(alternate_truth_operator(), splines)  # type: ignore[arg-type]


def sample_neutral_prior(pixel_count: int, seed: int) -> np.ndarray:
    if pixel_count < 16:
        raise ValueError("pixel_count must be at least 16")
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(BASE_MEAN, BASE_COVARIANCE, size=pixel_count)


def simulate_pseudo_roll(
    config: PseudoRollConfig,
    operator: AffineColorOperator | AffineMonotoneSplineOperator | None = None,
) -> PseudoRoll:
    truth = operator or default_truth_operator()
    rng = np.random.default_rng(config.seed)
    source_frames: list[np.ndarray] = []
    target_frames: list[np.ndarray] = []
    gains: list[float] = []
    white_balance_gains: list[np.ndarray] = []
    for _ in range(config.frames):
        scene_shift = rng.normal(0.0, config.scene_mean_sigma, size=3)
        source = rng.multivariate_normal(
            BASE_MEAN + scene_shift,
            BASE_COVARIANCE,
            size=config.pixels_per_frame,
        )
        gain = float(np.exp(rng.normal(0.0, config.exposure_sigma)))
        if config.white_balance_sigma:
            log_white_balance = rng.normal(0.0, config.white_balance_sigma, size=3)
            log_white_balance -= log_white_balance.mean()
            white_balance = np.exp(log_white_balance)
        else:
            white_balance = np.ones(3, dtype=np.float64)
        target = truth.apply(source) * gain * white_balance
        if config.sensor_noise_sigma:
            target += rng.normal(0.0, config.sensor_noise_sigma, size=target.shape)
        source_frames.append(source)
        target_frames.append(target)
        gains.append(gain)
        white_balance_gains.append(white_balance)
    return PseudoRoll(
        tuple(target_frames),
        tuple(source_frames),
        truth,
        tuple(gains),
        tuple(white_balance_gains),
        config,
    )


def mix_target_frames(first: PseudoRoll, second: PseudoRoll) -> tuple[np.ndarray, ...]:
    """Build a hostile shuffled-label group without exposing source pairs."""
    if len(first.target_frames) != len(second.target_frames):
        raise ValueError("rolls must have equal frame counts")
    cutoff = max(1, len(first.target_frames) // 2)
    return first.target_frames[:cutoff] + second.target_frames[cutoff:]


def partition_pixels(pixels: np.ndarray, frame_count: int) -> tuple[np.ndarray, ...]:
    """Partition one fixed pixel pool without changing its concatenated values."""
    values = np.asarray(pixels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("pixels must have shape (N, 3)")
    if frame_count < 1 or len(values) % frame_count:
        raise ValueError("frame_count must be positive and divide the pixel count")
    return tuple(np.split(values, frame_count, axis=0))


def shuffle_frame_boundaries(
    frames: tuple[np.ndarray, ...],
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Destroy true frame boundaries while preserving pixels and frame sizes."""
    if not frames:
        raise ValueError("at least one frame is required")
    lengths = [len(frame) for frame in frames]
    if min(lengths) < 4:
        raise ValueError("every frame must contain at least four pixels")
    pixels = np.concatenate(frames, axis=0)
    shuffled = pixels[np.random.default_rng(seed).permutation(len(pixels))]
    cuts = np.cumsum(lengths[:-1])
    return tuple(np.split(shuffled, cuts, axis=0))
