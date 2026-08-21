"""Metadata-only DNG dual-illuminant camera-to-PCS matrix mechanics.

This product includes DNG technology under license by Adobe.

The equations and constants in this private research primitive follow Adobe's
DNG SDK 1.7.1 build 2652.  It intentionally supports only three-channel A/D65
dual-illuminant profiles and never reads raster samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PCS_XY = (0.3457, 0.3585)
ILLUMINANT_TEMPERATURES = {17: 2850.0, 21: 6500.0}

# Wyszecki & Stiles table copied from Adobe DNG SDK 1.7.1 dng_temperature.cpp.
_TEMP_TABLE = (
    (0, 0.18006, 0.26352, -0.24341),
    (10, 0.18066, 0.26589, -0.25479),
    (20, 0.18133, 0.26846, -0.26876),
    (30, 0.18208, 0.27119, -0.28539),
    (40, 0.18293, 0.27407, -0.30470),
    (50, 0.18388, 0.27709, -0.32675),
    (60, 0.18494, 0.28021, -0.35156),
    (70, 0.18611, 0.28342, -0.37915),
    (80, 0.18740, 0.28668, -0.40955),
    (90, 0.18880, 0.28997, -0.44278),
    (100, 0.19032, 0.29326, -0.47888),
    (125, 0.19462, 0.30141, -0.58204),
    (150, 0.19962, 0.30921, -0.70471),
    (175, 0.20525, 0.31647, -0.84901),
    (200, 0.21142, 0.32312, -1.0182),
    (225, 0.21807, 0.32909, -1.2168),
    (250, 0.22511, 0.33439, -1.4512),
    (275, 0.23247, 0.33904, -1.7298),
    (300, 0.24010, 0.34308, -2.0637),
    (325, 0.24702, 0.34655, -2.4681),
    (350, 0.25591, 0.34951, -2.9641),
    (375, 0.26400, 0.35200, -3.5814),
    (400, 0.27218, 0.35407, -4.3633),
    (425, 0.28039, 0.35577, -5.3762),
    (450, 0.28863, 0.35714, -6.7262),
    (475, 0.29685, 0.35823, -8.5955),
    (500, 0.30505, 0.35907, -11.324),
    (525, 0.31320, 0.35968, -15.628),
    (550, 0.32129, 0.36011, -23.325),
    (575, 0.32931, 0.36038, -40.770),
    (600, 0.33724, 0.36051, -116.45),
)


class DngForwardMatrixError(ValueError):
    """Raised when metadata is outside the deliberately narrow P94 contract."""


@dataclass(frozen=True)
class DngForwardMatrixResult:
    camera_to_pcs: np.ndarray
    camera_white: np.ndarray
    forward_matrix: np.ndarray
    interpolation_weight_first: float
    neutral_xy: tuple[float, float]
    neutral_iterations: int
    reference_neutral: np.ndarray


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise DngForwardMatrixError(f"{name} must be finite with shape {shape}")
    return result


def _round_sdk(value: np.ndarray, factor: float = 10000.0) -> np.ndarray:
    scaled = value * factor
    rounded = np.where(scaled > 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return rounded / factor


def xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = (float(xy[0]), float(xy[1]))
    if not np.isfinite(x) or not np.isfinite(y) or y <= 0.0:
        raise DngForwardMatrixError("xy must be finite with positive y")
    return np.asarray([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


PCS_XYZ = xy_to_xyz(PCS_XY)


def xyz_to_xy(xyz: object) -> tuple[float, float]:
    value = _array(xyz, (3,), "xyz")
    total = float(np.sum(value))
    if total <= 0.0:
        raise DngForwardMatrixError("xyz sum must be positive")
    return float(value[0] / total), float(value[1] / total)


def legacy_temperature(xy: tuple[float, float]) -> float:
    """Return the SDK legacy correlated temperature used for interpolation."""

    x, y = xy
    denominator = 1.5 - x + 6.0 * y
    if denominator <= 0.0:
        raise DngForwardMatrixError("xy is outside the temperature domain")
    u = 2.0 * x / denominator
    v = 3.0 * y / denominator
    last_dt = 0.0
    for index in range(1, 31):
        r, table_u, table_v, tangent = _TEMP_TABLE[index]
        du = 1.0
        dv = tangent
        length = float(np.sqrt(1.0 + dv * dv))
        du /= length
        dv /= length
        uu = u - table_u
        vv = v - table_v
        dt = -uu * dv + vv * du
        if dt <= 0.0 or index == 30:
            dt = min(dt, 0.0)
            dt = -dt
            fraction = 0.0 if index == 1 else dt / (last_dt + dt)
            reciprocal = _TEMP_TABLE[index - 1][0] * fraction + r * (1.0 - fraction)
            if reciprocal <= 0.0:
                raise DngForwardMatrixError("temperature reciprocal is non-positive")
            return 1.0e6 / reciprocal
        last_dt = dt
    raise AssertionError("temperature table traversal did not terminate")


def normalize_color_matrix(matrix: object) -> np.ndarray:
    result = _array(matrix, (3, 3), "color_matrix").copy()
    maximum = float(np.max(result @ PCS_XYZ))
    if maximum > 0.0 and (maximum < 0.99 or maximum > 1.01):
        result /= maximum
    return _round_sdk(result)


def normalize_forward_matrix(matrix: object) -> np.ndarray:
    result = _round_sdk(_array(matrix, (3, 3), "forward_matrix"))
    mapped_one = result @ np.ones(3, dtype=np.float64)
    if np.any(mapped_one == 0.0):
        raise DngForwardMatrixError("forward matrix maps camera one to zero")
    result = np.diag(PCS_XYZ / mapped_one) @ result
    if not np.all(np.isfinite(result)):
        raise DngForwardMatrixError("normalized forward matrix is non-finite")
    return result


def _weight_first(temperature: float, first: float, second: float) -> float:
    if temperature <= first:
        return 1.0
    if temperature >= second:
        return 0.0
    inv = 1.0 / temperature
    return (inv - 1.0 / second) / (1.0 / first - 1.0 / second)


def build_dual_illuminant_camera_to_pcs(
    *,
    color_matrix1: object,
    color_matrix2: object,
    forward_matrix1: object,
    forward_matrix2: object,
    calibration_illuminant1: int,
    calibration_illuminant2: int,
    as_shot_neutral: object,
    camera_calibration1: object | None = None,
    camera_calibration2: object | None = None,
    analog_balance: object | None = None,
) -> DngForwardMatrixResult:
    """Construct the narrow P94 three-channel dual-illuminant transform."""

    illuminants = (int(calibration_illuminant1), int(calibration_illuminant2))
    if set(illuminants) != set(ILLUMINANT_TEMPERATURES):
        raise DngForwardMatrixError("only one A and one D65 calibration are supported")
    neutral = _array(as_shot_neutral, (3,), "as_shot_neutral")
    if np.any(neutral <= 0.0):
        raise DngForwardMatrixError("as_shot_neutral must be positive")
    analog = (
        np.ones(3, dtype=np.float64)
        if analog_balance is None
        else _array(analog_balance, (3,), "analog_balance")
    )
    if np.any(analog <= 0.0):
        raise DngForwardMatrixError("analog_balance must be positive")

    identities = np.eye(3, dtype=np.float64)
    calibrations = [
        identities
        if camera_calibration1 is None
        else _array(camera_calibration1, (3, 3), "camera_calibration1"),
        identities
        if camera_calibration2 is None
        else _array(camera_calibration2, (3, 3), "camera_calibration2"),
    ]
    colors = [
        normalize_color_matrix(color_matrix1),
        normalize_color_matrix(color_matrix2),
    ]
    forwards = [
        normalize_forward_matrix(forward_matrix1),
        normalize_forward_matrix(forward_matrix2),
    ]
    temperatures = [ILLUMINANT_TEMPERATURES[value] for value in illuminants]
    if temperatures[0] > temperatures[1]:
        temperatures.reverse()
        colors.reverse()
        forwards.reverse()
        calibrations.reverse()
    analog_matrix = np.diag(analog)
    colors = [
        analog_matrix @ calibration @ color
        for calibration, color in zip(calibrations, colors, strict=True)
    ]

    def interpolate(
        xy: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        weight = _weight_first(legacy_temperature(xy), temperatures[0], temperatures[1])
        color = weight * colors[0] + (1.0 - weight) * colors[1]
        forward = weight * forwards[0] + (1.0 - weight) * forwards[1]
        calibration = weight * calibrations[0] + (1.0 - weight) * calibrations[1]
        return color, forward, calibration, weight

    white_xy = PCS_XY
    iterations = 30
    for index in range(30):
        color, _, _, _ = interpolate(white_xy)
        try:
            next_xy = xyz_to_xy(np.linalg.solve(color, neutral))
        except np.linalg.LinAlgError as exc:
            raise DngForwardMatrixError(
                "interpolated color matrix is singular"
            ) from exc
        if abs(next_xy[0] - white_xy[0]) + abs(next_xy[1] - white_xy[1]) < 1.0e-7:
            white_xy = next_xy
            iterations = index + 1
            break
        if index == 29:
            next_xy = (
                (white_xy[0] + next_xy[0]) * 0.5,
                (white_xy[1] + next_xy[1]) * 0.5,
            )
        white_xy = next_xy

    color, forward, calibration, weight = interpolate(white_xy)
    camera_white = color @ xy_to_xyz(white_xy)
    maximum = float(np.max(camera_white))
    if maximum == 0.0:
        raise DngForwardMatrixError("camera white is zero")
    camera_white = np.clip(camera_white / maximum, 0.001, 1.0)
    try:
        individual_to_reference = np.linalg.inv(analog_matrix @ calibration)
    except np.linalg.LinAlgError as exc:
        raise DngForwardMatrixError("camera calibration is singular") from exc
    reference_neutral = individual_to_reference @ camera_white
    if np.any(reference_neutral == 0.0):
        raise DngForwardMatrixError("reference neutral contains zero")
    camera_to_pcs = forward @ np.diag(1.0 / reference_neutral) @ individual_to_reference
    determinant = float(np.linalg.det(camera_to_pcs))
    if not np.all(np.isfinite(camera_to_pcs)) or determinant == 0.0:
        raise DngForwardMatrixError("camera-to-PCS transform is invalid")
    return DngForwardMatrixResult(
        camera_to_pcs=camera_to_pcs,
        camera_white=camera_white,
        forward_matrix=forward,
        interpolation_weight_first=weight,
        neutral_xy=white_xy,
        neutral_iterations=iterations,
        reference_neutral=reference_neutral,
    )


__all__ = [
    "ILLUMINANT_TEMPERATURES",
    "PCS_XY",
    "PCS_XYZ",
    "DngForwardMatrixError",
    "DngForwardMatrixResult",
    "build_dual_illuminant_camera_to_pcs",
    "legacy_temperature",
    "normalize_color_matrix",
    "normalize_forward_matrix",
    "xy_to_xyz",
    "xyz_to_xy",
]
