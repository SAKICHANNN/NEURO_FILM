"""Strict controls-only structural evaluator for U5.R2AJ0C0B.

The module intentionally has no archive-path input and never reads an
external CLUT. It establishes the parser, interpolation, metric and evidence
semantics that a later, separately frozen primary-bank leaf may consume.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile
from scipy.stats import qmc

from src.eval.spectral_film_lut_bank import matched_basic_output, median_delta_e76
from src.eval.velvia_datasheet_witness import (
    _RGB_TO_XYZ,
    encoded_srgb_to_linear,
    xyz_to_lab,
)
from src.roll2film.baselines import fit_joint_basic_adjustment


CONFIG_SCHEMA = "u5-r2aj0c0b-hald-structural-controls-v2"
MANIFEST_SCHEMA = "u5-r2aj0c0b-hald-structural-control-manifest-v2"
REPORT_SCHEMA = "u5-r2aj0c0b-hald-structural-control-report-v2"
REPEAT_SCHEMA = "u5-r2aj0c0b-hald-structural-control-repeat-decision-v2"
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "cbb25c54e4de593f5f5ba3cee01007289855dbb2caf47ecb8528de6f1f9aa19e"
)
EXPECTED_V1_CLOSE_SHA256 = (
    "a655f1bf6b0707ae3ab55fd9f02b970320598df210c62de7a30576699c3dd98d"
)
SINGLE_RUN_STATE = "control-run-only-non-promoting"
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


class HaldStructuralError(RuntimeError):
    """Raised when a frozen C0B invariant is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HaldStructuralError("value is not canonical finite JSON") from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HaldStructuralError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise HaldStructuralError(f"non-finite JSON constant: {value}")


def strict_json_loads(payload: bytes) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except HaldStructuralError:
        raise
    except Exception as exc:
        raise HaldStructuralError("invalid strict JSON") from exc


def load_config_snapshot(
    path: Path, *, expected_raw_sha256: str | None = None
) -> tuple[dict[str, Any], str, str]:
    raw = path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    if expected_raw_sha256 is not None and raw_sha != expected_raw_sha256:
        raise HaldStructuralError("config raw SHA-256 changed")
    value = strict_json_loads(raw)
    if not isinstance(value, dict):
        raise HaldStructuralError("config must be an object")
    canonical_sha = canonical_sha256(value)
    if canonical_sha != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise HaldStructuralError("config canonical SHA-256 is not frozen v2")
    validate_config(value)
    return value, raw_sha, canonical_sha


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise HaldStructuralError("unsupported C0B config schema")
    schemas = config.get("evidence_schemas")
    if not isinstance(schemas, dict):
        raise HaldStructuralError("missing evidence schemas")
    expected = {
        "manifest": MANIFEST_SCHEMA,
        "report": REPORT_SCHEMA,
        "repeat_decision": REPEAT_SCHEMA,
    }
    for key, value in expected.items():
        if schemas.get(key) != value:
            raise HaldStructuralError(f"evidence schema mismatch: {key}")
    if canonical_sha256(config) != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise HaldStructuralError("config object differs from frozen v2")


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise HaldStructuralError("evidence destination is not a regular file")
        if path.read_bytes() == payload:
            return hashlib.sha256(payload).hexdigest()
        raise HaldStructuralError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise HaldStructuralError("temporary evidence path already exists")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def synthetic_cube(side: int) -> np.ndarray:
    if isinstance(side, bool) or not isinstance(side, int) or side < 2:
        raise HaldStructuralError("probe cube side must be an integer >= 2")
    axis = np.linspace(0.0, 1.0, side, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def quantize_rgb8(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )


def _linear_to_encoded(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def _lab(encoded: np.ndarray) -> np.ndarray:
    linear = encoded_srgb_to_linear(np.asarray(encoded, dtype=np.float64))
    return xyz_to_lab(linear @ _RGB_TO_XYZ.T)


def gradient_population(samples: int) -> np.ndarray:
    if samples != 4097:
        raise HaldStructuralError("gradient sample count differs from frozen v2")
    axis = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    lines = [np.stack([axis, axis, axis], axis=1)]
    for varying in range(3):
        others = [index for index in range(3) if index != varying]
        for first in (0.0, 1.0):
            for second in (0.0, 1.0):
                line = np.empty((samples, 3), dtype=np.float64)
                line[:, varying] = axis
                line[:, others[0]] = first
                line[:, others[1]] = second
                lines.append(line)
    for fixed in range(3):
        others = [index for index in range(3) if index != fixed]
        for fixed_value in (0.0, 1.0):
            for reverse in (False, True):
                line = np.empty((samples, 3), dtype=np.float64)
                line[:, fixed] = fixed_value
                line[:, others[0]] = axis
                line[:, others[1]] = 1.0 - axis if reverse else axis
                lines.append(line)
    result = np.stack(lines, axis=0)
    if result.shape != (25, 4097, 3):
        raise HaldStructuralError("gradient population shape mismatch")
    return result


def finite_difference_probes() -> np.ndarray:
    points = []
    for red_cell in range(3):
        for green_cell in range(3):
            for blue_cell in range(3):
                for red_fraction in (0.25, 0.5, 0.75):
                    for green_fraction in (0.25, 0.5, 0.75):
                        for blue_fraction in (0.25, 0.5, 0.75):
                            points.append(
                                (
                                    (red_cell + red_fraction) / 3.0,
                                    (green_cell + green_fraction) / 3.0,
                                    (blue_cell + blue_fraction) / 3.0,
                                )
                            )
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (729, 3):
        raise HaldStructuralError("finite-difference population mismatch")
    return result


def _integer_cube_root(value: int) -> int:
    estimate = int(round(value ** (1.0 / 3.0)))
    for candidate in range(max(0, estimate - 2), estimate + 3):
        if candidate**3 == value:
            return candidate
    raise HaldStructuralError("Hald raster side is not an integer cube")


@dataclass(frozen=True)
class HaldCube:
    table: np.ndarray

    def __post_init__(self) -> None:
        table = np.asarray(self.table)
        if (
            table.ndim != 4
            or table.shape[-1] != 3
            or len(set(table.shape[:3])) != 1
            or table.shape[0] < 2
        ):
            raise HaldStructuralError("Hald table must have shape NxNxNx3")
        if table.dtype != np.uint8 and table.dtype != np.float64:
            raise HaldStructuralError("Hald table must be uint8 or float64")
        if table.dtype == np.float64 and (
            not np.all(np.isfinite(table))
            or np.any(table < 0.0)
            or np.any(table > 1.0)
        ):
            raise HaldStructuralError("float Hald table must be finite in [0,1]")
        object.__setattr__(self, "table", table)

    @property
    def side(self) -> int:
        return int(self.table.shape[0])

    def normalized(self) -> np.ndarray:
        if self.table.dtype == np.uint8:
            return self.table.astype(np.float64) / 255.0
        return self.table

    @classmethod
    def from_raster(cls, raster: np.ndarray) -> "HaldCube":
        image = np.asarray(raster)
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[-1] != 3
            or image.shape[0] != image.shape[1]
        ):
            raise HaldStructuralError("Hald raster must be square RGB uint8")
        level = _integer_cube_root(int(image.shape[0]))
        if level <= 1:
            raise HaldStructuralError("Hald level must exceed one")
        cube_side = level**2
        flat_bgr = image.reshape(cube_side, cube_side, cube_side, 3)
        return cls(flat_bgr.transpose(2, 1, 0, 3))

    def to_raster(self) -> np.ndarray:
        level = math.isqrt(self.side)
        if level * level != self.side or level <= 1:
            raise HaldStructuralError("cube side is not a legal squared Hald level")
        image_side = level**3
        return np.ascontiguousarray(
            self.table.transpose(2, 1, 0, 3).reshape(image_side, image_side, 3)
        )

    def apply(self, points: np.ndarray, *, batch_rows: int = 65536) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim < 2 or values.shape[-1] != 3:
            raise HaldStructuralError("Hald inputs must have shape (...,3)")
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise HaldStructuralError("Hald inputs must be finite in [0,1]")
        original_shape = values.shape
        flat = values.reshape(-1, 3)
        output = np.empty_like(flat)
        for start in range(0, len(flat), batch_rows):
            stop = min(len(flat), start + batch_rows)
            output[start:stop] = self._apply_batch(flat[start:stop])
        return output.reshape(original_shape)

    def _corners(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        coordinates = points * (self.side - 1)
        lower = np.floor(coordinates).astype(np.int64)
        lower = np.minimum(lower, self.side - 2)
        fraction = coordinates - lower
        red, green, blue = lower.T
        table = self.table

        def gather(dr: int, dg: int, db: int) -> np.ndarray:
            sample = table[red + dr, green + dg, blue + db]
            if table.dtype == np.uint8:
                return sample.astype(np.float64) / 255.0
            return np.asarray(sample, dtype=np.float64)

        corners = tuple(
            gather(dr, dg, db)
            for dr, dg, db in (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
            )
        )
        return lower, fraction, corners

    def _apply_batch(self, points: np.ndarray) -> np.ndarray:
        _lower, fraction, corners = self._corners(points)
        fr, fg, fb = fraction.T
        c000, c100, c010, c001, c110, c101, c011, c111 = corners
        return (
            ((1 - fr) * (1 - fg) * (1 - fb))[:, None] * c000
            + (fr * (1 - fg) * (1 - fb))[:, None] * c100
            + ((1 - fr) * fg * (1 - fb))[:, None] * c010
            + ((1 - fr) * (1 - fg) * fb)[:, None] * c001
            + (fr * fg * (1 - fb))[:, None] * c110
            + (fr * (1 - fg) * fb)[:, None] * c101
            + ((1 - fr) * fg * fb)[:, None] * c011
            + (fr * fg * fb)[:, None] * c111
        )

    def jacobian(
        self, points: np.ndarray, *, batch_rows: int = 65536
    ) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise HaldStructuralError("Jacobian probes must have shape Nx3")
        output = np.empty((len(values), 3, 3), dtype=np.float64)
        for start in range(0, len(values), batch_rows):
            stop = min(len(values), start + batch_rows)
            output[start:stop] = self._jacobian_batch(values[start:stop])
        return output

    def _jacobian_batch(self, points: np.ndarray) -> np.ndarray:
        _lower, fraction, corners = self._corners(points)
        fr, fg, fb = fraction.T
        c000, c100, c010, c001, c110, c101, c011, c111 = corners
        scale = self.side - 1
        dr = scale * (
            ((1 - fg) * (1 - fb))[:, None] * (c100 - c000)
            + (fg * (1 - fb))[:, None] * (c110 - c010)
            + ((1 - fg) * fb)[:, None] * (c101 - c001)
            + (fg * fb)[:, None] * (c111 - c011)
        )
        dg = scale * (
            ((1 - fr) * (1 - fb))[:, None] * (c010 - c000)
            + (fr * (1 - fb))[:, None] * (c110 - c100)
            + ((1 - fr) * fb)[:, None] * (c011 - c001)
            + (fr * fb)[:, None] * (c111 - c101)
        )
        db = scale * (
            ((1 - fr) * (1 - fg))[:, None] * (c001 - c000)
            + (fr * (1 - fg))[:, None] * (c101 - c100)
            + ((1 - fr) * fg)[:, None] * (c011 - c010)
            + (fr * fg)[:, None] * (c111 - c110)
        )
        return np.stack((dr, dg, db), axis=-1)


def identity_table(side: int, *, quantized: bool) -> np.ndarray:
    if side < 2:
        raise HaldStructuralError("identity side must be >=2")
    if quantized:
        codes = np.rint(
            np.linspace(0.0, 1.0, side, dtype=np.float64) * 255.0
        ).astype(np.uint8)
        result = np.empty((side, side, side, 3), dtype=np.uint8)
        result[..., 0] = codes[:, None, None]
        result[..., 1] = codes[None, :, None]
        result[..., 2] = codes[None, None, :]
        return result
    return synthetic_cube(side)


def _basic_control(grid: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
    source = encoded_srgb_to_linear(grid)
    flat = source.reshape(-1, 3)
    pivot = float(np.median(flat @ LUMA))
    values = np.asarray(parameters, dtype=np.float64)
    exposure = float(np.exp(values[0]))
    wb_logs = np.array(
        [values[1], values[2], -values[1] - values[2]], dtype=np.float64
    )
    diagonal = np.diag(exposure * np.exp(wb_logs))
    contrast = float(np.exp(values[3]))
    saturation = float(np.exp(values[4]))
    saturation_matrix = (
        saturation * np.eye(3)
        + (1.0 - saturation) * np.ones((3, 1)) @ LUMA[None, :]
    )
    matrix = saturation_matrix @ (contrast * diagonal)
    bias = saturation_matrix @ np.full(3, pivot * (1.0 - contrast))
    return _linear_to_encoded(source @ matrix.T + bias)


def generate_controls(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    side = int(config["generated_hald_conformance"]["control_cube_side"])
    grid = synthetic_cube(side)
    hue = config["control_generation"]["hue_operator"]
    matrices = {
        "warm": np.asarray(hue["warm_matrix"], dtype=np.float64),
        "cool": np.asarray(hue["cool_matrix"], dtype=np.float64),
    }
    result: dict[str, np.ndarray] = {}
    rows = config["control_generation"]["controls"]
    for row in rows:
        control_id = str(row["id"])
        kind = str(row["kind"])
        if kind == "analytic_identity":
            table = grid.copy()
        elif kind == "quantized_identity":
            table = quantize_rgb8(grid)
        elif kind == "joint_basic":
            table = quantize_rgb8(_basic_control(grid, row["parameters"]))
        elif kind == "hue_operator":
            strength = float(row["strength"])
            matrix = matrices[str(row["matrix"])]
            output = grid + (
                0.75
                * strength
                * grid
                * (1.0 - grid)
                * np.einsum("ij,...j->...i", matrix, grid)
            )
            table = quantize_rgb8(output)
        elif kind == "exact_duplicate":
            source_id = str(row["source_id"])
            if source_id not in result:
                raise HaldStructuralError("duplicate source must precede duplicate")
            table = result[source_id].copy()
        elif kind == "axis_swap":
            table = quantize_rgb8(grid[..., list(row["order"])])
        elif kind == "hard_clip":
            low = float(row["low"])
            high = float(row["high"])
            output = np.where(grid <= low, 0.0, np.where(grid >= high, 1.0, grid))
            table = quantize_rgb8(output)
        elif kind == "staircase":
            levels = int(row["levels"])
            output = np.rint(grid * (levels - 1)) / (levels - 1)
            table = quantize_rgb8(output)
        elif kind == "single_cell_spike":
            table = quantize_rgb8(grid)
            index = tuple(int(value) for value in row["index"])
            table[index] = np.asarray(row["replacement_rgb8"], dtype=np.uint8)
        else:
            raise HaldStructuralError(f"unsupported control kind: {kind}")
        result[control_id] = table
    expected = config["independent_expected_hashes"]["control_source_tables"]
    observed = {key: array_sha256(value) for key, value in result.items()}
    if observed != expected:
        raise HaldStructuralError("independent control source-table hashes differ")
    return result


def _population_bundle(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    populations = {
        "style": synthetic_cube(
            int(config["synthetic_population"]["style_basic_probe_cube_side"])
        ),
        "residual": synthetic_cube(
            int(config["synthetic_population"]["residual_novelty_probe_cube_side"])
        ),
        "sobol": qmc.Sobol(d=3, scramble=False).random_base2(m=16),
        "gradients": gradient_population(4097),
        "finite_difference": finite_difference_probes(),
    }
    expected = config["independent_expected_hashes"]["populations"]
    names = {
        "style": "style_basic_probe_sha256",
        "residual": "residual_novelty_probe_sha256",
        "sobol": "sobol_probe_sha256",
        "gradients": "gradient_population_sha256",
        "finite_difference": "finite_difference_probe_sha256",
    }
    for name, key in names.items():
        if array_sha256(populations[name]) != expected[key]:
            raise HaldStructuralError(f"independent population hash differs: {name}")
    return populations


def _native_difference_signature(table: np.ndarray) -> np.ndarray:
    if table.dtype == np.float64:
        source = quantize_rgb8(table)
    else:
        source = np.asarray(table, dtype=np.uint8)
    first = [
        np.diff(source.astype(np.int16), axis=axis).reshape(-1)
        for axis in range(3)
    ]
    second = [
        np.diff(source.astype(np.int16), n=2, axis=axis).reshape(-1)
        for axis in range(3)
    ]
    return np.concatenate((*first, *second))


def _fit_basic_once(
    style_source: np.ndarray,
    style_target: np.ndarray,
    residual_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_linear = encoded_srgb_to_linear(style_source.reshape(-1, 3))
    target_linear = encoded_srgb_to_linear(style_target.reshape(-1, 3))
    operator = fit_joint_basic_adjustment(source_linear, target_linear)
    style_basic = _linear_to_encoded(
        operator.apply(source_linear).reshape(style_source.shape)
    )
    residual_linear = encoded_srgb_to_linear(residual_source.reshape(-1, 3))
    residual_basic = _linear_to_encoded(
        operator.apply(residual_linear).reshape(residual_source.shape)
    )
    return style_basic, residual_basic


def _safety_metrics(
    cube: HaldCube,
    *,
    style: np.ndarray,
    sobol: np.ndarray,
    gradients: np.ndarray,
    gates: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, np.ndarray]]:
    style_output = cube.apply(style)
    jacobian = cube.jacobian(sobol)
    determinants = np.linalg.det(jacobian)
    spectral_norms = np.linalg.svd(jacobian, compute_uv=False)[..., 0]
    gradient_output = cube.apply(gradients.reshape(-1, 3)).reshape(gradients.shape)
    input_steps = np.linalg.norm(np.diff(gradients, axis=1), axis=-1)
    output_steps = np.linalg.norm(np.diff(gradient_output, axis=1), axis=-1)
    gains = output_steps / input_steps
    second = np.diff(gradient_output, n=2, axis=1)
    neutral_linear_luma = (
        encoded_srgb_to_linear(gradient_output[0]) @ LUMA
    )
    neutral_reversal = float(
        max(0.0, -float(np.min(np.diff(neutral_linear_luma))))
    )
    margin = float(gates["interior_input_margin"])
    interior = np.all((style > margin) & (style < 1.0 - margin), axis=-1)
    endpoint_epsilon = float(gates["endpoint_epsilon"])
    interior_output = style_output[interior]
    probe_endpoint_fraction = float(
        np.mean(
            (interior_output <= endpoint_epsilon)
            | (interior_output >= 1.0 - endpoint_epsilon)
        )
    )
    native_axis = np.linspace(0.0, 1.0, cube.side, dtype=np.float64)
    native_interior_axis = (native_axis > margin) & (
        native_axis < 1.0 - margin
    )
    native_output = cube.normalized()[
        np.ix_(
            native_interior_axis,
            native_interior_axis,
            native_interior_axis,
            np.ones(3, dtype=bool),
        )
    ]
    native_endpoint_fraction = float(
        np.mean(
            (native_output <= endpoint_epsilon)
            | (native_output >= 1.0 - endpoint_epsilon)
        )
    )
    endpoint_fraction = max(probe_endpoint_fraction, native_endpoint_fraction)
    native_signature = _native_difference_signature(cube.table)
    native_side = cube.side
    first_count = sum(
        int(np.prod(np.diff(np.empty((native_side,) * 3 + (3,)), axis=axis).shape))
        for axis in range(3)
    )
    first_values = native_signature[:first_count]
    second_values = native_signature[first_count:]
    metrics: dict[str, Any] = {
        "finite": bool(
            np.all(np.isfinite(style_output))
            and np.all(np.isfinite(jacobian))
            and np.all(np.isfinite(gradient_output))
        ),
        "output_minimum": float(np.min(style_output)),
        "output_maximum": float(np.max(style_output)),
        "probe_new_interior_endpoint_fraction": probe_endpoint_fraction,
        "native_new_interior_endpoint_fraction": native_endpoint_fraction,
        "new_interior_endpoint_fraction": endpoint_fraction,
        "negative_jacobian_fraction": float(
            np.mean(
                determinants < float(gates["negative_jacobian_threshold"])
            )
        ),
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "maximum_jacobian_spectral_norm": float(np.max(spectral_norms)),
        "neutral_luma_reversal": neutral_reversal,
        "gradient_rgb_gain_p99": float(np.quantile(gains, 0.99)),
        "maximum_gradient_rgb_gain": float(np.max(gains)),
        "maximum_gradient_channel_second_difference": float(
            np.max(np.abs(second))
        ),
        "native_maximum_adjacent_code_step": int(
            np.max(np.abs(first_values))
        ),
        "native_maximum_second_code_difference": int(
            np.max(np.abs(second_values))
        ),
    }
    checks = {
        "finite": metrics["finite"] is bool(gates["all_outputs_finite"]),
        "range": metrics["output_minimum"] >= float(gates["output_minimum"])
        and metrics["output_maximum"] <= float(gates["output_maximum"]),
        "new_interior_endpoint_fraction": endpoint_fraction
        <= float(gates["maximum_new_interior_endpoint_fraction"]),
        "negative_jacobian_fraction": metrics["negative_jacobian_fraction"]
        <= float(gates["maximum_negative_jacobian_fraction"]),
        "minimum_jacobian_determinant": metrics[
            "minimum_jacobian_determinant"
        ]
        >= float(gates["minimum_jacobian_determinant"]),
        "maximum_jacobian_spectral_norm": metrics[
            "maximum_jacobian_spectral_norm"
        ]
        <= float(gates["maximum_jacobian_spectral_norm"]),
        "neutral_luma_reversal": neutral_reversal
        <= float(gates["maximum_neutral_luma_reversal"]),
        "gradient_rgb_gain_p99": metrics["gradient_rgb_gain_p99"]
        <= float(gates["maximum_gradient_rgb_gain_p99"]),
        "maximum_gradient_rgb_gain": metrics["maximum_gradient_rgb_gain"]
        <= float(gates["maximum_gradient_rgb_gain"]),
        "maximum_gradient_channel_second_difference": metrics[
            "maximum_gradient_channel_second_difference"
        ]
        <= float(gates["maximum_gradient_channel_second_difference"]),
        "native_maximum_adjacent_code_step": metrics[
            "native_maximum_adjacent_code_step"
        ]
        <= int(gates["native_maximum_adjacent_code_step"]),
        "native_maximum_second_code_difference": metrics[
            "native_maximum_second_code_difference"
        ]
        <= int(gates["native_maximum_second_code_difference"]),
    }
    signatures = {
        "style_output": style_output,
        "jacobian": jacobian,
        "gradient": gradient_output,
        "native_difference": native_signature,
    }
    return metrics, checks, signatures


def _strength_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-30 or norm_b <= 1e-30:
        return {
            "effect_cosine": 0.0,
            "minimum_explained_energy_fraction": 0.0,
            "bilateral_normalized_residual": 1.0,
        }
    cosine = float(np.dot(a, b) / (norm_a * norm_b))

    def direction(target: np.ndarray, basis: np.ndarray) -> tuple[float, float]:
        scale = float(np.dot(target, basis) / np.dot(basis, basis))
        residual = target - scale * basis
        energy = float(np.dot(target, target))
        explained = float(1.0 - np.dot(residual, residual) / energy)
        normalized = float(np.linalg.norm(residual) / np.linalg.norm(target))
        return explained, normalized

    explained_ab, residual_ab = direction(a, b)
    explained_ba, residual_ba = direction(b, a)
    return {
        "effect_cosine": cosine,
        "minimum_explained_energy_fraction": min(explained_ab, explained_ba),
        "bilateral_normalized_residual": max(residual_ab, residual_ba),
    }


def _components(
    ids: Sequence[str],
    output_hashes: Mapping[str, str],
    effects: Mapping[str, np.ndarray],
    strength_gates: Mapping[str, Any],
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    parent = {name: name for name in ids}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            exact = output_hashes[left] == output_hashes[right]
            metrics = _strength_metrics(effects[left], effects[right])
            strength = (
                metrics["effect_cosine"]
                >= float(strength_gates["minimum_positive_effect_cosine"])
                and metrics["minimum_explained_energy_fraction"]
                >= float(strength_gates["minimum_explained_energy_fraction"])
                and metrics["bilateral_normalized_residual"]
                <= float(strength_gates["maximum_bilateral_normalized_residual"])
            )
            if exact or strength:
                union(left, right)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "exact_signature": exact,
                    "strength_equivalent": strength,
                    "metrics": metrics,
                }
            )
    grouped: dict[str, list[str]] = {}
    for name in ids:
        grouped.setdefault(find(name), []).append(name)
    components = sorted(
        (sorted(values) for values in grouped.values()),
        key=lambda values: values[0],
    )
    return components, pairs


def _hald_conformance(
    config: Mapping[str, Any], populations: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    expected = config["independent_expected_hashes"]["identity_hald_rasters"]
    tolerance = float(
        config["oracle_tolerances"]["identity_maximum_absolute_channel_error"]
    )
    records = []
    probe_arrays = (
        populations["style"].reshape(-1, 3),
        populations["residual"].reshape(-1, 3),
        populations["sobol"],
        populations["gradients"].reshape(-1, 3),
    )
    for geometry in config["generated_hald_conformance"]["legal_geometries"]:
        level = int(geometry["level"])
        side = int(geometry["cube_side"])
        image_side = int(geometry["image_side"])
        table = identity_table(side, quantized=True)
        cube = HaldCube(table)
        raster = cube.to_raster()
        key = f"L{level}_N{side}_S{image_side}"
        raster_hash = array_sha256(raster)
        parsed = HaldCube.from_raster(raster)
        roundtrip = bool(np.array_equal(parsed.table, table))
        maximum_error = 0.0
        for probes in probe_arrays:
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(parsed.apply(probes) - probes))),
            )
        record = {
            "level": level,
            "cube_side": side,
            "image_side": image_side,
            "raster_array_sha256": raster_hash,
            "table_roundtrip_exact": roundtrip,
            "maximum_identity_channel_error": maximum_error,
            "checks": {
                "expected_raster_hash": raster_hash == expected[key],
                "table_roundtrip_exact": roundtrip,
                "identity_error": maximum_error <= tolerance,
            },
        }
        records.append(record)
    for illegal in (7, 9, 26, 28, 63, 65, 215, 217):
        try:
            HaldCube.from_raster(np.zeros((illegal, illegal, 3), dtype=np.uint8))
        except HaldStructuralError:
            continue
        raise HaldStructuralError(f"illegal Hald raster side accepted: {illegal}")
    return {
        "records": records,
        "all_checks_pass": all(
            all(record["checks"].values()) for record in records
        ),
    }


def _jacobian_conformance(
    config: Mapping[str, Any], populations: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    probes = populations["finite_difference"]
    identity = HaldCube(synthetic_cube(4))
    swap = HaldCube(synthetic_cube(4)[..., [1, 0, 2]])
    identity_error = float(
        np.max(
            np.abs(
                identity.jacobian(probes)
                - np.broadcast_to(np.eye(3), (len(probes), 3, 3))
            )
        )
    )
    swap_expected = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    swap_error = float(
        np.max(
            np.abs(
                swap.jacobian(probes)
                - np.broadcast_to(swap_expected, (len(probes), 3, 3))
            )
        )
    )
    grid = synthetic_cube(4)
    warm_matrix = np.asarray(
        config["control_generation"]["hue_operator"]["warm_matrix"],
        dtype=np.float64,
    )
    warm_table = grid + (
        0.75
        * grid
        * (1.0 - grid)
        * np.einsum("ij,...j->...i", warm_matrix, grid)
    )
    expected_hash = config["independent_expected_hashes"][
        "finite_difference_warm_float64_table_sha256"
    ]
    if array_sha256(warm_table) != expected_hash:
        raise HaldStructuralError("finite-difference table hash differs")
    warm = HaldCube(warm_table)
    analytic = warm.jacobian(probes)
    step = float(
        config["oracle_tolerances"]["finite_difference"]["central_step"]
    )
    finite = _decimal_finite_difference(warm, probes, step)
    difference = np.abs(analytic - finite)
    floor = float(
        config["oracle_tolerances"]["finite_difference"][
            "relative_denominator_floor"
        ]
    )
    relative = difference / np.maximum(np.abs(finite), floor)
    absolute_error = float(np.max(difference))
    relative_error = float(np.max(relative))
    tolerances = config["oracle_tolerances"]
    checks = {
        "identity": identity_error
        <= float(tolerances["jacobian_identity_maximum_absolute_error"]),
        "axis_swap": swap_error
        <= float(tolerances["jacobian_axis_swap_maximum_absolute_error"]),
        "finite_difference_absolute": absolute_error
        <= float(tolerances["finite_difference"]["maximum_absolute_error"]),
        "finite_difference_relative": relative_error
        <= float(tolerances["finite_difference"]["maximum_relative_error"]),
    }
    return {
        "identity_maximum_absolute_error": identity_error,
        "axis_swap_maximum_absolute_error": swap_error,
        "finite_difference_maximum_absolute_error": absolute_error,
        "finite_difference_maximum_relative_error": relative_error,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _decimal_finite_difference(
    cube: HaldCube, probes: np.ndarray, step: float
) -> np.ndarray:
    """Independent high-precision central difference for the N=4 oracle.

    Windows NumPy longdouble is float64. Decimal avoids cancellation around
    analytically near-zero Jacobian entries at the deliberately tiny frozen
    step, while still evaluating the trilinear interpolation definition
    independently from :meth:`HaldCube.jacobian`.
    """

    if cube.side != 4 or cube.table.dtype != np.float64:
        raise HaldStructuralError("decimal finite difference requires float64 N=4")
    result = np.empty((len(probes), 3, 3), dtype=np.float64)
    step_decimal = Decimal(str(step))
    scale = Decimal(cube.side - 1)
    table = cube.table
    with localcontext() as context:
        context.prec = 50
        for point_index, point in enumerate(probes):
            base_coordinates = [
                Decimal(str(float(point[channel]))) * scale
                for channel in range(3)
            ]
            lower = [int(value) for value in np.floor(np.asarray(point) * 3.0)]
            lower = [min(value, 2) for value in lower]
            for input_channel in range(3):
                output_values = []
                for direction in (-1, 1):
                    fractions = []
                    for channel in range(3):
                        coordinate = base_coordinates[channel]
                        if channel == input_channel:
                            coordinate += Decimal(direction) * step_decimal * scale
                        fractions.append(coordinate - Decimal(lower[channel]))
                    output = [Decimal(0), Decimal(0), Decimal(0)]
                    for dr in (0, 1):
                        for dg in (0, 1):
                            for db in (0, 1):
                                weight = (
                                    (fractions[0] if dr else 1 - fractions[0])
                                    * (fractions[1] if dg else 1 - fractions[1])
                                    * (fractions[2] if db else 1 - fractions[2])
                                )
                                sample = table[
                                    lower[0] + dr,
                                    lower[1] + dg,
                                    lower[2] + db,
                                ]
                                for output_channel in range(3):
                                    output[output_channel] += weight * Decimal(
                                        str(float(sample[output_channel]))
                                    )
                    output_values.append(output)
                for output_channel in range(3):
                    derivative = (
                        output_values[1][output_channel]
                        - output_values[0][output_channel]
                    ) / (2 * step_decimal)
                    result[
                        point_index, output_channel, input_channel
                    ] = float(derivative)
    return result


def _rgb16_conformance(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["rgb16_tiff_conformance"]
    source = np.asarray(spec["flat_samples"], dtype=np.uint16).reshape(
        tuple(spec["shape"])
    )
    expected_hash = config["independent_expected_hashes"][
        "rgb16_tiff_source_array_sha256"
    ]
    if array_sha256(source) != expected_hash:
        raise HaldStructuralError("RGB16 source-array hash differs")
    stream = io.BytesIO()
    tifffile.imwrite(
        stream,
        source,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=[(274, "H", 1, 1, False)],
    )
    stream.seek(0)
    with tifffile.TiffFile(stream) as tif:
        page = tif.pages[0]
        decoded = page.asarray()
        bits = page.tags["BitsPerSample"].value
        if isinstance(bits, int):
            bits = (bits, bits, bits)
        orientation = page.tags["Orientation"].value
        checks = {
            "pages": len(tif.pages) == int(spec["pages"]),
            "series": len(tif.series) == int(spec["series"]),
            "dtype": decoded.dtype == np.uint16,
            "shape": list(decoded.shape) == list(spec["shape"]),
            "samples_exact": bool(np.array_equal(decoded, source)),
            "bits_per_sample": list(bits) == list(spec["bits_per_sample"]),
            "photometric": int(page.photometric) == 2,
            "planar_configuration": int(page.planarconfig) == 1,
            "orientation": int(orientation) == int(spec["orientation"]),
            "extra_samples": len(page.extrasamples) == 0,
        }
        tags = {
            "bits_per_sample": list(bits),
            "photometric": int(page.photometric),
            "planar_configuration": int(page.planarconfig),
            "orientation": int(orientation),
            "extra_samples": [int(value) for value in page.extrasamples],
        }
    return {
        "source_array_sha256": array_sha256(source),
        "decoded_array_sha256": array_sha256(decoded),
        "tag_facts_sha256": canonical_sha256(tags),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _source_contract_checks(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    records = {}
    for key in ("basic_fit", "matched_basic", "lab"):
        contract = (
            config["colour_math_contract"][key]
            if key != "lab"
            else config["colour_math_contract"]["lab"]
        )
        path = root / str(contract["source_path"])
        raw_sha = sha256_file(path)
        blob = subprocess.check_output(
            ["git", "hash-object", "--", str(path.relative_to(root))],
            cwd=root,
            text=True,
        ).strip()
        records[key] = {
            "source_path": str(path.relative_to(root)).replace("\\", "/"),
            "source_sha256": raw_sha,
            "git_blob": blob,
            "checks": {
                "source_sha256": raw_sha == contract["source_sha256"],
                "git_blob": blob == contract["git_blob"],
            },
        }
    return {
        "records": records,
        "all_checks_pass": all(
            all(record["checks"].values()) for record in records.values()
        ),
    }


def evaluate_controls(
    config: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate only generated controls; no external path is accepted."""

    validate_config(config)
    populations = _population_bundle(config)
    controls = generate_controls(config)
    gates = config["automatic_gates"]
    style = populations["style"]
    residual = populations["residual"]
    analytic_style = style
    analytic_residual = residual
    records = []
    effects: dict[str, np.ndarray] = {}
    residual_signatures: dict[str, np.ndarray] = {}
    output_hashes: dict[str, str] = {}
    matched_basic_reference_checked = False
    rows = config["control_generation"]["controls"]
    for row in rows:
        control_id = str(row["id"])
        cube = HaldCube(controls[control_id])
        metrics, safety_checks, signatures = _safety_metrics(
            cube,
            style=style,
            sobol=populations["sobol"],
            gradients=populations["gradients"],
            gates=gates,
        )
        residual_output = cube.apply(residual)
        style_basic, residual_basic = _fit_basic_once(
            analytic_style, signatures["style_output"], analytic_residual
        )
        if control_id == "warm_strength_100":
            reference = matched_basic_output(
                analytic_style, signatures["style_output"]
            )
            if not np.allclose(reference, style_basic, atol=1e-12, rtol=1e-12):
                raise HaldStructuralError("matched-basic pinned API conformance failed")
            matched_basic_reference_checked = True
        residual_signature = _lab(residual_output) - _lab(residual_basic)
        metrics["style_delta_e76_median"] = median_delta_e76(
            analytic_style, signatures["style_output"]
        )
        metrics["joint_basic_residual_delta_e76_median"] = median_delta_e76(
            style_basic, signatures["style_output"]
        )
        style_check = metrics["style_delta_e76_median"] >= float(
            gates["minimum_style_delta_e76_median"]
        )
        non_basic_check = metrics[
            "joint_basic_residual_delta_e76_median"
        ] >= float(gates["minimum_joint_basic_residual_delta_e76_median"])
        structurally_safe = all(safety_checks.values())
        eligible = structurally_safe and style_check and non_basic_check
        role = str(row["expected_role"])
        if role == "exact_reference":
            expectation = structurally_safe and not eligible
        elif role == "structurally_safe_basic_ineligible":
            expectation = structurally_safe and not non_basic_check and not eligible
        elif role.startswith("structurally_safe_non_basic_eligible"):
            expectation = structurally_safe and eligible
        elif role == "reject":
            intended = [str(value) for value in row["intended_rejection_checks"]]
            expectation = (not structurally_safe) and any(
                not safety_checks[name] for name in intended
            )
        else:
            raise HaldStructuralError(f"unknown expected role: {role}")
        hashes = {
            "source_table_array_sha256": array_sha256(controls[control_id]),
            "style_output_array_sha256": array_sha256(signatures["style_output"]),
            "residual_signature_array_sha256": array_sha256(residual_signature),
            "jacobian_signature_array_sha256": array_sha256(signatures["jacobian"]),
            "gradient_signature_array_sha256": array_sha256(signatures["gradient"]),
            "native_difference_signature_array_sha256": array_sha256(
                signatures["native_difference"]
            ),
        }
        output_hashes[control_id] = hashes["style_output_array_sha256"]
        effects[control_id] = residual_output - analytic_residual
        residual_signatures[control_id] = residual_signature
        records.append(
            {
                "control_id": control_id,
                "kind": str(row["kind"]),
                "expected_role": role,
                "hashes": hashes,
                "metrics": metrics,
                "safety_checks": safety_checks,
                "structurally_safe": structurally_safe,
                "style_check": style_check,
                "non_basic_check": non_basic_check,
                "eligible": eligible,
                "expectation_pass": expectation,
            }
        )
    if not matched_basic_reference_checked:
        raise HaldStructuralError("matched-basic API conformance was not executed")
    eligible_ids = [row["control_id"] for row in records if row["eligible"]]
    expected_eligible = [
        "warm_strength_050",
        "warm_strength_075",
        "warm_strength_100",
        "warm_strength_100_duplicate",
        "cool_strength_100",
    ]
    if eligible_ids != expected_eligible:
        raise HaldStructuralError("eligible control population differs")
    expected_components = sorted(
        [
            sorted(config["control_expectations"]["warm_strength_component_exact"]),
            sorted(config["control_expectations"]["cool_component_exact"]),
        ],
        key=lambda values: values[0],
    )
    permutation_orders = {
        "config_order": eligible_ids,
        "reverse_order": list(reversed(eligible_ids)),
        "sha256_id_order": sorted(
            eligible_ids,
            key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
        ),
    }
    components_by_order = {}
    pairwise_by_order = {}
    for name, order in permutation_orders.items():
        components, pairwise = _components(
            order,
            output_hashes,
            effects,
            config["equivalence"]["strength"],
        )
        components_by_order[name] = components
        pairwise_by_order[name] = pairwise
    components_pass = all(
        value == expected_components for value in components_by_order.values()
    )
    novelty_rows = []
    novelty_pass = True
    minimum_novelty = float(
        config["equivalence"]["novelty"]["minimum_distinct_distance"]
    )
    for left, right in config["equivalence"]["novelty"]["required_distinct_pairs"]:
        distance = float(
            np.median(
                np.linalg.norm(
                    residual_signatures[left] - residual_signatures[right],
                    axis=-1,
                )
            )
        )
        passed = distance >= minimum_novelty
        novelty_pass = novelty_pass and passed
        novelty_rows.append(
            {
                "left": left,
                "right": right,
                "residual_signature_delta_e76_median": distance,
                "check": passed,
            }
        )
    hald = _hald_conformance(config, populations)
    jacobian = _jacobian_conformance(config, populations)
    rgb16 = _rgb16_conformance(config)
    sources = _source_contract_checks(root, config)
    expectation_pass = all(row["expectation_pass"] for row in records)
    checks = {
        "independent_source_hashes": sources["all_checks_pass"],
        "legal_hald_conformance": hald["all_checks_pass"],
        "analytic_jacobian_conformance": jacobian["all_checks_pass"],
        "rgb16_tiff_conformance": rgb16["all_checks_pass"],
        "all_control_expectations": expectation_pass,
        "eligible_population_exact": eligible_ids == expected_eligible,
        "components_permutation_invariant": components_pass,
        "post_collapse_warm_cool_novelty": novelty_pass,
        "matched_basic_api_conformance": matched_basic_reference_checked,
    }
    population_hashes = {
        "style_basic_probe_sha256": array_sha256(populations["style"]),
        "residual_novelty_probe_sha256": array_sha256(populations["residual"]),
        "sobol_probe_sha256": array_sha256(populations["sobol"]),
        "gradient_population_sha256": array_sha256(populations["gradients"]),
        "finite_difference_probe_sha256": array_sha256(
            populations["finite_difference"]
        ),
        "native_geometry_sha256": canonical_sha256(
            config["generated_hald_conformance"]["legal_geometries"]
        ),
    }
    manifest_body = {
        "population_hashes": population_hashes,
        "control_source_tables": [
            {
                "control_id": row["control_id"],
                "source_table_array_sha256": row["hashes"][
                    "source_table_array_sha256"
                ],
            }
            for row in records
        ],
        "source_contracts": sources,
        "hald_conformance": hald,
        "jacobian_conformance": jacobian,
        "rgb16_tiff_conformance": rgb16,
    }
    report_body = {
        "records": records,
        "eligible_control_ids": eligible_ids,
        "equivalence": {
            "expected_components": expected_components,
            "components_by_order": components_by_order,
            "config_order_pairwise": pairwise_by_order["config_order"],
            "novelty": novelty_rows,
        },
        "checks": checks,
        "all_control_expectations_pass": all(checks.values()),
    }
    return manifest_body, report_body


def _runtime_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": importlib.metadata.version("numpy"),
        "Pillow": importlib.metadata.version("Pillow"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit-image": importlib.metadata.version("scikit-image"),
        "tifffile": importlib.metadata.version("tifffile"),
    }


def _tracked_clean(root: Path) -> bool:
    return (
        subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode == 0
        and subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode
        == 0
    )


def _parent_binding(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    parent_spec = config["parent_evidence"]
    parent_path = root / str(parent_spec["decision_path"])
    parent_raw = parent_path.read_bytes()
    parent_sha = hashlib.sha256(parent_raw).hexdigest()
    parent = strict_json_loads(parent_raw)
    if parent_sha != parent_spec["decision_sha256"]:
        raise HaldStructuralError("AJ0B2 decision hash mismatch")
    if parent["decision"] != parent_spec["required_decision"]:
        raise HaldStructuralError("AJ0B2 decision state mismatch")
    if parent["archive"]["sha256"] != parent_spec["archive_sha256"]:
        raise HaldStructuralError("AJ0B2 archive identity mismatch")
    if (
        parent["inventory"]["primary_paths_sha256"]
        != parent_spec["primary_paths_sha256"]
    ):
        raise HaldStructuralError("AJ0B2 primary identity mismatch")
    close_path = root / str(config["supersedes"]["close_decision_path"])
    close_raw = close_path.read_bytes()
    close_sha = hashlib.sha256(close_raw).hexdigest()
    close = strict_json_loads(close_raw)
    if close_sha != EXPECTED_V1_CLOSE_SHA256:
        raise HaldStructuralError("C0 v1 close decision hash mismatch")
    if close["decision"] != config["supersedes"]["required_close_decision"]:
        raise HaldStructuralError("C0 v1 close state mismatch")
    if any(
        int(close["execution"][key]) != 0
        for key in (
            "formal_process_runs",
            "archive_bytes_read",
            "primary_cluts_decoded",
            "primary_candidate_metrics",
            "photograph_renders",
        )
    ):
        raise HaldStructuralError("C0 v1 close contains nonzero access")
    return {
        "aj0b2": {
            "decision_path": str(parent_spec["decision_path"]),
            "decision_sha256": parent_sha,
            "decision": parent["decision"],
            "result_commit": parent_spec["result_commit"],
            "archive_sha256": parent["archive"]["sha256"],
            "primary_paths_sha256": parent["inventory"][
                "primary_paths_sha256"
            ],
        },
        "c0_v1_close": {
            "decision_path": str(config["supersedes"]["close_decision_path"]),
            "decision_sha256": close_sha,
            "decision": close["decision"],
            "zero_access": True,
        },
    }


def run_single_control_process(
    *,
    root: Path,
    config_path: Path,
    output_dir: Path,
    expected_config_raw_sha256: str,
    software_commit: str,
) -> dict[str, str]:
    config, pre_sha, canonical_sha = load_config_snapshot(
        config_path, expected_raw_sha256=expected_config_raw_sha256
    )
    if not _tracked_clean(root):
        raise HaldStructuralError("tracked worktree must be clean for formal run")
    runtime = _runtime_versions()
    if runtime != config["runtime"]["packages"] | {"python": config["runtime"]["python"]}:
        raise HaldStructuralError("runtime versions differ from frozen config")
    parent = _parent_binding(root, config)
    manifest_body, report_body = evaluate_controls(config, root=root)
    _config_after, post_sha, post_canonical = load_config_snapshot(
        config_path, expected_raw_sha256=pre_sha
    )
    if post_sha != pre_sha or post_canonical != canonical_sha:
        raise HaldStructuralError("config changed during control run")
    access = {
        "archive_open_calls": 0,
        "archive_bytes_read": 0,
        "primary_open_calls": 0,
        "primary_cluts_decoded": 0,
        "primary_candidate_metrics": 0,
        "external_root_control_pixels": 0,
        "photograph_renders": 0,
    }
    common = {
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "state": SINGLE_RUN_STATE,
        "software_commit": software_commit,
        "tracked_worktree_clean": True,
        "config": {
            "path": str(config_path.relative_to(root)).replace("\\", "/"),
            "raw_sha256_before": pre_sha,
            "raw_sha256_after": post_sha,
            "canonical_sha256": canonical_sha,
        },
        "parent_evidence": parent,
        "runtime": runtime,
        "ordered_control_ids_sha256": config["control_generation"][
            "ordered_control_ids_sha256"
        ],
        "access_ledger": access,
        "primary_evaluation_performed": False,
        "photograph_rendered": False,
        "visual_review_allowed": False,
        "evidence_promoted": False,
        "c1_contract_design_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    manifest = {"schema_version": MANIFEST_SCHEMA, **common, **manifest_body}
    report = {"schema_version": REPORT_SCHEMA, **common, **report_body}
    validate_manifest(manifest, config)
    validate_report(report, config)
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.json"
    manifest_sha = write_canonical_json(manifest_path, manifest)
    report_sha = write_canonical_json(report_path, report)
    return {"manifest_sha256": manifest_sha, "report_sha256": report_sha}


def _expect_keys(
    value: Mapping[str, Any], expected: Iterable[str], *, label: str
) -> None:
    observed = set(value)
    wanted = set(expected)
    if observed != wanted:
        raise HaldStructuralError(
            f"{label} keys differ; missing={sorted(wanted-observed)}, "
            f"extra={sorted(observed-wanted)}"
        )


def _validate_common(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> set[str]:
    common_keys = {
        "experiment_id",
        "node",
        "state",
        "software_commit",
        "tracked_worktree_clean",
        "config",
        "parent_evidence",
        "runtime",
        "ordered_control_ids_sha256",
        "access_ledger",
        "primary_evaluation_performed",
        "photograph_rendered",
        "visual_review_allowed",
        "evidence_promoted",
        "c1_contract_design_allowed",
        "claim_ceiling",
    }
    if value["experiment_id"] != config["experiment_id"] or value["node"] != config["node"]:
        raise HaldStructuralError("evidence experiment identity mismatch")
    if value["state"] != SINGLE_RUN_STATE:
        raise HaldStructuralError("single-run state may not promote")
    if not isinstance(value["software_commit"], str) or len(value["software_commit"]) != 40:
        raise HaldStructuralError("invalid software commit")
    for key in (
        "tracked_worktree_clean",
        "primary_evaluation_performed",
        "photograph_rendered",
        "visual_review_allowed",
        "evidence_promoted",
        "c1_contract_design_allowed",
    ):
        if not isinstance(value[key], bool):
            raise HaldStructuralError(f"evidence Boolean has wrong type: {key}")
    if value["tracked_worktree_clean"] is not True or any(
        value[key] is not False
        for key in (
            "primary_evaluation_performed",
            "photograph_rendered",
            "visual_review_allowed",
            "evidence_promoted",
            "c1_contract_design_allowed",
        )
    ):
        raise HaldStructuralError("single-run evidence exceeds claim boundary")
    _expect_keys(
        value["config"],
        {"path", "raw_sha256_before", "raw_sha256_after", "canonical_sha256"},
        label="config binding",
    )
    if value["config"]["raw_sha256_before"] != value["config"]["raw_sha256_after"]:
        raise HaldStructuralError("config TOCTOU binding failed")
    if value["config"]["canonical_sha256"] != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise HaldStructuralError("canonical config identity mismatch")
    _expect_keys(
        value["parent_evidence"],
        {"aj0b2", "c0_v1_close"},
        label="parent evidence",
    )
    _expect_keys(
        value["parent_evidence"]["aj0b2"],
        {
            "decision_path",
            "decision_sha256",
            "decision",
            "result_commit",
            "archive_sha256",
            "primary_paths_sha256",
        },
        label="AJ0B2 parent",
    )
    _expect_keys(
        value["parent_evidence"]["c0_v1_close"],
        {"decision_path", "decision_sha256", "decision", "zero_access"},
        label="C0 v1 close parent",
    )
    parent = value["parent_evidence"]["aj0b2"]
    parent_spec = config["parent_evidence"]
    expected_parent = {
        "decision_path": parent_spec["decision_path"],
        "decision_sha256": parent_spec["decision_sha256"],
        "decision": parent_spec["required_decision"],
        "result_commit": parent_spec["result_commit"],
        "archive_sha256": parent_spec["archive_sha256"],
        "primary_paths_sha256": parent_spec["primary_paths_sha256"],
    }
    if parent != expected_parent:
        raise HaldStructuralError("AJ0B2 parent binding mismatch")
    close = value["parent_evidence"]["c0_v1_close"]
    expected_close = {
        "decision_path": config["supersedes"]["close_decision_path"],
        "decision_sha256": EXPECTED_V1_CLOSE_SHA256,
        "decision": config["supersedes"]["required_close_decision"],
        "zero_access": True,
    }
    if close != expected_close:
        raise HaldStructuralError("C0 v1 close binding mismatch")
    if value["parent_evidence"]["c0_v1_close"]["zero_access"] is not True:
        raise HaldStructuralError("C0 v1 close must bind zero access")
    _expect_keys(
        value["runtime"],
        {"python", "numpy", "Pillow", "scipy", "scikit-image", "tifffile"},
        label="runtime",
    )
    expected_runtime = config["runtime"]["packages"] | {
        "python": config["runtime"]["python"]
    }
    if value["runtime"] != expected_runtime:
        raise HaldStructuralError("evidence runtime mismatch")
    _expect_keys(
        value["access_ledger"],
        {
            "archive_open_calls",
            "archive_bytes_read",
            "primary_open_calls",
            "primary_cluts_decoded",
            "primary_candidate_metrics",
            "external_root_control_pixels",
            "photograph_renders",
        },
        label="access ledger",
    )
    for key, count in value["access_ledger"].items():
        if isinstance(count, bool) or not isinstance(count, int) or count != 0:
            raise HaldStructuralError(f"access ledger must be integer zero: {key}")
    if value["ordered_control_ids_sha256"] != config["control_generation"][
        "ordered_control_ids_sha256"
    ]:
        raise HaldStructuralError("ordered control identity mismatch")
    if value["claim_ceiling"] != config["claim_ceiling"]:
        raise HaldStructuralError("claim ceiling mismatch")
    return common_keys


def validate_manifest(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    common = _validate_common(value, config)
    _expect_keys(
        value,
        common
        | {
            "schema_version",
            "population_hashes",
            "control_source_tables",
            "source_contracts",
            "hald_conformance",
            "jacobian_conformance",
            "rgb16_tiff_conformance",
        },
        label="manifest",
    )
    if value["schema_version"] != MANIFEST_SCHEMA:
        raise HaldStructuralError("manifest schema mismatch")
    if len(value["control_source_tables"]) != 16:
        raise HaldStructuralError("manifest control count mismatch")
    expected_ids = [
        row["id"] for row in config["control_generation"]["controls"]
    ]
    observed_ids = [
        row["control_id"] for row in value["control_source_tables"]
    ]
    if observed_ids != expected_ids:
        raise HaldStructuralError("manifest control order mismatch")
    for row in value["control_source_tables"]:
        _expect_keys(
            row,
            {"control_id", "source_table_array_sha256"},
            label="manifest control row",
        )
        if row["source_table_array_sha256"] != config[
            "independent_expected_hashes"
        ]["control_source_tables"][row["control_id"]]:
            raise HaldStructuralError("manifest source table hash mismatch")


def validate_report(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    common = _validate_common(value, config)
    _expect_keys(
        value,
        common
        | {
            "schema_version",
            "records",
            "eligible_control_ids",
            "equivalence",
            "checks",
            "all_control_expectations_pass",
        },
        label="report",
    )
    if value["schema_version"] != REPORT_SCHEMA:
        raise HaldStructuralError("report schema mismatch")
    if not isinstance(value["all_control_expectations_pass"], bool):
        raise HaldStructuralError("report pass fact must be Boolean")
    expected_ids = [
        row["id"] for row in config["control_generation"]["controls"]
    ]
    if [row["control_id"] for row in value["records"]] != expected_ids:
        raise HaldStructuralError("report control order mismatch")
    required_hashes = set(
        config["evidence_schemas"]["required_per_control_hashes"]
    )
    for row in value["records"]:
        _expect_keys(
            row,
            {
                "control_id",
                "kind",
                "expected_role",
                "hashes",
                "metrics",
                "safety_checks",
                "structurally_safe",
                "style_check",
                "non_basic_check",
                "eligible",
                "expectation_pass",
            },
            label="report control row",
        )
        _expect_keys(row["hashes"], required_hashes, label="control hashes")
        for key in (
            "structurally_safe",
            "style_check",
            "non_basic_check",
            "eligible",
            "expectation_pass",
        ):
            if not isinstance(row[key], bool):
                raise HaldStructuralError(f"control Boolean type mismatch: {key}")
        for metric_name, metric in row["metrics"].items():
            if metric_name == "finite":
                if not isinstance(metric, bool):
                    raise HaldStructuralError("finite metric must be Boolean")
                continue
            if isinstance(metric, bool) or not isinstance(metric, (int, float)):
                raise HaldStructuralError("control metric must be numeric non-Boolean")
            if isinstance(metric, float) and not math.isfinite(metric):
                raise HaldStructuralError("control metric must be finite")
    canonical_json_bytes(value)


def _load_evidence(
    directory: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    manifest_bytes = (directory / "manifest.json").read_bytes()
    report_bytes = (directory / "report.json").read_bytes()
    manifest = strict_json_loads(manifest_bytes)
    report = strict_json_loads(report_bytes)
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise HaldStructuralError("evidence roots must be objects")
    validate_manifest(manifest, config)
    validate_report(report, config)
    return manifest, report, manifest_bytes, report_bytes


def finalize_repeat_evidence(
    *,
    first_dir: Path,
    second_dir: Path,
    config: Mapping[str, Any],
    config_raw_sha256: str,
    software_commit: str,
    root: Path,
) -> dict[str, Any]:
    first_manifest, first_report, first_manifest_bytes, first_report_bytes = (
        _load_evidence(first_dir, config)
    )
    second_manifest, second_report, second_manifest_bytes, second_report_bytes = (
        _load_evidence(second_dir, config)
    )
    manifests_equal = first_manifest_bytes == second_manifest_bytes
    reports_equal = first_report_bytes == second_report_bytes
    identities_equal = all(
        first_manifest[key] == second_manifest[key]
        and first_report[key] == second_report[key]
        for key in (
            "software_commit",
            "config",
            "parent_evidence",
            "runtime",
            "ordered_control_ids_sha256",
            "access_ledger",
        )
    )
    child_checks = (
        first_report["all_control_expectations_pass"]
        and second_report["all_control_expectations_pass"]
    )
    manifest_body, report_body = evaluate_controls(config, root=root)
    common_keys = (
        "experiment_id",
        "node",
        "state",
        "software_commit",
        "tracked_worktree_clean",
        "config",
        "parent_evidence",
        "runtime",
        "ordered_control_ids_sha256",
        "access_ledger",
        "primary_evaluation_performed",
        "photograph_rendered",
        "visual_review_allowed",
        "evidence_promoted",
        "c1_contract_design_allowed",
        "claim_ceiling",
    )
    expected_common = {key: first_manifest[key] for key in common_keys}
    expected_manifest_bytes = canonical_json_bytes(
        {"schema_version": MANIFEST_SCHEMA, **expected_common, **manifest_body}
    )
    expected_report_bytes = canonical_json_bytes(
        {"schema_version": REPORT_SCHEMA, **expected_common, **report_body}
    )
    strict_reconstruction = (
        first_manifest_bytes == expected_manifest_bytes
        and second_manifest_bytes == expected_manifest_bytes
        and first_report_bytes == expected_report_bytes
        and second_report_bytes == expected_report_bytes
    )
    automatic_pass = (
        manifests_equal
        and reports_equal
        and identities_equal
        and child_checks
        and strict_reconstruction
        and first_manifest["software_commit"] == software_commit
        and first_manifest["config"]["raw_sha256_before"] == config_raw_sha256
    )
    decision = {
        "schema_version": REPEAT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "decision": (
            "controls_only_conformance_pass"
            if automatic_pass
            else "close_controls_or_repeat_failure"
        ),
        "software_commit": software_commit,
        "config": {
            "raw_sha256": config_raw_sha256,
            "canonical_sha256": EXPECTED_CONFIG_CANONICAL_SHA256,
        },
        "child_evidence": {
            "manifest_sha256": hashlib.sha256(first_manifest_bytes).hexdigest(),
            "report_sha256": hashlib.sha256(first_report_bytes).hexdigest(),
            "manifest_bytes_exact": manifests_equal,
            "report_bytes_exact": reports_equal,
            "identities_exact": identities_equal,
            "both_child_control_expectations_pass": child_checks,
            "strict_reconstruction_exact": strict_reconstruction,
        },
        "access_ledger": first_manifest["access_ledger"],
        "automatic_pass": bool(automatic_pass),
        "c1_contract_design_allowed": bool(automatic_pass),
        "primary_evaluation_allowed": False,
        "photograph_rendered": False,
        "visual_review_allowed": False,
        "evidence_promoted": bool(automatic_pass),
        "claim_ceiling": config["claim_ceiling"],
    }
    validate_repeat_decision(decision, config)
    return decision


def validate_repeat_decision(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _expect_keys(
        value,
        {
            "schema_version",
            "experiment_id",
            "node",
            "decision",
            "software_commit",
            "config",
            "child_evidence",
            "access_ledger",
            "automatic_pass",
            "c1_contract_design_allowed",
            "primary_evaluation_allowed",
            "photograph_rendered",
            "visual_review_allowed",
            "evidence_promoted",
            "claim_ceiling",
        },
        label="repeat decision",
    )
    if value["schema_version"] != REPEAT_SCHEMA:
        raise HaldStructuralError("repeat schema mismatch")
    if value["experiment_id"] != config["experiment_id"] or value["node"] != config["node"]:
        raise HaldStructuralError("repeat experiment identity mismatch")
    if (
        not isinstance(value["software_commit"], str)
        or len(value["software_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in value["software_commit"])
    ):
        raise HaldStructuralError("invalid repeat software commit")
    _expect_keys(
        value["config"],
        {"raw_sha256", "canonical_sha256"},
        label="repeat config",
    )
    _expect_keys(
        value["child_evidence"],
        {
            "manifest_sha256",
            "report_sha256",
            "manifest_bytes_exact",
            "report_bytes_exact",
            "identities_exact",
            "both_child_control_expectations_pass",
            "strict_reconstruction_exact",
        },
        label="repeat child evidence",
    )
    for key in (
        "manifest_bytes_exact",
        "report_bytes_exact",
        "identities_exact",
        "both_child_control_expectations_pass",
        "strict_reconstruction_exact",
    ):
        if not isinstance(value["child_evidence"][key], bool):
            raise HaldStructuralError(f"repeat child Boolean mismatch: {key}")
    for key in ("manifest_sha256", "report_sha256"):
        digest = value["child_evidence"][key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise HaldStructuralError(f"invalid repeat child digest: {key}")
    for key in (
        "automatic_pass",
        "c1_contract_design_allowed",
        "primary_evaluation_allowed",
        "photograph_rendered",
        "visual_review_allowed",
        "evidence_promoted",
    ):
        if not isinstance(value[key], bool):
            raise HaldStructuralError(f"repeat Boolean type mismatch: {key}")
    if any(
        value[key] is not False
        for key in (
            "primary_evaluation_allowed",
            "photograph_rendered",
            "visual_review_allowed",
        )
    ):
        raise HaldStructuralError("repeat decision exceeds C0B boundary")
    if value["automatic_pass"] != value["c1_contract_design_allowed"]:
        raise HaldStructuralError("C1 design authority must equal repeat pass")
    if value["evidence_promoted"] != value["automatic_pass"]:
        raise HaldStructuralError("repeat promotion fact mismatch")
    expected_decision = (
        "controls_only_conformance_pass"
        if value["automatic_pass"]
        else "close_controls_or_repeat_failure"
    )
    if value["decision"] != expected_decision:
        raise HaldStructuralError("repeat decision label contradicts pass fact")
    raw_config_digest = value["config"]["raw_sha256"]
    if (
        not isinstance(raw_config_digest, str)
        or len(raw_config_digest) != 64
        or any(character not in "0123456789abcdef" for character in raw_config_digest)
        or value["config"]["canonical_sha256"] != EXPECTED_CONFIG_CANONICAL_SHA256
    ):
        raise HaldStructuralError("repeat config binding mismatch")
    expected_access_keys = {
        "archive_open_calls",
        "archive_bytes_read",
        "primary_open_calls",
        "primary_cluts_decoded",
        "primary_candidate_metrics",
        "external_root_control_pixels",
        "photograph_renders",
    }
    _expect_keys(value["access_ledger"], expected_access_keys, label="repeat access ledger")
    for count in value["access_ledger"].values():
        if isinstance(count, bool) or not isinstance(count, int) or count != 0:
            raise HaldStructuralError("repeat decision contains nonzero access")
    if value["claim_ceiling"] != config["claim_ceiling"]:
        raise HaldStructuralError("repeat claim ceiling mismatch")
    canonical_json_bytes(value)
