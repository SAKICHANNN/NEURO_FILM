"""Original-coordinate gradient windows for staged film-effect fields."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


GRADIENT_WINDOW_VERSION = "coordinate-gradient-window-v1"
ScalarWindowReader = Callable[[int, int, int, int], np.ndarray]


@dataclass(frozen=True)
class GradientWindowMetadata:
    version: str
    source_shape: tuple[int, int]
    core_bounds: tuple[int, int, int, int]
    expanded_bounds: tuple[int, int, int, int]
    core_shape: tuple[int, int]
    expanded_shape: tuple[int, int]
    halo: int
    expanded_input_bytes: int
    output_bytes: int


def _source_shape(value: object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("source_shape must be an (H, W) tuple")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 2 for item in value):
        raise ValueError("source_shape dimensions must be integers >= 2")
    return value


def _bounds(
    source_shape: tuple[int, int],
    y0: object,
    y1: object,
    x0: object,
    x1: object,
) -> tuple[int, int, int, int]:
    values = (y0, y1, x0, x1)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("window bounds must be integers")
    height, width = source_shape
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("window bounds are outside source_shape or unordered")
    return y0, y1, x0, x1


def array_scalar_window_reader(field: np.ndarray) -> ScalarWindowReader:
    """Return a strict global-coordinate reader over one finite float32 field."""

    if not isinstance(field, np.ndarray) or field.ndim != 2:
        raise ValueError("field must be a 2-D numpy array")
    _source_shape(tuple(int(item) for item in field.shape))
    if field.dtype != np.float32 or not np.isfinite(field).all():
        raise ValueError("field must be finite float32")

    def read(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        resolved = _bounds(tuple(int(item) for item in field.shape), y0, y1, x0, x1)
        region = field[resolved[0] : resolved[1], resolved[2] : resolved[3]].view()
        region.setflags(write=False)
        return region

    return read


def coordinate_gradient_window(
    reader: ScalarWindowReader,
    source_shape: tuple[int, int],
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[np.ndarray, np.ndarray, GradientWindowMetadata]:
    """Reproduce full-frame NumPy gradient values for one global core window."""

    if not callable(reader):
        raise ValueError("reader must be callable")
    shape = _source_shape(source_shape)
    core = _bounds(shape, y0, y1, x0, x1)
    expanded = (
        max(0, core[0] - 1),
        min(shape[0], core[1] + 1),
        max(0, core[2] - 1),
        min(shape[1], core[3] + 1),
    )
    try:
        field = reader(*expanded)
    except Exception as error:
        raise ValueError("reader failed for the requested expanded bounds") from error
    expected_shape = (expanded[1] - expanded[0], expanded[3] - expanded[2])
    if not isinstance(field, np.ndarray) or field.ndim != 2:
        raise ValueError("reader must return a 2-D numpy array")
    if field.shape != expected_shape:
        raise ValueError(
            f"reader returned shape {field.shape}, expected expanded shape {expected_shape}"
        )
    if field.dtype != np.float32 or not np.isfinite(field).all():
        raise ValueError("reader must return finite float32")

    readonly = field.view()
    readonly.setflags(write=False)
    gradient_y, gradient_x = np.gradient(readonly)
    local_y0 = core[0] - expanded[0]
    local_y1 = local_y0 + (core[1] - core[0])
    local_x0 = core[2] - expanded[2]
    local_x1 = local_x0 + (core[3] - core[2])
    gradient_y = np.ascontiguousarray(
        gradient_y[local_y0:local_y1, local_x0:local_x1], dtype=np.float32
    )
    gradient_x = np.ascontiguousarray(
        gradient_x[local_y0:local_y1, local_x0:local_x1], dtype=np.float32
    )
    gradient_y.setflags(write=False)
    gradient_x.setflags(write=False)
    metadata = GradientWindowMetadata(
        version=GRADIENT_WINDOW_VERSION,
        source_shape=shape,
        core_bounds=core,
        expanded_bounds=expanded,
        core_shape=(core[1] - core[0], core[3] - core[2]),
        expanded_shape=expected_shape,
        halo=1,
        expanded_input_bytes=int(field.nbytes),
        output_bytes=int(gradient_y.nbytes + gradient_x.nbytes),
    )
    return gradient_y, gradient_x, metadata


def coordinate_gradient_window_from_array(
    field: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[np.ndarray, np.ndarray, GradientWindowMetadata]:
    """Convenience wrapper retaining the same strict reader contract."""

    if not isinstance(field, np.ndarray) or field.ndim != 2:
        raise ValueError("field must be a 2-D numpy array")
    shape = tuple(int(item) for item in field.shape)
    return coordinate_gradient_window(
        array_scalar_window_reader(field),
        shape,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
    )
