"""Deterministic halo-aware execution for explicitly local image operators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class TiledRenderError(ValueError):
    """Raised when a tiled-render contract is invalid or violated."""


@dataclass(frozen=True)
class TileWindow:
    """One output core and its clipped image-space halo window."""

    core_y0: int
    core_y1: int
    core_x0: int
    core_x1: int
    expanded_y0: int
    expanded_y1: int
    expanded_x0: int
    expanded_x1: int

    @property
    def core_shape(self) -> tuple[int, int]:
        return self.core_y1 - self.core_y0, self.core_x1 - self.core_x0

    @property
    def expanded_shape(self) -> tuple[int, int]:
        return self.expanded_y1 - self.expanded_y0, self.expanded_x1 - self.expanded_x0

    @property
    def core_offsets(self) -> tuple[int, int, int, int]:
        y0 = self.core_y0 - self.expanded_y0
        x0 = self.core_x0 - self.expanded_x0
        return y0, y0 + self.core_shape[0], x0, x0 + self.core_shape[1]


@dataclass(frozen=True)
class TiledExecutionMetadata:
    """Auditable bounds and topology for a completed tiled execution."""

    input_shape: tuple[int, int, int]
    tile_size: int
    halo: int
    tile_count: int
    max_expanded_shape: tuple[int, int, int]


LocalTileOperator = Callable[[np.ndarray, TileWindow], np.ndarray]


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TiledRenderError(f"{label} must be an integer >= {minimum}")
    return value


def plan_tile_windows(height: int, width: int, *, tile_size: int, halo: int) -> tuple[TileWindow, ...]:
    """Plan deterministic row-major core windows with clipped image-space halos."""

    height = _integer(height, "height", minimum=1)
    width = _integer(width, "width", minimum=1)
    tile_size = _integer(tile_size, "tile_size", minimum=1)
    halo = _integer(halo, "halo", minimum=0)

    windows: list[TileWindow] = []
    for core_y0 in range(0, height, tile_size):
        core_y1 = min(core_y0 + tile_size, height)
        for core_x0 in range(0, width, tile_size):
            core_x1 = min(core_x0 + tile_size, width)
            windows.append(
                TileWindow(
                    core_y0=core_y0,
                    core_y1=core_y1,
                    core_x0=core_x0,
                    core_x1=core_x1,
                    expanded_y0=max(0, core_y0 - halo),
                    expanded_y1=min(height, core_y1 + halo),
                    expanded_x0=max(0, core_x0 - halo),
                    expanded_x1=min(width, core_x1 + halo),
                )
            )
    return tuple(windows)


def execute_tiled_local_operator(
    image: np.ndarray,
    operator: LocalTileOperator,
    *,
    tile_size: int,
    halo: int,
) -> tuple[np.ndarray, TiledExecutionMetadata]:
    """Execute a declared-local operator and stitch only each tile's core.

    The caller is responsible for supplying a halo that covers the operator's
    finite spatial support. Operators with global statistics, normalization or
    coordinate-dependent randomness require a separate context contract.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise TiledRenderError("image must be an HWC numpy array")
    if any(size <= 0 for size in image.shape):
        raise TiledRenderError("image dimensions must be non-zero")
    if not np.issubdtype(image.dtype, np.floating):
        raise TiledRenderError("image dtype must be floating point")
    if not np.isfinite(image).all():
        raise TiledRenderError("image must contain only finite values")
    if not callable(operator):
        raise TiledRenderError("operator must be callable")

    windows = plan_tile_windows(image.shape[0], image.shape[1], tile_size=tile_size, halo=halo)
    output = np.empty_like(image)
    max_height = 0
    max_width = 0

    for index, window in enumerate(windows):
        tile = image[
            window.expanded_y0 : window.expanded_y1,
            window.expanded_x0 : window.expanded_x1,
            :,
        ].view()
        tile.setflags(write=False)
        rendered = operator(tile, window)
        if not isinstance(rendered, np.ndarray) or rendered.ndim != 3:
            raise TiledRenderError(f"operator output {index} must be an HWC numpy array")
        if rendered.shape != tile.shape:
            raise TiledRenderError(
                f"operator output {index} shape {rendered.shape} does not match tile shape {tile.shape}"
            )
        if rendered.dtype != image.dtype:
            raise TiledRenderError(
                f"operator output {index} dtype {rendered.dtype} does not match input dtype {image.dtype}"
            )
        if not np.isfinite(rendered).all():
            raise TiledRenderError(f"operator output {index} contains non-finite values")

        tile_y0, tile_y1, tile_x0, tile_x1 = window.core_offsets
        output[
            window.core_y0 : window.core_y1,
            window.core_x0 : window.core_x1,
            :,
        ] = rendered[tile_y0:tile_y1, tile_x0:tile_x1, :]
        max_height = max(max_height, tile.shape[0])
        max_width = max(max_width, tile.shape[1])

    metadata = TiledExecutionMetadata(
        input_shape=tuple(int(value) for value in image.shape),
        tile_size=tile_size,
        halo=halo,
        tile_count=len(windows),
        max_expanded_shape=(max_height, max_width, image.shape[2]),
    )
    return output, metadata
