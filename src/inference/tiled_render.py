"""Deterministic halo-aware execution for explicitly local image operators."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
RowStripeConsumer = Callable[[int, np.ndarray], None]


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
    workers: int = 1,
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
    workers = _integer(workers, "workers", minimum=1)

    windows = plan_tile_windows(image.shape[0], image.shape[1], tile_size=tile_size, halo=halo)
    output = np.empty_like(image)
    max_height = 0
    max_width = 0

    def render_one(index: int, window: TileWindow) -> tuple[int, TileWindow, np.ndarray]:
        tile = image[
            window.expanded_y0 : window.expanded_y1,
            window.expanded_x0 : window.expanded_x1,
            :,
        ].view()
        tile.setflags(write=False)
        rendered = operator(tile, window)
        return index, window, rendered

    def stitch(rendered_row: tuple[int, TileWindow, np.ndarray]) -> None:
        nonlocal max_height, max_width
        index, window, rendered = rendered_row
        tile_shape = (
            window.expanded_y1 - window.expanded_y0,
            window.expanded_x1 - window.expanded_x0,
            image.shape[2],
        )
        if not isinstance(rendered, np.ndarray) or rendered.ndim != 3:
            raise TiledRenderError(f"operator output {index} must be an HWC numpy array")
        if rendered.shape != tile_shape:
            raise TiledRenderError(
                f"operator output {index} shape {rendered.shape} does not match tile shape {tile_shape}"
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
        max_height = max(max_height, tile_shape[0])
        max_width = max(max_width, tile_shape[1])

    if workers == 1:
        for index, window in enumerate(windows):
            stitch(render_one(index, window))
    else:
        worker_count = min(workers, len(windows))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for start in range(0, len(windows), worker_count):
                batch = tuple(
                    enumerate(windows[start : start + worker_count], start=start)
                )
                futures = [
                    executor.submit(render_one, index, window) for index, window in batch
                ]
                for future in futures:
                    stitch(future.result())

    metadata = TiledExecutionMetadata(
        input_shape=tuple(int(value) for value in image.shape),
        tile_size=tile_size,
        halo=halo,
        tile_count=len(windows),
        max_expanded_shape=(max_height, max_width, image.shape[2]),
    )
    return output, metadata


def stream_tiled_local_operator_rows(
    image: np.ndarray,
    operator: LocalTileOperator,
    consumer: RowStripeConsumer,
    *,
    tile_size: int,
    halo: int,
    workers: int = 1,
) -> TiledExecutionMetadata:
    """Execute row-major tiles and emit one bounded full-width core stripe.

    This primitive preserves the exact tile windows and stitch order of
    :func:`execute_tiled_local_operator` without allocating its
    full-resolution output array. The consumer must synchronously consume each
    read-only stripe before returning.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise TiledRenderError("image must be an HWC numpy array")
    if any(size <= 0 for size in image.shape):
        raise TiledRenderError("image dimensions must be non-zero")
    if not np.issubdtype(image.dtype, np.floating):
        raise TiledRenderError("image dtype must be floating point")
    if not np.isfinite(image).all():
        raise TiledRenderError("image must contain only finite values")
    if not callable(operator) or not callable(consumer):
        raise TiledRenderError("operator and consumer must be callable")
    workers = _integer(workers, "workers", minimum=1)

    windows = plan_tile_windows(
        image.shape[0], image.shape[1], tile_size=tile_size, halo=halo
    )
    max_height = 0
    max_width = 0
    cursor = 0
    while cursor < len(windows):
        core_y0 = windows[cursor].core_y0
        core_y1 = windows[cursor].core_y1
        stripe = np.empty(
            (core_y1 - core_y0, image.shape[1], image.shape[2]),
            dtype=image.dtype,
        )
        row_windows: list[tuple[int, TileWindow]] = []
        while cursor < len(windows) and windows[cursor].core_y0 == core_y0:
            row_windows.append((cursor, windows[cursor]))
            cursor += 1

        def render_one(
            indexed_window: tuple[int, TileWindow],
        ) -> tuple[int, TileWindow, np.ndarray]:
            index, window = indexed_window
            tile = image[
                window.expanded_y0 : window.expanded_y1,
                window.expanded_x0 : window.expanded_x1,
                :,
            ].view()
            tile.setflags(write=False)
            rendered = operator(tile, window)
            tile_shape = (
                window.expanded_y1 - window.expanded_y0,
                window.expanded_x1 - window.expanded_x0,
                image.shape[2],
            )
            if not isinstance(rendered, np.ndarray) or rendered.ndim != 3:
                raise TiledRenderError(
                    f"operator output {index} must be an HWC numpy array"
                )
            if rendered.shape != tile_shape:
                raise TiledRenderError(
                    f"operator output {index} shape {rendered.shape} does not "
                    f"match tile shape {tile_shape}"
                )
            if rendered.dtype != image.dtype:
                raise TiledRenderError(
                    f"operator output {index} dtype {rendered.dtype} does not "
                    f"match input dtype {image.dtype}"
                )
            if not np.isfinite(rendered).all():
                raise TiledRenderError(
                    f"operator output {index} contains non-finite values"
                )
            return index, window, rendered

        if workers == 1:
            rendered_tiles = [render_one(item) for item in row_windows]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(row_windows))) as executor:
                rendered_tiles = list(executor.map(render_one, row_windows))
        for _index, window, rendered in rendered_tiles:
            tile_shape = (
                window.expanded_y1 - window.expanded_y0,
                window.expanded_x1 - window.expanded_x0,
                image.shape[2],
            )
            tile_y0, tile_y1, tile_x0, tile_x1 = window.core_offsets
            stripe[
                :,
                window.core_x0 : window.core_x1,
                :,
            ] = rendered[tile_y0:tile_y1, tile_x0:tile_x1, :]
            max_height = max(max_height, tile_shape[0])
            max_width = max(max_width, tile_shape[1])
        stripe.setflags(write=False)
        consumer(core_y0, stripe)

    return TiledExecutionMetadata(
        input_shape=tuple(int(value) for value in image.shape),
        tile_size=tile_size,
        halo=halo,
        tile_count=len(windows),
        max_expanded_shape=(max_height, max_width, image.shape[2]),
    )
