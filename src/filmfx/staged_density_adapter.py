"""Isolated research adapter for staged density-halation execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compositor import composite_layers
from .layers import FilmLayer
from .staged_density import (
    STAGED_DENSITY_VERSION,
    StagedDensityExecutionMetadata,
    execute_staged_density_halation_default,
)


STAGED_DENSITY_ADAPTER_VERSION = "staged-density-halation-research-adapter-v1"


@dataclass(frozen=True)
class StagedDensityAdapterMetadata:
    """Scalar-only ownership and execution record for the research adapter."""

    version: str
    executor_version: str
    executor: StagedDensityExecutionMetadata
    composite_row_chunk: int
    output_margin: int
    input_bytes: int
    private_layer_bytes: int
    returned_composite_bytes: int
    public_ndarray_count: int
    scratch_bytes: int
    lifecycle: tuple[str, ...]
    renderer_integration_allowed: bool


def _integer(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _readonly_view(value: np.ndarray) -> np.ndarray:
    view = value.view()
    view.setflags(write=False)
    return view


def _composite_rows(
    base_rgb: np.ndarray,
    layer: FilmLayer,
    *,
    row_chunk: int,
    output_margin: int,
) -> np.ndarray:
    if layer.mode != "screen" or layer.rgb is None or layer.alpha is None:
        raise ValueError("staged density executor must return a complete screen layer")
    if layer.rgb.shape != base_rgb.shape:
        raise ValueError("staged density RGB shape does not match base")
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    if alpha.shape != (*base_rgb.shape[:2], 1):
        raise ValueError("staged density alpha shape does not match base")

    output = np.empty_like(base_rgb, dtype=np.float32)
    for y0 in range(0, base_rgb.shape[0], row_chunk):
        y1 = min(y0 + row_chunk, base_rgb.shape[0])
        row_layer = FilmLayer(
            name=layer.name,
            mode="screen",
            rgb=_readonly_view(layer.rgb[y0:y1]),
            alpha=_readonly_view(alpha[y0:y1]),
        )
        rendered = composite_layers(
            _readonly_view(base_rgb[y0:y1]),
            [row_layer],
            output_margin=output_margin,
        )
        if (
            not isinstance(rendered, np.ndarray)
            or rendered.dtype != np.float32
            or rendered.shape != output[y0:y1].shape
            or not np.isfinite(rendered).all()
        ):
            raise ValueError("row compositor returned an invalid float32 RGB result")
        output[y0:y1] = rendered
    return output


def render_staged_density_halation_research(
    base_rgb: np.ndarray,
    *,
    tile_size: int,
    source_row_chunk: int = 64,
    coarse_row_chunk: int = 7,
    composite_row_chunk: int = 64,
    output_margin: int = 0,
) -> tuple[np.ndarray, StagedDensityAdapterMetadata]:
    """Render the fixed staged-density candidate without production integration."""

    resolved_row_chunk = _integer(
        composite_row_chunk, "composite_row_chunk", 1, 2**31 - 1
    )
    resolved_margin = _integer(output_margin, "output_margin", 0, 32)
    layer, executor_metadata = execute_staged_density_halation_default(
        base_rgb,
        tile_size=tile_size,
        source_row_chunk=source_row_chunk,
        coarse_row_chunk=coarse_row_chunk,
        name="staged_density_halation_research",
    )
    output = _composite_rows(
        base_rgb,
        layer,
        row_chunk=resolved_row_chunk,
        output_margin=resolved_margin,
    )
    if executor_metadata.version != STAGED_DENSITY_VERSION:
        raise ValueError("unexpected staged density executor version")
    private_layer_bytes = int(layer.rgb.nbytes + layer.alpha.nbytes)
    metadata = StagedDensityAdapterMetadata(
        version=STAGED_DENSITY_ADAPTER_VERSION,
        executor_version=executor_metadata.version,
        executor=executor_metadata,
        composite_row_chunk=resolved_row_chunk,
        output_margin=resolved_margin,
        input_bytes=int(base_rgb.nbytes),
        private_layer_bytes=private_layer_bytes,
        returned_composite_bytes=int(output.nbytes),
        public_ndarray_count=1,
        scratch_bytes=0,
        lifecycle=(
            "caller_input_retained",
            "private_layer_created",
            "composite_rows_written",
            "private_layer_released_on_return",
            "composite_returned",
        ),
        renderer_integration_allowed=False,
    )
    return output, metadata
