"""Stream combined spatial physics into a hash-bound native CHW input file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import PhysicalDomain, PhysicalDomainArray
from .native_thomas_input import RelativeLayerLogExposure
from .native_thomas_spatial_chain import (
    NativeThomasSpatialChain,
    iter_native_thomas_spatial_fft_row_cores,
)


def stream_fft_spatial_log_exposure_chw(
    exposure: PhysicalDomainArray,
    chain: NativeThomasSpatialChain,
    *,
    destination: Path,
    tile_rows: int,
) -> dict[str, Any]:
    """Create one complete CHW log-exposure file from ordered spatial cores."""
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.channels != (
        "red-sensitive",
        "green-sensitive",
        "blue-sensitive",
    ):
        raise ValueError("streamed native input requires sensitive-layer order")
    path = Path(destination)
    if not path.parent.is_dir() or path.exists():
        raise ValueError("streamed native input destination must be absent")
    height, width, _ = exposure.values.shape
    mapped: np.memmap | None = None
    completed = False
    try:
        mapped = np.memmap(
            path,
            dtype="<f4",
            mode="w+",
            shape=(3, height, width),
            order="C",
        )
        next_row = 0
        for y0, y1, core in iter_native_thomas_spatial_fft_row_cores(
            exposure, chain, tile_rows=tile_rows, order="forward"
        ):
            if y0 != next_row or core.shape != (y1 - y0, width, 3):
                raise RuntimeError("streamed native input coverage drift")
            if np.any(core <= 0.0):
                raise RuntimeError("streamed native input requires positive exposure")
            for channel in range(3):
                np.log10(core[..., channel], out=mapped[channel, y0:y1])
            next_row = y1
        if next_row != height:
            raise RuntimeError("streamed native input incomplete")
        mapped.flush()
        del mapped
        mapped = None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1048576), b""):
                digest.update(chunk)
        completed = True
        return {
            "schema": "neuro_film.streamed_relative_layer_log_exposure.v1",
            "path": str(path.resolve()),
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
            "shape_chw": [3, height, width],
            "dtype": "float32-little-endian",
            "tile_rows": tile_rows,
        }
    finally:
        if mapped is not None:
            mapped.flush()
            del mapped
        if not completed:
            path.unlink(missing_ok=True)


def map_streamed_log_exposure(receipt: dict[str, Any]) -> RelativeLayerLogExposure:
    """Open a completed streamed receipt through the typed native boundary."""
    if receipt.get("schema") != "neuro_film.streamed_relative_layer_log_exposure.v1":
        raise ValueError("streamed native input receipt schema drift")
    shape = receipt.get("shape_chw")
    if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 3:
        raise ValueError("streamed native input receipt shape drift")
    return RelativeLayerLogExposure.map_chw(
        Path(receipt["path"]),
        height=int(shape[1]),
        width=int(shape[2]),
        expected_sha256=receipt["sha256"],
    )


__all__ = ["map_streamed_log_exposure", "stream_fft_spatial_log_exposure_chw"]
