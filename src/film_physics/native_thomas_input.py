"""Typed relative layer log-exposure input for the native Thomas runtime."""

from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import PhysicalDomain, PhysicalDomainArray

NATIVE_THOMAS_INPUT_SCHEMA = "neuro_film.relative_layer_log_exposure.v1"


@dataclass(frozen=True)
class RelativeLayerLogExposure:
    """Immutable CHW log10 layer exposure with an explicit RGB layer order.

    This is deliberately distinct from ``PhysicalDomainArray`` layer exposure,
    whose values are linear.  Construction never estimates layer exposure from
    a WorkingImage or other display/scene RGB representation.
    """

    values_chw: np.ndarray
    channels: tuple[str, str, str] = ("red", "green", "blue")
    _copy: InitVar[bool] = True

    def __post_init__(self, _copy: bool) -> None:
        source = self.values_chw
        values = np.asarray(source)
        if values.dtype != np.float32:
            raise TypeError("relative layer log exposure must be float32")
        if values.ndim != 3 or values.shape[0] != 3 or values.size == 0:
            raise ValueError("relative layer log exposure must be non-empty CHW RGB")
        if tuple(self.channels) != ("red", "green", "blue"):
            raise ValueError("relative layer log exposure requires red/green/blue order")
        if not np.all(np.isfinite(values)):
            raise ValueError("relative layer log exposure must be finite")
        mapped = isinstance(source, np.memmap)
        if not _copy and (
            not values.flags.c_contiguous
            or (not values.flags.owndata and not mapped)
        ):
            raise ValueError("adopted relative layer log exposure must own contiguous data")
        owned = (
            np.array(values, dtype=np.float32, copy=True, order="C")
            if _copy
            else source
        )
        owned.setflags(write=False)
        object.__setattr__(self, "values_chw", owned)

    @classmethod
    def from_chw(cls, values_chw: np.ndarray) -> RelativeLayerLogExposure:
        """Copy one native-layout source into an immutable typed value."""
        return cls(values_chw)

    @classmethod
    def adopt_chw(cls, values_chw: np.ndarray) -> RelativeLayerLogExposure:
        """Transfer ownership of one contiguous CHW array without a full-frame copy."""
        return cls(values_chw, _copy=False)

    @classmethod
    def map_chw(
        cls,
        path: Path,
        *,
        height: int,
        width: int,
        expected_sha256: str,
    ) -> RelativeLayerLogExposure:
        """Map one exact little-endian float32 CHW file without copying it."""
        source = Path(path)
        expected_bytes = 3 * height * width * np.dtype("<f4").itemsize
        if height <= 0 or width <= 0 or source.stat().st_size != expected_bytes:
            raise ValueError("mapped relative layer log exposure size drift")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1048576), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("mapped relative layer log exposure hash drift")
        mapped = np.memmap(
            source,
            dtype="<f4",
            mode="r",
            shape=(3, height, width),
            order="C",
        )
        return cls(mapped, _copy=False)

    @classmethod
    def from_layer_exposure(
        cls, exposure: PhysicalDomainArray
    ) -> RelativeLayerLogExposure:
        """Convert explicit positive linear layer exposure to log10 ABI input."""
        if not isinstance(exposure, PhysicalDomainArray):
            raise TypeError("exposure must be a PhysicalDomainArray")
        exposure.require(PhysicalDomain.LAYER_EXPOSURE)
        if exposure.channels != (
            "red-sensitive",
            "green-sensitive",
            "blue-sensitive",
        ):
            raise ValueError("layer exposure requires red/green/blue-sensitive order")
        if np.any(exposure.values <= 0.0):
            raise ValueError("layer exposure must be strictly positive before log10")
        height, width, _ = exposure.values.shape
        values_chw = np.empty((3, height, width), dtype=np.float32)
        for channel in range(3):
            np.log10(exposure.values[..., channel], out=values_chw[channel])
        return cls.adopt_chw(values_chw)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": NATIVE_THOMAS_INPUT_SCHEMA,
            "domain": "film-layer-log10-relative-exposure",
            "unit": "log10-relative-layer-exposure",
            "channels": list(self.channels),
            "dtype": self.values_chw.dtype.name,
            "shape_chw": list(self.values_chw.shape),
            "sha256": hashlib.sha256(self.values_chw.tobytes()).hexdigest(),
        }


__all__ = ["NATIVE_THOMAS_INPUT_SCHEMA", "RelativeLayerLogExposure"]
