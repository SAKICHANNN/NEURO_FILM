"""Memory-mapped scene-linear NPY row source for research transactions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class ReplayableSceneLinearNpyError(ValueError):
    """Raised when an explicit scene-linear NPY source fails closed."""


class ReplayableSceneLinearNpyRows:
    """Expose bounded rows from one immutable-shape float32 NPY array."""

    def __init__(
        self,
        path: Path,
        *,
        expected_file_sha256: str,
        expected_pixel_sha256: str,
    ) -> None:
        self.path = Path(path).resolve(strict=True)
        if not self.path.is_file():
            raise ReplayableSceneLinearNpyError("NPY source must be a file")
        if not _is_sha(expected_file_sha256) or not _is_sha(expected_pixel_sha256):
            raise ReplayableSceneLinearNpyError("NPY identities must be SHA-256")
        if _sha256_file(self.path) != expected_file_sha256:
            raise ReplayableSceneLinearNpyError("NPY file identity drift")
        pixels = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(pixels, np.memmap)
            or pixels.dtype != np.dtype("<f4")
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or pixels.size == 0
            or not pixels.flags.c_contiguous
        ):
            raise ReplayableSceneLinearNpyError(
                "NPY source must be C-order little-endian float32 HxWx3"
            )
        self._pixels = pixels
        self.expected_file_sha256 = expected_file_sha256
        self.expected_pixel_sha256 = expected_pixel_sha256
        self.height, self.width, _ = pixels.shape

    def rows(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count <= 0 or start + count > self.height:
            raise ReplayableSceneLinearNpyError("NPY row request outside source")
        rows = np.ascontiguousarray(self._pixels[start : start + count])
        if not np.all(np.isfinite(rows)) or np.any(rows < 0) or np.any(rows > 1):
            raise ReplayableSceneLinearNpyError("NPY row values outside [0,1]")
        return rows

    def verify_file_identity(self) -> dict[str, Any]:
        if _sha256_file(self.path) != self.expected_file_sha256:
            raise ReplayableSceneLinearNpyError("NPY file changed during use")
        return {
            "schema": "neuro_film.replayable_scene_linear_npy.v1",
            "path": str(self.path),
            "file_sha256": self.expected_file_sha256,
            "pixel_sha256": self.expected_pixel_sha256,
            "shape": [self.height, self.width, 3],
            "dtype": "float32-little-endian",
            "domain": "scene-linear-relative-exposure",
            "working_space": "linear-srgb-d65",
        }

    def close(self) -> None:
        """Release the memory map before a caller removes or replaces the file."""

        mapping = getattr(self._pixels, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = ["ReplayableSceneLinearNpyError", "ReplayableSceneLinearNpyRows"]
