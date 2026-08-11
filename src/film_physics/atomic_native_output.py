"""Atomic create-only publication for native byte-stream outputs."""

from __future__ import annotations

import ctypes
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Self

import numpy as np

from .native_gauge_profile import NativeGaugeProfileF32V1
from .native_granularity_amplitude import NativeGranularityAmplitudeProfileV1
from .native_thomas_export_profile import (
    reconstruct_native_thomas_export_profile,
    validate_native_thomas_export_profile,
)
from .native_thomas_field import NativeThomasFieldProfileV1

NativeByteSink = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
)


class AtomicNativeOutputError(RuntimeError):
    """Raised when a native byte stream cannot be published atomically."""


class AtomicNativeOutputSink:
    """Write native chunks to a sibling temporary and publish without overwrite."""

    def __init__(self, path: Path, *, maximum_bytes: int) -> None:
        if (
            isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or maximum_bytes <= 0
            or not path.is_absolute()
            or not path.parent.is_dir()
            or path.exists()
        ):
            raise AtomicNativeOutputError("atomic native output preflight failed")
        self.path = path
        self.maximum_bytes = maximum_bytes
        self.temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.stage"
        self._handle = self.temporary.open("xb")
        self._digest = hashlib.sha256()
        self._bytes_written = 0
        self._closed = False

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    def write(self, payload: bytes) -> bool:
        if self._closed:
            raise AtomicNativeOutputError("atomic native output sink is closed")
        if not isinstance(payload, bytes) or not payload:
            raise AtomicNativeOutputError("native output chunk must be non-empty bytes")
        if self._bytes_written + len(payload) > self.maximum_bytes:
            return False
        try:
            self._handle.write(payload)
        except OSError:
            return False
        self._digest.update(payload)
        self._bytes_written += len(payload)
        return True

    def finish(self) -> dict[str, int | str]:
        if self._closed or self._bytes_written == 0:
            raise AtomicNativeOutputError("atomic native output is incomplete")
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            os.link(self.temporary, self.path)
            result: dict[str, int | str] = {
                "path": str(self.path),
                "sha256": self._digest.hexdigest(),
                "bytes": self._bytes_written,
            }
            self.temporary.unlink()
            self._closed = True
            return result
        except (AtomicNativeOutputError, OSError, ValueError) as exc:
            self.abort()
            raise AtomicNativeOutputError("atomic native output publication failed") from exc

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._handle.close()
        finally:
            self.temporary.unlink(missing_ok=True)
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if not self._closed:
            self.abort()


def publish_native_thomas_rgb16_png(
    library: Any,
    amplitude: NativeGranularityAmplitudeProfileV1,
    fields: tuple[
        NativeThomasFieldProfileV1,
        NativeThomasFieldProfileV1,
        NativeThomasFieldProfileV1,
    ],
    gauge: NativeGaugeProfileF32V1,
    exposure: np.ndarray,
    *,
    row_partition: int,
    destination: Path,
    maximum_output_bytes: int,
) -> dict[str, Any]:
    """Stream the native P8CQ PNG directly into one atomic-create destination."""
    values = np.asarray(exposure)
    if (
        values.dtype != np.float32
        or values.ndim != 3
        or values.shape[0] != 3
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
        or isinstance(row_partition, bool)
        or not isinstance(row_partition, int)
        or row_partition <= 0
    ):
        raise AtomicNativeOutputError("native Thomas publication input drift")
    height, width = values.shape[1:]
    workspace_bytes = ctypes.c_size_t()
    status = library.nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1(
        height, width, row_partition, ctypes.byref(workspace_bytes)
    )
    if status != 0 or workspace_bytes.value == 0:
        raise AtomicNativeOutputError("native Thomas workspace request failed")
    workspace = ctypes.create_string_buffer(workspace_bytes.value)
    field_array = (NativeThomasFieldProfileV1 * 3)(*fields)
    means = (ctypes.c_double * 3)(-13.0, -13.0, -13.0)
    callback_error: list[Exception] = []
    sink = AtomicNativeOutputSink(
        destination, maximum_bytes=maximum_output_bytes
    )

    @NativeByteSink
    def callback(
        _context: int,
        payload: ctypes.POINTER(ctypes.c_uint8),
        count: int,
    ) -> int:
        try:
            return int(sink.write(ctypes.string_at(payload, count)))
        except (AtomicNativeOutputError, OSError, ValueError) as exc:
            callback_error.append(exc)
            return 0

    try:
        status = library.nf_thomas_rgb16_png_cached_parallel_apply_v1(
            ctypes.byref(amplitude),
            field_array,
            ctypes.byref(gauge),
            height,
            width,
            row_partition,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.size,
            workspace,
            workspace_bytes.value,
            callback,
            None,
            means,
        )
        if status != 0 or callback_error:
            raise AtomicNativeOutputError("native Thomas PNG stream failed")
        published = sink.finish()
    except BaseException:
        sink.abort()
        raise
    return {
        **published,
        "workspace_bytes": workspace_bytes.value,
        "raw_field_means": [float(value) for value in means],
    }


def publish_native_thomas_profile_rgb16_png(
    library: Any,
    profile: dict[str, Any],
    exposure: np.ndarray,
    *,
    expected_profile_sha256: str,
    row_partition: int,
    destination: Path,
    maximum_output_bytes: int,
) -> dict[str, Any]:
    """Validate one canonical profile and atomically publish its native PNG."""
    try:
        profile_sha256 = validate_native_thomas_export_profile(profile)
        if profile_sha256 != expected_profile_sha256:
            raise ValueError("native Thomas expected profile identity drift")
        amplitude, fields, gauge = reconstruct_native_thomas_export_profile(profile)
    except (KeyError, TypeError, ValueError) as exc:
        raise AtomicNativeOutputError("native Thomas export profile rejected") from exc
    return {
        **publish_native_thomas_rgb16_png(
            library,
            amplitude,
            fields,
            gauge,
            exposure,
            row_partition=row_partition,
            destination=destination,
            maximum_output_bytes=maximum_output_bytes,
        ),
        "profile_sha256": profile_sha256,
    }


__all__ = [
    "AtomicNativeOutputError",
    "AtomicNativeOutputSink",
    "NativeByteSink",
    "publish_native_thomas_profile_rgb16_png",
    "publish_native_thomas_rgb16_png",
]
