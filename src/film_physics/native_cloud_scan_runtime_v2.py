"""Windowed cloud scan runtime with replayable row-source ingress."""

from __future__ import annotations

import ctypes
import hashlib
from collections.abc import Callable

import numpy as np

from .native_abi_layouts import NativeGaussianProfileV1
from .native_cloud_scan_runtime import NativeCloudScanRuntimeError, OutputSink
from .native_standard_runtime import _load_gaussian, _pointer

ForwardRows = Callable[[int, int], np.ndarray]
WindowedPhysicalRows = Callable[[ForwardRows, int, int], np.ndarray]
SourceRows = Callable[[int, int], np.ndarray]


class WindowedNativeCloudScanRuntime:
    """Supply exact forward-scattered row windows to a physical provider."""

    def __init__(
        self,
        *,
        gaussian_library,
        forward_scatter_profile: NativeGaussianProfileV1,
        physical_rows: WindowedPhysicalRows,
        physical_component_sha256: str,
        tile_rows: int,
    ) -> None:
        if (
            tile_rows <= 0
            or not callable(physical_rows)
            or len(physical_component_sha256) != 64
        ):
            raise ValueError("invalid windowed cloud scan runtime construction")
        self._gaussian = _load_gaussian(gaussian_library)
        self._forward = forward_scatter_profile
        self._physical_rows = physical_rows
        self.physical_component_sha256 = physical_component_sha256
        self.tile_rows = tile_rows
        self._gaussian.nf_gaussian_f32_required_halo_v1.argtypes = [
            ctypes.POINTER(NativeGaussianProfileV1),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._gaussian.nf_gaussian_f32_required_halo_v1.restype = ctypes.c_int
        fp = ctypes.POINTER(ctypes.c_float)
        size = ctypes.c_size_t
        self._gaussian.nf_gaussian_row_window_f32_apply_v1.argtypes = [
            ctypes.POINTER(NativeGaussianProfileV1),
            size, size, size, size, fp, size, size,
            fp, size, fp, size, fp, size,
        ]
        self._gaussian.nf_gaussian_row_window_f32_apply_v1.restype = ctypes.c_int

    def render_to_sink(
        self, scene_linear: np.ndarray, *, output_sink: OutputSink
    ) -> dict:
        source = np.asarray(scene_linear)
        if (
            source.dtype != np.float32
            or source.ndim != 3
            or source.shape[-1] != 3
            or source.size == 0
            or not source.flags.c_contiguous
            or not np.all(np.isfinite(source))
            or np.any(source < 0)
            or np.any(source > 1)
            or not callable(output_sink)
        ):
            raise ValueError(
                "windowed cloud source must be finite C-contiguous "
                "float32 HxWx3 in [0,1]"
            )
        height, width, _ = source.shape
        source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
        previous = source.flags.writeable
        source.flags.writeable = False
        try:
            receipt = self.render_rows_to_sink(
                height=height,
                width=width,
                source_rows=lambda start, count: source[start : start + count],
                expected_input_sha256=source_sha,
                output_sink=output_sink,
            )
            if hashlib.sha256(memoryview(source).cast("B")).hexdigest() != source_sha:
                raise NativeCloudScanRuntimeError("windowed cloud source mutated")
            return {**receipt, "full_source_frame_retained": True}
        finally:
            source.flags.writeable = previous

    def render_rows_to_sink(
        self,
        *,
        height: int,
        width: int,
        source_rows: SourceRows,
        expected_input_sha256: str,
        output_sink: OutputSink,
    ) -> dict:
        """Verify one source pass, then render from the same replayable rows."""

        if (
            height <= 0
            or width <= 0
            or not callable(source_rows)
            or not callable(output_sink)
            or len(expected_input_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_input_sha256)
        ):
            raise ValueError("invalid windowed cloud row source")

        def read_rows(start: int, count: int) -> np.ndarray:
            rows = np.asarray(source_rows(start, count))
            if (
                rows.dtype != np.float32
                or rows.shape != (count, width, 3)
                or not rows.flags.c_contiguous
                or not np.all(np.isfinite(rows))
                or np.any(rows < 0)
                or np.any(rows > 1)
            ):
                raise NativeCloudScanRuntimeError(
                    "windowed cloud source row drift"
                )
            return rows

        input_digest = hashlib.sha256()
        for y0 in range(0, height, self.tile_rows):
            y1 = min(height, y0 + self.tile_rows)
            input_digest.update(memoryview(read_rows(y0, y1 - y0)).cast("B"))
        source_sha = input_digest.hexdigest()
        if source_sha != expected_input_sha256:
            raise NativeCloudScanRuntimeError("windowed cloud source hash drift")

        halo = ctypes.c_uint32()
        if self._gaussian.nf_gaussian_f32_required_halo_v1(
            ctypes.byref(self._forward), ctypes.byref(halo)
        ) != 0:
            raise NativeCloudScanRuntimeError("forward halo failed")
        maximum_forward_values = 0

        def forward_rows(start: int, count: int) -> np.ndarray:
            nonlocal maximum_forward_values
            input_start = max(0, start - halo.value)
            input_end = min(height, start + count + halo.value)
            window = read_rows(input_start, input_end - input_start)
            work = np.empty_like(window)
            rendered = np.empty_like(window)
            core = np.empty((count, width, 3), np.float32)
            maximum_forward_values = max(
                maximum_forward_values,
                window.size + work.size + rendered.size + core.size,
            )
            status = self._gaussian.nf_gaussian_row_window_f32_apply_v1(
                ctypes.byref(self._forward), height, width,
                input_start, window.shape[0], _pointer(window), start, count,
                _pointer(work), work.size, _pointer(rendered), rendered.size,
                _pointer(core), core.size,
            )
            if status != 0:
                raise NativeCloudScanRuntimeError(
                    f"forward window failed: {status}"
                )
            return core

        output_digest = hashlib.sha256()
        consumed = 0
        tiles = 0
        for y0 in range(0, height, self.tile_rows):
            y1 = min(height, y0 + self.tile_rows)
            rows = np.ascontiguousarray(
                self._physical_rows(forward_rows, y0, y1 - y0), np.float32
            )
            if rows.shape != (y1 - y0, width, 3):
                raise NativeCloudScanRuntimeError(
                    "windowed cloud row shape drift"
                )
            rows.flags.writeable = False
            output_sink(y0, y1, rows)
            output_digest.update(rows.tobytes())
            consumed = y1
            tiles += 1
        if consumed != height:
            raise NativeCloudScanRuntimeError("windowed cloud output incomplete")
        return {
            "input_sha256": source_sha,
            "output_sha256": output_digest.hexdigest(),
            "shape": [height, width, 3],
            "tile_rows": self.tile_rows,
            "submitted_tiles": tiles,
            "maximum_forward_workspace_values": maximum_forward_values,
            "full_forward_frame_retained": False,
            "full_source_frame_retained": False,
            "source_passes": 2,
            "physical_component_sha256": self.physical_component_sha256,
            "production_default_changed": False,
        }


__all__ = ["WindowedNativeCloudScanRuntime"]
