"""Opt-in row runtime for the correct-domain stochastic cloud scan chain."""

from __future__ import annotations

import ctypes
import hashlib
from typing import Callable

import numpy as np

from src.film_physics.native_abi_layouts import NativeGaussianProfileV1
from src.film_physics.native_standard_runtime import _load_gaussian, _pointer


PhysicalRows = Callable[[np.ndarray, int, int], np.ndarray]
OutputSink = Callable[[int, int, np.ndarray], None]


class NativeCloudScanRuntimeError(RuntimeError):
    """Raised when the opt-in cloud scan runtime fails closed."""


class NativeCloudScanRuntime:
    """Forward-scatter once, then stream exact cloud/spatial scan rows."""

    def __init__(
        self,
        *,
        gaussian_library,
        forward_scatter_profile: NativeGaussianProfileV1,
        physical_rows: PhysicalRows,
        physical_component_sha256: str,
        tile_rows: int,
    ) -> None:
        if (
            tile_rows <= 0 or not callable(physical_rows)
            or len(physical_component_sha256) != 64
            or any(c not in "0123456789abcdef" for c in physical_component_sha256)
        ):
            raise ValueError("invalid native cloud scan runtime construction")
        self._gaussian = _load_gaussian(gaussian_library)
        self._forward = forward_scatter_profile
        self._physical_rows = physical_rows
        self.physical_component_sha256 = physical_component_sha256
        self.tile_rows = tile_rows

    def render_to_sink(self, scene_linear: np.ndarray, *, output_sink: OutputSink) -> dict:
        source = np.asarray(scene_linear)
        if (
            source.dtype != np.float32 or source.ndim != 3 or source.shape[-1] != 3
            or source.size == 0 or not source.flags.c_contiguous
            or not np.all(np.isfinite(source)) or np.any(source < 0) or np.any(source > 1)
            or not callable(output_sink)
        ):
            raise ValueError("cloud scan source must be finite C-contiguous float32 HxWx3 in [0,1]")
        source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
        previous = source.flags.writeable; source.flags.writeable = False
        try:
            height, width, _ = source.shape
            workspace=np.empty_like(source);forward=np.empty_like(source)
            status=self._gaussian.nf_gaussian_f32_apply_v1(
                ctypes.byref(self._forward),_pointer(source),height,width,
                _pointer(workspace),workspace.size,_pointer(forward))
            if status != 0: raise NativeCloudScanRuntimeError(f"forward scatter failed: {status}")
            digest=hashlib.sha256();consumed=0;tiles=0
            for y0 in range(0,height,self.tile_rows):
                y1=min(height,y0+self.tile_rows)
                rows=np.ascontiguousarray(
                    self._physical_rows(forward,y0,y1-y0),dtype=np.float32)
                if rows.shape!=(y1-y0,width,3): raise NativeCloudScanRuntimeError("cloud scan row shape drift")
                rows.flags.writeable=False;output_sink(y0,y1,rows)
                digest.update(rows.tobytes());consumed=y1;tiles+=1
            if consumed!=height: raise NativeCloudScanRuntimeError("cloud scan output incomplete")
            if hashlib.sha256(memoryview(source).cast("B")).hexdigest()!=source_sha:
                raise NativeCloudScanRuntimeError("cloud scan source mutated")
        finally: source.flags.writeable=previous
        return {"input_sha256":source_sha,"output_sha256":digest.hexdigest(),
            "shape":[height,width,3],"tile_rows":self.tile_rows,"submitted_tiles":tiles,
            "physical_component_sha256":self.physical_component_sha256,
            "production_default_changed":False,"claim_ceiling":"opt-in synthetic-profile scan-linear row runtime only"}


__all__=["NativeCloudScanRuntime","NativeCloudScanRuntimeError"]
