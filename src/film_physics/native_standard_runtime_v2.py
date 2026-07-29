"""Memory-bounded implementation of the exact native Standard v1 receipt."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .native_standard_runtime import (
    NativeStandardRuntime,
    NativeStandardRuntimeError,
    OutputSink,
)


def sha256_c_contiguous_array(values: np.ndarray) -> str:
    """Hash one C-contiguous array without materializing its full byte copy."""

    if not isinstance(values, np.ndarray) or not values.flags.c_contiguous:
        raise ValueError("zero-copy array hashing requires C-contiguous ndarray")
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


class MemoryBoundNativeStandardRuntime(NativeStandardRuntime):
    """Exact native Standard runtime without full-input hash copies."""

    def render_to_sink(
        self,
        scene_linear: np.ndarray,
        *,
        output_sink: OutputSink,
    ) -> dict[str, Any]:
        source = np.asarray(scene_linear)
        if (
            source.dtype != np.float32
            or source.ndim != 3
            or source.shape[-1] != 3
            or source.size == 0
            or not source.flags.c_contiguous
            or not np.all(np.isfinite(source))
            or np.any(source < 0.0)
            or np.any(source > 1.0)
        ):
            raise ValueError(
                "native Standard source must be C-contiguous finite "
                "float32 HxWx3 in [0,1]"
            )
        if not callable(output_sink):
            raise ValueError("native Standard output_sink must be callable")
        height, width, _ = source.shape
        input_sha = sha256_c_contiguous_array(source)
        previous_writeable = source.flags.writeable
        source.flags.writeable = False
        try:
            context = self._build_source_context(source)
            receipt = self._render_physical_and_display(
                source,
                context=context,
                output_sink=output_sink,
            )
            if sha256_c_contiguous_array(source) != input_sha:
                raise NativeStandardRuntimeError(
                    "native Standard source mutated during render"
                )
        finally:
            source.flags.writeable = previous_writeable
        core = {
            "schema": "neuro_film.native_standard_stream_receipt.v1",
            "package_sha256": self.package_sha256,
            "artifact_sha256": self.artifact_sha256,
            "bundle_sha256": self.bundle_sha256,
            "input": {
                "array_sha256": input_sha,
                "dtype": "float32",
                "shape": [height, width, 3],
                "domain": "scene-linear-relative-exposure",
                "working_space": "linear-srgb",
            },
            "output": {
                "array_sha256": receipt["output_sha256"],
                "dtype": "float32",
                "shape": [height, width, 3],
                "domain": "display-encoded-rgb",
                "quantized": False,
            },
            "execution": {
                **self.policy,
                "submitted_tiles": receipt["submitted_tiles"],
                "consumed_tiles": receipt["consumed_tiles"],
                "maximum_observed_in_flight": receipt[
                    "maximum_observed_in_flight"
                ],
            },
            "production_default_changed": False,
            "claim_ceiling": (
                "opt-in exact-package Windows x64 native Standard "
                "float32 row stream; sink staging only, no decoder, "
                "encoder, durable commit, mobile, calibration or "
                "production promotion"
            ),
        }
        return {
            **core,
            "receipt_sha256": hashlib.sha256(
                json.dumps(
                    core,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest(),
        }


__all__ = [
    "MemoryBoundNativeStandardRuntime",
    "sha256_c_contiguous_array",
]
