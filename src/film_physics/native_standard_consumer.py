"""WorkingImage ingress for the opt-in native Standard row runtime."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.preprocess.types import WorkingImage

from .native_standard_runtime import NativeStandardRuntime, OutputSink


def render_native_standard_working_image_to_sink(
    runtime: NativeStandardRuntime,
    working: WorkingImage,
    *,
    output_sink: OutputSink,
) -> dict[str, Any]:
    """Validate one scene-linear WorkingImage and stream ordered output rows."""

    if not isinstance(runtime, NativeStandardRuntime):
        raise TypeError("runtime must be a NativeStandardRuntime")
    if not isinstance(working, WorkingImage):
        raise TypeError("working must be a WorkingImage")
    if working.transfer_state != "scene_linear":
        raise ValueError(
            "native Standard ingress requires scene_linear WorkingImage"
        )
    if working.working_space not in {
        "linear_srgb",
        "linear_srgb_d65",
    }:
        raise ValueError(
            "native Standard ingress requires D65 linear sRGB"
        )
    if not working.orientation_applied:
        raise ValueError(
            "native Standard ingress requires applied orientation"
        )
    if working.alpha_policy != "absent":
        raise ValueError(
            "native Standard ingress requires absent alpha"
        )

    runtime_receipt = runtime.render_to_sink(
        working.pixels,
        output_sink=output_sink,
    )
    input_sha = runtime_receipt["input"]["array_sha256"]
    ingress = {
        "schema": "neuro_film.native_standard_working_image_receipt.v1",
        "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
        "input": {
            "array_sha256": input_sha,
            "working_space": working.working_space,
            "transfer_state": working.transfer_state,
            "source_transfer_state": working.source_transfer_state,
            "source_profile_kind": working.source_profile.kind,
            "source_profile_description": working.source_profile.description,
            "source_profile_bytes_length": working.source_profile.bytes_length,
            "orientation_applied": working.orientation_applied,
            "alpha_policy": working.alpha_policy,
            "bit_depth_in": working.bit_depth_in,
            "source_path": str(working.source_path),
            "warning_codes": [item.code for item in working.warnings],
        },
        "output": runtime_receipt["output"],
        "production_default_changed": False,
        "claim_ceiling": (
            "opt-in scene-linear D65 linear-sRGB WorkingImage to exact "
            "native Standard float32 row staging; no decoder, encoder, "
            "durable commit, mobile, calibration or production promotion"
        ),
    }
    return {
        **ingress,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                ingress,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "runtime_receipt": runtime_receipt,
    }


__all__ = ["render_native_standard_working_image_to_sink"]
