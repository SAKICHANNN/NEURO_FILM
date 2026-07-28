"""Compile the artifact neutral-axis gauge for the native float32 ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
from typing import Any

import numpy as np


MAX_GAUGE_KNOTS = 1025
GAUGE_PAYLOAD_SCHEMA = "neuro_film.compiled_neutral_axis_gauge.v1"


class NativeGaugeProfileF32V1(ctypes.Structure):
    _KnotRow = ctypes.c_double * MAX_GAUGE_KNOTS
    _KnotTable = _KnotRow * 3
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_component_sha256", ctypes.c_char * 65),
        ("knot_count", ctypes.c_uint32 * 3),
        ("x_knots", _KnotTable),
        ("y_knots", _KnotTable),
        ("derivatives", _KnotTable),
    ]


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def validate_native_gauge_payload(payload: dict[str, Any]) -> None:
    if (
        set(payload)
        != {
            "schema",
            "base_component_sha256",
            "inverse_neutral_splines",
        }
        or payload.get("schema") != GAUGE_PAYLOAD_SCHEMA
        or not isinstance(payload["base_component_sha256"], str)
        or len(payload["base_component_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in payload["base_component_sha256"]
        )
        or not isinstance(payload["inverse_neutral_splines"], list)
        or len(payload["inverse_neutral_splines"]) != 3
    ):
        raise ValueError("unsupported native gauge payload")
    for spline in payload["inverse_neutral_splines"]:
        if (
            set(spline)
            != {"schema", "x_knots", "y_knots", "derivatives"}
            or spline["schema"]
            != "roll2film.rational_quadratic_spline.v1"
        ):
            raise ValueError("native gauge spline schema drift")
        x = np.asarray(spline["x_knots"], dtype=np.float64)
        y = np.asarray(spline["y_knots"], dtype=np.float64)
        derivatives = np.asarray(
            spline["derivatives"], dtype=np.float64
        )
        if (
            x.ndim != 1
            or len(x) < 2
            or len(x) > MAX_GAUGE_KNOTS
            or y.shape != x.shape
            or derivatives.shape != x.shape
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(derivatives))
            or np.any(derivatives <= 0.0)
            or np.any(np.diff(x) <= 0.0)
            or np.any(np.diff(y) <= 0.0)
            or x[0] != 0.0
            or x[-1] != 1.0
            or y[0] != 0.0
            or y[-1] != 1.0
        ):
            raise ValueError("invalid native gauge spline")


def native_gauge_profile_struct(
    payload: dict[str, Any],
) -> NativeGaugeProfileF32V1:
    validate_native_gauge_payload(payload)
    profile = NativeGaugeProfileF32V1()
    profile.struct_size = ctypes.sizeof(NativeGaugeProfileF32V1)
    profile.abi_version = 1
    profile.source_component_sha256 = _payload_sha256(payload).encode(
        "ascii"
    )
    for channel, spline in enumerate(
        payload["inverse_neutral_splines"]
    ):
        count = len(spline["x_knots"])
        profile.knot_count[channel] = count
        for index in range(count):
            profile.x_knots[channel][index] = float(
                spline["x_knots"][index]
            )
            profile.y_knots[channel][index] = float(
                spline["y_knots"][index]
            )
            profile.derivatives[channel][index] = float(
                spline["derivatives"][index]
            )
    return profile


def native_gauge_payload_sha256(payload: dict[str, Any]) -> str:
    validate_native_gauge_payload(payload)
    return _payload_sha256(payload)
