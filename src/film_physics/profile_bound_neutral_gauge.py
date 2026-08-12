"""Compile a deterministic inverse neutral gauge from one physical profile."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from src.roll2film.splines import RationalQuadraticSpline

from .native_gauge_profile import GAUGE_PAYLOAD_SCHEMA, validate_native_gauge_payload


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def compile_profile_bound_neutral_gauge(
    levels: np.ndarray,
    response_rgb: np.ndarray,
    *,
    base_component_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile three strict inverse RQS curves without isotonic repair."""

    source = np.asarray(levels, dtype=np.float64)
    response = np.asarray(response_rgb, dtype=np.float64)
    if (
        source.ndim != 1
        or source.size < 3
        or response.shape != (source.size, 3)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(response))
        or source[0] != 0.0
        or source[-1] != 1.0
        or np.any(np.diff(source) <= 0.0)
        or np.any(response < 0.0)
        or np.any(response > 1.0)
    ):
        raise ValueError("invalid profile-bound neutral response")

    splines: list[dict[str, Any]] = []
    channel_metrics: list[dict[str, float | bool]] = []
    for channel in range(3):
        values = response[:, channel]
        differences = np.diff(values)
        strictly_increasing = bool(np.all(differences > 0.0))
        x = np.concatenate(([0.0], values[1:-1], [1.0]))
        y = source.copy()
        full_strict = bool(np.all(np.diff(x) > 0.0))
        if not strictly_increasing or not full_strict:
            raise ValueError(f"channel {channel} neutral response is not strictly increasing")
        spline = RationalQuadraticSpline.from_knots(x, y)
        inverse_secants = np.diff(y) / np.diff(x)
        splines.append(spline.to_dict())
        channel_metrics.append(
            {
                "strictly_increasing": strictly_increasing,
                "response_span": float(values[-1] - values[0]),
                "minimum_response_step": float(np.min(differences)),
                "maximum_inverse_secant_slope": float(np.max(inverse_secants)),
            }
        )
    payload = {
        "schema": GAUGE_PAYLOAD_SCHEMA,
        "base_component_sha256": base_component_sha256,
        "inverse_neutral_splines": splines,
    }
    validate_native_gauge_payload(payload)
    return payload, {
        "channels": channel_metrics,
        "payload_sha256": _canonical_sha256(payload),
    }


def apply_profile_bound_neutral_gauge(
    payload: dict[str, Any], values: np.ndarray
) -> np.ndarray:
    """Apply a validated compiled gauge in float64 for research evaluation."""

    validate_native_gauge_payload(payload)
    source = np.asarray(values, dtype=np.float64)
    if (
        source.ndim < 1
        or source.shape[-1] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise ValueError("profile-bound gauge input must be finite RGB in [0,1]")
    output = np.stack(
        [
            RationalQuadraticSpline.from_dict(payload["inverse_neutral_splines"][channel]).apply(
                source[..., channel]
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("profile-bound neutral gauge produced nonfinite output")
    return output


__all__ = [
    "apply_profile_bound_neutral_gauge",
    "compile_profile_bound_neutral_gauge",
]
