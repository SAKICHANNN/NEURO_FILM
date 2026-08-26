"""Private consumer for the exact R1CZ shared HDR D-PCT payload format."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

PAYLOAD_FORMAT = "zhuise.rgb-dpct-reference-white.portable-json.v3"
BUNDLE_SCHEMA = "https://zhuise.local/schemas/paired-reference-shared-hdr-bundle-v1"
COMPOSITE_FORMAT = "zhuise.hdr-dpct-reference-white.bundle"
DPCT_FORMAT = "zhuise.dpct.transform"

_ENVELOPE_KEYS = {
    "schema",
    "algorithm_id",
    "algorithm_version",
    "capability_id",
    "profile_id",
    "reference_white_nits",
    "build_source_view_id",
    "reference_view_id",
    "legacy_bundle_id",
    "payload_format",
    "payload_byte_length",
    "payload_sha256",
    "bundle_id",
}
_DPCT_KEYS = {
    "format",
    "version",
    "scale",
    "strength",
    "identity",
    "extrapolation_slope_min",
    "extrapolation_slope_max",
    "source_knots",
    "reference_knots",
}


class SharedHdrDpctPayloadError(ValueError):
    """Raised when a shared HDR payload or envelope fails closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _exact_mapping(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise SharedHdrDpctPayloadError(f"{label} fields differ")
    return value


def _sha_identity(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise SharedHdrDpctPayloadError(f"{label} must be a SHA-256 identity")
    return value


@dataclass(frozen=True)
class SharedHdrDpctPayload:
    """Verified exact payload plus the envelope that binds it."""

    envelope: dict[str, Any]
    payload: dict[str, Any]
    payload_bytes: bytes
    source_knots: np.ndarray
    reference_knots: np.ndarray


def load_shared_hdr_dpct_payload(
    payload_bytes: bytes, envelope: Mapping[str, Any]
) -> SharedHdrDpctPayload:
    """Strictly verify and deserialize one shared HDR D-PCT payload."""

    if not isinstance(payload_bytes, bytes) or not payload_bytes:
        raise SharedHdrDpctPayloadError("payload must be non-empty bytes")
    bound = dict(_exact_mapping(envelope, _ENVELOPE_KEYS, "bundle envelope"))
    if bound["schema"] != BUNDLE_SCHEMA or bound["payload_format"] != PAYLOAD_FORMAT:
        raise SharedHdrDpctPayloadError("bundle schema or payload format differs")
    if bound["payload_byte_length"] != len(payload_bytes):
        raise SharedHdrDpctPayloadError("payload byte length differs")
    if _sha_identity(bound["payload_sha256"], "payload_sha256") != _sha256(
        payload_bytes
    ):
        raise SharedHdrDpctPayloadError("payload SHA-256 differs")
    for field in (
        "build_source_view_id",
        "reference_view_id",
        "legacy_bundle_id",
        "bundle_id",
    ):
        _sha_identity(bound[field], field)
    identity = dict(bound)
    bundle_id = identity.pop("bundle_id")
    if _sha256(_canonical_json_bytes(identity)) != bundle_id:
        raise SharedHdrDpctPayloadError("bundle ID differs")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SharedHdrDpctPayloadError("payload is not UTF-8 JSON") from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != payload_bytes:
        raise SharedHdrDpctPayloadError("payload is not canonical JSON")
    _exact_mapping(payload, {"format", "version", "fit_contract", "dpct"}, "payload")
    if payload["format"] != COMPOSITE_FORMAT or payload["version"] != 3:
        raise SharedHdrDpctPayloadError("composite payload format/version differs")
    fit = _exact_mapping(
        payload["fit_contract"],
        {
            "working_profile",
            "signed_log_reference_nits",
            "maximum_fit_samples",
            "fit_selection",
            "output_bounds_nits",
        },
        "fit contract",
    )
    expected_fit = {
        "working_profile": bound["profile_id"],
        "signed_log_reference_nits": bound["reference_white_nits"],
        "maximum_fit_samples": 65536,
        "fit_selection": "rows-plus-2cols-mod-4-equals-0",
        "output_bounds_nits": [0.0, 10000.0],
    }
    if dict(fit) != expected_fit:
        raise SharedHdrDpctPayloadError("fit contract differs")
    dpct = _exact_mapping(payload["dpct"], _DPCT_KEYS, "D-PCT payload")
    if (
        dpct["format"] != DPCT_FORMAT
        or dpct["version"] != 1
        or dpct["extrapolation_slope_min"] != 0.0
        or dpct["extrapolation_slope_max"] != 1.0
        or not isinstance(dpct["identity"], bool)
    ):
        raise SharedHdrDpctPayloadError("D-PCT format or policy differs")
    scale = float(dpct["scale"])
    strength = float(dpct["strength"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise SharedHdrDpctPayloadError("D-PCT scale is invalid")
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise SharedHdrDpctPayloadError("D-PCT strength is invalid")
    source_knots = np.asarray(dpct["source_knots"], dtype=np.float32)
    reference_knots = np.asarray(dpct["reference_knots"], dtype=np.float32)
    if (
        source_knots.shape != (33, 3)
        or reference_knots.shape != (33, 3)
        or not np.isfinite(source_knots).all()
        or not np.isfinite(reference_knots).all()
        or np.any(np.diff(source_knots, axis=0) <= 0.0)
        or np.any(np.diff(reference_knots, axis=0) <= 0.0)
    ):
        raise SharedHdrDpctPayloadError("D-PCT knots are invalid")
    return SharedHdrDpctPayload(
        envelope=bound,
        payload=payload,
        payload_bytes=payload_bytes,
        source_knots=np.array(source_knots, copy=True, order="C"),
        reference_knots=np.array(reference_knots, copy=True, order="C"),
    )


def _signed_log32(values: np.ndarray, scale: np.float32) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values) / scale)


def _signed_exp32(values: np.ndarray, scale: np.float32) -> np.ndarray:
    return np.sign(values) * np.expm1(np.abs(values)) * scale


def _interp_channel32(
    values: np.ndarray, x_knots: np.ndarray, y_knots: np.ndarray
) -> np.ndarray:
    flat = values.reshape(-1)
    indices = np.searchsorted(x_knots, flat, side="right") - 1
    interior = np.clip(indices, 0, len(x_knots) - 2)
    x0 = x_knots[interior]
    x1 = x_knots[interior + 1]
    y0 = y_knots[interior]
    y1 = y_knots[interior + 1]
    denominator = x1 - x0
    safe_denominator = np.where(denominator > 0.0, denominator, np.float32(1.0))
    bounded = np.minimum(np.maximum(flat, x0), x1)
    alpha = (bounded - x0) / safe_denominator
    result = y0 + alpha * (y1 - y0)
    below = flat < x_knots[0]
    above = flat > x_knots[-1]
    if np.any(below):
        numerator = y_knots[1] - y_knots[0]
        knot_denominator = x_knots[1] - x_knots[0]
        slope = (
            np.float32(0.0)
            if numerator <= 0.0 or knot_denominator <= 0.0
            else np.float32(1.0)
            if numerator >= knot_denominator
            else numerator / knot_denominator
        )
        result[below] = y_knots[0] + slope * (flat[below] - x_knots[0])
    if np.any(above):
        numerator = y_knots[-1] - y_knots[-2]
        knot_denominator = x_knots[-1] - x_knots[-2]
        slope = (
            np.float32(0.0)
            if numerator <= 0.0 or knot_denominator <= 0.0
            else np.float32(1.0)
            if numerator >= knot_denominator
            else numerator / knot_denominator
        )
        result[above] = y_knots[-1] + slope * (flat[above] - x_knots[-1])
    return result.reshape(values.shape).astype(np.float32, copy=False)


def apply_shared_hdr_dpct_payload(
    source: np.ndarray, bundle: SharedHdrDpctPayload
) -> np.ndarray:
    """Apply one verified shared bundle to an independent absolute-nits source."""

    value = np.asarray(source)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[2] != 3:
        raise SharedHdrDpctPayloadError("source must be float32 HxWx3")
    if value.size == 0 or not np.isfinite(value).all():
        raise SharedHdrDpctPayloadError("source must be finite and non-empty")
    dpct = bundle.payload["dpct"]
    if dpct["identity"] or float(dpct["strength"]) == 0.0:
        transformed = value.copy()
    else:
        scale = np.float32(dpct["scale"])
        shaped = _signed_log32(value, scale)
        mapped = np.empty_like(shaped)
        for channel in range(3):
            mapped[..., channel] = _interp_channel32(
                shaped[..., channel],
                bundle.source_knots[:, channel],
                bundle.reference_knots[:, channel],
            )
        transformed = _signed_exp32(mapped, scale)
        strength = np.float32(dpct["strength"])
        if strength != np.float32(1.0):
            transformed = _signed_exp32(
                (np.float32(1.0) - strength) * shaped
                + strength * _signed_log32(transformed, scale),
                scale,
            )
    if not np.isfinite(transformed).all():
        raise SharedHdrDpctPayloadError("payload apply produced non-finite output")
    return np.array(
        np.clip(transformed, np.float32(0.0), np.float32(10000.0)),
        dtype=np.float32,
        order="C",
        copy=True,
    )


__all__ = [
    "BUNDLE_SCHEMA",
    "PAYLOAD_FORMAT",
    "SharedHdrDpctPayload",
    "SharedHdrDpctPayloadError",
    "apply_shared_hdr_dpct_payload",
    "load_shared_hdr_dpct_payload",
]
