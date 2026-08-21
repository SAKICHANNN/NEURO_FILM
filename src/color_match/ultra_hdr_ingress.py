"""Private Ultra HDR v2 decoded-payload to MatchView canonicalizer.

This module does not decode media.  It accepts bytes already produced by the
pinned libultrahdr public C API and independently binds the source, decoder,
decoded payload, absolute-light conversion and consumer-owned MatchView.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_adapter import PreparedMatchViewV1, validate_prepared_match_view
from .core_contracts import (
    MATCH_PROFILE_ABSOLUTE_REC2020,
    MatchViewV1,
    make_match_view,
)

ULTRAHDR_INGRESS_SCHEMA_ID = "neuro-film.ultrahdr-absolute-rec2020-ingress.v1"
ULTRAHDR_DECODER_VERSION = "2.0.0"
ULTRAHDR_EXTERNAL_PROFILE_ID = "zhuise.display-linear-bt2020-d65-absolute-cdm2-f32.v2"
ULTRAHDR_RENDER_BRIDGE_ID = "neuro-film.ultrahdr-v2-public-c-api-203nit.v1"
ULTRAHDR_REFERENCE_WHITE_NITS = 203.0
ULTRAHDR_MAXIMUM_RELATIVE_LINEAR = 10000.0 / 203.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PreparedUltraHDRMatchViewV1:
    """Consumer-owned absolute-light view plus immutable provenance facts."""

    schema_id: str
    ingress_id: str
    source_asset_sha256: str
    decoded_payload_sha256: str
    decoder_version: str
    producer_profile_id: str
    prepared_view: PreparedMatchViewV1

    @property
    def descriptor(self) -> MatchViewV1:
        return self.prepared_view.descriptor

    @property
    def pixels(self) -> np.ndarray:
        return self.prepared_view.pixels


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bytes_like(
    value: bytes | bytearray | memoryview,
    label: str,
) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ReferenceMatchContractError(f"{label} must be bytes-like")
    return bytes(value)


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReferenceMatchContractError(f"{label} must be a lowercase SHA-256")
    return value


def _dimension(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceMatchContractError(f"{label} must be a positive integer")
    return value


def _pixel_sha256(pixels: np.ndarray) -> str:
    encoded = np.asarray(pixels, dtype=">f4", order="C")
    return _sha256_bytes(encoded.tobytes(order="C"))


def _ingress_payload(
    *,
    source_asset_sha256: str,
    decoded_payload_sha256: str,
    decoder_version: str,
    producer_profile_id: str,
    descriptor: MatchViewV1,
) -> dict[str, object]:
    return {
        "schema_id": ULTRAHDR_INGRESS_SCHEMA_ID,
        "source_asset_sha256": source_asset_sha256,
        "decoded_payload_sha256": decoded_payload_sha256,
        "decoder_version": decoder_version,
        "producer_profile_id": producer_profile_id,
        "consumer_profile_id": MATCH_PROFILE_ABSOLUTE_REC2020,
        "reference_white_nits": ULTRAHDR_REFERENCE_WHITE_NITS,
        "render_bridge_id": ULTRAHDR_RENDER_BRIDGE_ID,
        "view_id": descriptor.view_id,
    }


def prepare_ultrahdr_match_view_v1(
    *,
    source_asset: bytes | bytearray | memoryview,
    expected_source_sha256: str,
    decoded_rgba16f: bytes | bytearray | memoryview,
    width: int,
    height: int,
    decoder_version: str,
    producer_profile_id: str,
) -> PreparedUltraHDRMatchViewV1:
    """Canonicalize pinned libultrahdr RGBA16F output without media I/O."""

    source_bytes = _bytes_like(source_asset, "Ultra HDR source asset")
    if not source_bytes:
        raise ReferenceMatchContractError("Ultra HDR source asset must be non-empty")
    expected_source = _require_sha256(
        expected_source_sha256,
        "expected_source_sha256",
    )
    source_sha256 = _sha256_bytes(source_bytes)
    if source_sha256 != expected_source:
        raise ReferenceMatchContractError("Ultra HDR source asset identity mismatch")
    if decoder_version != ULTRAHDR_DECODER_VERSION:
        raise ReferenceMatchContractError("unsupported Ultra HDR decoder version")
    if producer_profile_id != ULTRAHDR_EXTERNAL_PROFILE_ID:
        raise ReferenceMatchContractError("unsupported Ultra HDR producer profile")

    validated_width = _dimension(width, "width")
    validated_height = _dimension(height, "height")
    payload = _bytes_like(
        decoded_rgba16f,
        "Ultra HDR decoded payload",
    )
    expected_length = validated_width * validated_height * 4 * 2
    if len(payload) != expected_length:
        raise ReferenceMatchContractError("Ultra HDR decoded payload length mismatch")

    rgba = np.frombuffer(payload, dtype="<f2").reshape(
        validated_height,
        validated_width,
        4,
    )
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    if not np.isfinite(rgba).all():
        raise ReferenceMatchContractError("Ultra HDR decoded payload must be finite")
    if np.any(rgb < np.float16(0.0)) or np.any(rgb > ULTRAHDR_MAXIMUM_RELATIVE_LINEAR):
        raise ReferenceMatchContractError(
            "Ultra HDR decoded RGB is outside the nominal linear range"
        )
    if not np.equal(alpha, np.float16(1.0)).all():
        raise ReferenceMatchContractError("Ultra HDR decoded alpha must be exactly one")

    pixels = np.array(rgb, dtype=np.float32, order="C", copy=True)
    pixels *= np.float32(ULTRAHDR_REFERENCE_WHITE_NITS)
    if not np.isfinite(pixels).all():
        raise ReferenceMatchContractError(
            "Ultra HDR absolute-light pixels must be finite"
        )
    pixels.flags.writeable = False

    decoded_payload_sha256 = _sha256_bytes(payload)
    provenance_fingerprint = canonical_sha256(
        {
            "schema_id": ULTRAHDR_INGRESS_SCHEMA_ID,
            "source_asset_sha256": source_sha256,
            "decoded_payload_sha256": decoded_payload_sha256,
            "decoder_version": decoder_version,
            "producer_profile_id": producer_profile_id,
            "reference_white_nits": ULTRAHDR_REFERENCE_WHITE_NITS,
            "conversion": "float32(float16-rgb)*float32(203.0)",
        }
    )
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_ABSOLUTE_REC2020,
        pixel_sha256=_pixel_sha256(pixels),
        shape=(validated_height, validated_width, 3),
        render_bridge_id=ULTRAHDR_RENDER_BRIDGE_ID,
        provenance_fingerprint=provenance_fingerprint,
        reference_white_nits=ULTRAHDR_REFERENCE_WHITE_NITS,
        alpha_mode="absent",
    )
    result = PreparedUltraHDRMatchViewV1(
        schema_id=ULTRAHDR_INGRESS_SCHEMA_ID,
        ingress_id=canonical_sha256(
            _ingress_payload(
                source_asset_sha256=source_sha256,
                decoded_payload_sha256=decoded_payload_sha256,
                decoder_version=decoder_version,
                producer_profile_id=producer_profile_id,
                descriptor=descriptor,
            )
        ),
        source_asset_sha256=source_sha256,
        decoded_payload_sha256=decoded_payload_sha256,
        decoder_version=decoder_version,
        producer_profile_id=producer_profile_id,
        prepared_view=PreparedMatchViewV1(
            descriptor=descriptor,
            pixels=pixels,
        ),
    )
    validate_prepared_ultrahdr_match_view_v1(result)
    return result


def validate_prepared_ultrahdr_match_view_v1(
    value: PreparedUltraHDRMatchViewV1,
) -> None:
    if not isinstance(value, PreparedUltraHDRMatchViewV1):
        raise ReferenceMatchContractError(
            "prepared Ultra HDR ingress has the wrong type"
        )
    if value.schema_id != ULTRAHDR_INGRESS_SCHEMA_ID:
        raise ReferenceMatchContractError("unsupported Ultra HDR ingress schema")
    _require_sha256(value.ingress_id, "ingress_id")
    _require_sha256(value.source_asset_sha256, "source_asset_sha256")
    _require_sha256(
        value.decoded_payload_sha256,
        "decoded_payload_sha256",
    )
    if value.decoder_version != ULTRAHDR_DECODER_VERSION:
        raise ReferenceMatchContractError("unsupported Ultra HDR decoder version")
    if value.producer_profile_id != ULTRAHDR_EXTERNAL_PROFILE_ID:
        raise ReferenceMatchContractError("unsupported Ultra HDR producer profile")
    validate_prepared_match_view(value.prepared_view)
    descriptor = value.descriptor
    if (
        descriptor.profile_id != MATCH_PROFILE_ABSOLUTE_REC2020
        or descriptor.reference_white_nits != ULTRAHDR_REFERENCE_WHITE_NITS
        or descriptor.render_bridge_id != ULTRAHDR_RENDER_BRIDGE_ID
        or descriptor.alpha_mode != "absent"
    ):
        raise ReferenceMatchContractError("Ultra HDR MatchView semantics mismatch")
    expected_ingress_id = canonical_sha256(
        _ingress_payload(
            source_asset_sha256=value.source_asset_sha256,
            decoded_payload_sha256=value.decoded_payload_sha256,
            decoder_version=value.decoder_version,
            producer_profile_id=value.producer_profile_id,
            descriptor=descriptor,
        )
    )
    if value.ingress_id != expected_ingress_id:
        raise ReferenceMatchContractError("Ultra HDR ingress identity mismatch")


__all__ = [
    "ULTRAHDR_DECODER_VERSION",
    "ULTRAHDR_EXTERNAL_PROFILE_ID",
    "ULTRAHDR_INGRESS_SCHEMA_ID",
    "ULTRAHDR_MAXIMUM_RELATIVE_LINEAR",
    "ULTRAHDR_REFERENCE_WHITE_NITS",
    "ULTRAHDR_RENDER_BRIDGE_ID",
    "PreparedUltraHDRMatchViewV1",
    "prepare_ultrahdr_match_view_v1",
    "validate_prepared_ultrahdr_match_view_v1",
]
