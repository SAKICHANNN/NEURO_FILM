"""Private libavif gain-map RGB16 PQ to absolute Rec.2020 MatchView bridge."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from src.preprocess.color_management import linear_rgb_matrix
from src.preprocess.rec2100_pq_transfer import pq_to_absolute_rec2020_cdm2

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_adapter import PreparedMatchViewV1, validate_prepared_match_view
from .core_contracts import MATCH_PROFILE_ABSOLUTE_REC2020, MatchViewV1, make_match_view

LIBAVIF_GAINMAP_INGRESS_SCHEMA_ID = (
    "neuro-film.libavif-gainmap-absolute-rec2020-ingress.v1"
)
LIBAVIF_GAINMAP_DECODER_VERSION = "1.4.2"
LIBAVIF_GAINMAP_RENDER_BRIDGE_ID = (
    "neuro-film.libavif-gainmap-pq-bt709-to-absolute-rec2020.v1"
)
LIBAVIF_GAINMAP_SOURCE_PROFILE_ID = "libavif.rgb16-pq-bt709-full.v1"
LIBAVIF_GAINMAP_REFERENCE_WHITE_NITS = 203.0
LIBAVIF_GAINMAP_COLOR_PRIMARIES = 1
LIBAVIF_GAINMAP_TRANSFER_CHARACTERISTICS = 16
LIBAVIF_GAINMAP_ROLES = frozenset({"base", "alternate"})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PreparedLibAvifGainMapMatchViewV1:
    """Consumer-owned absolute-light view plus exact libavif provenance."""

    schema_id: str
    ingress_id: str
    source_asset_sha256: str
    encoded_sample_sha256: str
    decoder_version: str
    source_profile_id: str
    higher_rendition_role: str
    prepared_view: PreparedMatchViewV1

    @property
    def descriptor(self) -> MatchViewV1:
        return self.prepared_view.descriptor

    @property
    def pixels(self) -> np.ndarray:
        return self.prepared_view.pixels


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bytes_like(value: bytes | bytearray | memoryview, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ReferenceMatchContractError(f"{label} must be bytes-like")
    return bytes(value)


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReferenceMatchContractError(f"{label} must be a lowercase SHA-256")
    return value


def _encoded_sample_sha256(samples: np.ndarray) -> str:
    encoded = np.asarray(samples, dtype="<u2", order="C")
    return _sha256_bytes(encoded.tobytes(order="C"))


def _pixel_sha256(pixels: np.ndarray) -> str:
    encoded = np.asarray(pixels, dtype=">f4", order="C")
    return _sha256_bytes(encoded.tobytes(order="C"))


def _ingress_payload(
    *,
    source_asset_sha256: str,
    encoded_sample_sha256: str,
    higher_rendition_role: str,
    descriptor: MatchViewV1,
) -> dict[str, object]:
    return {
        "schema_id": LIBAVIF_GAINMAP_INGRESS_SCHEMA_ID,
        "source_asset_sha256": source_asset_sha256,
        "encoded_sample_sha256": encoded_sample_sha256,
        "decoder_version": LIBAVIF_GAINMAP_DECODER_VERSION,
        "source_profile_id": LIBAVIF_GAINMAP_SOURCE_PROFILE_ID,
        "source_color_primaries": LIBAVIF_GAINMAP_COLOR_PRIMARIES,
        "source_transfer_characteristics": LIBAVIF_GAINMAP_TRANSFER_CHARACTERISTICS,
        "source_full_range": True,
        "higher_rendition_role": higher_rendition_role,
        "consumer_profile_id": MATCH_PROFILE_ABSOLUTE_REC2020,
        "reference_white_nits": LIBAVIF_GAINMAP_REFERENCE_WHITE_NITS,
        "render_bridge_id": LIBAVIF_GAINMAP_RENDER_BRIDGE_ID,
        "view_id": descriptor.view_id,
    }


def prepare_libavif_gainmap_match_view_v1(
    *,
    source_asset: bytes | bytearray | memoryview,
    expected_source_sha256: str,
    encoded_rgb16: np.ndarray,
    decoder_version: str,
    source_color_primaries: int,
    source_transfer_characteristics: int,
    source_full_range: bool,
    higher_rendition_role: str,
) -> PreparedLibAvifGainMapMatchViewV1:
    """Convert exact full-range PQ BT.709 RGB16 into an absolute MatchView."""

    source_bytes = _bytes_like(source_asset, "libavif source asset")
    if not source_bytes:
        raise ReferenceMatchContractError("libavif source asset must be non-empty")
    expected_source = _require_sha256(expected_source_sha256, "expected_source_sha256")
    source_sha256 = _sha256_bytes(source_bytes)
    if source_sha256 != expected_source:
        raise ReferenceMatchContractError("libavif source asset identity mismatch")
    if decoder_version != LIBAVIF_GAINMAP_DECODER_VERSION:
        raise ReferenceMatchContractError("unsupported libavif decoder version")
    if source_color_primaries != LIBAVIF_GAINMAP_COLOR_PRIMARIES:
        raise ReferenceMatchContractError("libavif source primaries must be BT.709")
    if source_transfer_characteristics != LIBAVIF_GAINMAP_TRANSFER_CHARACTERISTICS:
        raise ReferenceMatchContractError("libavif source transfer must be PQ")
    if source_full_range is not True:
        raise ReferenceMatchContractError("libavif RGB16 source must be full range")
    if higher_rendition_role not in LIBAVIF_GAINMAP_ROLES:
        raise ReferenceMatchContractError("higher_rendition_role is unsupported")
    if not isinstance(encoded_rgb16, np.ndarray):
        raise ReferenceMatchContractError("encoded_rgb16 must be a numpy ndarray")
    if encoded_rgb16.dtype != np.uint16:
        raise ReferenceMatchContractError("encoded_rgb16 must have dtype uint16")
    if encoded_rgb16.ndim != 3 or encoded_rgb16.shape[2] != 3:
        raise ReferenceMatchContractError("encoded_rgb16 must be HxWx3")
    if encoded_rgb16.shape[0] <= 0 or encoded_rgb16.shape[1] <= 0:
        raise ReferenceMatchContractError("encoded_rgb16 dimensions must be positive")

    sample_sha256 = _encoded_sample_sha256(encoded_rgb16)
    encoded = encoded_rgb16.astype(np.float64) / 65535.0
    absolute_srgb = pq_to_absolute_rec2020_cdm2(encoded)
    matrix = linear_rgb_matrix("linear_srgb", "linear_rec2020")
    absolute_rec2020 = np.matmul(absolute_srgb, matrix.T)
    if not np.isfinite(absolute_rec2020).all():
        raise ReferenceMatchContractError("libavif bridge produced nonfinite pixels")
    if np.any(absolute_rec2020 < 0.0) or np.any(absolute_rec2020 > 10000.0):
        raise ReferenceMatchContractError("libavif bridge escaped absolute RGB bounds")
    pixels = np.array(absolute_rec2020, dtype=np.float32, order="C", copy=True)
    pixels.flags.writeable = False

    provenance_fingerprint = canonical_sha256(
        {
            "schema_id": LIBAVIF_GAINMAP_INGRESS_SCHEMA_ID,
            "source_asset_sha256": source_sha256,
            "encoded_sample_sha256": sample_sha256,
            "decoder_version": decoder_version,
            "source_profile_id": LIBAVIF_GAINMAP_SOURCE_PROFILE_ID,
            "source_color_primaries": source_color_primaries,
            "source_transfer_characteristics": source_transfer_characteristics,
            "source_full_range": source_full_range,
            "higher_rendition_role": higher_rendition_role,
            "conversion": "rgb16/65535->bt2100-pq-eotf->linear-srgb-d65-to-linear-rec2020-d65->float32",
        }
    )
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_ABSOLUTE_REC2020,
        pixel_sha256=_pixel_sha256(pixels),
        shape=tuple(int(value) for value in pixels.shape),
        render_bridge_id=LIBAVIF_GAINMAP_RENDER_BRIDGE_ID,
        provenance_fingerprint=provenance_fingerprint,
        reference_white_nits=LIBAVIF_GAINMAP_REFERENCE_WHITE_NITS,
        alpha_mode="absent",
    )
    result = PreparedLibAvifGainMapMatchViewV1(
        schema_id=LIBAVIF_GAINMAP_INGRESS_SCHEMA_ID,
        ingress_id=canonical_sha256(
            _ingress_payload(
                source_asset_sha256=source_sha256,
                encoded_sample_sha256=sample_sha256,
                higher_rendition_role=higher_rendition_role,
                descriptor=descriptor,
            )
        ),
        source_asset_sha256=source_sha256,
        encoded_sample_sha256=sample_sha256,
        decoder_version=decoder_version,
        source_profile_id=LIBAVIF_GAINMAP_SOURCE_PROFILE_ID,
        higher_rendition_role=higher_rendition_role,
        prepared_view=PreparedMatchViewV1(descriptor=descriptor, pixels=pixels),
    )
    validate_prepared_libavif_gainmap_match_view_v1(result)
    return result


def validate_prepared_libavif_gainmap_match_view_v1(
    value: PreparedLibAvifGainMapMatchViewV1,
) -> None:
    if not isinstance(value, PreparedLibAvifGainMapMatchViewV1):
        raise ReferenceMatchContractError("prepared libavif ingress has the wrong type")
    if value.schema_id != LIBAVIF_GAINMAP_INGRESS_SCHEMA_ID:
        raise ReferenceMatchContractError("unsupported libavif ingress schema")
    _require_sha256(value.ingress_id, "ingress_id")
    _require_sha256(value.source_asset_sha256, "source_asset_sha256")
    _require_sha256(value.encoded_sample_sha256, "encoded_sample_sha256")
    if value.decoder_version != LIBAVIF_GAINMAP_DECODER_VERSION:
        raise ReferenceMatchContractError("unsupported libavif decoder version")
    if value.source_profile_id != LIBAVIF_GAINMAP_SOURCE_PROFILE_ID:
        raise ReferenceMatchContractError("unsupported libavif source profile")
    if value.higher_rendition_role not in LIBAVIF_GAINMAP_ROLES:
        raise ReferenceMatchContractError("unsupported libavif rendition role")
    validate_prepared_match_view(value.prepared_view)
    descriptor = value.descriptor
    if (
        descriptor.profile_id != MATCH_PROFILE_ABSOLUTE_REC2020
        or descriptor.reference_white_nits != LIBAVIF_GAINMAP_REFERENCE_WHITE_NITS
        or descriptor.render_bridge_id != LIBAVIF_GAINMAP_RENDER_BRIDGE_ID
        or descriptor.alpha_mode != "absent"
    ):
        raise ReferenceMatchContractError("libavif MatchView semantics mismatch")
    expected_ingress_id = canonical_sha256(
        _ingress_payload(
            source_asset_sha256=value.source_asset_sha256,
            encoded_sample_sha256=value.encoded_sample_sha256,
            higher_rendition_role=value.higher_rendition_role,
            descriptor=descriptor,
        )
    )
    if value.ingress_id != expected_ingress_id:
        raise ReferenceMatchContractError("libavif ingress identity mismatch")


__all__ = [
    "LIBAVIF_GAINMAP_COLOR_PRIMARIES",
    "LIBAVIF_GAINMAP_DECODER_VERSION",
    "LIBAVIF_GAINMAP_INGRESS_SCHEMA_ID",
    "LIBAVIF_GAINMAP_REFERENCE_WHITE_NITS",
    "LIBAVIF_GAINMAP_RENDER_BRIDGE_ID",
    "LIBAVIF_GAINMAP_SOURCE_PROFILE_ID",
    "LIBAVIF_GAINMAP_TRANSFER_CHARACTERISTICS",
    "PreparedLibAvifGainMapMatchViewV1",
    "prepare_libavif_gainmap_match_view_v1",
    "validate_prepared_libavif_gainmap_match_view_v1",
]
