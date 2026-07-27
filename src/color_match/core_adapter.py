"""Neuro-Film WorkingImage adapter for the external-core consumer contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from src.preprocess.types import WorkingImage

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_contracts import (
    MATCH_PROFILE_DISPLAY_REC2020,
    MATCH_PROFILE_DISPLAY_SRGB,
    CapabilitiesV1,
    MatchViewV1,
    make_match_view,
    validate_capabilities,
    validate_match_view,
)


WORKING_IMAGE_BRIDGE_ID = "neuro-film.working-image-direct.v1"
_WORKING_PROFILE = {
    "linear_srgb": MATCH_PROFILE_DISPLAY_SRGB,
    "linear_rec2020": MATCH_PROFILE_DISPLAY_REC2020,
}


@dataclass(frozen=True)
class PreparedMatchViewV1:
    """One isolated read-only float32 buffer and its bound descriptor."""

    descriptor: MatchViewV1
    pixels: np.ndarray


def _canonical_metadata(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ReferenceMatchContractError(
                f"{label} metadata must be finite"
            )
        return value
    if isinstance(value, np.generic):
        return _canonical_metadata(value.item(), label)
    if isinstance(value, (bytes, bytearray)):
        return {"bytes_hex": bytes(value).hex()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ReferenceMatchContractError(
                f"{label} metadata keys must be strings"
            )
        return {
            key: _canonical_metadata(value[key], f"{label}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_metadata(item, f"{label}[]")
            for item in value
        ]
    raise ReferenceMatchContractError(
        f"{label} metadata contains unsupported type "
        f"{type(value).__name__}"
    )


def _pixel_sha256(pixels: np.ndarray) -> str:
    """Hash exact row-major IEEE-754 binary32 network-order pixel bits."""

    encoded = np.asarray(pixels, dtype=">f4", order="C")
    return hashlib.sha256(encoded.tobytes(order="C")).hexdigest()


def _provenance_fingerprint(image: WorkingImage) -> str:
    payload = {
        "working_space": image.working_space,
        "transfer_state": image.transfer_state,
        "source_transfer_state": image.source_transfer_state,
        "source_profile": {
            "kind": image.source_profile.kind,
            "description": image.source_profile.description,
            "bytes_length": image.source_profile.bytes_length,
        },
        "hdr_metadata": _canonical_metadata(
            image.hdr_metadata,
            "hdr_metadata",
        ),
        "orientation_applied": image.orientation_applied,
        "alpha_policy": image.alpha_policy,
        "bit_depth_in": image.bit_depth_in,
        "warnings": [
            {"code": warning.code, "message": warning.message}
            for warning in image.warnings
        ],
    }
    return canonical_sha256(payload)


def _validate_working_image(image: WorkingImage) -> str:
    if not isinstance(image, WorkingImage):
        raise ReferenceMatchContractError(
            "core adapter input must be WorkingImage"
        )
    if image.transfer_state != "display_linear":
        raise ReferenceMatchContractError(
            "core adapter currently requires display-linear pixels"
        )
    if image.working_space not in _WORKING_PROFILE:
        raise ReferenceMatchContractError(
            "core adapter working space is unsupported"
        )
    if image.orientation_applied is not True:
        raise ReferenceMatchContractError(
            "core adapter requires applied orientation"
        )
    if image.alpha_policy != "absent":
        raise ReferenceMatchContractError(
            "core adapter requires an alpha-free WorkingImage"
        )
    return _WORKING_PROFILE[image.working_space]


def prepare_working_image_match_view(
    image: WorkingImage,
) -> PreparedMatchViewV1:
    """Copy a supported WorkingImage into an isolated core-consumer view."""

    profile_id = _validate_working_image(image)
    pixels = np.array(image.pixels, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(pixels).all():
        raise ReferenceMatchContractError(
            "core adapter pixels must be finite"
        )
    pixels.flags.writeable = False
    descriptor = make_match_view(
        profile_id=profile_id,
        pixel_sha256=_pixel_sha256(pixels),
        shape=tuple(int(value) for value in pixels.shape),
        render_bridge_id=WORKING_IMAGE_BRIDGE_ID,
        provenance_fingerprint=_provenance_fingerprint(image),
        alpha_mode="absent",
    )
    prepared = PreparedMatchViewV1(
        descriptor=descriptor,
        pixels=pixels,
    )
    validate_prepared_match_view(prepared)
    return prepared


def validate_prepared_match_view(value: PreparedMatchViewV1) -> None:
    if not isinstance(value, PreparedMatchViewV1):
        raise ReferenceMatchContractError(
            "prepared view must be PreparedMatchViewV1"
        )
    validate_match_view(value.descriptor)
    pixels = value.pixels
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.float32:
        raise ReferenceMatchContractError(
            "prepared pixels must be a float32 ndarray"
        )
    if tuple(pixels.shape) != value.descriptor.shape:
        raise ReferenceMatchContractError(
            "prepared pixel shape does not match descriptor"
        )
    if tuple(pixels.strides) != value.descriptor.strides_bytes:
        raise ReferenceMatchContractError(
            "prepared pixel strides do not match descriptor"
        )
    if not pixels.flags.c_contiguous:
        raise ReferenceMatchContractError(
            "prepared pixels must be C-contiguous"
        )
    if pixels.flags.writeable:
        raise ReferenceMatchContractError(
            "prepared pixels must be read-only"
        )
    if not np.isfinite(pixels).all():
        raise ReferenceMatchContractError(
            "prepared pixels must be finite"
        )
    if _pixel_sha256(pixels) != value.descriptor.pixel_sha256:
        raise ReferenceMatchContractError(
            "prepared pixel identity does not match descriptor"
        )


def validate_prepared_view_support(
    prepared: PreparedMatchViewV1,
    capabilities: CapabilitiesV1,
    *,
    algorithm_id: str,
    contract_schema_id: str,
    capability_requirements: Sequence[str] = (),
) -> None:
    """Require exact advertised support before any external-core invocation."""

    validate_prepared_match_view(prepared)
    validate_capabilities(capabilities)
    if prepared.descriptor.profile_id not in (
        capabilities.supported_profile_ids
    ):
        raise ReferenceMatchContractError(
            "prepared match profile is not advertised"
        )
    if algorithm_id not in capabilities.supported_algorithm_ids:
        raise ReferenceMatchContractError(
            "requested algorithm is not advertised"
        )
    if contract_schema_id not in capabilities.contract_schema_ids:
        raise ReferenceMatchContractError(
            "requested contract schema is not advertised"
        )
    missing = set(capability_requirements) - set(
        capabilities.feature_flags
    )
    if missing:
        raise ReferenceMatchContractError(
            f"missing required core capabilities: {sorted(missing)}"
        )


__all__ = [
    "WORKING_IMAGE_BRIDGE_ID",
    "PreparedMatchViewV1",
    "prepare_working_image_match_view",
    "validate_prepared_match_view",
    "validate_prepared_view_support",
]
