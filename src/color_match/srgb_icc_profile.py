"""Pinned sRGB ICC bytes shared by consumer conformance contracts."""

from __future__ import annotations

import base64
import binascii
import hashlib

from .contracts import ReferenceMatchContractError


SRGB_ICC_PROFILE_ID = "neuro-film.srgb-icc-profile.v1"
SRGB_ICC_PROFILE_SHA256 = (
    "217fe48ec958c667f8eef725aa27198f465df95d7662593b90d0a1cc30114356"
)
SRGB_ICC_PROFILE_SIZE_BYTES = 588
SRGB_ICC_PROFILE_CLAIM_CEILING = (
    "exact-profile-bytes-for-conformance-not-proof-of-application"
)
_SRGB_ICC_PROFILE_BASE64 = (
    "AAACTGxjbXMEQAAAbW50clJHQiBYWVogB9AAAQABAAAAAAAAYWNzcE1TRlQAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1sY21zAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALZGVzYwAAAQgAAAA2"
    "Y3BydAAAAUAAAABMd3RwdAAAAYwAAAAUY2hhZAAAAaAAAAAsclhZWgAAAcwAAAAU"
    "YlhZWgAAAeAAAAAUZ1hZWgAAAfQAAAAUclRSQwAAAggAAAAgZ1RSQwAAAggAAAAg"
    "YlRSQwAAAggAAAAgY2hybQAAAigAAAAkbWx1YwAAAAAAAAABAAAADGVuVVMAAAAa"
    "AAAAHABzAFIARwBCACAAYgB1AGkAbAB0AC0AaQBuAABtbHVjAAAAAAAAAAEAAAAM"
    "ZW5VUwAAADAAAAAcAE4AbwAgAGMAbwBwAHkAcgBpAGcAaAB0ACwAIAB1AHMAZQAg"
    "AGYAcgBlAGUAbAB5WFlaIAAAAAAAAPbWAAEAAAAA0y1zZjMyAAAAAAABDEIAAAXe"
    "///zJQAAB5MAAP2Q///7of///aIAAAPcAADAblhZWiAAAAAAAABvoAAAOPUAAAOQ"
    "WFlaIAAAAAAAACSfAAAPhAAAtsNYWVogAAAAAAAAYpcAALeHAAAY2XBhcmEAAAAA"
    "AAMAAAACZmYAAPKnAAANWQAAE9AAAApbY2hybQAAAAAAAwAAAACj1wAAVHsAAEzN"
    "AACZmgAAJmYAAA9c"
)


def _validate_profile(profile: bytes) -> None:
    if (
        not isinstance(profile, bytes)
        or len(profile) != SRGB_ICC_PROFILE_SIZE_BYTES
        or hashlib.sha256(profile).hexdigest()
        != SRGB_ICC_PROFILE_SHA256
        or int.from_bytes(profile[0:4], "big")
        != SRGB_ICC_PROFILE_SIZE_BYTES
        or profile[12:16] != b"mntr"
        or profile[16:20] != b"RGB "
        or profile[20:24] != b"XYZ "
        or profile[36:40] != b"acsp"
    ):
        raise ReferenceMatchContractError(
            "pinned sRGB ICC profile identity/header is invalid"
        )


def srgb_icc_profile_v1() -> bytes:
    try:
        profile = base64.b64decode(
            _SRGB_ICC_PROFILE_BASE64,
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise ReferenceMatchContractError(
            "pinned sRGB ICC profile encoding is invalid"
        ) from exc
    _validate_profile(profile)
    return profile


def srgb_icc_profile_conformance_v1() -> dict[str, object]:
    profile = srgb_icc_profile_v1()
    return {
        "schema_id": "neuro-film.srgb-icc-profile-conformance.v1",
        "profile_id": SRGB_ICC_PROFILE_ID,
        "profile_sha256": SRGB_ICC_PROFILE_SHA256,
        "profile_size_bytes": SRGB_ICC_PROFILE_SIZE_BYTES,
        "profile_base64": base64.b64encode(profile).decode("ascii"),
        "icc_declared_size_bytes": int.from_bytes(profile[0:4], "big"),
        "device_class": profile[12:16].decode("ascii"),
        "data_color_space": profile[16:20].decode("ascii"),
        "profile_connection_space": profile[20:24].decode("ascii"),
        "icc_signature": profile[36:40].decode("ascii"),
        "claim_ceiling": SRGB_ICC_PROFILE_CLAIM_CEILING,
    }


__all__ = [
    "SRGB_ICC_PROFILE_CLAIM_CEILING",
    "SRGB_ICC_PROFILE_ID",
    "SRGB_ICC_PROFILE_SHA256",
    "SRGB_ICC_PROFILE_SIZE_BYTES",
    "srgb_icc_profile_conformance_v1",
    "srgb_icc_profile_v1",
]
