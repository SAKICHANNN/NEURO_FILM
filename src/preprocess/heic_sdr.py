"""Metadata-first admission and decode for a strict ordinary SDR HEIC subset."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageCms

_HEVC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx"}
_HEIF_BRANDS = _HEVC_BRANDS | {b"mif1", b"msf1"}
_EXPECTED_PI_HEIF_VERSION = "1.4.0"
_STRICT_SRGB_NCLX = {
    "color_primaries": 1,
    "transfer_characteristics": 13,
    "matrix_coefficients": 6,
    "full_range_flag": True,
}
_GAIN_MAP_MARKERS = (
    b"http://ns.adobe.com/hdr-gain-map/1.0/",
    b"hdrgm:version",
    b"urn:com:apple:photo:2020:aux:hdrgainmap",
    b"urn:iso:std:iso:ts:21496:-1",
)


class StrictSdrHeicError(ValueError):
    """Raised when HEIC cannot prove the frozen ordinary-SDR subset."""


@dataclass(frozen=True)
class StrictSdrHeicInfo:
    """HEIC facts proven before primary image samples are materialized."""

    width: int
    height: int
    mode: str
    bit_depth: int
    frame_count: int
    primary_index: int
    colour_kind: str
    icc_profile: bytes | None
    nclx: tuple[int, int, int, bool] | None
    exif_orientation: int | None
    mimetype: str


def _ftyp_brands(path: Path) -> tuple[bytes, tuple[bytes, ...]] | None:
    try:
        with Path(path).open("rb") as handle:
            header = handle.read(16)
            if len(header) < 16 or header[4:8] != b"ftyp":
                return None
            size = int.from_bytes(header[:4], "big")
            if size < 16 or size > 4096:
                return None
            handle.seek(0)
            payload = handle.read(size)
    except OSError:
        return None
    if len(payload) != size or (size - 16) % 4:
        return None
    major = payload[8:12]
    compatible = tuple(payload[index : index + 4] for index in range(16, size, 4))
    return major, compatible


def looks_like_heic(path: Path) -> bool:
    """Return whether the file-type box declares an HEVC HEIF family."""

    brands = _ftyp_brands(Path(path))
    if brands is None:
        return False
    major, compatible = brands
    return bool(({major, *compatible} & _HEVC_BRANDS) and major in _HEIF_BRANDS)


def _pi_heif() -> Any:
    try:
        import pi_heif
    except ImportError as exc:
        raise StrictSdrHeicError(
            "pi-heif 1.4.0 decode-only runtime is unavailable"
        ) from exc
    if getattr(pi_heif, "__version__", None) != _EXPECTED_PI_HEIF_VERSION:
        raise StrictSdrHeicError("strict SDR HEIC requires exact pi-heif 1.4.0")
    return pi_heif


def _exif_orientation(exif: bytes | None) -> int | None:
    if not exif:
        return None
    try:
        parsed = Image.Exif()
        parsed.load(exif)
        orientation = parsed.get(274)
        return int(orientation) if orientation else None
    except (AttributeError, OSError, SyntaxError, TypeError, ValueError):
        return None


def _validate_icc(profile: bytes) -> None:
    try:
        ImageCms.ImageCmsProfile(BytesIO(profile))
    except Exception as exc:
        raise StrictSdrHeicError("HEIC embedded ICC profile is invalid") from exc


def _nclx_tuple(value: object) -> tuple[int, int, int, bool] | None:
    if not isinstance(value, dict):
        return None
    try:
        return (
            int(value["color_primaries"]),
            int(value["transfer_characteristics"]),
            int(value["matrix_coefficients"]),
            bool(value["full_range_flag"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _inspect_opened(file: Any, mimetype: str) -> StrictSdrHeicInfo:
    if mimetype != "image/heic":
        raise StrictSdrHeicError(
            "strict SDR HEIC requires a still-image image/heic container"
        )
    if len(file) != 1:
        raise StrictSdrHeicError("strict SDR HEIC requires exactly one top-level image")
    if file.primary_index != 0:
        raise StrictSdrHeicError("strict SDR HEIC requires primary index zero")
    image = file[0]
    info = image.info
    if info.get("primary") is not True:
        raise StrictSdrHeicError("strict SDR HEIC primary-image identity is invalid")
    bit_depth = info.get("bit_depth")
    if isinstance(bit_depth, bool) or bit_depth != 8:
        raise StrictSdrHeicError("strict SDR HEIC requires exactly 8-bit samples")
    if image.mode != "RGB":
        raise StrictSdrHeicError("strict SDR HEIC requires exactly RGB without alpha")
    if info.get("aux"):
        raise StrictSdrHeicError("strict SDR HEIC forbids auxiliary or gain-map images")
    if info.get("depth_images"):
        raise StrictSdrHeicError("strict SDR HEIC forbids depth images")
    forbidden_metadata = (
        "content_light_level",
        "mastering_display_colour_volume",
        "ambient_viewing_environment",
    )
    if any(info.get(key) for key in forbidden_metadata):
        raise StrictSdrHeicError("strict SDR HEIC forbids HDR display metadata")
    metadata_payload = b"".join(
        bytes(value)
        for value in (info.get("exif"), info.get("xmp"))
        if isinstance(value, (bytes, bytearray))
    ).lower()
    if any(marker in metadata_payload for marker in _GAIN_MAP_MARKERS):
        raise StrictSdrHeicError("strict SDR HEIC forbids gain-map metadata")

    profile_value = info.get("icc_profile")
    profile = bytes(profile_value) if profile_value else None
    nclx = _nclx_tuple(info.get("nclx_profile"))
    if profile is not None and nclx is not None:
        raise StrictSdrHeicError(
            "HEIC contains ambiguous ICC and NCLX colour identities"
        )
    if profile is not None:
        if info.get("icc_profile_type") not in {"prof", "rICC"}:
            raise StrictSdrHeicError("HEIC embedded ICC profile type is unsupported")
        _validate_icc(profile)
        colour_kind = "icc"
    elif nclx is not None:
        expected = tuple(_STRICT_SRGB_NCLX.values())
        if nclx != expected:
            raise StrictSdrHeicError(
                "strict SDR HEIC requires NCLX primaries=1 transfer=13 matrix=6 full-range"
            )
        colour_kind = "nclx"
    else:
        raise StrictSdrHeicError(
            "strict SDR HEIC requires explicit ICC or NCLX colour identity"
        )

    width, height = image.size
    if width < 1 or height < 1:
        raise StrictSdrHeicError("strict SDR HEIC dimensions are invalid")
    orientation = _exif_orientation(info.get("exif"))
    if orientation is not None and orientation not in range(1, 9):
        raise StrictSdrHeicError("HEIC EXIF orientation is invalid")
    return StrictSdrHeicInfo(
        width=int(width),
        height=int(height),
        mode=image.mode,
        bit_depth=bit_depth,
        frame_count=len(file),
        primary_index=file.primary_index,
        colour_kind=colour_kind,
        icc_profile=profile,
        nclx=nclx,
        exif_orientation=orientation,
        mimetype=mimetype,
    )


def inspect_strict_sdr_heic(path: Path) -> StrictSdrHeicInfo:
    """Inspect strict HEIC metadata without materializing primary image samples."""

    path = Path(path)
    if not looks_like_heic(path):
        raise StrictSdrHeicError("input is not an HEVC HEIF container")
    decoder = _pi_heif()
    try:
        mimetype = decoder.get_file_mimetype(path)
        opened = decoder.open_heif(path, convert_hdr_to_8bit=False)
    except (EOFError, OSError, RuntimeError, SyntaxError, TypeError, ValueError) as exc:
        raise StrictSdrHeicError(f"invalid HEIC input: {exc}") from exc
    return _inspect_opened(opened, mimetype)


def decode_strict_sdr_heic(path: Path, expected: StrictSdrHeicInfo) -> Image.Image:
    """Decode the frozen subset after repeating and matching metadata inspection."""

    path = Path(path)
    decoder = _pi_heif()
    try:
        mimetype = decoder.get_file_mimetype(path)
        opened = decoder.open_heif(path, convert_hdr_to_8bit=False)
        actual = _inspect_opened(opened, mimetype)
        if actual != expected:
            raise StrictSdrHeicError(
                "HEIC metadata changed between inspection and decode"
            )
        image = opened[0]
        data = bytes(image.data)
    except StrictSdrHeicError:
        raise
    except (EOFError, OSError, RuntimeError, SyntaxError, TypeError, ValueError) as exc:
        raise StrictSdrHeicError(f"HEIC pixel decode failed: {exc}") from exc
    expected_bytes = actual.width * actual.height * 3
    if len(data) != expected_bytes:
        raise StrictSdrHeicError("HEIC decoded byte length does not match metadata")
    result = Image.frombytes("RGB", (actual.width, actual.height), data)
    if actual.icc_profile is not None:
        result.info["icc_profile"] = actual.icc_profile
    return result
