"""Raster image inspection and SDR decode helpers."""

from __future__ import annotations

import zlib
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .color_management import REC2020_SDR_CICP, rec2020_to_linear_rec2020
from .output_encode import (
    normalized_icc_profile_sha256,
    srgb_icc_profile_fingerprint_sha256,
)
from .prophoto_icc import (
    ProPhotoICCError,
    decode_prophoto_rgb16_to_linear_rec2020,
)
from .types import DecodeWarning, InputInspection, SourceProfile, WorkingImage

_MODE_BIT_DEPTH = {
    "1": 1,
    "L": 8,
    "LA": 8,
    "P": 8,
    "RGB": 8,
    "RGBA": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "I;16": 16,
    "I;16L": 16,
    "I;16B": 16,
    "I": 32,
    "F": 32,
}

_UNSUPPORTED_DYNAMIC_RANGE_FORMATS = {"HEIF", "HEIC", "AVIF"}
_GAIN_MAP_PAYLOAD_MARKERS = (
    b"http://ns.adobe.com/hdr-gain-map/1.0/",
    b"hdrgm:version",
    b"item:semantic=\"gainmap\"",
    b"urn:com:apple:photo:2020:aux:hdrgainmap",
)
_PAYLOAD_SCAN_BYTES = 4 * 1024 * 1024
_MAX_METADATA_PAYLOAD_BYTES = 1024 * 1024
_MAX_DECODED_METADATA_BYTES = 4 * 1024 * 1024
_JPEG_SCANNED_METADATA_MARKERS = {0xE1, 0xE2, 0xFE}
_JPEG_STANDALONE_MARKERS = {0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)}
_PNG_TEXT_CHUNKS = {b"tEXt", b"zTXt", b"iTXt"}


def _orientation(image: Image.Image) -> int | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    value = exif.get(274)
    return int(value) if value else None


def _png_cicp(path: Path) -> bytes | None:
    found: bytes | None = None
    seen_idat = False
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        while True:
            length_bytes = handle.read(4)
            if len(length_bytes) != 4:
                raise ValueError("truncated PNG chunk length")
            length = int.from_bytes(length_bytes, "big")
            if length > 0x7FFFFFFF:
                raise ValueError("invalid PNG chunk length")
            chunk_type = handle.read(4)
            if len(chunk_type) != 4:
                raise ValueError("truncated PNG chunk type")
            if chunk_type == b"cICP":
                if seen_idat:
                    raise ValueError("PNG cICP must precede IDAT")
                if found is not None:
                    raise ValueError("PNG must not contain duplicate cICP chunks")
                payload = handle.read(length)
                crc_bytes = handle.read(4)
                if len(payload) != length or len(crc_bytes) != 4:
                    raise ValueError("truncated PNG cICP chunk")
                expected = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
                if int.from_bytes(crc_bytes, "big") != expected:
                    raise ValueError("PNG cICP CRC mismatch")
                if length != 4:
                    raise ValueError("PNG cICP payload must contain four bytes")
                found = payload
            else:
                handle.seek(length + 4, 1)
            if chunk_type == b"IDAT":
                seen_idat = True
            if chunk_type == b"IEND":
                return found


def _source_profile(image: Image.Image, cicp: bytes | None = None) -> SourceProfile:
    if cicp is not None:
        return SourceProfile("cicp", f"PNG cICP {cicp.hex()}", len(cicp))
    icc = image.info.get("icc_profile")
    if icc:
        return SourceProfile("icc", "embedded ICC profile", len(icc))
    if image.format in {"HEIF", "HEIC", "AVIF"}:
        return SourceProfile("nclx", "container color metadata may be present; backend support limited")
    return SourceProfile("assumed_srgb", "no embedded ICC profile; assuming sRGB")


def _hdr_metadata(image: Image.Image, cicp: bytes | None = None) -> dict[str, Any]:
    keys = {}
    if cicp is not None:
        keys["cicp"] = cicp.hex()
    for key, value in image.info.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("hdr", "gain", "cicp", "nclx", "mastering", "icc")):
            if isinstance(value, bytes):
                keys[key] = f"bytes:{len(value)}"
            else:
                keys[key] = str(value)
    return keys


def _png_bit_depth(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(26)
    if len(header) < 26 or not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR":
        raise ValueError("invalid PNG IHDR")
    return int(header[24])


def _bounded_payload_markers(path: Path) -> list[str]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        payload = handle.read(_PAYLOAD_SCAN_BYTES)
        if size > _PAYLOAD_SCAN_BYTES:
            handle.seek(max(0, size - _PAYLOAD_SCAN_BYTES))
            payload += handle.read(_PAYLOAD_SCAN_BYTES)
    lowered = payload.lower()
    return [marker.decode("ascii") for marker in _GAIN_MAP_PAYLOAD_MARKERS if marker in lowered]


def _recognized_payload_markers(payload: bytes) -> list[str]:
    lowered = payload.lower()
    return [marker.decode("ascii") for marker in _GAIN_MAP_PAYLOAD_MARKERS if marker in lowered]


def _jpeg_metadata_markers(path: Path) -> list[str]:
    """Scan bounded JPEG metadata segments before entropy-coded image data."""
    matches: list[str] = []
    inspected_bytes = 0
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError("invalid JPEG SOI")
        while True:
            prefix = handle.read(1)
            if not prefix:
                raise ValueError("truncated JPEG before SOS/EOI")
            if prefix != b"\xff":
                raise ValueError("invalid JPEG marker prefix")
            while True:
                code_bytes = handle.read(1)
                if not code_bytes:
                    raise ValueError("truncated JPEG marker")
                if code_bytes != b"\xff":
                    break
            code = code_bytes[0]
            if code == 0x00:
                raise ValueError("stuffed JPEG byte outside entropy data")
            if code == 0xDA or code == 0xD9:
                return list(dict.fromkeys(matches))
            if code in _JPEG_STANDALONE_MARKERS:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                raise ValueError("truncated JPEG segment length")
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                raise ValueError("invalid JPEG segment length")
            payload_length = length - 2
            if code in _JPEG_SCANNED_METADATA_MARKERS:
                if payload_length > _MAX_METADATA_PAYLOAD_BYTES:
                    raise ValueError("JPEG metadata payload exceeds scan budget")
                inspected_bytes += payload_length
                if inspected_bytes > _MAX_DECODED_METADATA_BYTES:
                    raise ValueError("JPEG metadata exceeds cumulative scan budget")
                payload = handle.read(payload_length)
                if len(payload) != payload_length:
                    raise ValueError("truncated JPEG metadata segment")
                matches.extend(_recognized_payload_markers(payload))
            else:
                handle.seek(payload_length, 1)


def _bounded_zlib_text(payload: bytes, maximum_bytes: int) -> bytes:
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(payload, maximum_bytes + 1)
    if len(decoded) > maximum_bytes or not decoder.eof or decoder.unconsumed_tail:
        raise ValueError("compressed PNG text exceeds scan budget or is incomplete")
    if decoder.unused_data:
        raise ValueError("compressed PNG text has trailing data")
    return decoded


def _png_text_payload(payload: bytes, chunk_type: bytes) -> bytes:
    if chunk_type == b"tEXt":
        if b"\x00" not in payload:
            raise ValueError("invalid PNG tEXt payload")
        return payload
    if chunk_type == b"zTXt":
        try:
            keyword, remainder = payload.split(b"\x00", 1)
        except ValueError as exc:
            raise ValueError("invalid PNG zTXt payload") from exc
        if not keyword or len(remainder) < 2 or remainder[0] != 0:
            raise ValueError("invalid PNG zTXt compression fields")
        return keyword + b"\x00" + _bounded_zlib_text(
            remainder[1:], _MAX_METADATA_PAYLOAD_BYTES
        )

    try:
        keyword, remainder = payload.split(b"\x00", 1)
    except ValueError as exc:
        raise ValueError("invalid PNG iTXt keyword") from exc
    if not keyword or len(remainder) < 2:
        raise ValueError("invalid PNG iTXt compression fields")
    compressed, method = remainder[0], remainder[1]
    if compressed not in {0, 1} or method != 0:
        raise ValueError("invalid PNG iTXt compression fields")
    remainder = remainder[2:]
    try:
        language, remainder = remainder.split(b"\x00", 1)
        translated, text = remainder.split(b"\x00", 1)
    except ValueError as exc:
        raise ValueError("invalid PNG iTXt language fields") from exc
    if compressed:
        text = _bounded_zlib_text(text, _MAX_METADATA_PAYLOAD_BYTES)
    return b"\x00".join((keyword, language, translated, text))


def _png_metadata_markers(path: Path) -> list[str]:
    """Scan PNG textual metadata chunks without retaining image payloads."""
    matches: list[str] = []
    decoded_bytes = 0
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("invalid PNG signature")
        while True:
            length_bytes = handle.read(4)
            if len(length_bytes) != 4:
                raise ValueError("truncated PNG chunk length")
            length = int.from_bytes(length_bytes, "big")
            if length > 0x7FFFFFFF:
                raise ValueError("invalid PNG chunk length")
            chunk_type = handle.read(4)
            if len(chunk_type) != 4:
                raise ValueError("truncated PNG chunk type")
            if chunk_type in _PNG_TEXT_CHUNKS:
                if length > _MAX_METADATA_PAYLOAD_BYTES:
                    raise ValueError("PNG text payload exceeds scan budget")
                payload = handle.read(length)
                crc_bytes = handle.read(4)
                if len(payload) != length or len(crc_bytes) != 4:
                    raise ValueError("truncated PNG text chunk")
                expected = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
                if int.from_bytes(crc_bytes, "big") != expected:
                    raise ValueError("PNG text CRC mismatch")
                decoded = _png_text_payload(payload, chunk_type)
                decoded_bytes += len(decoded)
                if decoded_bytes > _MAX_DECODED_METADATA_BYTES:
                    raise ValueError("PNG text exceeds cumulative scan budget")
                matches.extend(_recognized_payload_markers(decoded))
            else:
                handle.seek(length, 1)
                if len(handle.read(4)) != 4:
                    raise ValueError("truncated PNG chunk")
            if chunk_type == b"IEND":
                return list(dict.fromkeys(matches))


def _structured_payload_markers(path: Path, format_name: str) -> tuple[list[str], list[str]]:
    try:
        if format_name == "JPEG":
            return _jpeg_metadata_markers(path), []
        if format_name == "PNG":
            return _png_metadata_markers(path), []
    except (OSError, ValueError, zlib.error):
        return [], [f"structured:{format_name.casefold()}_metadata_untrusted"]
    return [], []


def unsupported_dynamic_range_signals(path: Path, inspection: InputInspection) -> list[str]:
    """Return deterministic signals that require an unavailable HDR decode path."""
    signals: list[str] = []
    if inspection.format_name in _UNSUPPORTED_DYNAMIC_RANGE_FORMATS:
        signals.append(f"container:{inspection.format_name}")
    for key in sorted(inspection.hdr_metadata, key=lambda value: str(value).casefold()):
        lowered = str(key).casefold()
        if lowered == "cicp" and inspection.hdr_metadata[key] == REC2020_SDR_CICP.hex():
            continue
        if lowered in {"icc", "icc_profile"}:
            continue
        if any(token in lowered for token in ("hdr", "gain", "cicp", "nclx", "mastering")):
            signals.append(f"metadata:{key}")
    if inspection.format_name in {"JPEG", "PNG"}:
        structured, failures = _structured_payload_markers(path, inspection.format_name)
        signals.extend(f"payload:{marker}" for marker in structured)
        signals.extend(failures)
        signals.extend(f"payload:{marker}" for marker in _bounded_payload_markers(path))
    return list(dict.fromkeys(signals))


def inspect_raster(path: Path) -> InputInspection:
    warnings: list[DecodeWarning] = []
    try:
        cicp = _png_cicp(path)
        with Image.open(path) as image:
            profile = _source_profile(image, cicp)
            if profile.kind == "assumed_srgb":
                warnings.append(DecodeWarning("assumed_srgb", profile.description))
            if image.format in {"HEIF", "HEIC", "AVIF"}:
                warnings.append(
                    DecodeWarning(
                        "limited_heif_hdr",
                        "HEIF/AVIF HDR or gain-map reconstruction is not implemented in this backend.",
                    )
                )
            bit_depth = _MODE_BIT_DEPTH.get(image.mode)
            if image.format == "TIFF":
                try:
                    with tifffile.TiffFile(path) as tif:
                        values = tif.pages[0].tags["BitsPerSample"].value
                    samples = (values,) if isinstance(values, int) else tuple(values)
                    if samples and len(set(int(value) for value in samples)) == 1:
                        bit_depth = int(samples[0])
                except (KeyError, OSError, tifffile.TiffFileError, TypeError, ValueError):
                    warnings.append(DecodeWarning("tiff_bit_depth_unknown", "TIFF BitsPerSample could not be read."))
            elif image.format == "PNG":
                try:
                    bit_depth = _png_bit_depth(path)
                except (OSError, ValueError):
                    warnings.append(DecodeWarning("png_bit_depth_unknown", "PNG IHDR bit depth could not be read."))
            inspection = InputInspection(
                path=path,
                exists=True,
                source_kind="raster",
                format_name=image.format or "unknown",
                mode=image.mode,
                width=image.width,
                height=image.height,
                bit_depth=bit_depth,
                has_alpha=image.mode in {"LA", "RGBA"} or "transparency" in image.info,
                orientation=_orientation(image),
                frame_count=getattr(image, "n_frames", 1),
                source_profile=profile,
                transfer_state="display_referred",
                hdr_metadata=_hdr_metadata(image, cicp),
                warnings=warnings,
            )
            signals = unsupported_dynamic_range_signals(path, inspection)
            if signals:
                if any(signal.startswith("metadata:cicp") for signal in signals):
                    message = "Unsupported PNG cICP color/dynamic-range signalling; signals="
                else:
                    message = "HDR/gain-map reconstruction is not implemented; signals="
                inspection.warnings.append(
                    DecodeWarning(
                        "unsupported_dynamic_range",
                        message + ",".join(signals),
                    )
                )
            return inspection
    except UnidentifiedImageError:
        return InputInspection(path=path, exists=path.exists(), source_kind="unknown")


def _convert_with_icc(image: Image.Image, warnings: list[DecodeWarning]) -> Image.Image:
    icc = image.info.get("icc_profile")
    if not icc:
        warnings.append(DecodeWarning("assumed_srgb", "No ICC profile; interpreting raster RGB as sRGB."))
        return image.convert("RGB")
    try:
        src = ImageCms.ImageCmsProfile(BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(image.convert("RGB"), src, dst, outputMode="RGB")
        return converted
    except Exception as exc:  # noqa: BLE001 - Pillow/LittleCMS exposes several profile failures.
        raise ValueError("embedded ICC conversion failed; refusing unprofiled RGB fallback") from exc


def _reject_or_strip_alpha(
    image: Image.Image,
    inspection: InputInspection,
    warnings: list[DecodeWarning],
) -> Image.Image:
    """Fail closed on real transparency; strip only a provably opaque channel."""
    if not inspection.has_alpha:
        return image
    alpha = image.convert("RGBA").getchannel("A")
    minimum, maximum = alpha.getextrema()
    if minimum < 255 or maximum < 255:
        raise ValueError(
            "raster alpha preservation/compositing is not implemented; "
            "refusing transparent hidden-RGB fallback"
        )
    warnings.append(
        DecodeWarning(
            "opaque_alpha_discarded",
            "Fully opaque alpha was verified and stripped before RGB rendering.",
        )
    )
    return image


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _is_supported_srgb_profile(profile: bytes) -> bool:
    try:
        return normalized_icc_profile_sha256(profile) == srgb_icc_profile_fingerprint_sha256()
    except ValueError:
        return False


def _load_srgb16_tiff(path: Path, inspection: InputInspection) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        array = page.asarray()
        profile_tag = page.tags.get(34675)
        profile = bytes(profile_tag.value) if profile_tag is not None else b""
        orientation_tag = page.tags.get("Orientation")
        orientation = int(orientation_tag.value) if orientation_tag is not None else 1
    if array.dtype != np.uint16 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("high-precision TIFF ingress requires contiguous uint16 RGB")
    if orientation != 1:
        raise ValueError("high-precision TIFF orientation handling is not implemented")
    if profile and not _is_supported_srgb_profile(profile):
        raise ValueError("16-bit TIFF embedded ICC conversion is not implemented for this profile")
    if inspection.source_profile.kind == "icc" and not profile:
        raise ValueError("TIFF ICC inspection/decode mismatch")
    return array.astype(np.float32) / 65535.0


def _load_prophoto16_tiff(path: Path, inspection: InputInspection) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        array = page.asarray()
        profile_tag = page.tags.get(34675)
        profile = bytes(profile_tag.value) if profile_tag is not None else b""
        orientation_tag = page.tags.get("Orientation")
        orientation = int(orientation_tag.value) if orientation_tag is not None else 1
    if array.dtype != np.uint16 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("high-precision TIFF ingress requires contiguous uint16 RGB")
    if orientation != 1:
        raise ValueError("high-precision TIFF orientation handling is not implemented")
    if inspection.source_profile.kind != "icc" or not profile:
        raise ValueError("TIFF ICC inspection/decode mismatch")
    try:
        return decode_prophoto_rgb16_to_linear_rec2020(array, profile)
    except ProPhotoICCError as exc:
        raise ValueError(
            "16-bit TIFF embedded ICC conversion is not implemented for this profile"
        ) from exc


def _load_srgb16_png(path: Path, inspection: InputInspection) -> np.ndarray:
    import cv2

    array_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array_bgr is None or array_bgr.dtype != np.uint16 or array_bgr.ndim != 3 or array_bgr.shape[2] != 3:
        raise ValueError("high-precision PNG ingress requires contiguous uint16 RGB")
    if inspection.orientation not in {None, 1}:
        raise ValueError("high-precision PNG orientation handling is not implemented")
    with Image.open(path) as image:
        profile = bytes(image.info.get("icc_profile") or b"")
    if profile and not _is_supported_srgb_profile(profile):
        raise ValueError("16-bit PNG embedded ICC conversion is not implemented for this profile")
    if inspection.source_profile.kind == "icc" and not profile:
        raise ValueError("PNG ICC inspection/decode mismatch")
    return array_bgr[..., ::-1].astype(np.float32) / 65535.0


def _load_rec2020_16_png(path: Path) -> np.ndarray:
    import cv2

    array_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        array_bgr is None
        or array_bgr.dtype != np.uint16
        or array_bgr.ndim != 3
        or array_bgr.shape[2] != 3
    ):
        raise ValueError("BT.2020 cICP ingress requires contiguous uint16 RGB PNG")
    encoded = array_bgr[..., ::-1].astype(np.float32) / 65535.0
    return rec2020_to_linear_rec2020(encoded)


def working_image_to_srgb_float(working: WorkingImage) -> np.ndarray:
    """Encode known linear-sRGB WorkingImage pixels to float display sRGB."""
    if working.working_space != "linear_srgb" or working.transfer_state not in {
        "display_linear",
        "scene_linear",
    }:
        raise ValueError(
            "sRGB adapter requires linear_srgb with display_linear or scene_linear pixels; "
            f"got {working.working_space}/{working.transfer_state}"
        )
    clipped = np.clip(working.pixels, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.asarray(np.clip(encoded, 0.0, 1.0), dtype=np.float32)


def working_image_to_legacy_srgb8(working: WorkingImage) -> Image.Image:
    """Explicit temporary adapter from WorkingImage to the 8-bit legacy renderer."""
    encoded = working_image_to_srgb_float(working)
    return Image.fromarray(np.rint(encoded * 255.0).astype(np.uint8), mode="RGB")


def load_raster_working_image(path: Path) -> WorkingImage:
    inspection = inspect_raster(path)
    warnings = list(inspection.warnings)
    if inspection.source_kind != "raster":
        raise ValueError(f"Unsupported raster input: {path}")
    if inspection.frame_count != 1:
        raise ValueError(
            "multi-frame raster rendering is not implemented; "
            f"refusing silent frame-zero fallback (frame_count={inspection.frame_count})"
        )
    dynamic_range_signals = unsupported_dynamic_range_signals(path, inspection)
    if dynamic_range_signals:
        if any(signal.startswith("metadata:cicp") for signal in dynamic_range_signals):
            raise ValueError(
                "unsupported PNG cICP color/dynamic-range signalling; refusing fallback: "
                + ",".join(dynamic_range_signals)
            )
        raise ValueError(
            "HDR/gain-map reconstruction is not implemented; refusing SDR fallback: "
            + ",".join(dynamic_range_signals)
        )
    supported_rec2020 = (
        inspection.source_profile.kind == "cicp"
        and inspection.hdr_metadata.get("cicp") == REC2020_SDR_CICP.hex()
    )
    if supported_rec2020:
        if inspection.has_alpha:
            raise ValueError("BT.2020 cICP alpha ingress is not implemented")
        if (
            inspection.format_name != "PNG"
            or inspection.bit_depth != 16
            or inspection.mode != "RGB"
        ):
            raise ValueError("BT.2020 cICP ingress is limited to 16-bit RGB PNG")
        pixels = _load_rec2020_16_png(path)
        return WorkingImage(
            pixels=pixels,
            working_space="linear_rec2020",
            transfer_state="display_linear",
            source_transfer_state=inspection.transfer_state,
            source_profile=inspection.source_profile,
            hdr_metadata=inspection.hdr_metadata,
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=inspection.bit_depth,
            source_path=path,
            warnings=warnings,
        )
    if (
        inspection.format_name == "TIFF"
        and inspection.bit_depth == 16
        and inspection.source_profile.kind == "icc"
    ):
        with tifffile.TiffFile(path) as tif:
            profile_tag = tif.pages[0].tags.get(34675)
            profile = bytes(profile_tag.value) if profile_tag is not None else b""
        if profile and not _is_supported_srgb_profile(profile):
            pixels = _load_prophoto16_tiff(path, inspection)
            warnings.append(
                DecodeWarning(
                    "embedded_prophoto_to_linear_rec2020",
                    "Decoded the supported ProPhoto RGB matrix-shaper ICC profile to "
                    "unclipped linear Rec.2020; gamut mapping was not applied.",
                )
            )
            return WorkingImage(
                pixels=pixels,
                working_space="linear_rec2020",
                transfer_state="display_linear",
                source_transfer_state=inspection.transfer_state,
                source_profile=inspection.source_profile,
                hdr_metadata=inspection.hdr_metadata,
                orientation_applied=True,
                alpha_policy="absent",
                bit_depth_in=inspection.bit_depth,
                source_path=path,
                warnings=warnings,
            )
        arr = _load_srgb16_tiff(path, inspection)
        alpha_policy = "absent"
    elif inspection.format_name == "TIFF" and inspection.bit_depth == 16:
        arr = _load_srgb16_tiff(path, inspection)
        alpha_policy = "absent"
    elif inspection.format_name == "PNG" and inspection.bit_depth == 16:
        arr = _load_srgb16_png(path, inspection)
        alpha_policy = "absent"
    else:
        with Image.open(path) as raw_image:
            oriented = ImageOps.exif_transpose(raw_image)
            oriented = _reject_or_strip_alpha(oriented, inspection, warnings)
            alpha_policy = "absent"
            rgb_image = _convert_with_icc(oriented, warnings)
            arr = np.asarray(rgb_image, dtype=np.float32) / 255.0
    pixels = _srgb_to_linear(np.clip(arr, 0.0, 1.0))
    return WorkingImage(
        pixels=pixels,
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state=inspection.transfer_state,
        source_profile=inspection.source_profile,
        hdr_metadata=inspection.hdr_metadata,
        orientation_applied=True,
        alpha_policy=alpha_policy,
        bit_depth_in=inspection.bit_depth,
        source_path=path,
        warnings=warnings,
    )
