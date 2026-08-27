"""Private source-bound create-only wrapper for the P289 libavif profile."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from src.film_physics.create_only_file import publish_create_only

LIBAVIF_GAINMAP_ENCODER_PROFILE_ID = (
    "neuro-film.libavif-gainmap-hdr-pq-sdr-srgb.create-only.v1"
)
_BASE_CICP = (1, 16, 0)
_ALTERNATE_CICP = (1, 13, 0)
_BASE_HEADROOM = 1.3
_ALTERNATE_HEADROOM = 0.0


class LibavifGainMapEncoderError(RuntimeError):
    """Raised when the private P289 encoder profile cannot publish safely."""


@dataclass(frozen=True)
class GainMapAvifReceiptV1:
    """Deterministic identity receipt for one create-only AVIF publication."""

    alternate_cicp: tuple[int, int, int]
    alternate_headroom: float
    base_cicp: tuple[int, int, int]
    base_headroom: float
    bytes: int
    decoder_sha256: str
    height: int
    hdr_endpoint_sha256: str
    output_sha256: str
    profile_id: str
    runtime_sha256: str
    sdr_endpoint_sha256: str
    width: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready deterministic mapping."""

        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_endpoint(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise LibavifGainMapEncoderError("endpoint PNG decode failed")
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint16:
        raise LibavifGainMapEncoderError(
            "endpoint must be one uint16 three-channel PNG"
        )
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise LibavifGainMapEncoderError("endpoint dimensions are empty")
    return np.ascontiguousarray(value)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )


def _headroom(text: str, role: str) -> float:
    match = re.search(rf"\* {role} headroom:\s+([-+0-9.eE]+)", text)
    if match is None:
        raise LibavifGainMapEncoderError("encoded gain-map headroom is absent")
    return float(match.group(1))


def _cicp(info: str, *, alternate: bool) -> tuple[int, int, int]:
    if alternate:
        marker = " * Alternate image:"
        if marker not in info:
            raise LibavifGainMapEncoderError("encoded alternate image is absent")
        section = info.split(marker, 1)[1]
    else:
        section = info.split(" * Alternate image:", 1)[0]
    values: list[int] = []
    for pattern in (
        r"Color Primaries\s*:\s*(\d+)",
        r"Transfer Char\.\s*:\s*(\d+)",
        r"Matrix Coeffs\.\s*:\s*(\d+)",
    ):
        match = re.search(pattern, section)
        if match is None:
            raise LibavifGainMapEncoderError("encoded CICP is incomplete")
        values.append(int(match.group(1)))
    return values[0], values[1], values[2]


def _validate_profile(
    *,
    base_cicp: tuple[int, int, int],
    alternate_cicp: tuple[int, int, int],
    base_headroom: float,
    alternate_headroom: float,
) -> None:
    if (
        base_cicp != _BASE_CICP
        or alternate_cicp != _ALTERNATE_CICP
        or not math.isfinite(base_headroom)
        or not math.isfinite(alternate_headroom)
        or base_headroom != _BASE_HEADROOM
        or alternate_headroom != _ALTERNATE_HEADROOM
    ):
        raise LibavifGainMapEncoderError("encoder profile identity drift")


def encode_gainmap_avif_create_only_v1(
    hdr_endpoint: Path,
    sdr_endpoint: Path,
    destination: Path,
    *,
    expected_hdr_sha256: str,
    expected_sdr_sha256: str,
    runtime: Path,
    expected_runtime_sha256: str,
    decoder: Path,
    expected_decoder_sha256: str,
    base_cicp: tuple[int, int, int],
    alternate_cicp: tuple[int, int, int],
    base_headroom: float,
    alternate_headroom: float,
) -> GainMapAvifReceiptV1:
    """Encode explicit HDR/SDR endpoints and atomically publish one AVIF."""

    paths = (hdr_endpoint, sdr_endpoint, destination, runtime, decoder)
    if not all(isinstance(path, Path) and path.is_absolute() for path in paths):
        raise LibavifGainMapEncoderError("all encoder paths must be absolute")
    if hdr_endpoint == sdr_endpoint or destination in {hdr_endpoint, sdr_endpoint}:
        raise LibavifGainMapEncoderError("encoder paths must be distinct")
    if (
        not hdr_endpoint.is_file()
        or not sdr_endpoint.is_file()
        or not runtime.is_file()
        or not decoder.is_file()
        or not destination.parent.is_dir()
        or destination.exists()
    ):
        raise LibavifGainMapEncoderError("encoder path preflight failed")
    _validate_profile(
        base_cicp=base_cicp,
        alternate_cicp=alternate_cicp,
        base_headroom=base_headroom,
        alternate_headroom=alternate_headroom,
    )
    hdr_sha256 = _sha256_file(hdr_endpoint)
    sdr_sha256 = _sha256_file(sdr_endpoint)
    runtime_sha256 = _sha256_file(runtime)
    decoder_sha256 = _sha256_file(decoder)
    if hdr_sha256 != expected_hdr_sha256 or sdr_sha256 != expected_sdr_sha256:
        raise LibavifGainMapEncoderError("endpoint identity mismatch")
    if runtime_sha256 != expected_runtime_sha256:
        raise LibavifGainMapEncoderError("runtime identity mismatch")
    if decoder_sha256 != expected_decoder_sha256:
        raise LibavifGainMapEncoderError("decoder identity mismatch")
    hdr_pixels = _decode_endpoint(hdr_endpoint)
    sdr_pixels = _decode_endpoint(sdr_endpoint)
    if hdr_pixels.shape != sdr_pixels.shape:
        raise LibavifGainMapEncoderError("endpoint dimensions differ")
    if np.array_equal(hdr_pixels, sdr_pixels):
        raise LibavifGainMapEncoderError("endpoint images are not distinct")

    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.stage"
    command = [
        str(runtime),
        "combine",
        str(hdr_endpoint),
        str(sdr_endpoint),
        str(stage),
        "--downscaling",
        "1",
        "--qgain-map",
        "100",
        "--depth-gain-map",
        "12",
        "--yuv-gain-map",
        "444",
        "--max-headroom",
        "1.3",
        "--cicp-base",
        "1/16/0",
        "--cicp-alternate",
        "1/13/0",
        "--speed",
        "10",
        "--qcolor",
        "100",
        "--qalpha",
        "100",
        "--grid",
        "1x1",
        "--yuv",
        "444",
        "--depth",
        "12",
        "--jobs",
        "1",
    ]
    try:
        result = _run(command)
        if result.returncode != 0 or not stage.is_file() or stage.stat().st_size == 0:
            raise LibavifGainMapEncoderError("official encoder failed")
        metadata = _run([str(runtime), "printmetadata", str(stage), "--jobs", "1"])
        info = _run([str(decoder), "--info", str(stage)])
        if metadata.returncode != 0 or info.returncode != 0:
            raise LibavifGainMapEncoderError("encoded media validation failed")
        metadata_text = metadata.stdout + "\n" + metadata.stderr
        info_text = info.stdout + "\n" + info.stderr
        if (
            _headroom(metadata_text, "Base") != base_headroom
            or _headroom(metadata_text, "Alternate") != alternate_headroom
            or _cicp(info_text, alternate=False) != base_cicp
            or _cicp(info_text, alternate=True) != alternate_cicp
            or "Gain map" not in info_text
        ):
            raise LibavifGainMapEncoderError("encoded media profile mismatch")
        output_sha256 = _sha256_file(stage)
        output_bytes = stage.stat().st_size
        publish_create_only(stage, destination)
        return GainMapAvifReceiptV1(
            alternate_cicp=alternate_cicp,
            alternate_headroom=alternate_headroom,
            base_cicp=base_cicp,
            base_headroom=base_headroom,
            bytes=output_bytes,
            decoder_sha256=decoder_sha256,
            height=int(hdr_pixels.shape[0]),
            hdr_endpoint_sha256=hdr_sha256,
            output_sha256=output_sha256,
            profile_id=LIBAVIF_GAINMAP_ENCODER_PROFILE_ID,
            runtime_sha256=runtime_sha256,
            sdr_endpoint_sha256=sdr_sha256,
            width=int(hdr_pixels.shape[1]),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise LibavifGainMapEncoderError("gain-map AVIF publication failed") from exc
    finally:
        stage.unlink(missing_ok=True)


__all__ = [
    "LIBAVIF_GAINMAP_ENCODER_PROFILE_ID",
    "GainMapAvifReceiptV1",
    "LibavifGainMapEncoderError",
    "encode_gainmap_avif_create_only_v1",
]
