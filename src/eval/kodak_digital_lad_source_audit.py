"""Exact source and safe-archive audit for Kodak Digital LAD assets."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile, ZipInfo

from pypdf import PdfReader


class KodakDigitalLadAuditError(ValueError):
    """Raised when a frozen source asset or archive boundary fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise KodakDigitalLadAuditError(f"missing source asset: {path}")
    if path.stat().st_size != expected_bytes:
        raise KodakDigitalLadAuditError(f"source byte count drift: {path}")
    if sha256_file(path) != expected_sha256:
        raise KodakDigitalLadAuditError(f"source SHA-256 drift: {path}")


def audit_html(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    required_urls: list[str],
) -> dict[str, Any]:
    verify_file(
        path, expected_bytes=expected_bytes, expected_sha256=expected_sha256
    )
    text = path.read_text(encoding="utf-8")
    missing = [url for url in required_urls if url not in text]
    if missing:
        raise KodakDigitalLadAuditError(f"official page link drift: {missing}")
    anchors = {
        "digital_lad_heading": 'id="digital-lad"' in text,
        "recorder_calibration": "calibration of a laser film recorder" in text,
        "dpx_format": "KODAK Digital LAD Test Image DPX Format" in text,
        "cineon_format": "KODAK Digital LAD Test Image Cineon Format" in text,
    }
    if not all(anchors.values()):
        raise KodakDigitalLadAuditError("official page Digital LAD anchor drift")
    return {
        "path": path.as_posix(),
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "required_urls_present": True,
        "anchor_presence": anchors,
    }


def audit_pdf(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_pages: int,
    required_phrases: dict[str, str],
) -> dict[str, Any]:
    verify_file(
        path, expected_bytes=expected_bytes, expected_sha256=expected_sha256
    )
    reader = PdfReader(path)
    if len(reader.pages) != expected_pages:
        raise KodakDigitalLadAuditError(f"PDF page count drift: {path}")
    normalized = " ".join(
        " ".join((page.extract_text() or "").split()) for page in reader.pages
    )
    folded = normalized.casefold()
    anchors = {
        key: phrase.casefold() in folded for key, phrase in required_phrases.items()
    }
    if not all(anchors.values()):
        missing = [key for key, present in anchors.items() if not present]
        raise KodakDigitalLadAuditError(f"PDF anchor drift: {path}: {missing}")
    return {
        "path": path.as_posix(),
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "page_count": len(reader.pages),
        "text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "anchor_presence": anchors,
    }


def _safe_member(info: ZipInfo) -> bool:
    path = PurePosixPath(info.filename.replace("\\", "/"))
    mode = info.external_attr >> 16
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and not stat.S_ISLNK(mode)
    )


def audit_zip_inventory(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    maximum_member_bytes: int,
    maximum_expansion_ratio: float,
) -> dict[str, Any]:
    verify_file(
        path, expected_bytes=expected_bytes, expected_sha256=expected_sha256
    )
    with ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        unique_names = len(set(names)) == len(names)
        safe_paths = all(_safe_member(info) for info in infos)
        total_uncompressed = sum(info.file_size for info in infos)
        total_compressed = sum(info.compress_size for info in infos)
        expansion_ratio = total_uncompressed / max(total_compressed, 1)
        largest_member_bytes = max((info.file_size for info in infos), default=0)
        members = [
            {
                "name": info.filename,
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "compression_method": info.compress_type,
                "safe_relative_non_symlink_path": _safe_member(info),
            }
            for info in infos
        ]
    return {
        "path": path.as_posix(),
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "member_count": len(members),
        "members": members,
        "unique_member_names": unique_names,
        "all_member_paths_safe": safe_paths,
        "total_uncompressed_bytes": total_uncompressed,
        "total_compressed_member_bytes": total_compressed,
        "expansion_ratio": expansion_ratio,
        "expansion_ratio_within_limit": expansion_ratio
        <= maximum_expansion_ratio,
        "largest_member_bytes": largest_member_bytes,
        "maximum_member_bytes": maximum_member_bytes,
        "largest_member_within_limit": largest_member_bytes
        <= maximum_member_bytes,
    }
