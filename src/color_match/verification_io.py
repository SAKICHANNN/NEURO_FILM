"""Bounded read-only file verification shared by staging verifiers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from src.inference import sha256_file

from .contracts import ReferenceMatchContractError


_HASH = re.compile(r"^[0-9a-f]{64}$")
MAX_REPORT_BYTES = 16 * 1024 * 1024


def read_hashed_utf8_report(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[str, str, str]:
    if (
        not isinstance(expected_sha256, str)
        or _HASH.fullmatch(expected_sha256) is None
    ):
        raise ReferenceMatchContractError(
            "expected_report_sha256 must be a lowercase SHA-256"
        )
    if not path.is_file():
        raise ReferenceMatchContractError(
            f"{label} must be an existing file"
        )
    try:
        size = path.stat().st_size
        raw = path.read_bytes()
    except OSError as exc:
        raise ReferenceMatchContractError(
            f"{label} is unreadable"
        ) from exc
    if size <= 0 or size > MAX_REPORT_BYTES or len(raw) != size:
        raise ReferenceMatchContractError(
            f"{label} violates the bounded size contract"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ReferenceMatchContractError(f"{label} hash mismatch")
    try:
        encoded = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ReferenceMatchContractError(
            f"{label} must be UTF-8"
        ) from exc
    return encoded, digest, str(path.resolve(strict=True))


def verify_hashed_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[str, str]:
    if not path.is_file():
        raise ReferenceMatchContractError(f"{label} is missing")
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise ReferenceMatchContractError(
            f"{label} is unreadable"
        ) from exc
    if digest != expected_sha256:
        raise ReferenceMatchContractError(f"{label} hash mismatch")
    return digest, str(path.resolve(strict=True))


__all__ = [
    "MAX_REPORT_BYTES",
    "read_hashed_utf8_report",
    "verify_hashed_file",
]
