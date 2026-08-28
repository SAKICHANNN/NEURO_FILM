"""Bounded create-only file transport for one scanner-chain profile bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_chain_bundle_execution import apply_scanner_chain_from_bundle
from src.film_physics.create_only_file import (
    PublishedFileIdentity,
    publish_create_only,
)
from src.film_physics.scanner_chain_profile import ScannerChainProfile

FILE_SCHEMA = "neuro_film.generic_scanner_chain_profile_file.v1"
MAXIMUM_FILE_BYTES = 4096
_FIELDS = {"schema", "profile_sha256", "profile"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase hexadecimal")


def encode_scanner_chain_profile_file(
    profile_bytes: bytes,
    *,
    expected_profile_sha256: str,
) -> bytes:
    """Encode one exact canonical profile into the bounded file envelope."""

    _validate_sha256(expected_profile_sha256, "expected_profile_sha256")
    profile = ScannerChainProfile.from_json_bytes(profile_bytes)
    if profile.canonical_bytes() != profile_bytes:
        raise ValueError("scanner chain profile bytes must be canonical")
    if profile.profile_sha256 != expected_profile_sha256:
        raise ValueError("scanner chain profile identity mismatch")
    payload = {
        "schema": FILE_SCHEMA,
        "profile_sha256": profile.profile_sha256,
        "profile": profile.to_dict(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(encoded) > MAXIMUM_FILE_BYTES:
        raise ValueError("scanner chain profile file exceeds bounded size")
    return encoded


def decode_scanner_chain_profile_file(
    file_bytes: bytes,
    *,
    expected_file_sha256: str,
    expected_profile_sha256: str,
) -> bytes:
    """Validate one exact file envelope and return canonical profile bytes."""

    if not isinstance(file_bytes, bytes):
        raise TypeError("scanner chain profile file must be bytes")
    _validate_sha256(expected_file_sha256, "expected_file_sha256")
    _validate_sha256(expected_profile_sha256, "expected_profile_sha256")
    if len(file_bytes) > MAXIMUM_FILE_BYTES:
        raise ValueError("scanner chain profile file exceeds bounded size")
    if _sha256(file_bytes) != expected_file_sha256:
        raise ValueError("scanner chain profile file identity mismatch")
    try:
        payload = json.loads(
            file_bytes.decode("ascii", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid scanner chain profile file JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("scanner chain profile file fields must match v1 exactly")
    if payload["schema"] != FILE_SCHEMA:
        raise ValueError("unsupported scanner chain profile file schema")
    if payload["profile_sha256"] != expected_profile_sha256:
        raise ValueError("scanner chain profile identity mismatch")
    if not isinstance(payload["profile"], dict):
        raise TypeError("scanner chain profile payload must be an object")
    profile = ScannerChainProfile.from_dict(payload["profile"])
    profile_bytes = profile.canonical_bytes()
    if profile.profile_sha256 != expected_profile_sha256:
        raise ValueError("scanner chain profile payload identity mismatch")
    if (
        encode_scanner_chain_profile_file(
            profile_bytes,
            expected_profile_sha256=expected_profile_sha256,
        )
        != file_bytes
    ):
        raise ValueError("scanner chain profile file must be canonical")
    return profile_bytes


def load_scanner_chain_profile_file(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_profile_sha256: str,
) -> bytes:
    """Read one bounded regular file without following symlinks."""

    source = Path(path)
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode) or source.is_symlink():
        raise ValueError("scanner chain profile path must be a regular non-symlink file")
    if before.st_size > MAXIMUM_FILE_BYTES:
        raise ValueError("scanner chain profile file exceeds bounded size")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, os.O_RDONLY | nofollow)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise ValueError("scanner chain profile file identity changed before read")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read(MAXIMUM_FILE_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = source.lstat()
    expected_identity = (before.st_dev, before.st_ino, before.st_size)
    if (after.st_dev, after.st_ino, after.st_size) != expected_identity or (
        current.st_dev,
        current.st_ino,
        current.st_size,
    ) != expected_identity:
        raise ValueError("scanner chain profile file identity changed during read")
    return decode_scanner_chain_profile_file(
        body,
        expected_file_sha256=expected_file_sha256,
        expected_profile_sha256=expected_profile_sha256,
    )


def publish_scanner_chain_profile_file_create_only(
    destination: Path,
    profile_bytes: bytes,
    *,
    expected_profile_sha256: str,
) -> PublishedFileIdentity:
    """Publish one canonical profile file without replacing a destination."""

    target = Path(destination)
    if not target.parent.is_dir():
        raise FileNotFoundError("scanner chain profile destination parent is absent")
    encoded = encode_scanner_chain_profile_file(
        profile_bytes,
        expected_profile_sha256=expected_profile_sha256,
    )
    stage = target.parent / f".{target.name}.{uuid.uuid4().hex}.stage"
    try:
        with stage.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return publish_create_only(stage, target)
    finally:
        try:
            stage.unlink()
        except FileNotFoundError:
            pass


def apply_scanner_chain_from_profile_file(
    transmittance: np.ndarray,
    path: Path,
    *,
    expected_file_sha256: str,
    expected_profile_sha256: str,
) -> np.ndarray:
    """Validate the file completely before delegating to U6.P6ZJ execution."""

    profile_bytes = load_scanner_chain_profile_file(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_profile_sha256=expected_profile_sha256,
    )
    return apply_scanner_chain_from_bundle(
        transmittance,
        profile_bytes,
        expected_profile_sha256=expected_profile_sha256,
    )


__all__ = [
    "FILE_SCHEMA",
    "MAXIMUM_FILE_BYTES",
    "apply_scanner_chain_from_profile_file",
    "decode_scanner_chain_profile_file",
    "encode_scanner_chain_profile_file",
    "load_scanner_chain_profile_file",
    "publish_scanner_chain_profile_file_create_only",
]
