"""Strict identity validation for a private repository-independent runtime capsule."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

CAPSULE_MANIFEST_ENV = "KMCFM_PRIVATE_CAPSULE_MANIFEST"
CAPSULE_MANIFEST_SHA256_ENV = "KMCFM_PRIVATE_CAPSULE_MANIFEST_SHA256"
CAPSULE_SCHEMA = "kmcfm.private-runtime-source-capsule.v1"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class RuntimeSourceCapsuleError(RuntimeError):
    """Raised when a private source capsule is absent, malformed, or changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RuntimeSourceCapsuleError(f"{label} keys mismatch")


def _relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeSourceCapsuleError(f"{label} must be a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise RuntimeSourceCapsuleError(f"{label} must be a normalized relative path")
    return path


def _file_row(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeSourceCapsuleError(f"{label} must be an object")
    if "bytes" not in value or "sha256" not in value:
        raise RuntimeSourceCapsuleError(f"{label} identity is incomplete")
    if not isinstance(value["bytes"], int) or isinstance(value["bytes"], bool):
        raise RuntimeSourceCapsuleError(f"{label}.bytes must be an integer")
    if value["bytes"] < 0 or not isinstance(value["sha256"], str):
        raise RuntimeSourceCapsuleError(f"{label} identity invalid")
    if not _HASH.fullmatch(value["sha256"]):
        raise RuntimeSourceCapsuleError(f"{label}.sha256 invalid")
    return value


def _verify_file(path: Path, row: Mapping[str, Any], label: str) -> None:
    try:
        details = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeSourceCapsuleError(f"{label} is unavailable") from exc
    if (
        not path.is_file()
        or details.st_size != row["bytes"]
        or _sha256(path) != row["sha256"]
    ):
        raise RuntimeSourceCapsuleError(f"{label} identity changed")


def has_git_identity(root: Path) -> bool:
    """Return whether a root exposes a Git identity entry of any kind."""

    return os.path.lexists(Path(root) / ".git")


def verify_runtime_source_capsule(
    root: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the launcher-bound capsule and return its immutable manifest."""

    root = Path(root).resolve(strict=True)
    if has_git_identity(root):
        raise RuntimeSourceCapsuleError("Git checkout cannot use capsule identity")
    env = os.environ if environment is None else environment
    manifest_text = env.get(CAPSULE_MANIFEST_ENV)
    expected_hash = env.get(CAPSULE_MANIFEST_SHA256_ENV)
    if not manifest_text or not expected_hash or not _HASH.fullmatch(expected_hash):
        raise RuntimeSourceCapsuleError("capsule launcher binding is unavailable")
    manifest = Path(manifest_text).resolve(strict=True)
    expected_manifest = root / "runtime-source-capsule.json"
    if manifest != expected_manifest:
        raise RuntimeSourceCapsuleError("capsule manifest path is not root-bound")
    encoded = manifest.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_hash:
        raise RuntimeSourceCapsuleError("capsule manifest identity changed")
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSourceCapsuleError("capsule manifest is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeSourceCapsuleError("capsule manifest must be an object")
    _keys(
        payload,
        {
            "schema",
            "source_commit",
            "runtime_scope",
            "archive",
            "external_files",
            "parent_runtime",
            "claim",
        },
        "capsule",
    )
    if payload["schema"] != CAPSULE_SCHEMA or not isinstance(
        payload["source_commit"], str
    ):
        raise RuntimeSourceCapsuleError("capsule schema or source commit invalid")
    if not _COMMIT.fullmatch(payload["source_commit"]):
        raise RuntimeSourceCapsuleError("capsule source commit invalid")
    scope = payload["runtime_scope"]
    if not isinstance(scope, list) or not scope:
        raise RuntimeSourceCapsuleError("capsule runtime scope invalid")
    normalized_scope = [
        _relative(item, "capsule.runtime_scope").as_posix() for item in scope
    ]
    if len(set(normalized_scope)) != len(normalized_scope):
        raise RuntimeSourceCapsuleError("capsule runtime scope duplicates")

    archive = payload["archive"]
    if not isinstance(archive, Mapping):
        raise RuntimeSourceCapsuleError("capsule archive must be an object")
    _keys(archive, {"path", "bytes", "sha256", "member_count"}, "capsule.archive")
    archive_path = root.joinpath(
        *_relative(archive["path"], "capsule.archive.path").parts
    )
    _verify_file(archive_path, _file_row(archive, "capsule.archive"), "capsule archive")
    if not isinstance(archive["member_count"], int) or archive["member_count"] < 1:
        raise RuntimeSourceCapsuleError("capsule archive member count invalid")

    external = payload["external_files"]
    if not isinstance(external, Mapping) or not external:
        raise RuntimeSourceCapsuleError("capsule external files invalid")
    for name, raw in external.items():
        relative = _relative(name, "capsule.external_files path")
        _verify_file(
            root.joinpath(*relative.parts),
            _file_row(raw, f"capsule.external_files.{name}"),
            f"capsule external file {name}",
        )

    parent = payload["parent_runtime"]
    if not isinstance(parent, Mapping):
        raise RuntimeSourceCapsuleError("capsule parent runtime invalid")
    _keys(parent, {"receipt", "python"}, "capsule.parent_runtime")
    runtime_root = root.parent.resolve(strict=True)
    for role in ("receipt", "python"):
        row = parent[role]
        if not isinstance(row, Mapping):
            raise RuntimeSourceCapsuleError(f"capsule parent {role} invalid")
        _keys(row, {"path", "bytes", "sha256"}, f"capsule.parent_runtime.{role}")
        relative = _relative(row["path"], f"capsule.parent_runtime.{role}.path")
        path = runtime_root.joinpath(*relative.parts).resolve(strict=True)
        try:
            path.relative_to(runtime_root)
        except ValueError as exc:
            raise RuntimeSourceCapsuleError(
                f"capsule parent {role} escaped runtime"
            ) from exc
        _verify_file(
            path, _file_row(row, f"capsule.parent_runtime.{role}"), f"parent {role}"
        )

    claim = payload["claim"]
    if not isinstance(claim, Mapping) or claim != {
        "calibrated_stock_response": False,
        "evidence_grade": "look-approximation",
        "mode": "film-inspired",
        "physical_film_reproduction": False,
        "public_release": False,
        "redistribution_authorized": False,
        "repository_independent_private_runtime": True,
    }:
        raise RuntimeSourceCapsuleError("capsule claim ceiling changed")
    return payload


def capsule_source_commit(root: Path) -> str:
    return str(verify_runtime_source_capsule(root)["source_commit"])


def validate_capsule_runtime_scope(
    root: Path, source_commit: str, runtime_scope: Sequence[str]
) -> None:
    payload = verify_runtime_source_capsule(root)
    if (
        payload["source_commit"] != source_commit
        or list(runtime_scope) != payload["runtime_scope"]
    ):
        raise RuntimeSourceCapsuleError("capsule runtime identity changed")


__all__ = [
    "CAPSULE_MANIFEST_ENV",
    "CAPSULE_MANIFEST_SHA256_ENV",
    "CAPSULE_SCHEMA",
    "RuntimeSourceCapsuleError",
    "capsule_source_commit",
    "has_git_identity",
    "validate_capsule_runtime_scope",
    "verify_runtime_source_capsule",
]
