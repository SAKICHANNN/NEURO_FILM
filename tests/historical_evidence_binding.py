from __future__ import annotations

import functools
import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@functools.cache
def _git_blob_ids(root_text: str, relative_path: str) -> frozenset[str]:
    root = Path(root_text)
    listing = subprocess.run(
        ["git", "rev-list", "--objects", "--all", "--", relative_path],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return frozenset(
        object_id
        for line in listing.stdout.splitlines()
        if " " in line
        for object_id, path in [line.split(" ", 1)]
        if path == relative_path
    )


@functools.cache
def _git_blob_sha256s(root_text: str, relative_path: str) -> frozenset[str]:
    root = Path(root_text)
    digests: set[str] = set()
    for object_id in _git_blob_ids(root_text, relative_path):
        blob = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        digests.add(_sha256(blob))
    return frozenset(digests)


def assert_historical_evidence_binding(
    root: Path, binding: Mapping[str, str]
) -> None:
    relative_path = str(binding["path"])
    expected = str(binding["sha256"])
    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise AssertionError(f"evidence binding path is not repository-relative: {relative_path}")

    current = root / Path(*normalized.parts)
    if current.is_file() and _sha256(current.read_bytes()) == expected:
        return

    historical = _git_blob_sha256s(str(root.resolve()), normalized.as_posix())
    if expected in historical:
        return

    recorded_blob = binding.get("git_blob")
    if isinstance(recorded_blob, str) and recorded_blob:
        object_ids = _git_blob_ids(str(root.resolve()), normalized.as_posix())
        if recorded_blob in object_ids:
            return

    raise AssertionError(
        f"bound SHA-256 {expected} for {normalized.as_posix()} is neither current "
        "nor present in immutable Git blob history, and its recorded git_blob "
        "does not identify that path in history"
    )
