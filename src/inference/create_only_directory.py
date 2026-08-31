"""Identity-safe materialization of one flat create-only directory."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.film_physics.create_only_file import (
    PublishedFileIdentity,
    remove_if_published,
)


class CreateOnlyDirectoryError(ValueError):
    """Raised when a flat create-only directory contract is invalid."""


@dataclass(frozen=True)
class _OwnedDirectoryIdentity:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class _OwnedFile:
    identity: PublishedFileIdentity
    length: int
    sha256: str


def _is_reparse_directory(path: Path, path_stat: os.stat_result) -> bool:
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    return bool(
        getattr(path_stat, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _validate_member_name(name: object) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        raise CreateOnlyDirectoryError(
            "create-only directory members must be simple file names"
        )
    return name


def _write_owned_file(path: Path, payload: bytes) -> _OwnedFile:
    identity: PublishedFileIdentity | None = None
    try:
        with path.open("xb") as handle:
            file_stat = os.fstat(handle.fileno())
            identity = PublishedFileIdentity(
                path=path,
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
            )
            if not stat.S_ISREG(file_stat.st_mode):
                raise CreateOnlyDirectoryError("created member is not a regular file")
            handle.write(payload)
            handle.flush()
        return _OwnedFile(
            identity=identity,
            length=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    except BaseException:
        if identity is not None:
            remove_if_published(identity)
        raise


def _require_owned_file(owned: _OwnedFile) -> None:
    try:
        current = owned.identity.path.lstat()
    except OSError as exc:
        raise CreateOnlyDirectoryError("created member became unavailable") from exc
    if (
        owned.identity.path.is_symlink()
        or not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino)
        != (owned.identity.device, owned.identity.inode)
        or current.st_size != owned.length
    ):
        raise CreateOnlyDirectoryError("created member identity drift")
    digest = hashlib.sha256()
    with owned.identity.path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != owned.sha256:
        raise CreateOnlyDirectoryError("created member payload drift")


def _remove_empty_directory_if_owned(identity: _OwnedDirectoryIdentity) -> bool:
    try:
        current = identity.path.lstat()
    except FileNotFoundError:
        return False
    if (
        identity.path.is_symlink()
        or _is_reparse_directory(identity.path, current)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (identity.device, identity.inode)
    ):
        return False
    try:
        identity.path.rmdir()
    except OSError:
        return False
    return True


def materialize_create_only_directory(
    destination: Path,
    files: Mapping[str, bytes],
) -> None:
    """Create exact flat files and remove only entries still owned on failure."""

    destination = Path(destination)
    normalized: list[tuple[str, bytes]] = []
    for raw_name, raw_payload in files.items():
        name = _validate_member_name(raw_name)
        if not isinstance(raw_payload, bytes):
            raise CreateOnlyDirectoryError("create-only payloads must be bytes")
        normalized.append((name, raw_payload))
    if not normalized:
        raise CreateOnlyDirectoryError("create-only directory must not be empty")
    if destination.exists() or destination.is_symlink():
        raise CreateOnlyDirectoryError("create-only destination already exists")

    destination.parent.mkdir(parents=True, exist_ok=True)
    owned_directory: _OwnedDirectoryIdentity | None = None
    owned_files: list[_OwnedFile] = []
    try:
        destination.mkdir(exist_ok=False)
        directory_stat = destination.lstat()
        if (
            destination.is_symlink()
            or _is_reparse_directory(destination, directory_stat)
            or not stat.S_ISDIR(directory_stat.st_mode)
        ):
            raise CreateOnlyDirectoryError("created destination is not a directory")
        owned_directory = _OwnedDirectoryIdentity(
            path=destination,
            device=directory_stat.st_dev,
            inode=directory_stat.st_ino,
        )
        for name, payload in normalized:
            owned_files.append(_write_owned_file(destination / name, payload))

        expected_names = {name for name, _payload in normalized}
        if {entry.name for entry in destination.iterdir()} != expected_names:
            raise CreateOnlyDirectoryError("created destination member set drift")
        for owned in owned_files:
            _require_owned_file(owned)
    except BaseException:
        for owned in reversed(owned_files):
            remove_if_published(owned.identity)
        if owned_directory is not None:
            _remove_empty_directory_if_owned(owned_directory)
        raise


__all__ = [
    "CreateOnlyDirectoryError",
    "materialize_create_only_directory",
]
