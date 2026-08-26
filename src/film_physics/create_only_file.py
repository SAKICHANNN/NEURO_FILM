"""Same-volume create-only file publication across supported filesystems."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublishedFileIdentity:
    """Filesystem identity captured for one successfully published entry."""

    path: Path
    device: int
    inode: int


def publish_create_only(
    source: Path,
    destination: Path,
) -> PublishedFileIdentity:
    """Publish one sibling stage without replacing an existing destination."""

    source_stat = source.lstat()
    if os.name == "nt":
        # Windows rename is same-volume and no-replace, and works on exFAT.
        # exFAT synthesizes a different st_ino after rename, so bind the
        # published path only after the successful move.
        os.rename(source, destination)
    else:
        os.link(source, destination)
        source.unlink()
    destination_stat = destination.lstat()
    if os.name != "nt" and (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ) != (source_stat.st_dev, source_stat.st_ino):
        raise OSError("published file identity drift")
    return PublishedFileIdentity(
        path=destination,
        device=destination_stat.st_dev,
        inode=destination_stat.st_ino,
    )


def remove_if_published(identity: PublishedFileIdentity) -> bool:
    """Remove the published entry only while its filesystem identity matches."""

    try:
        current = identity.path.lstat()
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != (identity.device, identity.inode):
        return False
    identity.path.unlink()
    return True


__all__ = [
    "PublishedFileIdentity",
    "publish_create_only",
    "remove_if_published",
]
