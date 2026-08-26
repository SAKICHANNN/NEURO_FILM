from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.film_physics.create_only_file import (
    publish_create_only,
    remove_if_published,
)


def test_publish_create_only_works_without_hardlinks_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.stage"
    destination = tmp_path / "payload.bin"
    source.write_bytes(b"exact-payload")

    if os.name == "nt":
        monkeypatch.setattr(
            os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("hardlinks unavailable")
            ),
        )
    identity = publish_create_only(source, destination)

    assert destination.read_bytes() == b"exact-payload"
    assert not source.exists()
    assert remove_if_published(identity) is True
    assert not destination.exists()


def test_publish_create_only_preserves_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "payload.stage"
    destination = tmp_path / "payload.bin"
    source.write_bytes(b"candidate")
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        publish_create_only(source, destination)

    assert source.read_bytes() == b"candidate"
    assert destination.read_bytes() == b"existing"


def test_remove_if_published_preserves_replacement(tmp_path: Path) -> None:
    source = tmp_path / "payload.stage"
    destination = tmp_path / "payload.bin"
    replacement = tmp_path / "replacement.stage"
    source.write_bytes(b"ours")
    identity = publish_create_only(source, destination)
    replacement.write_bytes(b"foreign")
    os.replace(replacement, destination)

    assert remove_if_published(identity) is False
    assert destination.read_bytes() == b"foreign"
