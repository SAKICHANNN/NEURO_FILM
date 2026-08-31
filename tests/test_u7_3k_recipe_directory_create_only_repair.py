from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.inference import create_only_directory as directory_module
from src.inference.recipe_export_request import (
    build_recipe_export_request_set,
    materialize_recipe_export_request_set,
)
from src.inference.recipe_workspace import (
    build_offline_recipe_workspace,
    materialize_offline_recipe_workspace,
)

Materializer = Callable[[Path, Path], object]


def test_windows_reparse_attribute_is_not_owned_directory(tmp_path: Path) -> None:
    path = tmp_path / "directory"
    path.mkdir()
    fake_stat = SimpleNamespace(st_file_attributes=0x400)
    assert directory_module._is_reparse_directory(path, fake_stat) is True


@pytest.fixture(params=["workspace", "request_set"])
def materializer(request: pytest.FixtureRequest) -> tuple[Materializer, Callable[[Path], dict[str, bytes]]]:
    if request.param == "workspace":
        return materialize_offline_recipe_workspace, build_offline_recipe_workspace
    return materialize_recipe_export_request_set, lambda root: build_recipe_export_request_set(root)["files"]


def test_successful_directories_retain_exact_builder_bytes(
    materializer: tuple[Materializer, Callable[[Path], dict[str, bytes]]],
    tmp_path: Path,
) -> None:
    materialize, build = materializer
    history = tmp_path / "history"
    history.mkdir()
    expected = build(history)
    destination = tmp_path / "published"

    materialize(history, destination)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == expected


def test_initial_foreign_directory_race_is_preserved(
    materializer: tuple[Materializer, Callable[[Path], dict[str, bytes]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize, _build = materializer
    history = tmp_path / "history"
    history.mkdir()
    destination = tmp_path / "published"
    original_mkdir = Path.mkdir

    def foreign_wins(path: Path, *args: object, **kwargs: object) -> None:
        if path == destination:
            original_mkdir(path, *args, **kwargs)
            (path / "foreign.bin").write_bytes(b"foreign-directory")
            raise FileExistsError("injected foreign directory")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", foreign_wins)
    with pytest.raises(FileExistsError, match="foreign directory"):
        materialize(history, destination)
    assert (destination / "foreign.bin").read_bytes() == b"foreign-directory"
    assert list(destination.iterdir()) == [destination / "foreign.bin"]


def test_foreign_added_member_survives_owned_cleanup(
    materializer: tuple[Materializer, Callable[[Path], dict[str, bytes]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize, _build = materializer
    history = tmp_path / "history"
    history.mkdir()
    destination = tmp_path / "published"
    original_write = directory_module._write_owned_file
    calls = 0

    def fail_after_owned_write(path: Path, payload: bytes) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            (destination / "foreign.bin").write_bytes(b"foreign-added")
            raise OSError("injected member failure")
        return original_write(path, payload)

    monkeypatch.setattr(directory_module, "_write_owned_file", fail_after_owned_write)
    with pytest.raises(OSError, match="member failure"):
        materialize(history, destination)
    assert (destination / "foreign.bin").read_bytes() == b"foreign-added"
    assert list(destination.iterdir()) == [destination / "foreign.bin"]


def test_foreign_replacement_survives_owned_cleanup(
    materializer: tuple[Materializer, Callable[[Path], dict[str, bytes]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize, _build = materializer
    history = tmp_path / "history"
    history.mkdir()
    destination = tmp_path / "published"
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"foreign-replacement")
    original_require = directory_module._require_owned_file
    calls = 0

    def replace_before_validation(owned: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            os.replace(replacement, owned.identity.path)  # type: ignore[attr-defined]
            raise directory_module.CreateOnlyDirectoryError("injected replacement")
        original_require(owned)  # type: ignore[arg-type]

    monkeypatch.setattr(directory_module, "_require_owned_file", replace_before_validation)
    with pytest.raises(directory_module.CreateOnlyDirectoryError, match="replacement"):
        materialize(history, destination)
    replaced = [path for path in destination.iterdir() if path.is_file()]
    assert len(replaced) == 1
    assert replaced[0].read_bytes() == b"foreign-replacement"


def test_failure_without_foreign_entries_leaves_zero_owned_residue(
    materializer: tuple[Materializer, Callable[[Path], dict[str, bytes]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize, _build = materializer
    history = tmp_path / "history"
    history.mkdir()
    destination = tmp_path / "published"

    def fail_write(_path: Path, _payload: bytes) -> object:
        raise OSError("injected clean failure")

    monkeypatch.setattr(directory_module, "_write_owned_file", fail_write)
    with pytest.raises(OSError, match="clean failure"):
        materialize(history, destination)
    assert not destination.exists()
