from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from scripts import open_product_desktop

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_root_and_existing_descendants_are_accepted() -> None:
    canonical = (ROOT / "tmp").resolve(strict=True)
    owned = canonical / "u7_14b_unit_accept"
    owned.mkdir()
    try:
        assert (
            open_product_desktop.resolve_product_scratch_root(ROOT / "tmp") == canonical
        )
        assert open_product_desktop.resolve_product_scratch_root(owned) == owned
        assert (
            open_product_desktop.resolve_product_scratch_root(
                ROOT / "tmp" / "." / owned.name
            )
            == owned
        )
    finally:
        owned.rmdir()


@pytest.mark.parametrize("relative", [".", "data", "outputs"])
def test_existing_repository_paths_outside_canonical_tmp_reject(
    relative: str,
) -> None:
    with pytest.raises(ValueError, match="repository tmp root"):
        open_product_desktop.resolve_product_scratch_root(ROOT / relative)


def test_normalized_escape_and_external_existing_directory_reject(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.bin"
    sentinel.write_bytes(b"foreign-sentinel")

    for candidate in (
        ROOT / "tmp" / ".." / "outputs",
        external,
        Path(ROOT.anchor),
    ):
        with pytest.raises(ValueError, match="repository tmp root"):
            open_product_desktop.resolve_product_scratch_root(candidate)
    assert sentinel.read_bytes() == b"foreign-sentinel"


def test_missing_path_and_regular_file_reject(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        open_product_desktop.resolve_product_scratch_root(missing)

    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"not-a-directory")
    with pytest.raises(ValueError, match="existing directory"):
        open_product_desktop.resolve_product_scratch_root(regular)


def test_invalid_cli_root_rejects_before_tk_import_or_input_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    missing_input = tmp_path / "must-not-be-read.png"
    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tkinter" or name.startswith("tkinter."):
            imported.append(name)
            raise AssertionError("tkinter imported before scratch rejection")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "open_product_desktop.py",
            "--input",
            str(missing_input),
            "--scratch-root",
            str(external),
        ],
    )

    assert open_product_desktop.main() == 2
    assert imported == []
    assert not missing_input.exists()
    assert list(external.iterdir()) == []
    assert "repository tmp root or its descendant" in capsys.readouterr().err


def test_escape_directory_link_rejects_when_supported(tmp_path: Path) -> None:
    canonical = (ROOT / "tmp").resolve(strict=True)
    external = tmp_path / "external"
    external.mkdir()
    link = canonical / "u7_14b_escape_link"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory-link creation is unavailable on this host")
    try:
        with pytest.raises(ValueError, match="repository tmp root"):
            open_product_desktop.resolve_product_scratch_root(link)
    finally:
        link.unlink()
