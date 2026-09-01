from __future__ import annotations

import argparse
import builtins
from pathlib import Path

import pytest

from scripts import open_product_desktop as desktop_entry


def test_initial_input_resolver_preserves_interactive_launch() -> None:
    assert desktop_entry.resolve_product_initial_input(None) is None


def test_initial_input_resolver_accepts_one_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    source.write_bytes(b"not-decoded-at-startup")

    assert desktop_entry.resolve_product_initial_input(source) == source.resolve()


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_invalid_initial_input_rejects_before_tk_without_traceback(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = tmp_path / kind
    if kind == "directory":
        candidate.mkdir()

    monkeypatch.setattr(
        desktop_entry,
        "parse_args",
        lambda: argparse.Namespace(
            input=candidate,
            scratch_root=desktop_entry.ROOT / "tmp",
            smoke_exit_ms=None,
        ),
    )
    real_import = builtins.__import__

    def reject_tk_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tkinter":
            raise AssertionError("invalid startup input reached Tk initialization")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_tk_import)

    assert desktop_entry.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "K-MCFM desktop rejected: initial input must be an existing file\n"
    )
    assert "Traceback" not in captured.err
