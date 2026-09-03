from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.inference.product_desktop import (
    ProductDesktopError,
    _load_product_runtime_scope,
    _validate_runtime_source_scope,
)

ROOT = Path(__file__).resolve().parents[1]
SCOPE_CONFIG = ROOT / "configs/u7_9d_private_runtime_source_scope_binding_v1.json"


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, tuple[str, ...], str]:
    root = tmp_path / "repository"
    (root / "src/preprocess").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "scripts").mkdir()
    files = {
        "src/preprocess/raster_decode.py": "VALUE = 1\n",
        "configs/profile.json": "{}\n",
        "scripts/render_film.py": "raise SystemExit(0)\n",
        "scripts/open_product_desktop.py": "raise SystemExit(0)\n",
        "requirements-product-v2.txt": "Pillow==12.1.1\n",
        "docs/outside.md": "outside\n",
    }
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    _git(root, "config", "user.email", "u7.21b@example.invalid")
    _git(root, "config", "user.name", "U7.21B")
    _git(root, "add", "--all")
    _git(root, "commit", "-q", "-m", "fixture")
    scope = (
        "src",
        "configs",
        "scripts/render_film.py",
        "scripts/open_product_desktop.py",
        "requirements-product-v2.txt",
    )
    return root, scope, _git(root, "rev-parse", "HEAD")


def test_default_scope_is_the_existing_u7_9d_authority() -> None:
    expected = tuple(json.loads(SCOPE_CONFIG.read_text("utf-8"))["runtime_scope"])
    assert _load_product_runtime_scope(ROOT) == expected


def test_runtime_scope_accepts_clean_head_and_ignores_outside_docs(
    tmp_path: Path,
) -> None:
    root, scope, commit = _repository(tmp_path)
    _validate_runtime_source_scope(root, commit, scope)
    (root / "docs/outside.md").write_text("changed outside\n", encoding="utf-8")
    _validate_runtime_source_scope(root, commit, scope)


@pytest.mark.parametrize("mutation", ["modified", "staged", "deleted", "untracked"])
def test_runtime_scope_rejects_every_worktree_drift(
    tmp_path: Path, mutation: str
) -> None:
    root, scope, commit = _repository(tmp_path)
    target = root / "src/preprocess/raster_decode.py"
    if mutation == "modified":
        target.write_text("VALUE = 2\n", encoding="utf-8")
    elif mutation == "staged":
        target.write_text("VALUE = 3\n", encoding="utf-8")
        _git(root, "add", target.relative_to(root).as_posix())
    elif mutation == "deleted":
        target.unlink()
    else:
        (root / "src/preprocess/injected.py").write_text(
            "VALUE = 4\n", encoding="utf-8"
        )
    with pytest.raises(ProductDesktopError, match="runtime source scope changed"):
        _validate_runtime_source_scope(root, commit, scope)


def test_runtime_scope_rejects_invalid_or_changed_authority(tmp_path: Path) -> None:
    root, scope, commit = _repository(tmp_path)
    with pytest.raises(ProductDesktopError, match="runtime source scope invalid"):
        _validate_runtime_source_scope(root, commit, ("../outside",))
    (root / "docs/new.md").write_text("new\n", encoding="utf-8")
    _git(root, "add", "docs/new.md")
    _git(root, "commit", "-q", "-m", "descendant")
    with pytest.raises(ProductDesktopError, match="preview renderer session changed"):
        _validate_runtime_source_scope(root, commit, scope)
