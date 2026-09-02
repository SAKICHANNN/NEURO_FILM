from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import install_product_runtime as installer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_9d_private_runtime_source_scope_binding_v1.json"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _project(tmp_path: Path) -> tuple[Path, str, Path]:
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "u7-9d@example.invalid")
    _git(root, "config", "user.name", "U7.9D")
    requirements = root / "requirements-product-v2.txt"
    _write(requirements, "fixture==1.0\n")
    _write(root / "src/runtime.py", "VALUE = 1\n")
    _write(root / "configs/product.json", "{}\n")
    _write(
        root / "scripts/render_film.py",
        "import json, subprocess\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parents[1]\n"
        "head = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'], check=True, "
        "capture_output=True, text=True).stdout.strip()\n"
        "print(json.dumps({'argv': __import__('sys').argv[1:], 'head': head}, "
        "sort_keys=True))\n",
    )
    _write(root / "scripts/open_product_desktop.py", "raise SystemExit(0)\n")
    _write(root / "docs/note.md", "initial\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root, _git(root, "rev-parse", "HEAD"), requirements


def _launcher(root: Path, commit: str, requirements: Path, target: Path) -> Path:
    target.write_text(
        installer._launcher_source(
            project_root=root,
            source_commit=commit,
            requirements_path=requirements,
            requirements_sha256=installer._sha256(requirements),
            runtime_scope=(
                "src",
                "configs",
                "scripts/render_film.py",
                "scripts/open_product_desktop.py",
                "requirements-product-v2.txt",
            ),
        ),
        encoding="utf-8",
        newline="\n",
    )
    return target


def _run(launcher: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(launcher), "probe"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_contract_freezes_exact_runtime_scope_and_claim() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    assert config["parent_receipt_schema"] == (
        "kmcfm.private-product-runtime-receipt.v3"
    )
    assert config["receipt_schema"] == "kmcfm.private-product-runtime-receipt.v4"
    assert config["runtime_scope"] == [
        "src",
        "configs",
        "scripts/render_film.py",
        "scripts/open_product_desktop.py",
        "requirements-product-v2.txt",
    ]
    assert config["policy"] == {
        "head_must_descend_from_installed_commit": True,
        "tracked_repository_must_be_clean": True,
        "runtime_scope_must_match_installed_commit": True,
        "runtime_scope_untracked_files_allowed": False,
        "committed_non_runtime_drift_allowed": True,
        "recipe_records_current_head": True,
    }
    assert all(config["gates"].values())
    assert config["claim"] == {
        "mode": "film-inspired",
        "evidence_grade": "look-approximation",
        "public_release": False,
        "standalone_runtime": False,
        "calibrated_stock_response": False,
        "physical_film_reproduction": False,
    }


def test_committed_docs_only_descendant_executes_and_records_current_head(
    tmp_path: Path,
) -> None:
    root, installed_commit, requirements = _project(tmp_path)
    launcher = _launcher(root, installed_commit, requirements, tmp_path / "launch.py")
    _write(root / "docs/note.md", "descendant\n")
    _git(root, "add", "docs/note.md")
    _git(root, "commit", "-m", "docs only")

    completed = _run(launcher)

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload == {"argv": ["probe"], "head": _git(root, "rev-parse", "HEAD")}
    assert payload["head"] != installed_commit


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("src/runtime.py", "runtime source scope drift"),
        ("configs/product.json", "runtime source scope drift"),
        ("scripts/render_film.py", "runtime source scope drift"),
        ("requirements-product-v2.txt", "product requirements identity drift"),
    ],
)
def test_committed_runtime_scope_drift_rejects(
    tmp_path: Path, path: str, message: str
) -> None:
    root, installed_commit, requirements = _project(tmp_path)
    launcher = _launcher(root, installed_commit, requirements, tmp_path / "launch.py")
    with (root / path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("# drift\n")
    _git(root, "add", path)
    _git(root, "commit", "-m", "runtime drift")

    completed = _run(launcher)

    assert completed.returncode == 2
    assert message in completed.stderr


def test_dirty_tracked_documentation_and_scoped_untracked_reject(tmp_path: Path) -> None:
    root, installed_commit, requirements = _project(tmp_path)
    launcher = _launcher(root, installed_commit, requirements, tmp_path / "launch.py")
    _write(root / "docs/note.md", "dirty\n")
    completed = _run(launcher)
    assert completed.returncode == 2
    assert "tracked repository drift" in completed.stderr

    _git(root, "restore", "docs/note.md")
    _write(root / "src/untracked.py", "VALUE = 2\n")
    completed = _run(launcher)
    assert completed.returncode == 2
    assert "untracked runtime source scope" in completed.stderr


def test_non_descendant_history_rejects(tmp_path: Path) -> None:
    root, installed_commit, requirements = _project(tmp_path)
    launcher = _launcher(root, installed_commit, requirements, tmp_path / "launch.py")
    _git(root, "checkout", "--orphan", "unrelated")
    _git(root, "rm", "-rf", ".")
    _write(requirements, "fixture==1.0\n")
    _write(root / "src/runtime.py", "VALUE = 1\n")
    _write(root / "configs/product.json", "{}\n")
    _write(root / "scripts/render_film.py", "print('unrelated')\n")
    _write(root / "scripts/open_product_desktop.py", "raise SystemExit(0)\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "unrelated")

    completed = _run(launcher)

    assert completed.returncode == 2
    assert "repository commit drift" in completed.stderr
