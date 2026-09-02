from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from scripts import install_product_runtime as installer

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts" / "render_film.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _launcher(path: Path) -> Path:
    requirements = ROOT / "requirements-product-v2.txt"
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    path.write_text(
        installer._launcher_source(
            project_root=ROOT,
            source_commit=commit,
            requirements_path=requirements,
            requirements_sha256=installer._sha256(requirements),
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _run(entry: Path, cwd: Path, output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(entry),
            "source photo.png",
            "--product-look",
            "ektar_100",
            "--look-amount",
            "0.65",
            "--write-recipe",
            "--output",
            output,
        ],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_installed_cli_relative_paths_resolve_against_caller_cwd(
    tmp_path: Path,
) -> None:
    """Foreign-cwd delivery must not reinterpret user paths under the repo."""

    caller = tmp_path / "用户 照片"
    caller.mkdir()
    Image.new("RGB", (17, 11), (43, 97, 151)).save(caller / "source photo.png")
    launcher = _launcher(tmp_path / "product-launch.py")

    direct = _run(RENDERER, caller, "direct result.png")
    assert direct.returncode == 0, direct.stderr
    installed = _run(launcher, caller, "installed result.png")

    assert installed.returncode == 0, installed.stderr
    assert installed.stderr == ""
    assert installed.stdout.strip() == "installed result.png"
    assert _sha256(caller / "installed result.png") == _sha256(
        caller / "direct result.png"
    )
    recipe = json.loads((caller / "installed result.recipe.json").read_text("utf-8"))
    assert recipe["input"]["path"] == str((caller / "source photo.png").resolve())
    assert recipe["output"]["path"] == str(
        (caller / "installed result.png").resolve()
    )
    assert recipe["claim"]["output_label"] == "film-inspired"
    assert recipe["claim"]["calibrated_reference_allowed"] is False
    assert not (ROOT / "installed result.png").exists()
    assert not (ROOT / "installed result.recipe.json").exists()


def test_launcher_keeps_repository_validation_independent_of_caller_cwd(
    tmp_path: Path,
) -> None:
    launcher = _launcher(tmp_path / "product-launch.py")
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    completed = subprocess.run(
        [sys.executable, "-I", str(launcher), "--list-product-looks"],
        cwd=foreign,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert [row["look_id"] for row in payload["looks"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
        "generic_bw",
    ]
