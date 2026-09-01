from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _artifacts(output: Path) -> tuple[Path, ...]:
    return (
        output,
        output.with_suffix(".recipe.json"),
        output.with_suffix(".metrics.json"),
        output.parent / f"{output.stem}_layers",
    )


@pytest.mark.parametrize("suffix", ["", ".webp", ".avif", ".heif", ".bmp", ".gif"])
def test_product_rejects_unsupported_8bit_extension_before_decode(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / "must-not-decode.png"
    output = tmp_path / f"unsupported{suffix}"
    completed = _run(
        str(source),
        "--product-look",
        "ektar_100",
        "--write-recipe",
        "--write-layers",
        "--write-metrics",
        "--output",
        str(output),
    )

    assert completed.returncode == 2
    expected = suffix or "<none>"
    assert f"unsupported 8-bit product output extension: {expected}" in completed.stderr
    assert str(source) not in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))


@pytest.mark.parametrize(
    "suffix", [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPEG", ".TIFF"]
)
def test_supported_product_8bit_extensions_cross_preflight(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / "accepted-then-read.png"
    output = tmp_path / f"accepted{suffix}"
    completed = _run(
        str(source),
        "--product-look",
        "ektar_100",
        "--output",
        str(output),
    )

    assert completed.returncode != 2
    assert "FileNotFoundError" in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))


@pytest.mark.parametrize("suffix", ["", ".webp", ".avif", ".heif", ".bmp", ".gif"])
def test_legacy_unsupported_extension_retains_late_behavior(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / "legacy-still-reaches-input.png"
    output = tmp_path / f"unsupported{suffix}"
    completed = _run(
        str(source),
        "--style",
        "ektar_100",
        "--output",
        str(output),
    )

    assert completed.returncode != 2
    assert "FileNotFoundError" in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))
