from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_film.py"


def _source(path: Path) -> None:
    yy, xx = np.mgrid[0:9, 0:13]
    rgb = np.stack(
        [
            (xx * 17 + yy * 3) % 256,
            (xx * 5 + yy * 29) % 256,
            (xx * 11 + yy * 7) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *(str(value) for value in arguments)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("case", ["missing-input", "existing-output"])
def test_product_cli_file_errors_are_one_safe_line_without_traceback(
    tmp_path: Path,
    case: str,
) -> None:
    source = tmp_path / "source.png"
    _source(source)
    source_before = source.read_bytes()
    output = tmp_path / "result.png"
    if case == "missing-input":
        source.unlink()
        expected = "input must be an existing file"
    else:
        output.write_bytes(b"foreign-output")
        expected = "product output destination must not already exist"

    completed = _run(
        source,
        "--product-look",
        "ektar_100",
        "--look-amount",
        "0.65",
        "--write-recipe",
        "--output",
        output,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == f"K-MCFM render stopped safely: {expected}\n"
    assert "Traceback" not in completed.stderr
    assert not output.with_suffix(".recipe.json").exists()
    if case == "missing-input":
        assert not output.exists()
    else:
        assert output.read_bytes() == b"foreign-output"
        assert source.read_bytes() == source_before


def test_argparse_error_remains_argparse_owned(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    _source(source)

    completed = _run(
        source,
        "--product-look",
        "not-a-look",
        "--output",
        output,
    )

    assert completed.returncode == 2
    assert "invalid choice: 'not-a-look'" in completed.stderr
    assert "K-MCFM render stopped safely:" not in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()
