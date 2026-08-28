from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.inference import list_product_looks

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_product_catalog_cli_is_exact_and_requires_no_media_paths() -> None:
    first = _run("--list-product-looks")
    second = _run("--list-product-looks")

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload == {
        "schema_id": "kmcfm.product-look-catalog.v1",
        "looks": list(list_product_looks()),
    }
    assert [row["look_id"] for row in payload["looks"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
        "generic_bw",
    ]
    assert [row["availability"] for row in payload["looks"]] == [
        "available",
        "available",
        "available",
        "blocked_severe_artifact",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ("missing.png", "--list-product-looks"),
        ("--list-product-looks", "--output", "must-not-exist.png"),
    ],
)
def test_product_catalog_cli_rejects_media_paths(args: tuple[str, ...]) -> None:
    completed = _run(*args)
    assert completed.returncode == 2
    assert "cannot be combined with input or --output" in completed.stderr
    assert not (ROOT / "must-not-exist.png").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((), "input is required unless --list-product-looks is used"),
        (("missing.png",), "--output is required unless --list-product-looks is used"),
    ],
)
def test_normal_render_still_requires_input_and_output(
    args: tuple[str, ...], message: str
) -> None:
    completed = _run(*args)
    assert completed.returncode == 2
    assert message in completed.stderr
