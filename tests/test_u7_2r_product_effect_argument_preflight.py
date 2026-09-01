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


@pytest.mark.parametrize("option", ["--grain", "--halation", "--dust"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.0001", "1.0001"])
def test_product_effect_strengths_reject_before_decode(
    tmp_path: Path, option: str, value: str
) -> None:
    source = tmp_path / "must-not-decode.png"
    output = tmp_path / f"{option[2:]}-{value}.png"
    effect_arguments = (
        (f"{option}={value}",) if value.startswith("-") else (option, value)
    )
    completed = _run(
        str(source),
        "--product-look",
        "ektar_100",
        *effect_arguments,
        "--write-recipe",
        "--write-layers",
        "--write-metrics",
        "--output",
        str(output),
    )
    assert completed.returncode == 2
    assert f"{option} must be finite and in [0,1]" in completed.stderr
    assert str(source) not in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))


@pytest.mark.parametrize("value", [str(-(2**31) - 1), str(2**31)])
def test_product_seed_rejects_outside_signed_int32_before_decode(
    tmp_path: Path, value: str
) -> None:
    source = tmp_path / "must-not-decode.png"
    output = tmp_path / f"seed-{value}.png"
    completed = _run(
        str(source),
        "--product-look",
        "ektar_100",
        "--seed",
        value,
        "--output",
        str(output),
    )
    assert completed.returncode == 2
    assert "--seed must be a signed 32-bit integer" in completed.stderr
    assert str(source) not in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))


@pytest.mark.parametrize(
    "arguments",
    [
        ("--grain", "0"),
        ("--grain", "1"),
        ("--halation", "0"),
        ("--halation", "1"),
        ("--dust", "0"),
        ("--dust", "1"),
        ("--seed", str(-(2**31))),
        ("--seed", str(2**31 - 1)),
    ],
)
def test_product_effect_boundary_values_cross_argument_preflight(
    tmp_path: Path, arguments: tuple[str, str]
) -> None:
    source = tmp_path / "accepted-then-read.png"
    output = tmp_path / "must-not-exist.png"
    completed = _run(
        str(source),
        "--product-look",
        "ektar_100",
        *arguments,
        "--output",
        str(output),
    )
    assert completed.returncode != 2
    assert "FileNotFoundError" in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))


@pytest.mark.parametrize(
    "arguments",
    [
        ("--grain", "nan"),
        ("--halation", "-0.1"),
        ("--dust", "1.1"),
        ("--seed", str(2**31)),
    ],
)
def test_non_product_cli_retains_historical_numeric_behavior(
    tmp_path: Path, arguments: tuple[str, str]
) -> None:
    source = tmp_path / "legacy-still-reaches-input.png"
    output = tmp_path / "must-not-exist.png"
    completed = _run(
        str(source),
        "--style",
        "ektar_100",
        *arguments,
        "--output",
        str(output),
    )
    assert completed.returncode != 2
    assert "FileNotFoundError" in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))
