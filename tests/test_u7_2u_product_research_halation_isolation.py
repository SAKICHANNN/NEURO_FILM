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


@pytest.mark.parametrize("look", ["velvia_50", "portra_400", "ektar_100"])
@pytest.mark.parametrize("with_research_controls", [False, True])
def test_product_rejects_staged_density_before_decode(
    tmp_path: Path, look: str, with_research_controls: bool
) -> None:
    source = tmp_path / "must-not-decode.png"
    output = tmp_path / f"{look}.png"
    arguments = [
        str(source),
        "--product-look",
        look,
        "--halation-model",
        "staged-density-research",
    ]
    if with_research_controls:
        arguments.extend(["--halation", "1", "--tile-size", "64"])
    arguments.extend(
        [
            "--write-recipe",
            "--write-layers",
            "--write-metrics",
            "--output",
            str(output),
        ]
    )

    completed = _run(*arguments)

    assert completed.returncode == 2
    assert (
        "--halation-model staged-density-research is research-only "
        "and cannot be combined with --product-look"
    ) in completed.stderr
    assert str(source) not in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))


@pytest.mark.parametrize("model", ["simple", "physical"])
def test_product_nonresearch_halation_models_cross_preflight(
    tmp_path: Path, model: str
) -> None:
    source = tmp_path / "accepted-then-read.png"
    output = tmp_path / f"{model}.png"
    completed = _run(
        str(source),
        "--product-look",
        "ektar_100",
        "--halation-model",
        model,
        "--output",
        str(output),
    )

    assert completed.returncode != 2
    assert "FileNotFoundError" in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))


def test_nonproduct_staged_density_route_crosses_product_isolation_preflight(
    tmp_path: Path,
) -> None:
    source = tmp_path / "research-still-reaches-input.png"
    output = tmp_path / "research.png"
    completed = _run(
        str(source),
        "--style",
        "ektar_100",
        "--use-render-profile",
        "--halation-model",
        "staged-density-research",
        "--halation",
        "1",
        "--tile-size",
        "64",
        "--output",
        str(output),
    )

    assert completed.returncode != 2
    assert "FileNotFoundError" in completed.stderr
    assert all(not path.exists() for path in _artifacts(output))
