from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"


def _run(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "must-not-decode.png"),
            *arguments,
            "--output",
            str(tmp_path / "out.png"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_predecode_rejection(
    completed: subprocess.CompletedProcess[str], tmp_path: Path
) -> None:
    assert completed.returncode == 2
    assert "must-not-decode.png" not in completed.stderr
    assert not (tmp_path / "out.png").exists()
    assert not (tmp_path / "out.recipe.json").exists()
    assert not (tmp_path / "out.metrics.json").exists()
    assert not (tmp_path / "out_layers").exists()


@pytest.mark.parametrize("look", ["velvia_50", "portra_400", "ektar_100"])
def test_product_rejects_expert_physical_controls_predecode(
    tmp_path: Path, look: str
) -> None:
    completed = _run(
        tmp_path,
        "--product-look",
        look,
        "--halation-model",
        "physical",
        "--halation-expert-controls",
        "--halation-background-gain",
        "nan",
    )

    _assert_predecode_rejection(completed, tmp_path)
    assert "requires locked controls" in completed.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--halation-amount", "nan"),
        ("--halation-amount", "inf"),
        ("--halation-amount", "-0.0001"),
        ("--halation-amount", "2.4001"),
        ("--halation-impact", "nan"),
        ("--halation-impact", "-inf"),
        ("--halation-impact", "-0.0001"),
        ("--halation-impact", "1.0001"),
        ("--halation-anti-halation", "nan"),
        ("--halation-anti-halation", "inf"),
        ("--halation-anti-halation", "-0.0001"),
        ("--halation-anti-halation", "1.0001"),
        ("--halation-source-selectivity", "nan"),
        ("--halation-source-selectivity", "-inf"),
        ("--halation-source-selectivity", "-0.0001"),
        ("--halation-source-selectivity", "1.0001"),
        ("--halation-diffusion", "nan"),
        ("--halation-diffusion", "inf"),
        ("--halation-diffusion", "-0.0001"),
        ("--halation-diffusion", "1.0001"),
        ("--halation-warm-core", "nan"),
        ("--halation-warm-core", "-inf"),
        ("--halation-warm-core", "-0.0001"),
        ("--halation-warm-core", "1.0001"),
        ("--halation-background-visibility", "nan"),
        ("--halation-background-visibility", "inf"),
        ("--halation-background-visibility", "-0.0001"),
        ("--halation-background-visibility", "1.0001"),
    ],
)
def test_product_rejects_invalid_locked_numeric_controls_predecode(
    tmp_path: Path, option: str, value: str
) -> None:
    completed = _run(
        tmp_path,
        "--product-look",
        "ektar_100",
        "--halation-model",
        "physical",
        f"{option}={value}",
    )

    _assert_predecode_rejection(completed, tmp_path)
    assert f"{option} must be finite" in completed.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ("--halation-preset", "does-not-exist"),
        ("--halation-type", "bw_clear_base"),
        (
            "--halation-model-family",
            "bw_density_halation",
            "--halation-color-response",
            "red_orange_core",
        ),
    ],
)
def test_product_rejects_invalid_locked_control_combinations_predecode(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    completed = _run(
        tmp_path,
        "--product-look",
        "ektar_100",
        "--halation-model",
        "physical",
        *arguments,
    )

    _assert_predecode_rejection(completed, tmp_path)
    assert "invalid product physical halation controls" in completed.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("--halation-amount", "0"),
        ("--halation-amount", "2.4"),
        ("--halation-impact", "0"),
        ("--halation-impact", "1"),
        ("--halation-preset", "vision3_clean"),
        ("--halation-preset", "cinestill_strong"),
        (
            "--halation-type",
            "bw_clear_base",
            "--halation-color-response",
            "neutral_density",
        ),
    ],
)
def test_valid_product_locked_controls_cross_preflight(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    completed = _run(
        tmp_path,
        "--product-look",
        "ektar_100",
        "--halation-model",
        "physical",
        *arguments,
    )

    assert completed.returncode != 2
    assert "FileNotFoundError" in completed.stderr
    assert not (tmp_path / "out.png").exists()


def test_nonproduct_expert_controls_cross_product_preflight(tmp_path: Path) -> None:
    completed = _run(
        tmp_path,
        "--style",
        "ektar_100",
        "--use-render-profile",
        "--halation-model",
        "physical",
        "--halation-expert-controls",
        "--halation-background-gain",
        "1.25",
    )

    assert completed.returncode != 2
    assert "FileNotFoundError" in completed.stderr
    assert not (tmp_path / "out.png").exists()
