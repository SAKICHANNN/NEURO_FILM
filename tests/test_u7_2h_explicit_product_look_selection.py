from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
CONFIG = ROOT / "configs/u7_2h_explicit_product_look_selection_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _run(source: Path, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            *arguments,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_contract_binds_parent_evidence_profiles_and_prechange_source() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in contract["parent_evidence"]:
        assert _sha(ROOT / binding["path"]) == binding["sha256"]
    assert _sha(PRODUCT_PROFILE) == contract["source_locks"]["product_profile_sha256"]
    assert _sha(LEGACY_PROFILE) == contract["source_locks"]["legacy_profile_sha256"]


def test_product_profile_requires_explicit_style_before_input_decode(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "must_not_decode.png"
    output = tmp_path / "must_not_exist.png"
    completed = _run(
        missing,
        output,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--write-recipe",
    )
    assert completed.returncode != 0
    assert "requires an explicit --style product look selection" in completed.stderr
    assert "must_not_decode" not in completed.stderr
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()


@pytest.mark.parametrize("style", ["velvia_50", "portra_400", "ektar_100"])
def test_explicit_product_look_retains_prechange_output_and_recipe_claim(
    tmp_path: Path, style: str
) -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "source.png"
    output = tmp_path / f"{style}.png"
    _source(source)
    assert _sha(source) == contract["prechange_oracle"]["fixture_sha256"]
    completed = _run(
        source,
        output,
        "--style",
        style,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--write-recipe",
    )
    assert completed.returncode == 0, completed.stderr
    assert _sha(output) == contract["prechange_oracle"]["product"][style][
        "output_sha256"
    ]
    recipe = json.loads(output.with_suffix(".recipe.json").read_text(encoding="utf-8"))
    assert recipe["render"]["style"] == style
    assert recipe["claim"]["output_label"] == "film-inspired"
    assert recipe["claim"]["calibrated_reference_allowed"] is False


def test_legacy_profile_keeps_omitted_velvia_default(tmp_path: Path) -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "source.png"
    implicit = tmp_path / "implicit.png"
    explicit = tmp_path / "explicit.png"
    _source(source)
    first = _run(
        source,
        implicit,
        "--use-render-profile",
        "--render-profile",
        str(LEGACY_PROFILE),
    )
    second = _run(
        source,
        explicit,
        "--style",
        "velvia_50",
        "--use-render-profile",
        "--render-profile",
        str(LEGACY_PROFILE),
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert implicit.read_bytes() == explicit.read_bytes()
    assert _sha(implicit) == contract["prechange_oracle"]["legacy_default"][
        "output_sha256"
    ]


def test_blocked_generic_bw_remains_unavailable(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "generic.png"
    _source(source)
    completed = _run(
        source,
        output,
        "--style",
        "generic_bw",
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
    )
    assert completed.returncode != 0
    assert "product look 'generic_bw' is unavailable" in completed.stderr
    assert not output.exists()
