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
CONFIG = ROOT / "configs/u7_2i_product_look_catalog_enforcement_v1.json"
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
        [sys.executable, str(SCRIPT), str(source), *arguments, "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _normalized_recipe_sha(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["software"]["commit"] = "<normalized>"
    payload["input"]["path"] = "<input>"
    payload["output"]["path"] = "<output>"
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def test_contract_binds_prechange_sources_and_u7_2h_parent() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    locks = config["source_locks"]
    assert _sha(ROOT / "src/inference/product_look_catalog.py") == locks[
        "product_look_catalog_sha256"
    ]
    assert _sha(PRODUCT_PROFILE) == locks["product_profile_sha256"]
    assert _sha(ROOT / "docs/evidence/U7_2H_EXPLICIT_PRODUCT_LOOK_SELECTION_RESULT.json") == locks[
        "parent_u7_2h_evidence_sha256"
    ]


@pytest.mark.parametrize(
    "style",
    [
        "hp5",
        "tri_x_400",
        "vision3_250d",
        "vision3_500t",
        "portra_800",
        "unknown",
        "",
    ],
)
def test_non_catalog_styles_reject_before_input_decode_and_leave_no_output(
    tmp_path: Path, style: str
) -> None:
    missing = tmp_path / "must_not_decode.png"
    output = tmp_path / f"{style}.png"
    completed = _run(
        missing,
        output,
        "--style",
        style,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--write-recipe",
    )
    assert completed.returncode != 0
    assert "only supports available product-catalog looks" in completed.stderr
    assert "must_not_decode" not in completed.stderr
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()


def test_legacy_profile_hp5_remains_exact(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    oracle = config["prechange_legacy_hp5_oracle"]
    source = tmp_path / "source.png"
    output = tmp_path / "hp5.png"
    _source(source)
    completed = _run(
        source,
        output,
        "--style",
        "hp5",
        "--use-render-profile",
        "--render-profile",
        str(LEGACY_PROFILE),
        "--write-recipe",
    )
    assert completed.returncode == 0, completed.stderr
    assert _sha(source) == oracle["fixture_sha256"]
    assert _sha(output) == oracle["output_sha256"]
    assert _normalized_recipe_sha(output.with_suffix(".recipe.json")) == oracle[
        "normalized_recipe_sha256"
    ]
