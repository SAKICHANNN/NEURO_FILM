from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.eval.rec2020_source_anchored_interior import (
    source_anchored_interior_residual as research_residual,
)
from src.inference.romm_rec2020_velvia import (
    PROFILE_ID,
    _source_anchored_interior_residual,
    load_profile,
    render_official_romm_velvia_rec2020,
)
from src.preprocess import (
    convert_official_romm_rgb16_to_rec2020_png,
    load_working_image,
)
from tests.test_romm_rec2020_product import _write_romm

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/romm_rec2020_velvia_v1.json"
SCRIPT = ROOT / "scripts/render_official_romm_rec2020_velvia.py"


def test_product_residual_is_exactly_the_qualified_research_primitive() -> None:
    rng = np.random.default_rng(1415)
    source = rng.uniform(0.01, 0.99, size=(9, 11, 3)).astype(np.float32)
    candidate = np.clip(
        source + rng.normal(0.0, 0.2, size=source.shape), 0.0, 1.0
    ).astype(np.float32)
    expected, expected_scale = research_residual(
        source, candidate, margin=2.0 / 65535.0
    )
    actual, actual_scale = _source_anchored_interior_residual(
        source, candidate, margin=2.0 / 65535.0
    )
    assert actual.tobytes() == expected.tobytes()
    assert actual_scale.tobytes() == expected_scale.tobytes()


def test_romm_rec2020_velvia_profile_is_evidence_bound() -> None:
    profile, profile_sha256 = load_profile(PROFILE, root=ROOT)
    assert profile["profile_id"] == PROFILE_ID
    assert profile_sha256
    assert profile["production_default_changed"] is False
    assert profile["effects"] == {"grain": False, "halation": False, "dust": False}


def test_romm_rec2020_velvia_render_is_deterministic_and_distinct(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tiff"
    baseline = tmp_path / "baseline.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_romm(source)
    convert_official_romm_rgb16_to_rec2020_png(source, baseline)
    receipt_a = render_official_romm_velvia_rec2020(
        source, first, profile_path=PROFILE, root=ROOT
    )
    receipt_b = render_official_romm_velvia_rec2020(
        source, second, profile_path=PROFILE, root=ROOT
    )
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != baseline.read_bytes()
    assert receipt_a == receipt_b
    assert receipt_a["output_claim"] == "film-inspired-look-approximation"
    assert receipt_a["output"]["exact_sample_readback"] is True
    assert receipt_a["look"]["style"] == "velvia_50"
    restored = load_working_image(first)
    assert restored.working_space == "linear_rec2020"


def test_romm_rec2020_velvia_cli_writes_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.tiff"
    output = tmp_path / "output.png"
    receipt = tmp_path / "receipt.json"
    _write_romm(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--output",
            str(output),
            "--receipt",
            str(receipt),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert completed.stdout.strip() == str(output)
    assert payload["profile_id"] == PROFILE_ID
    assert payload["production_default_changed"] is False
