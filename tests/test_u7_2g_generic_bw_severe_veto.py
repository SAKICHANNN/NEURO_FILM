from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.inference import (
    list_product_looks,
    render_product_look_rgb,
    replay_style_safe_recipe_to_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u7_2g_generic_bw_severe_veto_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"


def test_contract_binds_failed_population_review_and_catalog_state() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = contract["parent_evidence"]
    evidence_path = ROOT / parent["path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == parent["sha256"]
    assert evidence["status"] == parent["required_status"]
    assert evidence["decision_counts"]["FAIL_CONFIRMED_SEVERE_ARTIFACT"] == 3

    rows = {row["look_id"]: row for row in list_product_looks()}
    assert [rows[look_id]["availability"] for look_id in contract["required_order"]] == [
        contract["availability"][look_id] for look_id in contract["required_order"]
    ]
    assert rows["generic_bw"]["availability_evidence_path"] == parent["path"]
    assert rows["generic_bw"]["availability_evidence_sha256"] == parent["sha256"]


def test_product_dispatch_blocks_generic_bw_before_pixel_validation() -> None:
    deliberately_invalid_pixels = np.empty((0, 0, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="unavailable.*3/16"):
        render_product_look_rgb(
            deliberately_invalid_pixels,
            profile={},
            look_id="generic_bw",
            look_amount=1.0,
            style_statistics={},
            guardrails={},
            seed=0,
        )


def test_recipe_replay_blocks_generic_bw_before_profile_or_input_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocked.png"
    with pytest.raises(ValueError, match="unavailable.*3/16"):
        replay_style_safe_recipe_to_file(
            {"render": {"style": "generic_bw"}},
            profile_path=tmp_path / "missing-profile.json",
            output_path=output,
            root=tmp_path,
        )
    assert not output.exists()


def test_cli_blocks_generic_bw_before_input_decode_or_artifact_creation(
    tmp_path: Path,
) -> None:
    missing_input = tmp_path / "missing.png"
    output = tmp_path / "blocked.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(missing_input),
            "--style",
            "generic_bw",
            "--use-render-profile",
            "--render-profile",
            str(PRODUCT_PROFILE),
            "--output",
            str(output),
            "--write-recipe",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "product look 'generic_bw' is unavailable" in completed.stderr
    assert "3/16" in completed.stderr
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()
