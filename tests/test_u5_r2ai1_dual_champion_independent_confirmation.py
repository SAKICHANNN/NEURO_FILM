from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from scripts.pipeline_color_baseline import apply_output_margin
from src.eval.dual_champion_composition import build_operators
from src.eval.dual_champion_independent_confirmation import (
    IndependentConfirmationError,
    paired_advantage,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ai1_dual_champion_independent_confirmation_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_contract_and_source_population_validate() -> None:
    validated = validate_contract(ROOT, _config())
    assert validated["candidate_ids"] == [
        "anchor56_chroma_margin4_challenger",
        "cyan_shadow_warm_highlight_like__s50",
        "density_then_anchor__density_s50",
    ]
    assert len(validated["eligible_ids"]) == 17
    assert len({row["make"] for row in validated["source_rows"].values()}) == 9
    assert "fujifilm_s2pro" not in validated["eligible_ids"]


def test_parent_operator_construction_replays_frozen_parent_pixels() -> None:
    config = _config()
    validated = validate_contract(ROOT, config)
    parent_config = validated["parent_config"]
    parent_validated = validated["parent_validated"]
    apply_anchor, apply_density = build_operators(
        parent_config,
        parent_validated,
    )
    sample = parent_validated["samples"]["01"]
    with Image.open(ROOT / sample["source_path"]) as image:
        source_u8 = np.asarray(
            ImageOps.exif_transpose(image).convert("RGB"),
            dtype=np.uint8,
        )
    source_float32 = source_u8.astype(np.float32) / 255.0
    source_float64 = source_u8.astype(np.float64) / 255.0

    anchor = apply_output_margin(
        apply_anchor(source_float32),
        int(parent_config["parent_anchor"]["output_margin"]),
    )
    anchor_pixels = np.rint(anchor * 255.0).astype(np.uint8)
    anchor_record = next(
        row
        for row in parent_validated["anchor_manifest"]["records"]
        if row["candidate_id"]
        == parent_config["parent_anchor"]["candidate_id"]
        and row["sample_id"] == "01"
    )
    with Image.open(ROOT / anchor_record["output"]) as image:
        archived_anchor = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert np.array_equal(anchor_pixels, archived_anchor)

    density = apply_density(
        source_float64,
        float(parent_config["parent_density"]["control_strength"]),
    )
    density_pixels = np.rint(density * 255.0).astype(np.uint8)
    density_record = next(
        row
        for row in parent_validated["density_manifest"]["records"]
        if row["candidate_id"]
        == parent_config["parent_density"]["candidate_id"]
        and row["sample_id"] == "01"
    )
    density_manifest_path = ROOT / parent_config["parent_density"]["manifest"]
    with Image.open(
        density_manifest_path.parent / density_record["output"]
    ) as image:
        archived_density = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert np.array_equal(density_pixels, archived_density)


def test_contract_rejects_excluded_row_or_threshold_drift() -> None:
    config = _config()
    config["confirmation_population"]["eligible_ids"][0] = "fujifilm_s2pro"
    with pytest.raises(IndependentConfirmationError, match="excluded"):
        validate_contract(ROOT, config)

    config = _config()
    config["metrics"]["minimum_confirmation_median_style_delta_e76"] += 0.01
    with pytest.raises(IndependentConfirmationError, match="threshold"):
        validate_contract(ROOT, config)


def test_paired_advantage_counts_image_and_make_wins() -> None:
    validated = validate_contract(ROOT, _config())
    candidate = []
    first_parent = []
    second_parent = []
    for sample_id, row in validated["source_rows"].items():
        shared = {"sample_id": sample_id, "make": row["make"]}
        candidate.append(
            {
                **shared,
                "median_style_delta_e76": 3.0,
                "median_non_basic_residual_delta_e76": 2.5,
            }
        )
        first_parent.append(
            {
                **shared,
                "median_style_delta_e76": 2.0,
                "median_non_basic_residual_delta_e76": 1.5,
            }
        )
        second_parent.append(
            {
                **shared,
                "median_style_delta_e76": 2.5,
                "median_non_basic_residual_delta_e76": 2.0,
            }
        )
    summary = paired_advantage(
        candidate_rows=candidate,
        parent_rows=[first_parent, second_parent],
        metrics=_config()["metrics"],
    )
    assert summary["median_style_gain_over_best_parent_delta_e76"] == 0.5
    assert (
        summary["median_non_basic_gain_over_best_parent_delta_e76"] == 0.5
    )
    assert summary["per_image_style_wins"] == 17
    assert summary["per_image_non_basic_wins"] == 17
    assert summary["camera_make_style_wins"] == 9
    assert summary["camera_make_non_basic_wins"] == 9

    with pytest.raises(IndependentConfirmationError, match="population"):
        paired_advantage(
            candidate_rows=candidate,
            parent_rows=[first_parent[:-1], second_parent],
            metrics=_config()["metrics"],
        )


def test_runner_help_loads_from_repo_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "run_u5_r2ai1_dual_champion_independent_confirmation.py"
            ),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--build-blind-sheets" in completed.stdout
