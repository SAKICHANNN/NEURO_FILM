from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.fivek_bilateral_affine_capacity import (
    _affine_design,
    _apply_safe_affine,
    evaluate_affine_capacity,
)
from src.eval.fivek_bilateral_affine_capacity_run import validate_contract
from src.eval.fivek_bilateral_gain_capacity import _gain_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bt1_fivek_bilateral_affine_capacity_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bt1_contract_is_development_only_and_parameter_matched() -> None:
    config = json.loads(CONFIG.read_text())
    validate_contract(ROOT, config)
    assert config["candidate"]["parameter_count"] == 384
    assert config["controls"]["closed_bt0_gain"]["parameter_count"] == 384
    assert config["confirmation_pixels_allowed"] is False
    assert config["learned_final_rgb_allowed"] is False


def test_affine_design_and_analytical_guard_are_bounded() -> None:
    source = np.full((5, 7, 3), 0.5)
    rank = np.linspace(0.0, 1.0, 35).reshape(5, 7)
    features = _gain_features((5, 7), rank, (2, 2, 8))
    design = _affine_design(features, source)
    assert design.shape == (35, 128)
    coefficients = np.full((128, 3), 1.5)
    output, dose = _apply_safe_affine(
        source,
        features,
        coefficients,
        {"epsilon": 1.0 / 4096.0},
    )
    assert np.all(np.isfinite(output))
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.min(dose) < 1.0


def test_affine_grid_scores_additive_local_effect_against_frozen_controls() -> None:
    config = json.loads(CONFIG.read_text())
    config["population"]["source_count_exact"] = 2
    config["evaluation"]["automatic_gates"]["source_count_exact"] = 2
    rows = []
    for index in range(2):
        y, x = np.mgrid[0:32, 0:32]
        source = np.stack(
            [
                0.08 + 0.7 * x / 31.0,
                0.10 + 0.65 * y / 31.0,
                0.15 + 0.45 * (x + y) / 62.0,
            ],
            axis=-1,
        )
        local = 0.06 * np.sin(2.0 * np.pi * x / 31.0) * (0.5 + 0.5 * y / 31.0)
        target = np.clip(source + np.stack([local, -0.5 * local, 0.25 * local], axis=-1), 0, 1)
        rows.append(
            {
                "pair_id": f"synthetic-{index}",
                "group": f"group-{index}",
                "target_variant": "aligned_expert",
                "source": linear_srgb_to_encoded(source),
                "target": linear_srgb_to_encoded(target),
            }
        )
    report = evaluate_affine_capacity(rows, config)
    assert np.isfinite(report["metrics"]["mean_improvement_over_closed_bt0_gain"])
    assert len(report["rows"]) == 4
    assert report["metrics"]["maximum_out_of_cube_fraction"] == 0.0


def test_tracked_evidence_binds_exact_formal_close() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/U5_R2BT1_FIVEK_BILATERAL_AFFINE_CAPACITY_RESULT.json").read_text()
    )
    reports = [ROOT / path for path in evidence["formal_reports"]["paths"]]
    assert {_sha256(path) for path in reports} == {evidence["formal_reports"]["sha256"]}
    report = json.loads(reports[0].read_text())
    assert report["stable_evidence_id"] == evidence["formal_reports"]["stable_evidence_id"]
    assert report["automatic_pass"] is False
    assert report["confirmation_rows_loaded"] == 0
