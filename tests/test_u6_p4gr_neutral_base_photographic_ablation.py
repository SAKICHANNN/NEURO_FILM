import json
from pathlib import Path

import numpy as np

from src.eval.neutral_base_photographic_ablation import (
    _new_boundary_fraction,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gr_neutral_base_photographic_ablation_v1.json"


def test_p4gr_freezes_fresh_photographic_ablation_before_execution():
    contract = load_contract(CONTRACT)
    assert contract["source"]["expected_rows"] == 9
    assert contract["source"]["expected_camera_makes"] == 9
    assert contract["source"]["required_rights_scope"].startswith("CC0_")
    assert contract["candidate"]["arms"] == [
        "source",
        "ao6-only",
        "physical-residual-only",
        "physical-residual-plus-ao6",
    ]
    assert contract["candidate"]["hard_clipping_allowed"] is False
    assert contract["execution"]["formal_processes"] == 2
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert "not scene exposure" in contract["claim_ceiling"]


def test_p4gr_new_boundary_ignores_boundaries_already_in_ao6():
    reference = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    candidate = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    assert _new_boundary_fraction(reference, candidate) == 1.0 / 3.0


def test_p4gr_formal_photographic_result_closes_severe_candidate():
    evidence = json.loads(
        (
            ROOT
            / "docs/evidence/U6_P4GR_NEUTRAL_BASE_PHOTOGRAPHIC_ABLATION_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["formal_processes"] == 2
    assert evidence["reports_byte_exact"] is True
    assert evidence["automatic_pass"] is False
    assert evidence["autonomous_visual_review"]["pass"] is False
    assert evidence["observations"]["all_rows_partition_exact"] is True
    assert evidence["observations"]["hard_clipping_used"] is False
    assert evidence["observations"]["maximum_limited_fraction"] > 0.05
    assert evidence["observations"]["total_isolated_excursion_count"] > 0
    assert evidence["decision"].startswith("close_neutral_base")
