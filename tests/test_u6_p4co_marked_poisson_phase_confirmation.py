from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval import marked_poisson_phase_confirmation as p4co

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4co_marked_poisson_phase_confirmation_v1.json"
EVIDENCE = ROOT / "docs/evidence/U6_P4CO_MARKED_POISSON_PHASE_CONFIRMATION_RESULT.json"


def test_contract_freezes_new_sources_and_bounded_topology_grid() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert set(contract["development"]["source_names"]).isdisjoint(
        contract["confirmation"]["source_names"]
    )
    assert contract["candidate"]["expected_parent_count_grid"] == [4, 8, 16, 32, 64, 128]
    assert contract["candidate"]["cluster_sigma_grid_pixels"] == [1.0, 2.0, 4.0, 8.0]
    assert contract["candidate"]["offspring_per_parent"] == 8
    assert contract["candidate"]["p4cm_log_scale_std"] == 0.2847407310619019
    assert contract["confirmation"]["metadata_lock_sha256"] == (
        "f0277eab5ba67e726965fb087d8ffe1a15642f3dc440beee46880937c4dc7c36"
    )


def test_contract_retains_exact_spectrum_and_strength_gate() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["confirmation_gates"][
        "minimum_median_feature_distance_improvement_over_p4cm"
    ] == 0.2
    assert contract["confirmation_gates"]["require_exact_power_projection"] is True
    assert contract["confirmation_gates"]["require_exact_acf_projection"] is True


def test_marked_phase_is_repeat_exact_and_power_preserving() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = p4co._marked_fields(contract, 4, 8, 2.0, seed_offset=7)
    second = p4co._marked_fields(contract, 4, 8, 2.0, seed_offset=7)
    assert np.array_equal(first, second)
    compat = p4co._p4cm_contract(contract)
    bases, _scales = p4co._base_and_scale_fields(compat, 4, seed_offset=7)
    errors = p4co._projection_errors(contract, bases, first)
    assert errors["maximum_per_sample_power_relative_error"] <= 5e-8
    assert errors["maximum_per_sample_acf_absolute_error"] <= 1e-12


def test_topology_parameters_change_high_order_features() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    sparse = p4co._marked_fields(contract, 8, 4, 1.0, seed_offset=11)
    broad = p4co._marked_fields(contract, 8, 64, 8.0, seed_offset=11)
    assert not np.array_equal(p4co._feature_matrix(sparse), p4co._feature_matrix(broad))


def test_evidence_retains_median_signal_but_closes_worst_tail() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["two_full_runs_byte_identical"] is True
    assert evidence["confirmation"]["sources_beating_p4cm"] == 3
    assert evidence["gates"]["minimum_20_percent_median_feature_improvement"] is True
    assert evidence["gates"]["worst_feature_distance"] is False
    assert evidence["projection"]["power_gate"] is True
    assert evidence["decision"] == "FAIL_CLOSED_MARKED_POISSON_PHASE_TRANSFER"
