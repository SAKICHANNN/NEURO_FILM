from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4co_marked_poisson_phase_confirmation_v1.json"


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
