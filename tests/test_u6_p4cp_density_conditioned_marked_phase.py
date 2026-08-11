from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cp_density_conditioned_marked_phase_v1.json"


def test_contract_freezes_source_observable_two_bin_conditioning() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert set(contract["development"]["source_names"]).isdisjoint(
        contract["confirmation"]["source_names"]
    )
    assert contract["patch_observation"]["density_split"] == 0.5
    assert contract["candidate"]["expected_parent_count_grid"] == [4, 8, 16, 32, 64, 128]
    assert contract["candidate"]["cluster_sigma_pixels"] == 1.0
    assert contract["confirmation"]["metadata_lock_sha256"] == (
        "3ee90ac134a40b271b1c69ca2b51dcdcfbca537fbea56a8377a9940f09d759ea"
    )


def test_contract_retains_tail_and_exact_spectrum_gates() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    gates = contract["confirmation_gates"]
    assert gates["minimum_median_source_improvement_over_fixed_p4co"] == 0.2
    assert gates["maximum_worst_source_distance_ratio_to_fixed_p4co"] == 1.0
    assert gates["require_exact_power_projection"] is True
    assert gates["require_exact_acf_projection"] is True
