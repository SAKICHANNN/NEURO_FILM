from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cn_hermite_phase_coupling_confirmation_v1.json"


def test_contract_freezes_disjoint_development_and_confirmation() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    development = set(contract["development"]["source_names"])
    confirmation = set(contract["confirmation"]["source_names"])
    assert len(development) == len(confirmation) == 4
    assert development.isdisjoint(confirmation)
    assert contract["confirmation"]["metadata_lock_sha256"] == (
        "ab54569f4109c36694d31feb94a06cdaf52d443c7e10f09d8b647287a4d1c865"
    )


def test_contract_preserves_power_and_unchanged_strength_gate() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["candidate"]["alpha_grid"] == [
        -0.5 + index / 32 for index in range(33)
    ]
    assert contract["confirmation_gates"][
        "minimum_median_feature_distance_improvement_over_p4cm"
    ] == 0.2
    assert contract["confirmation_gates"]["require_exact_power_projection"] is True
    assert contract["confirmation_gates"]["require_exact_acf_projection"] is True
