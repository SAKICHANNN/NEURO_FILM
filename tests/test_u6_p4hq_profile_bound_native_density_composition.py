from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4hq_contract_binds_both_retained_native_primitives() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u6_p4hq_profile_bound_native_density_composition_v1.json"
        ).read_text("utf-8")
    )
    assert contract["parents"]["p4hn_evidence"]["required_decision"].startswith(
        "retain_source_observable"
    )
    assert contract["parents"]["p4hp_evidence"]["required_decision"].startswith(
        "retain_hybrid_native"
    )
    assert contract["candidate"]["hard_clipping_allowed"] is False
    assert contract["candidate"]["profile_refit_allowed"] is False
