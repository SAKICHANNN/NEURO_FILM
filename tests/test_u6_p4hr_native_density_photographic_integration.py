from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4hr_contract_preserves_p4he_and_bounds_claim() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u6_p4hr_native_density_photographic_integration_v1.json"
        ).read_text("utf-8")
    )
    assert contract["parents"]["p4hq_evidence"]["required_decision"].startswith(
        "retain_profile_bound"
    )
    assert contract["automatic_gates"]["require_all_p4he_photographic_gates"]
    assert contract["candidate"]["p4hj_pixel_identity_required"] is False
    assert contract["candidate"]["cohort_refit_allowed"] is False
