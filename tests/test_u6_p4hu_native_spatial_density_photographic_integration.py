from __future__ import annotations

from pathlib import Path

from src.eval.native_density_photographic_integration import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_p4hu_reuses_retained_native_spatial_and_fast_density_mechanisms() -> None:
    contract = load_contract(
        ROOT / "configs/u6_p4hu_native_spatial_density_photographic_integration_v1.json"
    )
    assert contract["candidate"]["spatial_backend"] == "native-thomas-field-f32-v1"
    assert contract["candidate"]["gamma_backend"] == "fast-hybrid-v1"
    assert (
        contract["parents"]["p8bs_evidence"]["required_decision"]
        == "retain_native_generic_thomas_field_primitive"
    )
    assert contract["parents"]["p4ht_evidence"]["required_decision"].startswith(
        "retain_fast_native"
    )
    assert contract["candidate"]["cohort_refit_allowed"] is False
