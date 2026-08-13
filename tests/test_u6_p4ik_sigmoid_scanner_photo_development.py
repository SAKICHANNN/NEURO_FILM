import json
from pathlib import Path

from src.eval.sigmoid_scanner_photo_development import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ik_sigmoid_scanner_photo_development_v1.json"


def test_p4ik_contract_freezes_existing_photo_population_and_gates() -> None:
    contract = load_contract(CONFIG)
    assert contract["source"]["expected_evaluation_rows"] == 11
    assert contract["sigmoid"] == {
        "fit_samples": 4097,
        "initial_slope": 8.0,
        "maximum_iterations": 20000,
        "maximum_curve_fit_rmse": 0.04,
    }
    assert contract["candidate"]["cohort_fitting_allowed"] is False
    assert contract["candidate"]["hard_clipping_allowed"] is False
    assert contract["automatic_gates"]["maximum_new_boundary_fraction_vs_matched_scanner_ao6"] == 0.0005
    evidence = json.loads(
        (ROOT / contract["parents"]["p4ij_evidence"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["decision"] == contract["parents"]["p4ij_evidence"]["required_decision"]
