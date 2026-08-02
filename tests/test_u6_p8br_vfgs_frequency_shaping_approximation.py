from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8br_vfgs_frequency_shaping_approximation_v1.json"


def test_p8br_contract_freezes_current_primary_sources_and_no_ml_runtime() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"].endswith("contract.v1")
    assert contract["primary_sources"]["vfgs"]["commit"] == (
        "fbf4bd95058e934fb7246edd5e7fb8d6c9ed0ec0"
    )
    assert contract["primary_sources"]["vfgs"]["core_sha256"] == (
        "8051fbefc6fef8395e678a4a105457fc799a16044e51a6123f53fd0a28e0946a"
    )
    assert contract["primary_sources"]["fga_nn"]["pdf_sha256"] == (
        "cf55d2ab2cadd7d80cd50dde4c1fc2db286b7e6b203fe1de7cc0f46749c9e30a"
    )
    assert contract["model"]["cutoff_candidates_inclusive"] == [2, 14]
    assert contract["model"]["soft_window_k"] == 1.118
    assert contract["model"]["learned_predictor_used"] is False
    assert contract["roles"]["refit_rescale_or_cutoff_change_on_confirmation"] is False
