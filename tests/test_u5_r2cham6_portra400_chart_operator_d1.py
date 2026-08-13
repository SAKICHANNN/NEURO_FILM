from __future__ import annotations

from pathlib import Path

from src.eval.portra400_chart_operator_d1 import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_cham6_contract_is_spatially_held_and_non_product() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cham6_portra400_chart_operator_d1_v1.json"
    )
    assert contract["sampling"]["fold_assignment"] == "spatial_block_index_modulo_4"
    assert (
        contract["sampling"]["grid_rows"] * contract["sampling"]["grid_columns"] == 64
    )
    assert contract["models"]["fit_domain"] == "linear_srgb_from_exact_srgb_decodes"
    assert (
        contract["models"]["candidate_parameterization"]["hard_output_clipping"]
        is False
    )
    assert "not natural-scene transfer" in contract["claim_ceiling"]
