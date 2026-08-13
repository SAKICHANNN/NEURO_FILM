from __future__ import annotations

from pathlib import Path

from src.eval.base_first_opponent_diffusion_photo import load_contract
from src.eval.p4hu_ao6_value import evaluate_base_first_source_arms

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4it_base_first_opponent_diffusion_photo_v1.json"


def test_p4it_contract_closes_tuning_and_binds_base_first_entry() -> None:
    contract = load_contract(CONFIG)
    assert contract["mechanism"]["P4HX_parameters_unchanged"] is True
    assert contract["mechanism"]["P4IA_envelope_unchanged"] is True
    assert contract["mechanism"]["cohort_fitting_allowed"] is False
    assert evaluate_base_first_source_arms.__doc__
