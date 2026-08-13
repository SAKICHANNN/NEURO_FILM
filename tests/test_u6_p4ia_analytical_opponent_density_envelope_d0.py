from pathlib import Path

from src.eval.analytical_opponent_density_envelope_d0 import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_p4ia_contract_freezes_adversarial_highlights() -> None:
    contract = load_contract(
        ROOT / "configs" / "u6_p4ia_analytical_opponent_density_envelope_d0_v1.json"
    )
    assert contract["development"]["highlight_floor"] == 0.84
    assert contract["mechanism"]["hard_clipping_allowed"] is False
    assert contract["gates"]["minimum_limited_fraction"] > 0.0
