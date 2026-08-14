from pathlib import Path

import pytest

from src.eval.historical_trix_wiener_density_source import load_contract, run_audit
from src.film_physics.bw_wiener_density_prior import BWWienerDensityPrior

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ao_historical_trix_wiener_density_source_v1.json"


def test_p2ao_historical_source_replays_exactly() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["table_copy_count"] == 2


def test_p2ao_profile_cannot_render_or_claim_current_stock() -> None:
    payload = run_audit(root=ROOT, contract=load_contract(CONTRACT))["profile"]
    payload.pop("schema")
    profile = BWWienerDensityPrior(
        **{
            **payload,
            "densities": tuple(payload["densities"]),
            "wiener_granularity_spectrum_cm2": tuple(
                payload["wiener_granularity_spectrum_cm2"]
            ),
        }
    )
    assert profile.current_400tx_claim is False
    with pytest.raises(ValueError, match="cannot render"):
        profile.render()
