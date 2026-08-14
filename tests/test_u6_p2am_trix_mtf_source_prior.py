from pathlib import Path

import pytest

from src.eval.trix_mtf_source_prior import load_contract, run_audit
from src.film_physics.bw_mtf_source_prior import BWMTFSourcePrior

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2am_trix_mtf_source_prior_v1.json"


def test_p2am_vector_profile_replays_exactly() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["profile"]["render_allowed"] is False
    assert first["profile"]["current_stock_measurement_claim"] is False


def test_p2am_source_prior_rejects_render_authority() -> None:
    contract = load_contract(CONTRACT)
    payload = run_audit(root=ROOT, contract=contract)["profile"]
    payload.pop("schema")
    profile = BWMTFSourcePrior(
        **{
            **payload,
            "frequencies_cycles_per_mm": tuple(payload["frequencies_cycles_per_mm"]),
            "response_fractions": tuple(payload["response_fractions"]),
        }
    )
    with pytest.raises(ValueError, match="cannot render"):
        profile.render()
