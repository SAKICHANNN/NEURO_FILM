from pathlib import Path

import numpy as np
import pytest

from src.eval.bw_characteristic_silver_chain import load_contract, run_audit
from src.film_physics.bw_characteristic_silver_chain import (
    build_bw_characteristic_silver_chain,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ai_bw_characteristic_silver_chain_v1.json"


def test_p2ai_chain_replays_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["domain_mismatch_rejected"] is True


def test_p2ai_rejects_non_2d_exposure() -> None:
    with pytest.raises(TypeError, match="BWCharacteristicSurface"):
        build_bw_characteristic_silver_chain(
            np.zeros((2, 2), dtype=np.float64),
            object(),
            development_time_minutes=9.0,
            radius_um=5.0,
            output_zoom=2,
            output_pixel_pitch_um=8.0,
            monte_carlo_samples=2,
            seed=1,
        )
