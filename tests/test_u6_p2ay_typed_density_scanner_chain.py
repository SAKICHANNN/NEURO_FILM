from pathlib import Path

import numpy as np
import pytest

from src.eval.typed_density_scanner_chain import load_contract, run_audit
from src.film_physics.bw_density_scanner_chain import (
    build_typed_neutral_density_scanner_chain,
)
from src.film_physics.spatial_response import SpatialResponseProfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ay_typed_density_scanner_chain_v1.json"


def test_p2ay_typed_scanner_chain_replays_and_decides() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is all(first["gate_results"].values())
    assert first["measurements"]["wrong_order_rms_difference"] > 0.0


def test_neutral_scanner_chain_rejects_invalid_density() -> None:
    profile = SpatialResponseProfile(
        6.35,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (4.445, 4.445, 4.445),
        3.0,
    )
    with pytest.raises(ValueError):
        build_typed_neutral_density_scanner_chain(
            np.asarray([[0.1, -0.1]], dtype=np.float64), profile
        )
