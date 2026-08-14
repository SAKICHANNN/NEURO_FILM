from pathlib import Path

import numpy as np
import pytest

from src.eval.bw_negative_direct_scan import load_contract, run_audit
from src.film_physics.bw_density_scanner_chain import (
    interpret_bw_negative_direct_scan,
)
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2az_bw_negative_direct_scan_v1.json"


def test_p2az_direct_scan_replays_and_decides() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is all(first["gate_results"].values())
    assert first["measurements"]["endpoint_fit_count_zero"] is True


def test_direct_scan_rejects_wrong_domain() -> None:
    wrong = PhysicalDomainArray(
        np.full((2, 2, 3), 0.5, dtype=np.float64),
        PhysicalDomain.TRANSMITTANCE,
        PhysicalUnit.TRANSMITTANCE,
        ("neutral", "neutral", "neutral"),
    )
    with pytest.raises(ValueError):
        interpret_bw_negative_direct_scan(wrong)
