from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.neutral_transmittance_structure_ao6_d0 import (
    apply_neutral_transmittance_structure,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4ip_neutral_transmittance_structure_ao6_d0_v1.json"


def test_p4ip_scalar_structure_is_bounded_and_ratio_preserving() -> None:
    source = np.linspace(0.01, 0.99, 32 * 24 * 3, dtype=np.float32).reshape(24, 32, 3)
    result = apply_neutral_transmittance_structure(source, source, 2, 0.018)
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0 and np.max(result) <= 1.0
    ratio = result.astype(np.float64) / source.astype(np.float64)
    assert np.max(np.ptp(ratio, axis=-1)) < 2e-7
    with pytest.raises(ValueError, match="invalid P4IP input"):
        apply_neutral_transmittance_structure(source.astype(np.float64), source, 0, 0.018)


def test_p4ip_formal_replay_and_gate_decision(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    first = evaluate(contract, ROOT, contact_path=tmp_path / "a.png")
    second = evaluate(contract, ROOT, contact_path=tmp_path / "b.png")
    assert first == second
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    assert first["decision"] in {contract["decision_if_pass"], contract["decision_if_fail"]}
