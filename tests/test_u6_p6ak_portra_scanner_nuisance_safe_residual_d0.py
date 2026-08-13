from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.portra_scanner_nuisance_safe_residual_d0 import (
    load_contract,
    maximum_safe_residual,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p6ak_portra_scanner_nuisance_safe_residual_d0_v1.json"


def test_maximum_safe_residual_is_bounded_and_collinear() -> None:
    source = np.asarray([[0.2, 0.4, 0.6], [0.3, 0.3, 0.3]], dtype=np.float64)
    raw = np.asarray([[1.4, 0.2, -0.2], [0.4, 0.2, 0.5]], dtype=np.float64)
    safe, alpha = maximum_safe_residual(source, raw)
    assert np.all((safe >= 0.0) & (safe <= 1.0))
    assert alpha[0] < 1.0
    assert alpha[1] == 1.0
    np.testing.assert_allclose(safe - source, alpha[:, None] * (raw - source), atol=2e-16, rtol=0.0)


def test_contract_forbids_clipping_and_refit() -> None:
    contract = load_contract(CONFIG)
    assert contract["execution"]["hard_clipping_allowed"] is False
    assert contract["execution"]["fit"].startswith("exact_p6aj")


def test_contract_rejects_operator_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["execution"]["operator"] = "clip"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_contract(path)
