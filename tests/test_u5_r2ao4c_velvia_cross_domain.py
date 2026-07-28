from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao4c_velvia_cross_domain import (
    CONFIG_SHA256,
    load_config,
)
from src.real_film.velvia_cross_domain import evaluate_cross_domain


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao4c_velvia_chart_palette_cross_domain_v1.json"


def test_config_identity_and_closed_claims() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    assert config["fit"]["restart_count"] == 3
    assert config["stock_response_claim_allowed"] is False
    assert config["production_integration_allowed"] is False


def test_config_hash_fail_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG, expected_sha256="0" * 64)


def test_evaluator_requires_exact_domains() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    rows = np.linspace(0.01, 0.9, 36).reshape(12, 3)
    with pytest.raises(ValueError, match="all three"):
        evaluate_cross_domain({"velvia_chart": (rows, rows)}, config)
