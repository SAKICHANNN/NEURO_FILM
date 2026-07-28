from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao5_combined_velvia_operator import (
    CONFIG_SHA256,
    load_config,
)
from src.real_film.combined_velvia_operator import (
    evaluate_combined_velvia_operator,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao5_combined_velvia_operator_v1.json"


def test_config_identity_and_claim_ceiling() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    assert config["inputs"]["combined_rows"] == 71
    assert config["fit"]["model"] == "one_matrix"
    assert config["stock_response_claim_allowed"] is False


def test_config_hash_fail_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG, expected_sha256="0" * 64)


def test_evaluator_requires_all_declared_domains() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    rows = np.linspace(0.01, 0.9, 36).reshape(12, 3)
    with pytest.raises(ValueError, match="exact chart"):
        evaluate_combined_velvia_operator(
            {"combined": (rows, rows)}, config
        )
