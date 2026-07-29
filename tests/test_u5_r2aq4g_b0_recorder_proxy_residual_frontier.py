from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_u5_r2aq4g_b0_recorder_proxy_residual_frontier import (
    CONFIG_SHA256,
)
from src.eval.recorder_proxy_b0_residual_frontier import (
    RecorderProxyB0ResidualError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq4g_b0_recorder_proxy_residual_frontier_v1.json"
)


def test_frozen_contract_binds_b0_bundle_and_ao6_control() -> None:
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    validated = validate_contract(ROOT, config)
    assert len(validated["samples"]) == 41
    assert len(validated["base_records"]) == 41
    assert len(validated["candidates"]) == 3
    assert config["operator_refit_allowed"] is False
    assert config["production_integration_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_contract_tamper_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["factorization"]["post_operator_clipping_allowed"] = True
    with pytest.raises(RecorderProxyB0ResidualError, match="contract"):
        validate_contract(ROOT, config)
