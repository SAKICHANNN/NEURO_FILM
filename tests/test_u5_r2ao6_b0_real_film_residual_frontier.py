from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_u5_r2ao6_b0_real_film_residual_frontier import (
    CONFIG_SHA256,
)
from src.eval.b0_real_film_residual_frontier import (
    B0RealFilmResidualFrontierError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao6_b0_real_film_residual_frontier_v1.json"


def test_ao6_contract_hash_lineage_and_bank() -> None:
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    validated = validate_contract(ROOT, config)
    assert len(validated["candidates"]) == config["candidate_count"] == 4
    assert len(validated["base_records"]) == 41
    assert config["base"]["candidate_id"] == (
        "density_then_anchor__density_s50"
    )
    assert config["operator_witness_id"] == "velvia_combined_one_matrix"
    assert config["operator_refit_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_ao6_bank_is_unique_bounded_and_increasing() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bank = config["residual_bank"]
    assert len({row["candidate_id"] for row in bank}) == len(bank)
    assert [row["tone_strength"] for row in bank] == sorted(
        row["tone_strength"] for row in bank
    )
    assert [row["chroma_strength"] for row in bank] == sorted(
        row["chroma_strength"] for row in bank
    )
    assert all(0.0 <= row["tone_strength"] <= 1.0 for row in bank)
    assert all(0.0 <= row["chroma_strength"] <= 2.0 for row in bank)


def test_ao6_rejects_refit_or_lineage_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator_refit_allowed"] = True
    with pytest.raises(B0RealFilmResidualFrontierError, match="boundary"):
        validate_contract(ROOT, config)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["base"]["manifest_sha256"] = "0" * 64
    with pytest.raises(B0RealFilmResidualFrontierError, match="hash mismatch"):
        validate_contract(ROOT, config)
