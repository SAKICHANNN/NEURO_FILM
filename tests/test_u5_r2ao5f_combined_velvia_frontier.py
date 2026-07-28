from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_u5_r2ao5f_combined_velvia_frontier import CONFIG_SHA256
from src.eval.factorized_chart_frontier import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ao5f_combined_velvia_factorized_frontier_v1.json"
)


def test_ao5f_contract_hash_and_lineage() -> None:
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    validated = validate_contract(ROOT, config)
    assert len(validated["candidates"]) == 4
    assert config["operator_witness_id"] == "velvia_combined_one_matrix"
    assert config["operator_refit_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_ao5f_reuses_exact_ao3_factor_bank() -> None:
    ao5 = json.loads(CONFIG.read_text(encoding="utf-8"))
    ao3 = json.loads(
        (
            ROOT
            / "configs/u5_r2ao3_factorized_chart_boundary_frontier_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert ao5["factorization"] == ao3["factorization"]
    assert ao5["factor_bank"] == ao3["factor_bank"]
    assert ao5["metrics"] == ao3["metrics"]
