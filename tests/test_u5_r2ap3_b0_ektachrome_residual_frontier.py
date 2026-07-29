from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_u5_r2ap3_b0_ektachrome_residual_frontier import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.b0_real_film_residual_frontier import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ap3_b0_ektachrome_residual_frontier_v1.json"
)


def test_ap3_contract_reuses_exact_ao6_bank() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    validated = validate_contract(ROOT, config)
    assert len(validated["samples"]) == 41
    assert [
        (row["tone_strength"], row["chroma_strength"])
        for row in validated["candidates"]
    ] == [
        (0.1, 0.25),
        (0.15, 0.35),
        (0.2, 0.5),
        (0.25, 0.65),
    ]
    assert config["operator_witness_id"] == (
        "ektachrome_palette_one_matrix"
    )
    assert config["operator_refit_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_ap3_config_hash_fails_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG, expected_sha256="0" * 64)
