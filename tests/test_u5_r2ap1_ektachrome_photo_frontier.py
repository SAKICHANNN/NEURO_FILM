from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_u5_r2ap1_ektachrome_photo_frontier import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.positive_film_frontier import candidate_bank, validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ap1_ektachrome_photo_frontier_v1.json"


def test_ap1_contract_and_lineage() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    validated = validate_contract(ROOT, config)
    assert len(validated["samples"]) == 41
    assert set(validated["operator"]["witnesses"]) == {
        "ektachrome_palette_one_matrix"
    }
    assert config["comparators"] == [
        "density_then_anchor__density_s50",
        "b0_plus_film_t15_c35",
    ]
    assert config["stock_response_claim_allowed"] is False
    assert config["production_integration_allowed"] is False


def test_ap1_candidate_bank_is_exact() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert candidate_bank(config) == [
        {
            "candidate_id": "ektachrome_palette_one_matrix__s50",
            "witness_id": "ektachrome_palette_one_matrix",
            "strength": 0.5,
        },
        {
            "candidate_id": "ektachrome_palette_one_matrix__s75",
            "witness_id": "ektachrome_palette_one_matrix",
            "strength": 0.75,
        },
        {
            "candidate_id": "ektachrome_palette_one_matrix__s100",
            "witness_id": "ektachrome_palette_one_matrix",
            "strength": 1.0,
        },
    ]


def test_ap1_config_hash_fails_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG, expected_sha256="0" * 64)
