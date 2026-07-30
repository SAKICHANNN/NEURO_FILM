from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_fresh_pair_acquisition import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay6s1_fivek_confirmation_acquisition_v1.json"
)


def test_contract_binds_exact_third_population_and_owned_root() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["preflight_manifest"]["rows"]) == 64
    assert config["preflight"]["expected_bytes"] == 3_667_932_324
    assert config["ownership"]["foreign_resource_mutation_allowed"] is False
    assert "dimension_mismatches" not in config["pass_gates"]
