from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_triangular_logit_transport_visual_product_value import (
    FiveKTriangularVisualError,
)
from src.eval.fivek_triangular_logit_transport_vs_ao6 import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bn7_triangular_logit_transport_vs_ao6_v1.json"


def test_bn7_contract_binds_challenger_and_fresh_source() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["challenger"]["pass"]
    assert len(validated["eligible_ids"]) == 10
    assert config["blind_protocol"]["primary_pair"][0] == "fixed_ao6_direct"


def test_bn7_contract_rejects_global_comparison() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["comparison_reference_arm"] = (
        "global_triangular_transport_then_fixed_ao6"
    )
    with pytest.raises(FiveKTriangularVisualError):
        validate_contract(ROOT, config)
