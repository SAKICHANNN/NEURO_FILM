from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_u6_p8bm_native_standard_strength_transaction import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u6_p8bm_native_standard_strength_transaction_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bm_native_standard_strength_transaction_decision_v1.json"
)


def test_p8bm_contract_binds_strength_parent_and_cc0_raw() -> None:
    config = json.loads(CONFIG.read_text())
    parent, _ = validate_contract(config)
    assert config["strength"] == 0.8
    assert config["strength_domain"] == "encoded-display-srgb"
    assert parent["result"]["autonomous_visual_review"][
        "preferred_global_development_strength"
    ] == 0.8
    assert hashlib.sha256(
        (ROOT / config["raw_fixture"]).read_bytes()
    ).hexdigest() == config["raw_fixture_sha256"]
    assert hashlib.sha256(
        (ROOT / config["legacy_quantized_sweep_output"]).read_bytes()
    ).hexdigest() == config[
        "legacy_quantized_sweep_output_sha256"
    ]
    assert config["raw_fixture_rights"].startswith("CC0")
    assert not config["production_default_changed"]


def test_p8bm_decision_binds_passing_formal_evidence() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["result"]["status"] == (
        "pass-local-versioned-strength-transaction"
    )
    assert decision["result"]["all_frozen_gates_pass"]
    assert decision["result"]["raw_to_png_repeat_exact"]
    assert decision["result"]["restart_verification_repeat_exact"]
    assert max(
        decision["result"]["legacy_sweep_max_abs_code_delta"]
    ) <= 1
    for path_key, hash_key in (
        ("contract", "contract_sha256"),
        ("benchmark", "benchmark_sha256"),
        ("formal_report", "formal_report_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / decision[path_key]).read_bytes()
        ).hexdigest() == decision[hash_key]
    assert decision["next_leaf"].startswith("U6.P8BN")
