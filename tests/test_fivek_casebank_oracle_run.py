from __future__ import annotations

import hashlib
import json

import pytest

from src.eval.fivek_casebank_oracle_run import (
    FiveKCasebankOracleRunError,
    run_oracle,
    validate_contract,
)


def _write_json(path, payload):
    data = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _contract(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    manifest_sha = _write_json(
        manifest_path,
        {
            "rows": [],
            "split_summary": {
                "development_rows": 3,
                "confirmation_rows": 2,
                "selection_used_target_or_pixels": False,
            },
        },
    )
    report_sha = _write_json(report_path, {"automatic_pass": True})
    evaluation = {
        "samples_per_confirmation_image": 32,
        "strength_doses": [0.0, 0.5, 1.0],
        "random_case_seed": 5,
        "bootstrap_seed": 7,
        "bootstrap_repetitions": 1000,
        "gates": {
            "minimum_mean_improvement_over_identity": 0.05,
            "minimum_win_fraction_over_identity": 0.6,
            "maximum_p95_ratio_to_identity": 1.0,
            "maximum_worst_ratio_to_identity": 1.1,
            "minimum_mean_improvement_over_global": 0.05,
            "minimum_win_fraction_over_global": 0.6,
            "maximum_p95_ratio_to_global": 1.0,
            "maximum_worst_ratio_to_global": 1.1,
            "minimum_mean_improvement_over_strength_oracle": 0.03,
            "minimum_win_fraction_over_strength_oracle": 0.55,
            "minimum_mean_improvement_over_random_case": 0.05,
            "minimum_win_fraction_over_random_case": 0.6,
            "minimum_bootstrap_lower_improvement": 0.02,
            "minimum_distinct_selected_cases": 8,
            "maximum_selected_case_share": 0.2,
        },
    }
    return {
        "status": "contract_frozen_implementation_ready",
        "experiment_id": "test",
        "parent_population": {
            "manifest": manifest_path.name,
            "manifest_sha256": manifest_sha,
            "report": report_path.name,
            "report_sha256": report_sha,
            "development_rows": 3,
            "confirmation_rows": 2,
        },
        "decode": {"maximum_side": 32},
        "operator": {
            "family": "monotone_triangular_logit_transport",
            "parameter_count": 14,
            "lower_bounds": [-1.0] * 14,
            "upper_bounds": [1.0] * 14,
            "hard_output_clipping_allowed": False,
            "spatial_or_semantic_features_allowed": False,
            "learned_final_rgb_allowed": False,
        },
        "target_variants": {
            "aligned_expert": {
                "target_variant": "aligned_expert",
                "evaluation": evaluation,
            },
            "filtered": {
                "target_variant": "filtered",
                "evaluation": evaluation,
            },
        },
        "required_pass_variants": ["aligned_expert", "filtered"],
        "router_training_allowed": False,
        "confirmation_target_use_for_selection_allowed": False,
        "final_rgb_learning_allowed": False,
        "production_integration_allowed": False,
        "film_or_stock_claim_allowed": False,
        "claim_ceiling": "test only",
    }


def test_runner_contract_binds_target_blind_parent(tmp_path) -> None:
    config = _contract(tmp_path)
    validated = validate_contract(tmp_path, config)
    assert validated["manifest"]["split_summary"]["development_rows"] == 3


def test_runner_executes_both_required_variants(tmp_path, monkeypatch) -> None:
    config = _contract(tmp_path)
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    calls = []

    def fake_load(manifest, *, split, target_variant, maximum_side):
        calls.append((split, target_variant, maximum_side))
        return [{"pair_id": f"{split}-1"}]

    def fake_evaluate(development, confirmation, inner_config):
        return {"automatic_pass": True, "target": len(calls)}

    monkeypatch.setattr(
        "src.eval.fivek_casebank_oracle_run.load_split_population", fake_load
    )
    monkeypatch.setattr(
        "src.eval.fivek_casebank_oracle_run.evaluate_offdiagonal_oracle",
        fake_evaluate,
    )
    output = tmp_path / "out" / "report.json"
    report = run_oracle(
        root=tmp_path,
        config=config,
        config_path=config_path,
        output_path=output,
        software_commit="a" * 40,
    )
    assert report["automatic_pass"] is True
    assert calls == [
        ("development", "aligned_expert", 32),
        ("confirmation", "aligned_expert", 32),
        ("development", "filtered", 32),
        ("confirmation", "filtered", 32),
    ]
    assert output.is_file()


def test_runner_rejects_router_training(tmp_path) -> None:
    config = _contract(tmp_path)
    config["router_training_allowed"] = True
    with pytest.raises(FiveKCasebankOracleRunError, match="boundary"):
        validate_contract(tmp_path, config)


def test_runner_rejects_operator_and_strength_control_drift(tmp_path) -> None:
    config = _contract(tmp_path)
    config["operator"]["hard_output_clipping_allowed"] = True
    with pytest.raises(FiveKCasebankOracleRunError, match="operator"):
        validate_contract(tmp_path, config)

    config = _contract(tmp_path)
    config["target_variants"]["filtered"]["evaluation"][
        "strength_doses"
    ] = [0.0, 0.5]
    with pytest.raises(FiveKCasebankOracleRunError, match="evaluation"):
        validate_contract(tmp_path, config)
