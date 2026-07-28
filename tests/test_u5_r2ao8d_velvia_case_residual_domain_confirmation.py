from __future__ import annotations

import copy
from pathlib import Path

import pytest

import src.eval.velvia_case_residual_domain_confirmation as confirmation_module
from scripts.run_u5_r2ao8d_velvia_case_residual_domain_confirmation import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.velvia_case_residual_domain_confirmation import (
    VelviaCaseResidualConfirmationError,
    evaluate_case_residual_domain_confirmation,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/u5_r2ao8d_velvia_case_residual_domain_confirmation_v1.json"
)


def _config() -> dict:
    return load_config(CONFIG_PATH, expected_sha256=CONFIG_SHA256)


def test_contract_fixes_one_candidate_and_three_new_splits() -> None:
    ao8c = validate_contract(ROOT, _config())
    assert ao8c["candidate_bank"][3]["candidate_id"] == (
        "case_knn_k3_distance"
    )
    assert [row["split_id"] for row in _config()["split_repetitions"]] == [
        "split_a",
        "split_b",
        "split_c",
    ]


def test_contract_rejects_shuffle_gate_drift() -> None:
    changed = copy.deepcopy(_config())
    changed["gates"][
        "maximum_within_domain_shuffle_empirical_p_value_each_split"
    ] = 1.0
    with pytest.raises(
        VelviaCaseResidualConfirmationError, match="contract"
    ):
        validate_contract(ROOT, changed)


def test_contract_rejects_candidate_reselection() -> None:
    changed = copy.deepcopy(_config())
    changed["fixed_candidate"]["selection_allowed"] = True
    with pytest.raises(
        VelviaCaseResidualConfirmationError, match="contract"
    ):
        validate_contract(ROOT, changed)


def test_config_hash_fails_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG_PATH, expected_sha256="0" * 64)


def test_structural_fit_failure_becomes_a_frozen_negative_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_split(**kwargs):
        split = kwargs["split"]
        raise VelviaCaseResidualConfirmationError(
            f"{split['split_id']} synthetic structural fit failure"
        )

    monkeypatch.setattr(confirmation_module, "_evaluate_split", fail_split)
    report = evaluate_case_residual_domain_confirmation(ROOT, _config())
    assert report["automatic_pass"] is False
    assert report["decision"] == "close_case_selector_as_split_unstable"
    assert len(report["execution_failures"]) == 3
    assert report["checks"]["all_split_results_present"] is False
