from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.editreward_consumed_source_aware_preference import (
    ScoreRow,
    _presentation_index,
    _validate_smoke_receipt,
    aggregate_score_locks,
    audit_score_locks,
    wrong_source_controls,
)


def _row(dataset: str, index: int, label: str) -> ScoreRow:
    source_prefix = "p399" if dataset == "p401" else "p402"
    path = Path(f"{dataset}-{index}-{label}.png")
    return ScoreRow(
        dataset=dataset,
        source_id=f"{source_prefix}_{index:02d}",
        presentation_id=f"R1-{index + 1:02d}",
        role=("landscape_nature", "architecture_interior", "night_artificial_light")[
            index // 4
        ],
        label=label,
        source_path=path,
        candidate_path=path,
        source_sha256=f"source-{dataset}-{index}",
        candidate_sha256=f"candidate-{dataset}-{index}-{label}",
        source_geometry=(8, 8),
        candidate_geometry=(8, 8),
    )


def test_presentation_index_accepts_only_frozen_round_one() -> None:
    assert _presentation_index("R1-01") == 0
    assert _presentation_index("R1-12") == 11
    for invalid in ("R2-01", "R1-00", "R1-13"):
        try:
            _presentation_index(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{invalid} must fail closed")


def test_wrong_source_control_is_deterministic_cyclic_and_candidate_blind() -> None:
    rows = []
    for dataset, labels in (("p401", "AB"), ("p402", "ABCD")):
        for index in range(12):
            rows.extend(_row(dataset, index, label) for label in labels)
    controls = wrong_source_controls(rows)
    assert len(controls) == 24
    assert len({candidate.row_id for candidate, _ in controls}) == 24
    for candidate, wrong_source in controls:
        source_index = int(candidate.source_id[-2:])
        expected_index = (source_index + 1) % 12
        assert wrong_source.source_id.endswith(f"{expected_index:02d}")
        assert candidate.source_id != wrong_source.source_id


def test_row_id_contains_no_private_candidate_identity() -> None:
    row = _row("p402", 3, "C")
    assert row.row_id == "p402:R1-04:C"
    assert "identity" not in row.row_id
    assert "complete" not in row.row_id


def test_mechanics_audit_accepts_exact_reverse_enumeration(tmp_path: Path) -> None:
    contract = {
        "experiment_id": "U5.R2EDITREWARD0",
        "fixed_instruction": "fixed",
        "external_asset": {
            "model_sha256": "model-sha",
            "tensor_count": 741,
            "parameter_count": 8296688740,
        },
        "mechanics_gates": {
            "canonical_reverse_same_asset_score_max_abs_error_max": 1e-5
        },
        "claim_ceiling": "test only",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), "utf-8")
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    primary = []
    wrong = []
    for dataset, labels in (("p401", "AB"), ("p402", "ABCD")):
        for index in range(12):
            for label_index, label in enumerate(labels):
                row = _row(dataset, index, label)
                primary.append(
                    {
                        "row_id": row.row_id,
                        "dataset": row.dataset,
                        "source_id": row.source_id,
                        "presentation_id": row.presentation_id,
                        "role": row.role,
                        "label": row.label,
                        "source_kind": "correct",
                        "source_sha256": row.source_sha256,
                        "candidate_sha256": row.candidate_sha256,
                        "source_geometry": [8, 8],
                        "candidate_geometry": [8, 8],
                        "mean": float(label_index),
                        "log_sigma": 0.0,
                    }
                )
            control_row = _row(dataset, index, labels[0])
            wrong.append(
                {
                    "row_id": control_row.row_id,
                    "dataset": control_row.dataset,
                    "source_id": control_row.source_id,
                    "presentation_id": control_row.presentation_id,
                    "role": control_row.role,
                    "label": control_row.label,
                    "source_kind": "cyclic-wrong",
                    "source_sha256": control_row.source_sha256,
                    "candidate_sha256": control_row.candidate_sha256,
                    "source_geometry": [8, 8],
                    "candidate_geometry": [8, 8],
                    "wrong_source_id": f"wrong-{dataset}-{index}",
                    "wrong_source_sha256": f"wrong-sha-{dataset}-{index}",
                    "mean": -1.0,
                    "log_sigma": 0.0,
                }
            )
    load = {
        "state_tensor_count": 741,
        "state_parameter_count": 8296688740,
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
        "device_map": {"": "cuda:0"},
    }
    paths = []
    for order in ("canonical", "reverse"):
        lock = {
            "schema": "neuro-film.u5-r2editreward0-blind-score-lock.v1",
            "status": "BLIND_SCORE_LOCK_COMPLETE",
            "smoke": False,
            "order": order,
            "contract_sha256": contract_sha,
            "model_sha256": "model-sha",
            "instruction_sha256": hashlib.sha256(b"fixed").hexdigest(),
            "inventory": {
                "primary_count": 72,
                "wrong_source_count": 24,
                "private_mapping_reads": 0,
                "direct_result_reads": 0,
            },
            "load": load,
            "primary": list(reversed(primary)) if order == "reverse" else primary,
            "wrong_source": list(reversed(wrong)) if order == "reverse" else wrong,
        }
        path = tmp_path / f"{order}.json"
        path.write_text(json.dumps(lock, sort_keys=True), "utf-8")
        paths.append(path)
    result = audit_score_locks(contract_path, paths)
    assert result["status"] == "PASS_MECHANICS_READY_FOR_PRIVATE_AGGREGATION"
    assert result["scientific_preference_metrics_computed"] is False
    assert result["metrics"]["maximum_fresh_process_score_error"] == 0.0


def test_mechanics_audit_rejects_changed_wrong_source_identity(tmp_path: Path) -> None:
    contract = {
        "experiment_id": "U5.R2EDITREWARD0",
        "fixed_instruction": "fixed",
        "external_asset": {
            "model_sha256": "model-sha",
            "tensor_count": 741,
            "parameter_count": 8296688740,
        },
        "mechanics_gates": {
            "canonical_reverse_same_asset_score_max_abs_error_max": 1e-5
        },
        "claim_ceiling": "test only",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), "utf-8")
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()

    primary = []
    wrong = []
    for dataset, labels in (("p401", "AB"), ("p402", "ABCD")):
        for index in range(12):
            for label_index, label in enumerate(labels):
                row = _row(dataset, index, label)
                primary.append(
                    {
                        "row_id": row.row_id,
                        "dataset": row.dataset,
                        "source_id": row.source_id,
                        "presentation_id": row.presentation_id,
                        "role": row.role,
                        "label": row.label,
                        "source_kind": "correct",
                        "source_sha256": row.source_sha256,
                        "candidate_sha256": row.candidate_sha256,
                        "source_geometry": [8, 8],
                        "candidate_geometry": [8, 8],
                        "mean": float(label_index),
                        "log_sigma": 0.0,
                    }
                )
            control_row = _row(dataset, index, labels[0])
            wrong.append(
                {
                    "row_id": control_row.row_id,
                    "dataset": control_row.dataset,
                    "source_id": control_row.source_id,
                    "presentation_id": control_row.presentation_id,
                    "role": control_row.role,
                    "label": control_row.label,
                    "source_kind": "cyclic-wrong",
                    "source_sha256": control_row.source_sha256,
                    "candidate_sha256": control_row.candidate_sha256,
                    "source_geometry": [8, 8],
                    "candidate_geometry": [8, 8],
                    "wrong_source_id": f"wrong-{dataset}-{index}",
                    "wrong_source_sha256": f"wrong-sha-{dataset}-{index}",
                    "mean": -1.0,
                    "log_sigma": 0.0,
                }
            )
    load = {
        "state_tensor_count": 741,
        "state_parameter_count": 8296688740,
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
        "device_map": {"": "cuda:0"},
    }
    paths = []
    for order in ("canonical", "reverse"):
        lock = {
            "schema": "neuro-film.u5-r2editreward0-blind-score-lock.v1",
            "status": "BLIND_SCORE_LOCK_COMPLETE",
            "smoke": False,
            "order": order,
            "contract_sha256": contract_sha,
            "model_sha256": "model-sha",
            "instruction_sha256": hashlib.sha256(b"fixed").hexdigest(),
            "inventory": {
                "primary_count": 72,
                "wrong_source_count": 24,
                "private_mapping_reads": 0,
                "direct_result_reads": 0,
            },
            "load": load,
            "primary": primary,
            "wrong_source": json.loads(json.dumps(wrong)),
        }
        if order == "reverse":
            lock["wrong_source"][0]["wrong_source_sha256"] = "changed"
        path = tmp_path / f"{order}.json"
        path.write_text(json.dumps(lock, sort_keys=True), "utf-8")
        paths.append(path)

    result = audit_score_locks(contract_path, paths)
    assert result["status"] == "INVALID_MECHANICS_EDITREWARD_LOCAL_PATH"
    assert result["gate_results"]["row_identities_exact"] is False


def test_aggregate_score_locks_passes_perfect_consumed_truth(tmp_path: Path) -> None:
    def write(name: str, payload: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload, sort_keys=True), "utf-8")
        return path

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    roles = ("landscape_nature", "architecture_interior", "night_artificial_light")
    p401_mapping = {
        "rows": [
            {
                "source_id": f"p399_{index:02d}",
                "round": 1,
                "presentation_id": f"R1-{index + 1:02d}",
                "label_to_variant": {"A": "candidate", "B": "identity"},
            }
            for index in range(12)
        ]
    }
    p401_result = {
        "source_results": [
            {
                "source_id": f"p399_{index:02d}",
                "preference_decision": "candidate" if index < 11 else "tie",
            }
            for index in range(12)
        ]
    }
    variants = ("identity", "calibration", "channel_balance", "color_curve")
    pairwise_winners = {
        "identity": [],
        "calibration": ["identity"],
        "channel_balance": ["identity", "calibration"],
        "color_curve": ["identity", "calibration", "channel_balance"],
    }
    p402_mapping = {
        "rows": [
            {
                "source_id": f"p402_{index:02d}",
                "round": 1,
                "presentation_id": f"R1-{index + 1:02d}",
                "label_to_candidate": dict(zip("ABCD", variants, strict=True)),
            }
            for index in range(12)
        ]
    }
    p402_result = {
        "rows": [
            {
                "source_id": f"p402_{index:02d}",
                "role": roles[index // 4],
                "pairwise_winners": pairwise_winners,
                "unique_winner": "color_curve" if index < 11 else None,
            }
            for index in range(12)
        ]
    }
    p401_mapping_path = write("p401_mapping.json", p401_mapping)
    p401_result_path = write("p401_result.json", p401_result)
    p402_mapping_path = write("p402_mapping.json", p402_mapping)
    p402_review_path = write("p402_review.json", {"bound": True})
    p402_result_path = write("p402_result.json", p402_result)

    contract = {
        "experiment_id": "U5.R2EDITREWARD0",
        "fixed_instruction": "fixed",
        "external_asset": {
            "model_sha256": "model-sha",
            "tensor_count": 741,
            "parameter_count": 8296688740,
        },
        "consumed_inputs": {
            "p401": {
                "private_mapping_sha256": digest(p401_mapping_path),
                "formal_report_sha256": digest(p401_result_path),
            },
            "p402": {
                "private_mapping_sha256": digest(p402_mapping_path),
                "review_binding_sha256": digest(p402_review_path),
                "formal_result_sha256": digest(p402_result_path),
            },
        },
        "mechanics_gates": {
            "canonical_reverse_same_asset_score_max_abs_error_max": 1e-5
        },
        "scientific_gates": {
            "p402_all_pair_concordant_min": 47,
            "p402_transform_vs_identity_concordant_min": 24,
            "p402_unique_winner_top1_agreement_min": 7,
            "p402_each_content_role_pairwise_concordant_min": 14,
            "p401_decisive_agreement_min": 9,
            "correct_source_score_over_wrong_source_count_min": 18,
            "correct_source_minus_wrong_source_median_min_exclusive": 0.0,
            "correct_source_each_role_over_wrong_min": 5,
        },
        "claim_ceiling": "test only",
    }
    contract_path = write("contract.json", contract)
    contract_sha = digest(contract_path)
    primary = []
    wrong = []
    for dataset, labels in (("p401", "AB"), ("p402", "ABCD")):
        for index in range(12):
            for label_index, label in enumerate(labels):
                row = _row(dataset, index, label)
                score = (
                    float(2 - label_index)
                    if dataset == "p401"
                    else float(label_index + 1)
                )
                primary.append(
                    {
                        "row_id": row.row_id,
                        "dataset": row.dataset,
                        "source_id": row.source_id,
                        "presentation_id": row.presentation_id,
                        "role": row.role,
                        "label": row.label,
                        "source_kind": "correct",
                        "source_sha256": row.source_sha256,
                        "candidate_sha256": row.candidate_sha256,
                        "source_geometry": [8, 8],
                        "candidate_geometry": [8, 8],
                        "mean": score,
                        "log_sigma": 0.0,
                    }
                )
            selected = _row(dataset, index, labels[0])
            wrong.append(
                {
                    "row_id": selected.row_id,
                    "dataset": selected.dataset,
                    "source_id": selected.source_id,
                    "presentation_id": selected.presentation_id,
                    "role": selected.role,
                    "label": selected.label,
                    "source_kind": "cyclic-wrong",
                    "source_sha256": selected.source_sha256,
                    "candidate_sha256": selected.candidate_sha256,
                    "source_geometry": [8, 8],
                    "candidate_geometry": [8, 8],
                    "wrong_source_id": f"wrong-{dataset}-{index}",
                    "wrong_source_sha256": f"wrong-sha-{dataset}-{index}",
                    "mean": 0.0,
                    "log_sigma": 0.0,
                }
            )
    load = {
        "state_tensor_count": 741,
        "state_parameter_count": 8296688740,
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
        "device_map": {"": "cuda:0"},
    }
    score_lock_paths = []
    for order in ("canonical", "reverse"):
        lock = {
            "schema": "neuro-film.u5-r2editreward0-blind-score-lock.v1",
            "status": "BLIND_SCORE_LOCK_COMPLETE",
            "smoke": False,
            "order": order,
            "contract_sha256": contract_sha,
            "model_sha256": "model-sha",
            "instruction_sha256": hashlib.sha256(b"fixed").hexdigest(),
            "inventory": {
                "primary_count": 72,
                "wrong_source_count": 24,
                "private_mapping_reads": 0,
                "direct_result_reads": 0,
            },
            "load": load,
            "primary": list(reversed(primary)) if order == "reverse" else primary,
            "wrong_source": list(reversed(wrong)) if order == "reverse" else wrong,
        }
        score_lock_paths.append(write(f"{order}.json", lock))

    result = aggregate_score_locks(
        contract_path=contract_path,
        score_lock_paths=score_lock_paths,
        p401_mapping_path=p401_mapping_path,
        p401_result_path=p401_result_path,
        p402_mapping_path=p402_mapping_path,
        p402_review_path=p402_review_path,
        p402_result_path=p402_result_path,
    )
    assert (
        result["status"]
        == "PASS_PRIVATE_CONSUMED_RETROSPECTIVE_SOURCE_AWARE_CONCORDANCE"
    )
    assert result["metrics"]["p402_all_pair_concordant"] == 72
    assert result["metrics"]["correct_source_over_wrong_count"] == 24


def test_aggregate_does_not_open_private_files_before_mechanics_pass(
    tmp_path: Path,
) -> None:
    contract = {
        "experiment_id": "U5.R2EDITREWARD0",
        "fixed_instruction": "fixed",
        "external_asset": {
            "model_sha256": "model-sha",
            "tensor_count": 741,
            "parameter_count": 8296688740,
        },
        "mechanics_gates": {
            "canonical_reverse_same_asset_score_max_abs_error_max": 1e-5
        },
        "claim_ceiling": "test only",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), "utf-8")
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    load = {
        "state_tensor_count": 741,
        "state_parameter_count": 8296688740,
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
        "device_map": {"": "cuda:0"},
    }
    score_locks = []
    for order in ("canonical", "reverse"):
        lock = {
            "schema": "neuro-film.u5-r2editreward0-blind-score-lock.v1",
            "status": "BLIND_SCORE_LOCK_COMPLETE",
            "smoke": False,
            "order": order,
            "contract_sha256": contract_sha,
            "model_sha256": "model-sha",
            "instruction_sha256": hashlib.sha256(b"fixed").hexdigest(),
            "inventory": {
                "primary_count": 72,
                "wrong_source_count": 24,
                "private_mapping_reads": 0,
                "direct_result_reads": 0,
            },
            "load": load,
            "primary": [],
            "wrong_source": [],
        }
        path = tmp_path / f"{order}.json"
        path.write_text(json.dumps(lock, sort_keys=True), "utf-8")
        score_locks.append(path)

    missing = tmp_path / "private-file-must-not-open.json"
    try:
        aggregate_score_locks(
            contract_path=contract_path,
            score_lock_paths=score_locks,
            p401_mapping_path=missing,
            p401_result_path=missing,
            p402_mapping_path=missing,
            p402_review_path=missing,
            p402_result_path=missing,
        )
    except RuntimeError as error:
        assert str(error) == "private aggregation is forbidden before mechanics pass"
    else:
        raise AssertionError("private aggregation must stop before opening truth files")


def test_formal_scoring_requires_passing_bound_smoke_receipt(tmp_path: Path) -> None:
    contract = {
        "external_asset": {"model_sha256": "model-sha"},
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), "utf-8")
    receipt = {
        "schema": "neuro-film.u5-r2editreward0-blind-score-lock.v1",
        "status": "MECHANICS_SMOKE_PASS",
        "smoke": True,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "model_sha256": "model-sha",
        "runtime": {"resource_gate_pass": True},
    }
    receipt_path = tmp_path / "smoke.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), "utf-8")
    _validate_smoke_receipt(receipt_path, contract_path, contract)

    receipt["runtime"]["resource_gate_pass"] = False
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), "utf-8")
    try:
        _validate_smoke_receipt(receipt_path, contract_path, contract)
    except ValueError as error:
        assert str(error) == "smoke receipt resource gate failed"
    else:
        raise AssertionError("failed smoke receipt must not authorize formal scoring")
