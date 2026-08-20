from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.editreward_consumed_source_aware_preference import (
    ScoreRow,
    _presentation_index,
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
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
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
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
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
