from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_u6_p7h_p4hu_ao6_value as runner


def _write_contract(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "neuro-film.u6-p7h-p4hu-ao6-value-contract.v1",
                "measurement": runner.EXPECTED_MEASUREMENT,
                "resource_gates": runner.EXPECTED_RESOURCE_GATES,
                "decision_if_pass": "PASS_FOR_BLIND_REVIEW_ONLY",
                "decision_if_fail": "FAIL_CLOSED_KEEP_AO6",
                "decision_if_infrastructure_invalid": "INVALID_NO_SCIENTIFIC_DECISION",
            }
        ),
        encoding="utf-8",
    )
    return path


def _worker_report(
    *,
    automatic_pass: bool = True,
    stable_id: str = "stable-science-id",
    output_sha: str = "a" * 64,
    rss: int = 500_000_000,
    wall: float = 30.0,
) -> dict[str, Any]:
    result = {
        "schema": "neuro-film.u6-p7h-p4hu-ao6-value-result.v1",
        "stable_evidence_id": stable_id,
        "automatic_pass": automatic_pass,
    }
    ordered = [
        {
            "source_id": "source-a",
            "arm_id": "fixed_ao6_colour_only_t15_c35",
            "output_path": "renders/arm/source-a.png",
            "output_sha256": output_sha,
            "output_size_bytes": 123,
            "output_rgb16_array_sha256": "b" * 64,
            "output_encoded_array_sha256": "c" * 64,
        }
    ]
    pngs = [
        {
            "path": "renders/arm/source-a.png",
            "sha256": output_sha,
            "size_bytes": 123,
        }
    ]
    return {
        "schema": runner.WORKER_REPORT_SCHEMA,
        "scientific_result": result,
        "scientific_stable_evidence_id": stable_id,
        "ordered_output_inventory": ordered,
        "ordered_output_inventory_sha256": "d" * 64,
        "png_inventory": pngs,
        "png_inventory_sha256": "e" * 64,
        "measurement": {
            "worker_wall_seconds": wall,
            "excluded_from_scientific_identity": True,
        },
        "monitor": {
            "peak_process_tree_rss_bytes": rss,
            "wall_seconds": wall,
            "sample_interval_seconds": 0.01,
            "excluded_from_scientific_identity": True,
        },
        "worker_report_sha256": "f" * 64,
    }


def _stub_workers(
    monkeypatch: pytest.MonkeyPatch, reports: list[dict[str, Any]]
) -> list[str]:
    calls: list[str] = []

    def fake_execute_worker(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["run_dir"].name)
        return reports[len(calls) - 1]

    monkeypatch.setattr(runner, "_execute_worker", fake_execute_worker)
    return calls


def test_two_exact_workers_pass_serially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _write_contract(tmp_path / "contract.json")
    reports = [_worker_report(), _worker_report()]
    calls = _stub_workers(monkeypatch, reports)

    result = runner.run(contract, tmp_path / "result")

    assert calls == ["run_a", "run_b"]
    assert result["status"] == "scientific_pass"
    assert result["automatic_pass"] is True
    assert result["blind_review_allowed"] is True
    assert result["decision"] == "PASS_FOR_BLIND_REVIEW_ONLY"
    assert result["fresh_worker_replay"]["exact"] is True


def test_scientific_fail_is_not_infrastructure_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _write_contract(tmp_path / "contract.json")
    _stub_workers(
        monkeypatch,
        [_worker_report(automatic_pass=False), _worker_report(automatic_pass=False)],
    )

    result = runner.run(contract, tmp_path / "result")

    assert result["status"] == "scientific_fail"
    assert result["scientific_result_valid"] is True
    assert result["decision"] == "FAIL_CLOSED_KEEP_AO6"
    assert result["blind_review_allowed"] is False


def test_replay_byte_mismatch_is_infrastructure_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _write_contract(tmp_path / "contract.json")
    _stub_workers(
        monkeypatch,
        [_worker_report(output_sha="a" * 64), _worker_report(output_sha="9" * 64)],
    )

    result = runner.run(contract, tmp_path / "result")

    assert result["status"] == "infrastructure_invalid"
    assert result["scientific_result_valid"] is False
    assert result["decision"] == "INVALID_NO_SCIENTIFIC_DECISION"
    assert result["blind_review_allowed"] is False
    assert "ordered_output_inventory" in result["infrastructure_error"][
        "mismatched_fields"
    ]


def test_resource_failure_is_separate_from_scientific_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _write_contract(tmp_path / "contract.json")
    too_large = runner.EXPECTED_RESOURCE_GATES[
        "maximum_peak_process_tree_rss_bytes"
    ] + 1
    _stub_workers(
        monkeypatch,
        [_worker_report(rss=too_large), _worker_report(rss=too_large)],
    )

    result = runner.run(contract, tmp_path / "result")

    assert result["status"] == "resource_fail"
    assert result["scientific_pass"] is True
    assert result["resource_pass"] is False
    assert result["decision"] == "FAIL_CLOSED_KEEP_AO6"
    assert result["blind_review_allowed"] is False
