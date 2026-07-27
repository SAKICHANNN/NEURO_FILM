from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from src.color_match.research.w1_evidence import (
    compute_w1_decision_id,
    inspect_w1_development_evidence,
    load_w1_intake_contract,
    w1_evidence_decision_to_json,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "configs" / "reference_match_w1_development_intake_v1.json"
)


def _gate(passed: bool) -> dict[str, bool]:
    result = {
        "operator_median": passed,
        "operator_p90": passed,
        "operator_families": passed,
        "replicate": passed,
        "beats_identity": passed,
        "beats_global": passed,
        "identity_false_positive": passed,
        "strength": passed,
        "owner_strength_order": passed,
        "owner_same_direction": passed,
        "range": passed,
        "jacobian": passed,
        "norm": passed,
        "inverse": passed,
        "replay": passed,
        "coefficient": passed,
        "content_probe": passed,
        "nuisance_probe": passed,
    }
    result["all_except_repeat"] = passed
    result["method_id_matches"] = True
    return result


def _report(contract: dict, commit: str, branch: str) -> dict:
    gate_results = {
        gate_id: _gate(False)
        for gate_id in contract["report"]["gate_result_ids"]
    }
    if branch == "pending_repeat_unseen_regression_passes":
        gate_results["unseen_single_regression"] = _gate(True)
    elif branch == "pending_repeat_single_fails_multi_passes":
        gate_results["unseen_four_regression"] = _gate(True)
    elif branch == "pending_repeat_seen_retrieval_only_passes":
        gate_results["seen_single_retrieval"] = _gate(True)
    return {
        "schema_version": contract["report"]["schema_version"],
        "experiment_id": contract["report"]["experiment_id"],
        "node": contract["report"]["node"],
        "config_sha256": contract["report"]["config_sha256"],
        "software_commit": commit,
        "reserved_confirmation_seeds_accessed": False,
        "gate_results": gate_results,
        "paired_upper_bound_pass": False,
        "decision_branch_before_repeat": branch,
        "claim_ceiling": contract["report"]["claim_ceiling"],
    }


@pytest.fixture
def pinned_repo(tmp_path: Path) -> tuple[Path, str, dict]:
    contract = load_w1_intake_contract(CONTRACT_PATH)
    repo = tmp_path / "external"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tests"],
        cwd=repo,
        check=True,
    )
    tracked = {}
    for relative_path in contract["source"]["tracked_files"]:
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode("utf-8"))
        tracked[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    contract = json.loads(json.dumps(contract))
    contract["source"]["required_ancestor_commit"] = commit
    contract["source"]["tracked_files"] = tracked
    return repo, commit, contract


def _write_reports(
    tmp_path: Path, report: dict
) -> tuple[Path, Path]:
    payload = (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
    left = tmp_path / "report_a.json"
    right = tmp_path / "report_b.json"
    left.write_bytes(payload)
    right.write_bytes(payload)
    return left, right


def test_missing_reports_keep_identity_default(tmp_path: Path) -> None:
    contract = load_w1_intake_contract(CONTRACT_PATH)
    decision = inspect_w1_development_evidence(
        contract,
        external_repo=tmp_path,
        report_a=tmp_path / "missing-a.json",
        report_b=tmp_path / "missing-b.json",
    )

    assert decision.status == "not-ready"
    assert decision.delivery_algorithm == "identity"
    assert decision.product_integration_open is False


def test_single_reference_development_pass_opens_only_confirmation(
    tmp_path: Path, pinned_repo: tuple[Path, str, dict]
) -> None:
    repo, commit, contract = pinned_repo
    reports = _write_reports(
        tmp_path,
        _report(
            contract,
            commit,
            "pending_repeat_unseen_regression_passes",
        ),
    )
    decision = inspect_w1_development_evidence(
        contract,
        external_repo=repo,
        report_a=reports[0],
        report_b=reports[1],
    )

    assert decision.status == "synthetic-confirmation-required"
    assert decision.single_reference_confirmation_open is True
    assert decision.product_integration_open is False
    assert decision.delivery_algorithm == "identity"


@pytest.mark.parametrize(
    ("branch", "status"),
    [
        (
            "pending_repeat_single_fails_multi_passes",
            "multi-reference-development-only",
        ),
        (
            "pending_repeat_seen_retrieval_only_passes",
            "reference-bank-development-only",
        ),
        ("all_practical_methods_fail", "development-route-closed"),
    ],
)
def test_non_single_arbitrary_reference_branches_fail_closed(
    tmp_path: Path,
    pinned_repo: tuple[Path, str, dict],
    branch: str,
    status: str,
) -> None:
    repo, commit, contract = pinned_repo
    reports = _write_reports(tmp_path, _report(contract, commit, branch))

    decision = inspect_w1_development_evidence(
        contract,
        external_repo=repo,
        report_a=reports[0],
        report_b=reports[1],
    )

    assert decision.status == status
    assert decision.product_integration_open is False
    assert decision.delivery_algorithm == "identity"


def test_mismatched_repeat_or_source_drift_is_rejected(
    tmp_path: Path, pinned_repo: tuple[Path, str, dict]
) -> None:
    repo, commit, contract = pinned_repo
    left, right = _write_reports(
        tmp_path,
        _report(
            contract,
            commit,
            "pending_repeat_unseen_regression_passes",
        ),
    )
    right.write_text("{}\n", encoding="utf-8")
    mismatch = inspect_w1_development_evidence(
        contract,
        external_repo=repo,
        report_a=left,
        report_b=right,
    )
    assert mismatch.status == "rejected-evidence"

    right.write_bytes(left.read_bytes())
    first_path = repo / next(iter(contract["source"]["tracked_files"]))
    first_path.write_text("drift", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "drift"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    drift_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    payload = json.loads(left.read_text(encoding="utf-8"))
    payload["software_commit"] = drift_commit
    left, right = _write_reports(tmp_path, payload)
    drift = inspect_w1_development_evidence(
        contract,
        external_repo=repo,
        report_a=left,
        report_b=right,
    )
    assert drift.status == "rejected-evidence"
    assert any("external-source-file-drift" in item for item in drift.reasons)


def test_incomplete_gate_payload_and_nonfinite_json_fail_closed(
    tmp_path: Path, pinned_repo: tuple[Path, str, dict]
) -> None:
    repo, commit, contract = pinned_repo
    payload = _report(
        contract,
        commit,
        "pending_repeat_unseen_regression_passes",
    )
    del payload["gate_results"]["unseen_single_regression"]["jacobian"]
    left, right = _write_reports(tmp_path, payload)
    incomplete = inspect_w1_development_evidence(
        contract,
        external_repo=repo,
        report_a=left,
        report_b=right,
    )
    assert incomplete.status == "rejected-evidence"
    assert any("invalid-gate-result" in item for item in incomplete.reasons)

    left.write_text('{"value": NaN}\n', encoding="utf-8")
    right.write_bytes(left.read_bytes())
    nonfinite = inspect_w1_development_evidence(
        contract,
        external_repo=repo,
        report_a=left,
        report_b=right,
    )
    assert nonfinite.status == "rejected-evidence"
    assert nonfinite.reasons == (
        "w1-development-report-is-not-valid-json",
    )


def test_decision_identity_detects_tampering(tmp_path: Path) -> None:
    contract = load_w1_intake_contract(CONTRACT_PATH)
    decision = inspect_w1_development_evidence(
        contract,
        external_repo=tmp_path,
        report_a=tmp_path / "a",
        report_b=tmp_path / "b",
    )
    assert w1_evidence_decision_to_json(decision)["decision_id"]

    tampered = replace(decision, status="eligible")
    assert compute_w1_decision_id(tampered) != decision.decision_id
    with pytest.raises(ValueError, match="decision_id"):
        w1_evidence_decision_to_json(tampered)
