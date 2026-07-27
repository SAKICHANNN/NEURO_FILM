"""Fail-closed intake for stable external W1 development evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from ..canonical import canonical_sha256


W1_INTAKE_SCHEMA_ID = (
    "neuro-film.reference-match-w1-development-intake.v1"
)
W1_DECISION_SCHEMA_ID = (
    "neuro-film.reference-match-w1-development-decision.v1"
)
_CONTRACT_KEYS = {
    "schema_id",
    "contract_id",
    "source",
    "report",
    "intake_policy",
    "claim_ceiling",
}
_REPORT_CONTRACT_KEYS = {
    "schema_version",
    "experiment_id",
    "node",
    "config_sha256",
    "claim_ceiling",
    "gate_result_ids",
    "allowed_pre_repeat_branches",
}
_INTAKE_POLICY_KEYS = {
    "byte_identical_reports_required",
    "reserved_confirmation_seeds_accessed_required",
    "development_pass_never_opens_product_integration",
    "unseen_single_requires_untouched_confirmation",
    "multi_reference_does_not_satisfy_single_reference_product",
    "seen_retrieval_is_bank_bounded_not_arbitrary_reference",
    "missing_or_invalid_evidence_keeps_identity_default",
}
_REQUIRED_INTAKE_POLICY = {
    "byte_identical_reports_required": True,
    "reserved_confirmation_seeds_accessed_required": False,
    "development_pass_never_opens_product_integration": True,
    "unseen_single_requires_untouched_confirmation": True,
    "multi_reference_does_not_satisfy_single_reference_product": True,
    "seen_retrieval_is_bank_bounded_not_arbitrary_reference": True,
    "missing_or_invalid_evidence_keeps_identity_default": True,
}
_STRUCTURE_KEYS = {
    "range",
    "jacobian",
    "norm",
    "inverse",
    "replay",
    "coefficient",
}
_PROBE_KEYS = {"content_probe", "nuisance_probe"}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DECISION_KEYS = {
    "schema_id",
    "contract_id",
    "status",
    "decision_branch",
    "external_software_commit",
    "report_sha256",
    "single_reference_confirmation_open",
    "multi_reference_development_only",
    "reference_bank_development_only",
    "product_integration_open",
    "delivery_algorithm",
    "reasons",
    "claim_ceiling",
    "decision_id",
}
_DECISION_STATUSES = {
    "not-ready",
    "rejected-evidence",
    "synthetic-confirmation-required",
    "multi-reference-development-only",
    "reference-bank-development-only",
    "development-route-closed",
}


@dataclass(frozen=True)
class W1EvidenceDecision:
    """Canonical result of inspecting two external development reports."""

    schema_id: str
    contract_id: str
    status: str
    decision_branch: str | None
    external_software_commit: str | None
    report_sha256: str | None
    single_reference_confirmation_open: bool
    multi_reference_development_only: bool
    reference_bank_development_only: bool
    product_integration_open: bool
    delivery_algorithm: str
    reasons: tuple[str, ...]
    claim_ceiling: str
    decision_id: str


def _decision_payload(decision: W1EvidenceDecision) -> dict[str, Any]:
    payload = asdict(decision)
    payload.pop("decision_id")
    payload["reasons"] = list(decision.reasons)
    return payload


def compute_w1_decision_id(decision: W1EvidenceDecision) -> str:
    """Return the canonical identity of all adjudication state."""

    return canonical_sha256(_decision_payload(decision))


def _make_decision(
    contract: dict[str, Any],
    *,
    status: str,
    branch: str | None = None,
    commit: str | None = None,
    report_sha256: str | None = None,
    single_confirmation: bool = False,
    multi_only: bool = False,
    bank_only: bool = False,
    reasons: tuple[str, ...],
) -> W1EvidenceDecision:
    provisional = W1EvidenceDecision(
        schema_id=W1_DECISION_SCHEMA_ID,
        contract_id=str(contract["contract_id"]),
        status=status,
        decision_branch=branch,
        external_software_commit=commit,
        report_sha256=report_sha256,
        single_reference_confirmation_open=single_confirmation,
        multi_reference_development_only=multi_only,
        reference_bank_development_only=bank_only,
        product_integration_open=False,
        delivery_algorithm="identity",
        reasons=reasons,
        claim_ceiling=str(contract["claim_ceiling"]),
        decision_id="",
    )
    return W1EvidenceDecision(
        **{
            **asdict(provisional),
            "decision_id": compute_w1_decision_id(provisional),
        }
    )


def load_w1_intake_contract(path: Path) -> dict[str, Any]:
    """Load and validate the pinned external-evidence boundary."""

    contract = json.loads(path.read_text(encoding="utf-8"))
    if set(contract) != _CONTRACT_KEYS:
        raise ValueError("W1 intake contract keys mismatch")
    if contract["schema_id"] != W1_INTAKE_SCHEMA_ID:
        raise ValueError("unsupported W1 intake contract")
    source = contract["source"]
    if set(source) != {
        "repository_role",
        "required_ancestor_commit",
        "tracked_files",
    }:
        raise ValueError("W1 intake source keys mismatch")
    if not _COMMIT_RE.fullmatch(str(source["required_ancestor_commit"])):
        raise ValueError("invalid W1 required ancestor")
    tracked = source["tracked_files"]
    if not isinstance(tracked, dict) or not tracked:
        raise ValueError("W1 tracked file inventory is empty")
    if any(
        not isinstance(path_value, str)
        or not path_value
        or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        for path_value, digest in tracked.items()
    ):
        raise ValueError("invalid W1 tracked file inventory")
    if set(contract["report"]) != _REPORT_CONTRACT_KEYS:
        raise ValueError("W1 intake report keys mismatch")
    if set(contract["intake_policy"]) != _INTAKE_POLICY_KEYS:
        raise ValueError("W1 intake policy keys mismatch")
    if contract["intake_policy"] != _REQUIRED_INTAKE_POLICY:
        raise ValueError("W1 intake policy must keep every fail-close rule")
    return contract


def _git_bytes(repo: Path, commit: str, relative_path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{relative_path}"],
        stderr=subprocess.STDOUT,
    )


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _source_is_pinned(
    contract: dict[str, Any], repo: Path, commit: str
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                str(contract["source"]["required_ancestor_commit"]),
                commit,
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        reasons.append("external-software-commit-is-not-approved-descendant")
        return False, tuple(reasons)
    for relative_path, expected in contract["source"][
        "tracked_files"
    ].items():
        try:
            actual = hashlib.sha256(
                _git_bytes(repo, commit, relative_path)
            ).hexdigest()
        except (OSError, subprocess.CalledProcessError):
            reasons.append(f"external-source-file-unavailable:{relative_path}")
            continue
        if actual != expected:
            reasons.append(f"external-source-file-drift:{relative_path}")
    return not reasons, tuple(reasons)


def _all_except_repeat_is_valid(result: dict[str, Any]) -> bool:
    values = [
        value
        for key, value in result.items()
        if key not in {"all_except_repeat", "method_id_matches"}
    ]
    return (
        bool(values)
        and all(isinstance(value, bool) for value in values)
        and result.get("all_except_repeat") is bool(all(values))
        and result.get("method_id_matches") is True
    )


def _passes_without(result: dict[str, Any], excluded: set[str]) -> bool:
    return all(
        value
        for key, value in result.items()
        if key not in excluded
        and key not in {"all_except_repeat", "method_id_matches"}
    )


def _recompute_pre_repeat_branch(report: dict[str, Any]) -> str:
    results = report["gate_results"]
    primary = tuple(results.values())
    structure_failure = any(
        _passes_without(result, _STRUCTURE_KEYS)
        and not all(result[key] for key in _STRUCTURE_KEYS)
        for result in primary
    )
    probe_failure = any(
        _passes_without(result, _PROBE_KEYS)
        and not all(result[key] for key in _PROBE_KEYS)
        for result in primary
    )
    if results["unseen_single_regression"]["all_except_repeat"]:
        return "pending_repeat_unseen_regression_passes"
    if results["unseen_four_regression"]["all_except_repeat"]:
        return "pending_repeat_single_fails_multi_passes"
    if results["seen_single_retrieval"]["all_except_repeat"]:
        return "pending_repeat_seen_retrieval_only_passes"
    if results["seen_four_retrieval"]["all_except_repeat"]:
        return "pending_repeat_single_fails_multi_passes"
    if structure_failure:
        return "structure_or_repeat_fails"
    if probe_failure:
        return "content_or_nuisance_control_fails"
    if report["paired_upper_bound_pass"] is True:
        return "paired_upper_bound_only_passes"
    return "all_practical_methods_fail"


def _validate_report(
    contract: dict[str, Any], report: dict[str, Any]
) -> tuple[str | None, tuple[str, ...]]:
    expected = contract["report"]
    reasons: list[str] = []
    for key in (
        "schema_version",
        "experiment_id",
        "node",
        "config_sha256",
        "claim_ceiling",
    ):
        if report.get(key) != expected[key]:
            reasons.append(f"report-field-mismatch:{key}")
    if report.get("reserved_confirmation_seeds_accessed") is not False:
        reasons.append("reserved-confirmation-seeds-accessed")
    commit = report.get("software_commit")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        reasons.append("invalid-external-software-commit")
        commit = None
    gate_results = report.get("gate_results")
    expected_gate_ids = set(expected["gate_result_ids"])
    if not isinstance(gate_results, dict) or set(gate_results) != expected_gate_ids:
        reasons.append("gate-result-inventory-mismatch")
    else:
        for gate_id, result in gate_results.items():
            required_gate_keys = _STRUCTURE_KEYS | _PROBE_KEYS
            if (
                not isinstance(result, dict)
                or not required_gate_keys.issubset(result)
                or not _all_except_repeat_is_valid(result)
            ):
                reasons.append(f"invalid-gate-result:{gate_id}")
        if not reasons:
            recomputed = _recompute_pre_repeat_branch(report)
            if report.get("decision_branch_before_repeat") != recomputed:
                reasons.append("decision-branch-does-not-recompute")
    if report.get("decision_branch_before_repeat") not in set(
        expected["allowed_pre_repeat_branches"]
    ):
        reasons.append("unsupported-decision-branch")
    if not isinstance(report.get("paired_upper_bound_pass"), bool):
        reasons.append("invalid-paired-upper-bound-result")
    return commit, tuple(reasons)


def inspect_w1_development_evidence(
    contract: dict[str, Any],
    *,
    external_repo: Path,
    report_a: Path,
    report_b: Path,
) -> W1EvidenceDecision:
    """Adjudicate stable W1 evidence without importing external code."""

    if not report_a.is_file() or not report_b.is_file():
        return _make_decision(
            contract,
            status="not-ready",
            reasons=("repeated-w1-development-reports-absent",),
        )
    bytes_a = report_a.read_bytes()
    bytes_b = report_b.read_bytes()
    if bytes_a != bytes_b:
        return _make_decision(
            contract,
            status="rejected-evidence",
            reasons=("w1-development-reports-not-byte-identical",),
        )
    report_sha256 = hashlib.sha256(bytes_a).hexdigest()
    try:
        report = json.loads(
            bytes_a,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _make_decision(
            contract,
            status="rejected-evidence",
            report_sha256=report_sha256,
            reasons=("w1-development-report-is-not-valid-json",),
        )
    if not isinstance(report, dict):
        return _make_decision(
            contract,
            status="rejected-evidence",
            report_sha256=report_sha256,
            reasons=("w1-development-report-is-not-an-object",),
        )
    commit, report_reasons = _validate_report(contract, report)
    if report_reasons or commit is None:
        return _make_decision(
            contract,
            status="rejected-evidence",
            branch=report.get("decision_branch_before_repeat"),
            commit=commit,
            report_sha256=report_sha256,
            reasons=report_reasons,
        )
    source_ok, source_reasons = _source_is_pinned(
        contract, external_repo, commit
    )
    if not source_ok:
        return _make_decision(
            contract,
            status="rejected-evidence",
            branch=report["decision_branch_before_repeat"],
            commit=commit,
            report_sha256=report_sha256,
            reasons=source_reasons,
        )
    branch = str(report["decision_branch_before_repeat"])
    if branch == "pending_repeat_unseen_regression_passes":
        return _make_decision(
            contract,
            status="synthetic-confirmation-required",
            branch=branch,
            commit=commit,
            report_sha256=report_sha256,
            single_confirmation=True,
            reasons=(
                "single-reference-development-pass-is-not-confirmation",
                "identity-default-retained-until-untouched-confirmation",
            ),
        )
    if branch == "pending_repeat_single_fails_multi_passes":
        return _make_decision(
            contract,
            status="multi-reference-development-only",
            branch=branch,
            commit=commit,
            report_sha256=report_sha256,
            multi_only=True,
            reasons=(
                "single-reference-product-requirement-not-satisfied",
                "identity-default-retained",
            ),
        )
    if branch == "pending_repeat_seen_retrieval_only_passes":
        return _make_decision(
            contract,
            status="reference-bank-development-only",
            branch=branch,
            commit=commit,
            report_sha256=report_sha256,
            bank_only=True,
            reasons=(
                "arbitrary-uploaded-reference-generalization-not-established",
                "identity-default-retained-outside-frozen-bank",
            ),
        )
    return _make_decision(
        contract,
        status="development-route-closed",
        branch=branch,
        commit=commit,
        report_sha256=report_sha256,
        reasons=(branch, "identity-default-retained"),
    )


def w1_evidence_decision_to_json(
    decision: W1EvidenceDecision,
) -> dict[str, Any]:
    """Serialize and revalidate one evidence decision."""

    validate_w1_evidence_decision(decision)
    payload = asdict(decision)
    payload["reasons"] = list(decision.reasons)
    return payload


def validate_w1_evidence_decision(
    decision: W1EvidenceDecision,
) -> None:
    """Reject serialized evidence state that could open delivery by drift."""

    if not isinstance(decision, W1EvidenceDecision):
        raise ValueError("W1 evidence decision type is invalid")
    if decision.schema_id != W1_DECISION_SCHEMA_ID:
        raise ValueError("unsupported W1 evidence decision schema")
    if not isinstance(decision.contract_id, str) or not decision.contract_id:
        raise ValueError("W1 evidence contract_id is empty")
    if (
        not isinstance(decision.status, str)
        or decision.status not in _DECISION_STATUSES
    ):
        raise ValueError("unsupported W1 evidence decision status")
    if (
        decision.decision_branch is not None
        and not isinstance(decision.decision_branch, str)
    ):
        raise ValueError("invalid W1 evidence decision branch")
    if (
        decision.external_software_commit is not None
        and (
            not isinstance(decision.external_software_commit, str)
            or not _COMMIT_RE.fullmatch(decision.external_software_commit)
        )
    ):
        raise ValueError("invalid W1 evidence software commit")
    if (
        decision.report_sha256 is not None
        and (
            not isinstance(decision.report_sha256, str)
            or not _SHA256_RE.fullmatch(decision.report_sha256)
        )
    ):
        raise ValueError("invalid W1 evidence report SHA-256")
    for field_name in (
        "single_reference_confirmation_open",
        "multi_reference_development_only",
        "reference_bank_development_only",
        "product_integration_open",
    ):
        if not isinstance(getattr(decision, field_name), bool):
            raise ValueError(f"W1 evidence {field_name} must be boolean")
    expected_flags = {
        "single_reference_confirmation_open":
            decision.status == "synthetic-confirmation-required",
        "multi_reference_development_only":
            decision.status == "multi-reference-development-only",
        "reference_bank_development_only":
            decision.status == "reference-bank-development-only",
    }
    if any(
        getattr(decision, field_name) is not expected
        for field_name, expected in expected_flags.items()
    ):
        raise ValueError("W1 evidence decision status/flags mismatch")
    if decision.product_integration_open:
        raise ValueError("W1 development evidence cannot open integration")
    if decision.delivery_algorithm != "identity":
        raise ValueError("W1 development evidence must retain identity")
    if (
        not isinstance(decision.reasons, tuple)
        or not decision.reasons
        or any(
            not isinstance(reason, str) or not reason
            for reason in decision.reasons
        )
    ):
        raise ValueError("W1 evidence reasons must be a non-empty tuple")
    if (
        not isinstance(decision.claim_ceiling, str)
        or not decision.claim_ceiling
    ):
        raise ValueError("W1 evidence claim ceiling is empty")
    if decision.status == "not-ready" and any(
        value is not None
        for value in (
            decision.decision_branch,
            decision.external_software_commit,
            decision.report_sha256,
        )
    ):
        raise ValueError("not-ready W1 evidence cannot bind external results")
    if (
        decision.status not in {"not-ready", "rejected-evidence"}
        and (
            decision.decision_branch is None
            or decision.external_software_commit is None
            or decision.report_sha256 is None
        )
    ):
        raise ValueError("completed W1 evidence must bind report provenance")
    if (
        not isinstance(decision.decision_id, str)
        or not _SHA256_RE.fullmatch(decision.decision_id)
        or compute_w1_decision_id(decision) != decision.decision_id
    ):
        raise ValueError("W1 evidence decision_id mismatch")


def w1_evidence_decision_from_dict(
    payload: dict[str, Any],
) -> W1EvidenceDecision:
    """Parse strict language-neutral W1 decision JSON."""

    if not isinstance(payload, dict) or set(payload) != _DECISION_KEYS:
        raise ValueError("W1 evidence decision keys mismatch")
    reasons = payload["reasons"]
    if not isinstance(reasons, list):
        raise ValueError("W1 evidence reasons must be an array")
    try:
        decision = W1EvidenceDecision(
            **{
                **payload,
                "reasons": tuple(reasons),
            }
        )
    except TypeError as exc:
        raise ValueError("W1 evidence decision types are invalid") from exc
    validate_w1_evidence_decision(decision)
    return decision


def w1_evidence_decision_from_json(encoded: str) -> W1EvidenceDecision:
    """Decode one finite JSON decision."""

    if not isinstance(encoded, str):
        raise ValueError("encoded W1 evidence decision must be a string")
    try:
        payload = json.loads(
            encoded,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("encoded W1 evidence decision is invalid") from exc
    return w1_evidence_decision_from_dict(payload)


def load_w1_evidence_decision(path: Path) -> W1EvidenceDecision:
    """Load a bounded committed W1 decision artifact."""

    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("W1 evidence decision file is missing or oversized")
    return w1_evidence_decision_from_json(path.read_text(encoding="utf-8"))


__all__ = [
    "W1_DECISION_SCHEMA_ID",
    "W1_INTAKE_SCHEMA_ID",
    "W1EvidenceDecision",
    "compute_w1_decision_id",
    "inspect_w1_development_evidence",
    "load_w1_evidence_decision",
    "load_w1_intake_contract",
    "validate_w1_evidence_decision",
    "w1_evidence_decision_from_dict",
    "w1_evidence_decision_from_json",
    "w1_evidence_decision_to_json",
]
