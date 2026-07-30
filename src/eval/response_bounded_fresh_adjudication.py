"""Score the frozen U6.P3O blind value and severe review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.fresh_native_standard_confirmation import sha256_file


SCHEMA = "neuro_film.u6_p3o_response_bounded_fresh_adjudication_contract.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P3O adjudication contract")
    return payload


def _load_exact(root: Path, record: dict[str, str]) -> Any:
    path = root / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"evidence hash mismatch: {record['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def score_blind_round(
    *,
    judgments: dict[str, str],
    mapping: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if set(judgments) != set(mapping):
        raise ValueError("blind judgment support mismatch")
    candidate_votes = 0
    parent_votes = 0
    ties = 0
    resolved: dict[str, str] = {}
    for sample_id, choice in judgments.items():
        if choice not in {"A", "B", "tie"}:
            raise ValueError("unsupported blind judgment")
        if set(mapping[sample_id]) != {"A", "B"} or set(
            mapping[sample_id].values()
        ) != {"bounded", "no_spatial"}:
            raise ValueError("invalid blind mapping")
        if choice == "tie":
            ties += 1
            resolved[sample_id] = "tie"
        elif mapping[sample_id][choice] == "bounded":
            candidate_votes += 1
            resolved[sample_id] = "bounded"
        else:
            parent_votes += 1
            resolved[sample_id] = "no_spatial"
    return {
        "candidate_votes": candidate_votes,
        "parent_votes": parent_votes,
        "ties": ties,
        "resolved": resolved,
    }


def adjudicate_response_bounded_fresh_confirmation(
    *,
    root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("production_integration_allowed")
    ):
        raise ValueError("invalid U6.P3O adjudication contract")
    experiment = _load_exact(root, contract["experiment_contract"])
    reports = [
        _load_exact(root, record) for record in contract["formal_runs"]
    ]
    review = _load_exact(root, contract["blind_review"])
    if (
        experiment.get("schema")
        != "neuro_film.u6_p3o_response_bounded_fresh_confirmation_contract.v1"
        or len(reports) != 2
        or reports[0] != reports[1]
        or reports[0].get("stable_evidence_id")
        != "0cd2ada8f836bb8f4ef97e8df5d5030a6296a56c8db8ac8e02c174ab105e14aa"
        or review.get("mapping_reviewed_before_judgment") is True
        or review["review_basis"]["mapping_reviewed_before_judgment"]
    ):
        raise ValueError("P3O formal evidence drift")
    report = reports[0]
    required = contract["required"]
    if bool(report["automatic_pass"]) != bool(required["automatic_pass"]):
        raise ValueError("P3O automatic result drift")
    if (
        required["repeat_report_sha256_exact"]
        and contract["formal_runs"][0]["sha256"]
        != contract["formal_runs"][1]["sha256"]
    ):
        raise ValueError("P3O report replay drift")

    report_rounds = {row["round"]: row for row in report["blind_rounds"]}
    review_rounds = {row["round"]: row for row in review["rounds"]}
    expected_rounds = list(
        range(
            1,
            len(experiment["visual_protocol"]["blind_value_round_seeds"]) + 1,
        )
    )
    if sorted(report_rounds) != expected_rounds or sorted(
        review_rounds
    ) != expected_rounds:
        raise ValueError("P3O blind round support drift")

    scored_rounds = []
    total_severe = int(
        review["direct_severe_review"]["confirmed_severe_artifact_count"]
    )
    passing_rounds = 0
    for round_index in expected_rounds:
        report_round = report_rounds[round_index]
        mapping_record = {
            "path": (
                "outputs/u6_p3o_response_bounded_fresh_confirmation_v1/"
                f"run_a/visual/blind_round_{round_index}_mapping.json"
            ),
            "sha256": report_round["mapping_sha256"],
        }
        mapping_payload = _load_exact(root, mapping_record)
        if (
            mapping_payload["round"] != round_index
            or mapping_payload["seed"] != report_round["seed"]
        ):
            raise ValueError("P3O blind mapping identity drift")
        review_round = review_rounds[round_index]
        scored = score_blind_round(
            judgments=review_round["judgments"],
            mapping=mapping_payload["mapping"],
        )
        scored["round"] = round_index
        scored["passed"] = scored["candidate_votes"] >= int(
            required["minimum_candidate_votes_per_round"]
        )
        passing_rounds += int(scored["passed"])
        total_severe += int(
            review_round["confirmed_severe_artifact_count"]
        )
        scored_rounds.append(scored)

    severe_pass = (
        total_severe == 0
        if required["zero_confirmed_severe_artifact"]
        else True
    )
    value_pass = passing_rounds >= int(required["minimum_passing_rounds"])
    if not report["automatic_pass"]:
        decision = "close_automatic_gate"
    elif not severe_pass:
        decision = "close_photographic_use_severe_failure"
    elif value_pass:
        decision = "retain_physical_product_challenger"
    else:
        decision = "close_product_value_retain_research_primitive"
    core = {
        "schema": "neuro_film.u6_p3o_response_bounded_fresh_adjudication_report.v1",
        "node": contract["node"],
        "formal_report_sha256": contract["formal_runs"][0]["sha256"],
        "formal_stable_evidence_id": report["stable_evidence_id"],
        "blind_review_sha256": contract["blind_review"]["sha256"],
        "automatic_pass": report["automatic_pass"],
        "severe_pass": severe_pass,
        "confirmed_severe_artifact_count": total_severe,
        "rounds": scored_rounds,
        "passing_rounds": passing_rounds,
        "minimum_passing_rounds": int(required["minimum_passing_rounds"]),
        "value_pass": value_pass,
        "decision": decision,
        "production_integration_allowed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        **core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(core)
        ).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "SCHEMA",
    "adjudicate_response_bounded_fresh_confirmation",
    "load_contract",
    "score_blind_round",
    "write_report",
]
