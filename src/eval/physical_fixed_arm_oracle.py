"""Hash-bound retrospective Oracle diagnostic for the frozen physical arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "neuro_film.u6_p8bq_fixed_arm_oracle_diagnostic_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p8bq_fixed_arm_oracle_diagnostic.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_parent(
    root: Path,
    parent: dict[str, str],
) -> dict[str, Any]:
    path = root / parent["path"]
    if sha256_file(path) != parent["sha256"]:
        raise ValueError(f"parent hash mismatch: {parent['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"parent must be an object: {parent['path']}")
    return value


def _development_votes(
    scoring: dict[str, Any],
    mapping: dict[str, Any],
    *,
    physical_arm: str,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    rounds = scoring.get("blind_choices")
    if not isinstance(rounds, dict) or set(rounds) != set(mapping):
        raise ValueError("development scoring/mapping rounds drift")
    for round_id, votes in rounds.items():
        round_mapping = mapping[round_id]
        if not isinstance(votes, dict) or set(votes) != set(round_mapping):
            raise ValueError("development scoring/mapping sources drift")
        for source_id, masked_choice in votes.items():
            decoded = round_mapping[source_id][masked_choice]
            if decoded not in {"colour_only", physical_arm}:
                raise ValueError("unsupported development arm")
            counts = result.setdefault(
                source_id,
                {"global": 0, "physical": 0, "censored": 0},
            )
            counts["physical" if decoded == physical_arm else "global"] += 1
    return result


def _fresh_votes(
    adjudication: dict[str, Any],
    *,
    global_arm: str,
    physical_arm: str,
    censor_arm: str,
) -> dict[str, dict[str, int]]:
    rows = adjudication.get("per_scene_vote_counts")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("fresh per-scene votes missing")
    result: dict[str, dict[str, int]] = {}
    required = {global_arm, physical_arm, censor_arm, "tie"}
    for source_id, counts in rows.items():
        if set(counts) != required or any(
            not isinstance(counts[key], int) or counts[key] < 0
            for key in required
        ):
            raise ValueError("fresh vote schema drift")
        if sum(counts.values()) != 3:
            raise ValueError("fresh source does not have three votes")
        result[source_id] = {
            "global": counts[global_arm],
            "physical": counts[physical_arm],
            "censored": counts[censor_arm] + counts["tie"],
        }
    return result


def _oracle_summary(
    votes: dict[str, dict[str, int]],
) -> dict[str, Any]:
    rows = []
    for source_id in sorted(votes):
        counts = votes[source_id]
        selected = (
            "physical"
            if counts["physical"] > counts["global"]
            else "global"
        )
        selected_score = counts[selected]
        rows.append(
            {
                "source_id": source_id,
                "global_votes": counts["global"],
                "physical_votes": counts["physical"],
                "censored_votes": counts["censored"],
                "oracle_selected_arm": selected,
                "oracle_observed_score": selected_score,
            }
        )
    global_score = sum(row["global_votes"] for row in rows)
    physical_score = sum(row["physical_votes"] for row in rows)
    oracle_score = sum(row["oracle_observed_score"] for row in rows)
    total_votes = sum(
        row["global_votes"]
        + row["physical_votes"]
        + row["censored_votes"]
        for row in rows
    )
    censor_count = sum(row["censored_votes"] for row in rows)
    return {
        "source_count": len(rows),
        "total_votes": total_votes,
        "global_observed_score": global_score,
        "physical_observed_score": physical_score,
        "oracle_observed_score": oracle_score,
        "oracle_gain_votes_over_global": oracle_score - global_score,
        "oracle_gain_fraction_of_all_votes": (
            (oracle_score - global_score) / total_votes
        ),
        "physical_selected_source_count": sum(
            row["oracle_selected_arm"] == "physical" for row in rows
        ),
        "censored_vote_count": censor_count,
        "censored_vote_fraction": censor_count / total_votes,
        "rows": rows,
    }


def evaluate(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if config.get("schema") != SCHEMA or config.get("node") != "U6.P8BQ":
        raise ValueError("unsupported U6.P8BQ contract")
    if config.get("forbidden") is None:
        raise ValueError("missing forbidden-action contract")
    parents = {
        key: _load_parent(root, value)
        for key, value in config["parents"].items()
    }
    identity = config["identity"]
    physical = identity["development_arm"]
    bundle_sha = identity["compiled_bundle_sha256"]

    p7f_decision = parents["p7f_decision"]
    p7f_adjudication = parents["p7f_development_adjudication"]
    p8a = parents["p8a_profile_compiler"]
    p8a_decision = parents["p8a_profile_compiler_decision"]
    p8b = parents["p8b_artifact_consumer"]
    p8b_decision = parents["p8b_artifact_consumer_decision"]
    fresh = parents["p8bp_fresh_adjudication"]
    fresh_report = parents["p8bp_fresh_report"]

    if (
        p7f_decision["automatic_result"]["selected_candidate_id"] != physical
        or p7f_adjudication["decision"] != "complete_pass"
        or p7f_adjudication["severe_confirmed_count"] != 0
        or p8a["parent_contract"]
        != "configs/u6_p7f_neutral_gauged_physical_chain_v1.json"
        or p8a_decision["decision"].startswith("reject")
        or p8b["parent_contract"]
        != "configs/u6_p8a_fixed_reference_profile_compiler_v1.json"
        or p8b_decision["bundle_sha256"] != bundle_sha
        or fresh_report["bundle_sha256"] != bundle_sha
        or fresh["confirmed_new_severe_count"] != 0
        or fresh["production_default_changed"]
    ):
        raise ValueError("physical candidate identity or frozen status drift")

    development_votes = _development_votes(
        parents["p7f_development_blind_scoring"],
        parents["p7f_development_private_mapping"],
        physical_arm=physical,
    )
    fresh_votes = _fresh_votes(
        fresh,
        global_arm=identity["fresh_global_arm"],
        physical_arm=identity["fresh_physical_arm"],
        censor_arm=identity["fresh_censor_arm"],
    )
    development = _oracle_summary(development_votes)
    confirmation = _oracle_summary(fresh_votes)

    if (
        development["source_count"] != 9
        or development["total_votes"] != 27
        or confirmation["source_count"] != 9
        or confirmation["total_votes"] != 27
    ):
        raise ValueError("frozen population size drift")

    gates = config["decision_gate"]
    checks = {
        "fresh_observed_gain_votes": (
            confirmation["oracle_gain_votes_over_global"]
            >= gates["minimum_fresh_observed_oracle_gain_votes"]
        ),
        "fresh_observed_gain_fraction": (
            confirmation["oracle_gain_fraction_of_all_votes"]
            >= gates["minimum_fresh_observed_oracle_gain_fraction"]
        ),
        "fresh_selected_source_support": (
            confirmation["physical_selected_source_count"]
            >= gates["minimum_fresh_physical_selected_sources"]
        ),
        "complete_pairwise_observation": (
            confirmation["censored_vote_count"] == 0
            if gates["require_zero_censoring_for_router_open"]
            else True
        ),
    }
    router_evidence_pass = all(checks.values())
    if confirmation["censored_vote_count"] > 0:
        decision = "censored_no_router"
        branch = config["branch_rule"]["censored"]
    elif router_evidence_pass:
        decision = "observed_oracle_gap_only"
        branch = config["branch_rule"]["pass"]
    else:
        decision = "close_frozen_physical_arm_routing"
        branch = config["branch_rule"]["fail"]

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "node": "U6.P8BQ",
        "parent_sha256": {
            key: value["sha256"]
            for key, value in sorted(config["parents"].items())
        },
        "candidate_identity": {
            "development_arm": physical,
            "fresh_arm": identity["fresh_physical_arm"],
            "bundle_sha256": bundle_sha,
            "identity_chain_valid": True,
        },
        "development": development,
        "fresh_confirmation": confirmation,
        "gate_checks": checks,
        "router_evidence_pass": router_evidence_pass,
        "decision": decision,
        "branch": branch,
        "production_default_changed": False,
        "router_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(report)
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
