"""Adjudicate the frozen U5.R2BH1 two-arm global-policy confirmation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.global_frontier import sha256_file


SCHEMA = "neuro_film.u5_r2bh1_fixed_global_policy_decision.v1"
ARMS = ("fixed_b0", "fixed_ao6_colour_only_t15_c35")


class FixedGlobalPolicyAdjudicationError(ValueError):
    """Raised when a frozen BH1 identity or observation is invalid."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def adjudicate_choices(
    *,
    choices_by_round: Sequence[Mapping[str, str]],
    mappings_by_round: Sequence[Mapping[str, Mapping[str, str]]],
    thresholds: Mapping[str, Any],
    confirmed_severe_count: int,
    automatic_gate_pass: bool,
    repeat_exact: bool,
) -> dict[str, Any]:
    """Apply the preregistered round and aggregate preference gates."""

    if len(choices_by_round) != 3 or len(mappings_by_round) != 3:
        raise FixedGlobalPolicyAdjudicationError("exactly three rounds required")
    sources = set(choices_by_round[0])
    if not sources:
        raise FixedGlobalPolicyAdjudicationError("empty source population")
    round_rows: list[dict[str, Any]] = []
    aggregate: Counter[str] = Counter()
    source_choices: dict[str, list[str]] = {
        source: [] for source in sorted(sources)
    }
    for index, (choices, mapping) in enumerate(
        zip(choices_by_round, mappings_by_round, strict=True), start=1
    ):
        if set(choices) != sources or set(mapping) != sources:
            raise FixedGlobalPolicyAdjudicationError(
                "source population drift across rounds"
            )
        counts: Counter[str] = Counter()
        resolved: dict[str, str] = {}
        for source in sorted(sources):
            if set(mapping[source]) != {"A", "B"} or set(
                mapping[source].values()
            ) != set(ARMS):
                raise FixedGlobalPolicyAdjudicationError(
                    "mapping does not cover the fixed arms"
                )
            label = choices[source]
            if label not in {"A", "B"}:
                raise FixedGlobalPolicyAdjudicationError(
                    "choice must be A or B"
                )
            arm = mapping[source][label]
            resolved[source] = arm
            counts[arm] += 1
            aggregate[arm] += 1
            source_choices[source].append(arm)
        round_rows.append(
            {
                "round": index,
                "counts": dict(counts),
                "winner": (
                    ARMS[0]
                    if counts[ARMS[0]] > counts[ARMS[1]]
                    else ARMS[1]
                    if counts[ARMS[1]] > counts[ARMS[0]]
                    else "tie"
                ),
                "resolved_choices": resolved,
            }
        )

    b0_round_wins = sum(row["winner"] == ARMS[0] for row in round_rows)
    gates = {
        "automatic_gate": {
            "value": bool(automatic_gate_pass),
            "threshold": True,
            "pass": bool(automatic_gate_pass),
        },
        "repeat_exact": {
            "value": bool(repeat_exact),
            "threshold": True,
            "pass": bool(repeat_exact),
        },
        "maximum_confirmed_severe_artifact_count": {
            "value": int(confirmed_severe_count),
            "threshold": int(
                thresholds["maximum_confirmed_severe_artifact_count"]
            ),
            "pass": int(confirmed_severe_count)
            <= int(thresholds["maximum_confirmed_severe_artifact_count"]),
        },
        "minimum_b0_round_wins": {
            "value": b0_round_wins,
            "threshold": int(thresholds["minimum_b0_round_wins"]),
            "pass": b0_round_wins
            >= int(thresholds["minimum_b0_round_wins"]),
        },
        "minimum_b0_aggregate_choices": {
            "value": aggregate[ARMS[0]],
            "threshold": int(thresholds["minimum_b0_aggregate_choices"]),
            "pass": aggregate[ARMS[0]]
            >= int(thresholds["minimum_b0_aggregate_choices"]),
        },
    }
    passed = all(row["pass"] for row in gates.values())
    return {
        "rounds": round_rows,
        "aggregate_counts": dict(aggregate),
        "b0_round_wins": b0_round_wins,
        "per_source_resolved_choices": source_choices,
        "gates": gates,
        "pass": passed,
        "selected_research_global_policy": ARMS[0] if passed else ARMS[1],
    }


def adjudicate_files(
    *,
    config_path: Path,
    observations_path: Path,
    full_resolution_review_path: Path,
    mapping_receipt_path: Path,
    render_report_path: Path,
    mapping_paths: Sequence[Path],
    adjudicator_software_commit: str,
) -> dict[str, Any]:
    if len(adjudicator_software_commit) != 40 or any(
        character not in "0123456789abcdef"
        for character in adjudicator_software_commit
    ):
        raise FixedGlobalPolicyAdjudicationError(
            "invalid adjudicator software commit"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    review = json.loads(
        full_resolution_review_path.read_text(encoding="utf-8")
    )
    receipt = json.loads(mapping_receipt_path.read_text(encoding="utf-8"))
    report = json.loads(render_report_path.read_text(encoding="utf-8"))
    if tuple(row["arm_id"] for row in config["fixed_arms"]) != ARMS:
        raise FixedGlobalPolicyAdjudicationError("fixed-arm drift")
    if (
        sha256_file(render_report_path)
        != observations["render_report_sha256"]
        or report["stable_evidence_id"]
        != observations["stable_render_evidence_id"]
        or not report["automatic_gate_pass"]
        or not report["blind_review_allowed"]
    ):
        raise FixedGlobalPolicyAdjudicationError("render evidence drift")
    if (
        receipt.get("status")
        != "mapping_identities_bound_after_observations_commit"
        or not receipt.get("mapping_revealed")
        or receipt.get("observations_sha256")
        != sha256_file(observations_path)
        or receipt.get("observations_commit")
        != "30534ce5a06ed3cefc9e37dfe774fed632b7de3b"
    ):
        raise FixedGlobalPolicyAdjudicationError(
            "mapping receipt does not bind frozen observations"
        )
    if len(mapping_paths) != 3 or [
        sha256_file(path) for path in mapping_paths
    ] != receipt["mapping_sha256_by_round"]:
        raise FixedGlobalPolicyAdjudicationError("mapping identity drift")
    if (
        review.get("status") != "full_resolution_severe_review_complete"
        or review.get("confirmed_severe_count") != 0
        or review.get("render_report_sha256")
        != sha256_file(render_report_path)
    ):
        raise FixedGlobalPolicyAdjudicationError(
            "full-resolution severe review is not closed"
        )

    observation_rounds = {
        int(row["round"]): row["choices"] for row in observations["rounds"]
    }
    mappings: list[dict[str, dict[str, str]]] = []
    for path in mapping_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        mappings.append(
            {
                str(row["source_id"]): {
                    "A": str(row["A"]),
                    "B": str(row["B"]),
                }
                for row in rows
            }
        )
    severe_count = (
        int(
            observations["blind_sheet_severe_review"][
                "confirmed_severe_count"
            ]
        )
        + int(review["confirmed_severe_count"])
    )
    result = adjudicate_choices(
        choices_by_round=[
            observation_rounds[index] for index in (1, 2, 3)
        ],
        mappings_by_round=mappings,
        thresholds=config["blind_protocol"],
        confirmed_severe_count=severe_count,
        automatic_gate_pass=bool(report["automatic_gate_pass"]),
        repeat_exact=observations["repeat_mismatch_count"] == 0,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "adjudicator_software_commit": adjudicator_software_commit,
        "status": (
            "b0_confirmation_pass_retain_b0_research_champion"
            if result["pass"]
            else "b0_confirmation_fail_retain_ao6_incumbent"
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "observations_sha256": sha256_file(observations_path),
            "full_resolution_review_sha256": sha256_file(
                full_resolution_review_path
            ),
            "mapping_receipt_sha256": sha256_file(mapping_receipt_path),
            "render_report_sha256": sha256_file(render_report_path),
            "render_stable_evidence_id": report["stable_evidence_id"],
            "blind_mapping_sha256": receipt["mapping_sha256_by_round"],
        },
        **result,
        "thresholds_changed": False,
        "additional_rounds_allowed": False,
        "arm_retuning_allowed": False,
        "training_allowed": False,
        "selector_or_router_training_allowed": False,
        "production_default_changed": False,
        "next_branch": (
            "retain_b0_as_research_global_champion"
            if result["pass"]
            else "retain_ao6_and_continue_distinct_algorithm_leaf"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    payload["stable_evidence_id"] = _canonical_sha256(payload)
    return payload
