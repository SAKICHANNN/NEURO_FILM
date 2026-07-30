"""Adjudicate frozen complete rankings for the U5.R2BH0 fixed bank."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.global_frontier import sha256_file


SCHEMA = "neuro_film.u5_r2bh0_fixed_bank_oracle_decision.v1"


class FixedBankAdjudicationError(ValueError):
    """Raised when a frozen ranking or mapping identity is incomplete."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _choose_lowest(
    scores: Mapping[str, int],
    tie_order: Sequence[str],
) -> str:
    minimum = min(scores.values())
    tied = {arm for arm, score in scores.items() if score == minimum}
    for arm in tie_order:
        if arm in tied:
            return arm
    raise FixedBankAdjudicationError("tie order does not cover all arms")


def adjudicate_rank_tables(
    *,
    rank_tables: Sequence[Mapping[str, Mapping[str, int]]],
    arms: Sequence[str],
    tie_order: Sequence[str],
    thresholds: Mapping[str, Any],
    confirmed_severe_count: int,
) -> dict[str, Any]:
    """Apply the preregistered leave-one-round-out product-value gate."""

    if len(rank_tables) != 3 or set(arms) != set(tie_order):
        raise FixedBankAdjudicationError("expected three rounds and one tie rank per arm")
    sources = set(rank_tables[0])
    expected_ranks = set(range(1, len(arms) + 1))
    if not sources:
        raise FixedBankAdjudicationError("empty source population")
    for table in rank_tables:
        if set(table) != sources:
            raise FixedBankAdjudicationError("source population drift across rounds")
        for source_scores in table.values():
            if set(source_scores) != set(arms):
                raise FixedBankAdjudicationError("arm population drift")
            if set(source_scores.values()) != expected_ranks:
                raise FixedBankAdjudicationError("ranking is not strict and complete")

    folds: list[dict[str, Any]] = []
    non_global_counts: dict[str, Counter[str]] = {
        source: Counter() for source in sources
    }
    for heldout_index, heldout in enumerate(rank_tables):
        development = [
            table
            for index, table in enumerate(rank_tables)
            if index != heldout_index
        ]
        global_scores = {
            arm: sum(
                table[source][arm]
                for table in development
                for source in sources
            )
            for arm in arms
        }
        global_arm = _choose_lowest(global_scores, tie_order)
        selected: dict[str, str] = {}
        for source in sorted(sources):
            source_scores = {
                arm: sum(table[source][arm] for table in development)
                for arm in arms
            }
            minimum = min(source_scores.values())
            tied = [
                arm for arm in arms if source_scores[arm] == minimum
            ]
            chosen = tied[0] if len(tied) == 1 else global_arm
            selected[source] = chosen
            if chosen != global_arm:
                non_global_counts[source][chosen] += 1

        global_heldout_total = sum(
            heldout[source][global_arm] for source in sources
        )
        selected_heldout_total = sum(
            heldout[source][selected[source]] for source in sources
        )
        folds.append(
            {
                "heldout_round": heldout_index + 1,
                "development_rounds": [
                    index + 1
                    for index in range(len(rank_tables))
                    if index != heldout_index
                ],
                "selected_global_arm": global_arm,
                "development_global_rank_sums": global_scores,
                "selected_source_arms": selected,
                "heldout_global_rank_total": global_heldout_total,
                "heldout_selected_rank_total": selected_heldout_total,
                "heldout_rank_gain": (
                    global_heldout_total - selected_heldout_total
                ),
            }
        )

    aggregate_global = sum(row["heldout_global_rank_total"] for row in folds)
    aggregate_selected = sum(row["heldout_selected_rank_total"] for row in folds)
    aggregate_gain = aggregate_global - aggregate_selected
    relative_gain = aggregate_gain / aggregate_global
    improved_folds = sum(row["heldout_rank_gain"] > 0 for row in folds)
    stable_non_global = {
        source: {
            "arm": counts.most_common(1)[0][0],
            "fold_count": counts.most_common(1)[0][1],
        }
        for source, counts in sorted(non_global_counts.items())
        if counts and counts.most_common(1)[0][1] >= 2
    }

    all_round_global_scores = {
        arm: sum(
            table[source][arm]
            for table in rank_tables
            for source in sources
        )
        for arm in arms
    }
    descriptive_global = _choose_lowest(all_round_global_scores, tie_order)
    descriptive_selected: dict[str, str] = {}
    for source in sorted(sources):
        scores = {
            arm: sum(table[source][arm] for table in rank_tables)
            for arm in arms
        }
        minimum = min(scores.values())
        tied = [arm for arm in arms if scores[arm] == minimum]
        descriptive_selected[source] = (
            tied[0] if len(tied) == 1 else descriptive_global
        )
    descriptive_global_total = sum(
        table[source][descriptive_global]
        for table in rank_tables
        for source in sources
    )
    descriptive_oracle_total = sum(
        table[source][descriptive_selected[source]]
        for table in rank_tables
        for source in sources
    )

    gates = {
        "minimum_improved_leave_one_round_out_folds": {
            "value": improved_folds,
            "threshold": int(
                thresholds["minimum_improved_leave_one_round_out_folds"]
            ),
            "pass": improved_folds
            >= int(thresholds["minimum_improved_leave_one_round_out_folds"]),
        },
        "minimum_aggregate_heldout_rank_gain": {
            "value": aggregate_gain,
            "threshold": int(
                thresholds["minimum_aggregate_heldout_rank_gain"]
            ),
            "pass": aggregate_gain
            >= int(thresholds["minimum_aggregate_heldout_rank_gain"]),
        },
        "minimum_aggregate_heldout_relative_rank_gain": {
            "value": relative_gain,
            "threshold": float(
                thresholds["minimum_aggregate_heldout_relative_rank_gain"]
            ),
            "pass": relative_gain
            >= float(
                thresholds[
                    "minimum_aggregate_heldout_relative_rank_gain"
                ]
            ),
        },
        "minimum_sources_with_stable_non_global_choice": {
            "value": len(stable_non_global),
            "threshold": int(
                thresholds["minimum_sources_with_stable_non_global_choice"]
            ),
            "pass": len(stable_non_global)
            >= int(
                thresholds["minimum_sources_with_stable_non_global_choice"]
            ),
        },
        "maximum_confirmed_severe_artifact_count": {
            "value": int(confirmed_severe_count),
            "threshold": int(
                thresholds["maximum_confirmed_severe_artifact_count"]
            ),
            "pass": int(confirmed_severe_count)
            <= int(thresholds["maximum_confirmed_severe_artifact_count"]),
        },
    }
    return {
        "leave_one_round_out": {
            "folds": folds,
            "improved_folds": improved_folds,
            "aggregate_global_rank_total": aggregate_global,
            "aggregate_selected_rank_total": aggregate_selected,
            "aggregate_rank_gain": aggregate_gain,
            "aggregate_relative_rank_gain": relative_gain,
            "stable_non_global_choices": stable_non_global,
        },
        "descriptive_all_round_oracle": {
            "selected_global_arm": descriptive_global,
            "global_arm_rank_sums": all_round_global_scores,
            "selected_source_arms": descriptive_selected,
            "global_rank_total": descriptive_global_total,
            "oracle_rank_total": descriptive_oracle_total,
            "rank_gain": descriptive_global_total - descriptive_oracle_total,
        },
        "gates": gates,
        "pass": all(row["pass"] for row in gates.values()),
    }


def adjudicate_files(
    *,
    config_path: Path,
    observations_path: Path,
    mapping_receipt_path: Path,
    render_report_path: Path,
    mapping_paths: Sequence[Path],
    adjudicator_software_commit: str,
) -> dict[str, Any]:
    if (
        len(adjudicator_software_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in adjudicator_software_commit
        )
    ):
        raise FixedBankAdjudicationError("invalid adjudicator software commit")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    mapping_receipt = json.loads(
        mapping_receipt_path.read_text(encoding="utf-8")
    )
    render_report = json.loads(render_report_path.read_text(encoding="utf-8"))
    if len(mapping_paths) != 3:
        raise FixedBankAdjudicationError("exactly three mapping files required")
    if (
        sha256_file(render_report_path)
        != observations["render_report_sha256"]
        or render_report["stable_evidence_id"]
        != observations["stable_render_evidence_id"]
        or not render_report["automatic_gate_pass"]
        or not render_report["blind_review_allowed"]
    ):
        raise FixedBankAdjudicationError("render evidence drift or automatic gate failure")

    if (
        mapping_receipt.get("status")
        != "mapping_identities_bound_after_rankings_commit"
        or not mapping_receipt.get("mapping_revealed")
        or mapping_receipt.get("observations_sha256")
        != sha256_file(observations_path)
    ):
        raise FixedBankAdjudicationError("mapping receipt does not bind observations")
    expected_mapping_hashes = mapping_receipt["mapping_sha256_by_round"]
    if [sha256_file(path) for path in mapping_paths] != expected_mapping_hashes:
        raise FixedBankAdjudicationError("blind mapping identity drift")
    arms = [row["arm_id"] for row in config["fixed_arms"]]
    rank_tables: list[dict[str, dict[str, int]]] = []
    observation_rounds = {
        int(row["round"]): row["rankings"]
        for row in observations["rounds"]
    }
    for round_index, mapping_path in enumerate(mapping_paths, start=1):
        mapping_rows = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping = {
            row["source_id"]: {
                label: arm
                for label, arm in row.items()
                if label in {"A", "B", "C", "D", "E"}
            }
            for row in mapping_rows
        }
        rankings = observation_rounds[round_index]
        if set(mapping) != set(rankings):
            raise FixedBankAdjudicationError("mapping and observation sources differ")
        table: dict[str, dict[str, int]] = {}
        for source, labels in rankings.items():
            if set(mapping[source].values()) != set(arms):
                raise FixedBankAdjudicationError("mapping does not cover fixed bank")
            table[source] = {
                mapping[source][label]: rank
                for rank, label in enumerate(labels, start=1)
            }
        rank_tables.append(table)

    result = adjudicate_rank_tables(
        rank_tables=rank_tables,
        arms=arms,
        tie_order=config["ranking_protocol"]["global_tie_break_order"],
        thresholds=config["ranking_protocol"],
        confirmed_severe_count=observations[
            "severe_artifact_review"
        ]["confirmed_severe_count"],
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "adjudicator_software_commit": adjudicator_software_commit,
        "status": (
            "oracle_pass_open_simplest_hard_selector"
            if result["pass"]
            else "oracle_fail_close_selector_and_router"
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "observations_sha256": sha256_file(observations_path),
            "mapping_receipt_sha256": sha256_file(mapping_receipt_path),
            "render_report_sha256": sha256_file(render_report_path),
            "render_stable_evidence_id": render_report[
                "stable_evidence_id"
            ],
            "blind_mapping_sha256": expected_mapping_hashes,
        },
        **result,
        "training_allowed": bool(result["pass"]),
        "training_scope_if_pass": (
            "simplest_hard_content_selector_with_ood_global_fallback"
            if result["pass"]
            else "none"
        ),
        "dense_blending_allowed": False,
        "direct_rgb_prediction_allowed": False,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_payload = dict(payload)
    payload["stable_evidence_id"] = _canonical_sha256(stable_payload)
    return payload
