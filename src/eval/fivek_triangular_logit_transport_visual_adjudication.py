"""Adjudicate the frozen BN6 triangular-transport visual comparison."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.fivek_triangular_logit_transport_visual_product_value import (
    validate_contract,
)
from src.eval.global_frontier import sha256_file


SCHEMA = "neuro_film.u5_r2bn6_triangular_logit_transport_visual_decision.v1"
GLOBAL_ARM = "global_triangular_transport_then_fixed_ao6"
ADAPTIVE_ARM = "adaptive_triangular_transport_then_fixed_ao6"


class FiveKTriangularVisualAdjudicationError(ValueError):
    """Raised when frozen BN6 evidence or provenance is invalid."""


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
    automatic_pass: bool,
    repeat_exact: bool,
) -> dict[str, Any]:
    """Apply the preregistered round, aggregate, and source-majority gates."""

    if len(choices_by_round) != 3 or len(mappings_by_round) != 3:
        raise FiveKTriangularVisualAdjudicationError(
            "exactly three blind rounds are required"
        )
    sources = set(choices_by_round[0])
    if not sources:
        raise FiveKTriangularVisualAdjudicationError("empty source set")
    aggregate: Counter[str] = Counter()
    source_adaptive_choices: Counter[str] = Counter()
    round_rows: list[dict[str, Any]] = []
    for round_index, (choices, mapping) in enumerate(
        zip(choices_by_round, mappings_by_round, strict=True), start=1
    ):
        if set(choices) != sources or set(mapping) != sources:
            raise FiveKTriangularVisualAdjudicationError(
                "source population drift across blind rounds"
            )
        counts: Counter[str] = Counter()
        resolved: dict[str, str] = {}
        for source_id in sorted(sources):
            labels = mapping[source_id]
            if set(labels) != {"A", "B"} or set(labels.values()) != {
                GLOBAL_ARM,
                ADAPTIVE_ARM,
            }:
                raise FiveKTriangularVisualAdjudicationError(
                    "mapping does not cover the exact BN6 arms"
                )
            choice = choices[source_id]
            if choice not in {"A", "B"}:
                raise FiveKTriangularVisualAdjudicationError(
                    "blind choice must be A or B"
                )
            arm = labels[choice]
            resolved[source_id] = arm
            counts[arm] += 1
            aggregate[arm] += 1
            if arm == ADAPTIVE_ARM:
                source_adaptive_choices[source_id] += 1
        winner = (
            ADAPTIVE_ARM
            if counts[ADAPTIVE_ARM] > counts[GLOBAL_ARM]
            else GLOBAL_ARM
            if counts[GLOBAL_ARM] > counts[ADAPTIVE_ARM]
            else "tie"
        )
        round_rows.append(
            {
                "round": round_index,
                "counts": dict(counts),
                "winner": winner,
                "resolved_choices": resolved,
            }
        )

    adaptive_round_wins = sum(
        row["winner"] == ADAPTIVE_ARM for row in round_rows
    )
    adaptive_source_majorities = sum(
        source_adaptive_choices[source_id] >= 2 for source_id in sources
    )
    gates = {
        "automatic_gate": {
            "value": bool(automatic_pass),
            "threshold": True,
            "pass": bool(automatic_pass),
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
        "minimum_adaptive_round_wins": {
            "value": adaptive_round_wins,
            "threshold": int(thresholds["minimum_adaptive_round_wins"]),
            "pass": adaptive_round_wins
            >= int(thresholds["minimum_adaptive_round_wins"]),
        },
        "minimum_adaptive_aggregate_choices": {
            "value": aggregate[ADAPTIVE_ARM],
            "threshold": int(thresholds["minimum_adaptive_aggregate_choices"]),
            "pass": aggregate[ADAPTIVE_ARM]
            >= int(thresholds["minimum_adaptive_aggregate_choices"]),
        },
        "minimum_adaptive_source_majorities": {
            "value": adaptive_source_majorities,
            "threshold": int(thresholds["minimum_adaptive_source_majorities"]),
            "pass": adaptive_source_majorities
            >= int(thresholds["minimum_adaptive_source_majorities"]),
        },
    }
    return {
        "rounds": round_rows,
        "aggregate_counts": dict(aggregate),
        "adaptive_round_wins": adaptive_round_wins,
        "adaptive_source_majorities": adaptive_source_majorities,
        "per_source_adaptive_choices": {
            source_id: source_adaptive_choices[source_id]
            for source_id in sorted(sources)
        },
        "gates": gates,
        "pass": all(gate["pass"] for gate in gates.values()),
    }


def adjudicate_files(
    *,
    root: Path,
    config_path: Path,
    observations_path: Path,
    full_resolution_review_path: Path,
    mapping_receipt_path: Path,
    render_report_paths: Sequence[Path],
    mapping_paths: Sequence[Path],
    adjudicator_software_commit: str,
) -> dict[str, Any]:
    """Validate frozen inputs, reveal exact mappings, and issue one decision."""

    if len(adjudicator_software_commit) != 40 or any(
        character not in "0123456789abcdef"
        for character in adjudicator_software_commit
    ):
        raise FiveKTriangularVisualAdjudicationError(
            "invalid adjudicator software commit"
        )
    if len(render_report_paths) != 2 or len(mapping_paths) != 3:
        raise FiveKTriangularVisualAdjudicationError(
            "two reports and three mappings are required"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_contract(root, config)
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    review = json.loads(full_resolution_review_path.read_text(encoding="utf-8"))
    receipt = json.loads(mapping_receipt_path.read_text(encoding="utf-8"))
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in render_report_paths
    ]
    report_hashes = [sha256_file(path) for path in render_report_paths]
    if (
        len(set(report_hashes)) != 1
        or reports[0] != reports[1]
        or report_hashes[0] != observations.get("report_sha256")
        or not reports[0].get("automatic_pass")
        or not reports[0].get("blind_review_allowed")
    ):
        raise FiveKTriangularVisualAdjudicationError(
            "render evidence is not exact and eligible"
        )
    if (
        observations.get("status")
        != "blind_observations_frozen_mapping_unread"
        or observations.get("mapping_files_read") is not False
        or observations.get("sheet_level_confirmed_severe_artifact_count") != 0
    ):
        raise FiveKTriangularVisualAdjudicationError(
            "blind observation boundary is invalid"
        )
    mapping_hashes = [sha256_file(path) for path in mapping_paths]
    if (
        receipt.get("status")
        != "mapping_identities_bound_after_observations_commit"
        or receipt.get("mapping_revealed") is not True
        or receipt.get("observations_sha256") != sha256_file(observations_path)
        or len(str(receipt.get("observations_commit", ""))) != 40
        or mapping_hashes != receipt.get("mapping_sha256_by_round")
    ):
        raise FiveKTriangularVisualAdjudicationError(
            "mapping receipt does not bind the frozen observations"
        )
    if (
        review.get("status") != "full_resolution_severe_review_complete"
        or review.get("render_report_sha256") != report_hashes[0]
        or int(review.get("confirmed_severe_count", -1)) < 0
    ):
        raise FiveKTriangularVisualAdjudicationError(
            "full-resolution severe review is not closed"
        )

    report_rows = {row["source_id"]: row for row in reports[0]["rows"]}
    for row in review["reviewed_outputs"]:
        try:
            expected = report_rows[row["source_id"]]["arms"][row["arm"]][
                "output_sha256"
            ]
        except KeyError as exc:
            raise FiveKTriangularVisualAdjudicationError(
                "reviewed output is outside the report"
            ) from exc
        if row["sha256"] != expected:
            raise FiveKTriangularVisualAdjudicationError(
                "reviewed output identity drift"
            )

    choices_by_round: list[dict[str, str]] = []
    for expected_round, round_payload in enumerate(observations["rounds"], start=1):
        if int(round_payload.get("round", -1)) != expected_round:
            raise FiveKTriangularVisualAdjudicationError("blind round order drift")
        choices = {
            row["source_id"]: row["choice"] for row in round_payload["choices"]
        }
        if len(choices) != len(round_payload["choices"]):
            raise FiveKTriangularVisualAdjudicationError(
                "duplicate source in blind observations"
            )
        choices_by_round.append(choices)
    mappings_by_round: list[dict[str, dict[str, str]]] = []
    for path in mapping_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = {
            row["source_id"]: {"A": row["A"], "B": row["B"]}
            for row in payload["rows"]
        }
        if len(mapping) != len(payload["rows"]):
            raise FiveKTriangularVisualAdjudicationError(
                "duplicate source in blind mapping"
            )
        mappings_by_round.append(mapping)

    result = adjudicate_choices(
        choices_by_round=choices_by_round,
        mappings_by_round=mappings_by_round,
        thresholds=config["blind_protocol"],
        confirmed_severe_count=(
            int(observations["sheet_level_confirmed_severe_artifact_count"])
            + int(review["confirmed_severe_count"])
        ),
        automatic_pass=bool(reports[0]["automatic_pass"]),
        repeat_exact=True,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "adjudicator_software_commit": adjudicator_software_commit,
        "status": (
            "adaptive_visual_value_pass_research_challenger"
            if result["pass"]
            else "adaptive_visual_value_fail_retain_ao6_incumbent"
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "observations_sha256": sha256_file(observations_path),
            "full_resolution_review_sha256": sha256_file(
                full_resolution_review_path
            ),
            "mapping_receipt_sha256": sha256_file(mapping_receipt_path),
            "render_report_sha256": report_hashes[0],
            "render_stable_evidence_id": reports[0]["stable_evidence_id"],
            "blind_mapping_sha256": mapping_hashes,
        },
        **result,
        "thresholds_changed": False,
        "additional_rounds_allowed": False,
        "operator_retuning_allowed": False,
        "selector_or_router_training_allowed": False,
        "production_default_changed": False,
        "next_branch": (
            "compare_adaptive_triangular_transport_against_ao6_incumbent"
            if result["pass"]
            else "close_bn6_visual_promotion_and_continue_distinct_explicit_or_physical_algorithm"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    payload["stable_evidence_id"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "ADAPTIVE_ARM",
    "GLOBAL_ARM",
    "FiveKTriangularVisualAdjudicationError",
    "adjudicate_choices",
    "adjudicate_files",
]
