"""Adjudicate the frozen BN7 triangular-transport versus AO6 comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.fivek_triangular_logit_transport_visual_adjudication import (
    DIRECT_ARM,
    FiveKTriangularVisualAdjudicationError,
    adjudicate_choices,
)
from src.eval.fivek_triangular_logit_transport_vs_ao6 import validate_contract
from src.eval.global_frontier import sha256_file


SCHEMA = "neuro_film.u5_r2bn7_triangular_logit_transport_vs_ao6_decision.v1"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


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
    """Validate exact evidence and apply the frozen AO6 head-to-head gates."""

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
        baseline_arm=DIRECT_ARM,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "adjudicator_software_commit": adjudicator_software_commit,
        "status": (
            "adaptive_beats_ao6_open_gold_stress_regression"
            if result["pass"]
            else "adaptive_fails_ao6_retain_incumbent"
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
            "freeze_gold_stress_product_safety_regression"
            if result["pass"]
            else "close_triangular_promotion_retain_ao6_and_continue_distinct_algorithm"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    payload["stable_evidence_id"] = _canonical_sha256(payload)
    return payload


__all__ = ["SCHEMA", "adjudicate_files"]
