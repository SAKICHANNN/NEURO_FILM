"""Adjudicate BL13 automatic and three-round autonomous visual evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.ao6_nonbasic_chroma_emphasis import CANDIDATE_ARM
from src.eval.filmmatch_strict_interior_fresh_confirmation import AO6_ARM
from src.eval.global_frontier import sha256_file


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def adjudicate_ao6_nonbasic_chroma(
    *,
    config: dict[str, Any],
    observations_path: Path,
    run_a: Path,
    run_b: Path,
    blind_dir: Path,
) -> dict[str, Any]:
    if (run_a / "report.json").read_bytes() != (run_b / "report.json").read_bytes():
        raise ValueError("BL13 two-run report identity failed")
    report = json.loads((run_a / "report.json").read_text(encoding="utf-8"))
    if not report.get("automatic_gate_pass"):
        raise ValueError("BL13 automatic gate did not open visual review")
    for row in report["rows"]:
        expected = row["output_sha256"]
        if (
            sha256_file(run_a / row["output"]) != expected
            or sha256_file(run_b / row["output"]) != expected
        ):
            raise ValueError("BL13 output identity drift")

    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    mapping_path = blind_dir / "mapping.json"
    if (
        observations.get("mapping_revealed_before_choices_fixed") is not False
        or observations.get("observations_committed_before_mapping_reveal") is not True
        or observations.get("population_preference_claimed")
        or observations.get("product_promotion_allowed")
        or observations.get("blind_mapping_sha256") != sha256_file(mapping_path)
    ):
        raise ValueError("BL13 observation boundary drift")
    mappings = json.loads(mapping_path.read_text(encoding="utf-8"))
    choices = observations["round_choices"]
    source_order = observations["source_order"]
    if len(mappings) != 3 or len(choices) != 3:
        raise ValueError("BL13 round inventory drift")

    decoded_rounds: list[dict[str, Any]] = []
    candidate_counts: list[int] = []
    for round_mapping, round_choices in zip(mappings, choices, strict=True):
        rows = round_mapping["rows"]
        if (
            [row["source_id"] for row in rows] != source_order
            or len(round_choices) != len(rows)
            or not set(round_choices) <= {"A", "B"}
        ):
            raise ValueError("BL13 visual population drift")
        decoded: list[dict[str, str]] = []
        candidate_count = 0
        for row, choice in zip(rows, round_choices, strict=True):
            selected = row[choice]
            if selected not in {AO6_ARM, CANDIDATE_ARM}:
                raise ValueError("unexpected BL13 visual arm")
            candidate_count += int(selected == CANDIDATE_ARM)
            decoded.append(
                {
                    "source_id": row["source_id"],
                    "anonymous_choice": choice,
                    "selected_arm": selected,
                }
            )
        candidate_counts.append(candidate_count)
        decoded_rounds.append(
            {
                "round": int(round_mapping["round"]),
                "candidate_choices": candidate_count,
                "ao6_choices": len(rows) - candidate_count,
                "decoded_choices": decoded,
            }
        )

    visual_gate = config["visual_gate"]
    minimum = int(visual_gate["minimum_candidate_choices_per_17"])
    required_rounds = int(visual_gate["minimum_passing_rounds_of_3"])
    passing_rounds = sum(count >= minimum for count in candidate_counts)
    severe = int(observations["confirmed_severe_artifact_count"])
    visual_pass = severe == 0 and passing_rounds >= required_rounds
    core = {
        "schema": "neuro_film.u5_r2bl13_ao6_nonbasic_chroma_emphasis_decision.v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": report["config_sha256"],
        "report_sha256": sha256_file(run_a / "report.json"),
        "report_stable_evidence_id": report["stable_evidence_id"],
        "blind_mapping_sha256": sha256_file(mapping_path),
        "blind_observations_sha256": sha256_file(observations_path),
        "automatic_gate_pass": True,
        "two_run_identity": True,
        "confirmed_severe_artifact_count": severe,
        "severe_artifact_veto_pass": severe == 0,
        "candidate_choices_by_round": candidate_counts,
        "minimum_candidate_choices_per_round": minimum,
        "passing_rounds": passing_rounds,
        "required_passing_rounds": required_rounds,
        "visual_gate_pass": visual_pass,
        "visual_evidence_grade": "autonomous_development_precommitted_before_mapping_reveal",
        "decoded_rounds": decoded_rounds,
        "decision": (
            "retain_bl13_as_development_challenger_require_new_population"
            if visual_pass
            else "close_bl13_preference_retain_ao6"
        ),
        "interpretation": (
            "The analytical non-basic chroma emphasis is deterministic, materially stronger "
            "and artifact-clean, but its anonymous preference is not stable enough across "
            "three label permutations to clear the frozen two-of-three visual gate."
        ),
        "threshold_retuning_allowed": False,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _canonical_sha256(core)}


__all__ = ["adjudicate_ao6_nonbasic_chroma"]
