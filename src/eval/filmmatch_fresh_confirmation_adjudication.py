"""Resolve BL8 preference and BL9 non-basic evidence without refitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CANDIDATE = "fixed_bl5_strict_interior_sigmoid"
REFERENCE = "fixed_ao6_colour_only_t15_c35"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adjudicate_fresh_confirmation(
    *,
    config: dict[str, Any],
    observations_path: Path,
    run_a: Path,
    run_b: Path,
    nonbasic_a_path: Path,
    nonbasic_b_path: Path,
) -> dict[str, Any]:
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if observations.get("revealed_mapping_at_capture") is not False:
        raise ValueError("blind observations were not sealed before reveal")
    severe_count = int(observations.get("severe_artifact_observations", -1))

    report_a_path = run_a / "report.json"
    report_b_path = run_b / "report.json"
    if report_a_path.read_bytes() != report_b_path.read_bytes():
        raise ValueError("two-run BL8 report identity failed")
    report = json.loads(report_a_path.read_text(encoding="utf-8"))
    if not report.get("automatic_gate_pass"):
        raise ValueError("BL8 automatic gate did not open visual adjudication")

    expected_sources = list(observations["source_order"])
    expected_pairs = [
        (source_id, arm)
        for source_id in expected_sources
        for arm in (REFERENCE, CANDIDATE)
    ]
    rows = {(row["source_id"], row["arm_id"]): row for row in report["rows"]}
    if set(rows) != set(expected_pairs):
        raise ValueError("BL8 report population or arms drift")
    output_hashes: dict[str, dict[str, str]] = {}
    for source_id, arm in expected_pairs:
        row = rows[(source_id, arm)]
        relative = Path(row["output"])
        expected = row["output_sha256"]
        if _sha256(run_a / relative) != expected or _sha256(run_b / relative) != expected:
            raise ValueError(f"render identity drift: {source_id}/{arm}")
        output_hashes.setdefault(source_id, {})[arm] = expected

    candidate_choices = 0
    candidate_round_wins = 0
    per_source_candidate_choices = dict.fromkeys(expected_sources, 0)
    decoded_rounds = []
    mapping_hashes = []
    for round_row in observations["rounds"]:
        round_index = int(round_row["round"])
        mapping_name = f"blind_round_{round_index}_mapping.json"
        mapping_a_path = run_a / "blind" / mapping_name
        mapping_b_path = run_b / "blind" / mapping_name
        if mapping_a_path.read_bytes() != mapping_b_path.read_bytes():
            raise ValueError("two-run mapping identity failed")
        mapping = json.loads(mapping_a_path.read_text(encoding="utf-8"))
        if [row["source_id"] for row in mapping] != expected_sources:
            raise ValueError("blind mapping population or order drift")
        choices = list(round_row["choices"])
        if len(choices) != len(mapping) or not set(choices) <= {"A", "B"}:
            raise ValueError("invalid blind choice inventory")
        for part_index, expected_hash in enumerate(round_row["sheet_sha256"], start=1):
            name = f"blind_round_{round_index}_part_{part_index}.png"
            if _sha256(run_a / "blind" / name) != expected_hash:
                raise ValueError("blind sheet identity drift")
            if _sha256(run_b / "blind" / name) != expected_hash:
                raise ValueError("two-run blind sheet identity failed")
        decoded = []
        candidate_count = 0
        for row, choice in zip(mapping, choices, strict=True):
            selected = row[choice]
            if selected not in {CANDIDATE, REFERENCE}:
                raise ValueError("unexpected blind arm")
            if selected == CANDIDATE:
                candidate_count += 1
                per_source_candidate_choices[row["source_id"]] += 1
            decoded.append(
                {"source_id": row["source_id"], "anonymous_choice": choice, "selected_arm": selected}
            )
        candidate_choices += candidate_count
        round_won = candidate_count > len(mapping) - candidate_count
        candidate_round_wins += int(round_won)
        mapping_hashes.append(_sha256(mapping_a_path))
        decoded_rounds.append(
            {
                "round": round_index,
                "candidate_choices": candidate_count,
                "reference_choices": len(mapping) - candidate_count,
                "candidate_won_round": round_won,
                "decoded": decoded,
            }
        )

    if nonbasic_a_path.read_bytes() != nonbasic_b_path.read_bytes():
        raise ValueError("two-run BL9 report identity failed")
    nonbasic = json.loads(nonbasic_a_path.read_text(encoding="utf-8"))
    if not nonbasic.get("automatic_gate_pass"):
        raise ValueError("BL9 non-basic audit did not pass")

    protocol = config["blind_protocol"]
    preference_gates = {
        "automatic_gate": True,
        "two_run_identity": True,
        "severe_artifact_veto": severe_count
        <= int(protocol["maximum_confirmed_severe_artifact_count"]),
        "minimum_candidate_round_wins": candidate_round_wins
        >= int(protocol["minimum_candidate_round_wins"]),
        "minimum_candidate_aggregate_choices": candidate_choices
        >= int(protocol["minimum_candidate_aggregate_choices"]),
    }
    preference_passed = all(preference_gates.values())
    core = {
        "schema": "neuro_film.u5_r2bl8_filmmatch_fresh_confirmation_decision.v1",
        "experiment_id": config["experiment_id"],
        "report_sha256": _sha256(report_a_path),
        "report_stable_evidence_id": report["stable_evidence_id"],
        "blind_observations_sha256": _sha256(observations_path),
        "blind_mapping_sha256": mapping_hashes,
        "nonbasic_report_sha256": _sha256(nonbasic_a_path),
        "nonbasic_stable_evidence_id": nonbasic["stable_evidence_id"],
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "candidate_round_wins": candidate_round_wins,
        "candidate_aggregate_choices": candidate_choices,
        "aggregate_choice_denominator": int(protocol["aggregate_choice_denominator"]),
        "candidate_source_majority_wins": sum(
            value >= 2 for value in per_source_candidate_choices.values()
        ),
        "per_source_candidate_choices": per_source_candidate_choices,
        "confirmed_severe_artifact_count": severe_count,
        "preference_gates": preference_gates,
        "preference_passed": preference_passed,
        "nonbasic_passed": True,
        "decision": (
            "retain_bl5_as_fresh_confirmed_challenger"
            if preference_passed
            else "retain_ao6_incumbent_close_bl5_preference_promotion"
        ),
        "interpretation": (
            "BL5 is materially non-basic on fresh images but loses the frozen preference gate; do not rescue by strength or capacity."
        ),
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
        "output_sha256": output_hashes,
        "rounds": decoded_rounds,
    }
    core["stable_evidence_id"] = _canonical_sha256(core)
    return core


__all__ = ["adjudicate_fresh_confirmation"]
