"""Adjudicate frozen CB11 versus AO6 on the CB14 OOD population."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "neuro_film.u5_r2cb14_characteristic_vs_ao6_ood_decision.v1"


class CharacteristicVsAo6OodAdjudicationError(ValueError):
    """Raised when frozen CB14 evidence is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def adjudicate_files(
    *,
    config_path: Path,
    observations_path: Path,
    mapping_receipt_path: Path,
    report_paths: Sequence[Path],
    render_dir: Path,
    adjudicator_software_commit: str,
) -> dict[str, Any]:
    """Validate exact CB14 evidence and apply the frozen blind gates."""

    if len(report_paths) != 2:
        raise CharacteristicVsAo6OodAdjudicationError("two reports are required")
    if len(adjudicator_software_commit) != 40 or any(
        value not in "0123456789abcdef" for value in adjudicator_software_commit
    ):
        raise CharacteristicVsAo6OodAdjudicationError("invalid adjudicator commit")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    receipt = json.loads(mapping_receipt_path.read_text(encoding="utf-8"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    report_hashes = [_sha256_file(path) for path in report_paths]
    if (
        reports[0] != reports[1]
        or len(set(report_hashes)) != 1
        or report_hashes[0] != observations.get("report_sha256")
        or reports[0].get("automatic_pass") is not True
    ):
        raise CharacteristicVsAo6OodAdjudicationError("render replay is not exact")
    if (
        observations.get("status") != "blind_observations_frozen_mapping_unread"
        or observations.get("mapping_files_read") is not False
        or observations.get("sheet_level_confirmed_severe_artifact_count")
        > config["protocol"]["maximum_confirmed_severe_artifact_count"]
    ):
        raise CharacteristicVsAo6OodAdjudicationError("blind boundary is invalid")
    if (
        receipt.get("status") != "mapping_revealed_after_blind_observations_commit"
        or receipt.get("mapping_revealed") is not True
        or receipt.get("observations_sha256") != _sha256_file(observations_path)
        or receipt.get("report_sha256") != report_hashes[0]
        or len(str(receipt.get("observations_commit", ""))) != 40
    ):
        raise CharacteristicVsAo6OodAdjudicationError("mapping receipt is invalid")

    expected_sheet_hashes = {
        Path(row["path"]).name: row["sha256"] for row in reports[0]["blind_sheets"]
    }
    if expected_sheet_hashes != observations.get("sheet_sha256"):
        raise CharacteristicVsAo6OodAdjudicationError("blind sheet identity drift")
    for name, expected in expected_sheet_hashes.items():
        if _sha256_file(render_dir / "blind" / name) != expected:
            raise CharacteristicVsAo6OodAdjudicationError("blind sheet bytes drift")

    report_rows = {row["id"]: row for row in reports[0]["rows"]}
    mappings = reports[0]["sealed_mappings"]
    aggregate: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    round_counts: list[dict[str, int]] = []
    for expected_round, round_payload in enumerate(observations["rounds"], start=1):
        if round_payload.get("round") != expected_round:
            raise CharacteristicVsAo6OodAdjudicationError("blind round order drift")
        round_mapping = mappings[str(expected_round)]
        seen: set[str] = set()
        current: Counter[str] = Counter()
        for choice in round_payload["choices"]:
            source_id = choice["source_id"]
            if source_id in seen or source_id not in report_rows:
                raise CharacteristicVsAo6OodAdjudicationError("source identity drift")
            seen.add(source_id)
            selected = round_mapping[source_id][0 if choice["choice"] == "A" else 1]
            if selected not in {"candidate", "ao6"}:
                raise CharacteristicVsAo6OodAdjudicationError("arm identity drift")
            current[selected] += 1
            aggregate[selected] += 1
            source_counts[source_id][selected] += 1
        if seen != set(report_rows):
            raise CharacteristicVsAo6OodAdjudicationError("incomplete blind round")
        round_counts.append(dict(current))

    thresholds = config["protocol"]
    candidate_round_wins = sum(
        row.get("candidate", 0) > row.get("ao6", 0) for row in round_counts
    )
    candidate_source_majorities = sum(
        row.get("candidate", 0) >= 2 for row in source_counts.values()
    )
    gates = {
        "automatic_boundary_gate": True,
        "sheet_level_severe_artifact_veto": True,
        "minimum_candidate_round_wins": candidate_round_wins
        >= thresholds["minimum_candidate_round_wins"],
        "minimum_candidate_aggregate_choices": aggregate["candidate"]
        >= thresholds["minimum_candidate_aggregate_choices"],
        "minimum_candidate_source_majorities": candidate_source_majorities
        >= thresholds["minimum_candidate_source_majorities"],
    }
    passed = all(gates.values())
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "adjudicator_software_commit": adjudicator_software_commit,
        "status": (
            "candidate_beats_ao6_on_second_ood_population"
            if passed
            else "candidate_fails_ao6_retain_incumbent"
        ),
        "inputs": {
            "config_sha256": _sha256_file(config_path),
            "observations_sha256": _sha256_file(observations_path),
            "mapping_receipt_sha256": _sha256_file(mapping_receipt_path),
            "render_report_sha256": report_hashes[0],
            "render_stable_evidence_id": reports[0]["stable_evidence_id"],
        },
        "measurements": {
            "candidate_round_wins": candidate_round_wins,
            "candidate_aggregate_choices": aggregate["candidate"],
            "aggregate_choice_count": sum(aggregate.values()),
            "candidate_source_majorities": candidate_source_majorities,
            "source_count": len(source_counts),
            "round_counts": round_counts,
        },
        "gates": gates,
        "pass": passed,
        "thresholds_changed": False,
        "additional_rounds_allowed": False,
        "operator_retuning_allowed": False,
        "production_default_changed": False,
        "full_resolution_review_required": passed,
        "next_branch": (
            "complete_missing_gold_face_then_product_review"
            if passed
            else "close_cb11_ood_challenger_retain_ao6"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    payload["stable_evidence_id"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "SCHEMA",
    "CharacteristicVsAo6OodAdjudicationError",
    "adjudicate_files",
]
