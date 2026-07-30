"""Adjudicate frozen blind observations for a two-factorization comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.global_frontier import sha256_file


def _load(root: Path, path: str, sha256: str) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != sha256:
        raise ValueError(f"factorization adjudication hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def adjudicate(
    *,
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if (
        config.get("experiment_id")
        != "u5.r2ba1v-hue-value-residual-adjudication-v1"
        or config.get("retuning_allowed")
        or config.get("additional_rounds_allowed")
    ):
        raise ValueError("BA1V adjudication boundary drift")
    visual = _load(
        root,
        config["visual_contract"]["path"],
        config["visual_contract"]["sha256"],
    )
    observations = _load(
        root,
        config["observations"]["path"],
        config["observations"]["sha256"],
    )
    build = _load(
        root, config["build_report"]["path"], config["build_report"]["sha256"]
    )
    mapping = _load(
        root, config["private_mapping"]["path"], config["private_mapping"]["sha256"]
    )
    if (
        observations.get("status")
        != "frozen_before_private_mapping_reveal"
        or observations.get("mapping_read_before_freeze")
        or build["private_mapping_sha256"]
        != config["private_mapping"]["sha256"]
    ):
        raise ValueError("blind freeze or mapping identity drift")

    round_rows = []
    minimum = int(
        visual["blind"]["minimum_candidate_preferences_per_passing_round"]
    )
    for round_name, choices in observations["blind_choices"].items():
        if choices.keys() != mapping[round_name].keys():
            raise ValueError("blind sample coverage drift")
        candidate_preferences = 0
        comparator_preferences = 0
        ties = 0
        for sample_id, choice in choices.items():
            if choice == "TIE":
                ties += 1
            elif mapping[round_name][sample_id][choice] == "candidate":
                candidate_preferences += 1
            else:
                comparator_preferences += 1
        round_rows.append(
            {
                "round": round_name,
                "candidate_preferences": candidate_preferences,
                "comparator_preferences": comparator_preferences,
                "ties": ties,
                "passes": candidate_preferences >= minimum,
            }
        )
    passing_rounds = sum(row["passes"] for row in round_rows)
    severe = int(observations["confirmed_severe_artifact_count"])
    required_rounds = int(visual["blind"]["minimum_passing_rounds"])
    passed = (
        passing_rounds >= required_rounds
        and severe
        <= int(visual["blind"]["maximum_confirmed_severe_artifacts"])
    )
    core = {
        "schema": "neuro-film.u5-r2ba1v-adjudication-report.v1",
        "experiment_id": config["experiment_id"],
        "candidate_id": visual["candidate"]["candidate_id"],
        "comparator_id": visual["comparator"]["candidate_id"],
        "rounds": round_rows,
        "passing_rounds": passing_rounds,
        "required_passing_rounds": required_rounds,
        "confirmed_severe_artifact_count": severe,
        "visual_pass": passed,
        "decision": (
            "retain_hue_value_as_development_champion"
            if passed
            else "close_hue_value_visual_preference_without_rescue"
        ),
        "claim_ceiling": visual["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


__all__ = ["adjudicate"]
