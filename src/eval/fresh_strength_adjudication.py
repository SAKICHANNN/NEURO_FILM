"""Hash-bound adjudication for a two-strength autonomous blind comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound_json(root: Path, binding: dict[str, str]) -> Any:
    path = root / binding["path"]
    if _sha256(path) != binding["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {binding['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def adjudicate_fresh_strength(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    if config.get("schema") != (
        "neuro_film.u6_p8bn_fresh_strength_adjudication.v1"
    ):
        raise ValueError("unsupported adjudication schema")

    experiment = _load_bound_json(root, config["experiment"])
    automatic = _load_bound_json(root, config["automatic_report"])
    observations = _load_bound_json(root, config["blind_observations"])
    mappings = [
        _load_bound_json(root, binding)
        for binding in config["blind_mappings"]
    ]

    if experiment["comparison"]["strengths"] != [0.8, 1.0]:
        raise ValueError("unexpected frozen strengths")
    if not automatic["automatic_gate_pass"]:
        raise ValueError("automatic gate did not open visual adjudication")
    if not automatic["blind_review_allowed"]:
        raise ValueError("blind review was not allowed")
    if not observations["mapping_unread_when_recorded"]:
        raise ValueError("observations were not recorded blind")

    rounds = observations["rounds"]
    if len(rounds) != len(mappings) or len(rounds) != 3:
        raise ValueError("exactly three blind rounds are required")

    round_results: list[dict[str, Any]] = []
    strength_round_wins = {"0.8": 0, "1.0": 0, "tie": 0}
    per_scene = {
        row["id"]: {"0.8": 0, "1.0": 0}
        for row in mappings[0]
    }
    expected_ids = set(per_scene)

    for round_index, (round_observation, mapping) in enumerate(
        zip(rounds, mappings, strict=True), start=1
    ):
        if round_observation["round"] != round_index:
            raise ValueError("round order mismatch")
        by_id = {row["id"]: row for row in mapping}
        votes = round_observation["votes"]
        if set(by_id) != expected_ids or set(votes) != expected_ids:
            raise ValueError("blind row identity mismatch")
        decoded: dict[str, float] = {}
        counts = {"0.8": 0, "1.0": 0}
        for source_id in sorted(expected_ids):
            side = votes[source_id]
            if side not in {"A", "B"}:
                raise ValueError("vote must be A or B")
            strength = float(by_id[source_id][side])
            if strength not in {0.8, 1.0}:
                raise ValueError("mapping contains an unfrozen strength")
            key = f"{strength:.1f}"
            decoded[source_id] = strength
            counts[key] += 1
            per_scene[source_id][key] += 1
        winner = (
            "0.8"
            if counts["0.8"] > counts["1.0"]
            else "1.0"
            if counts["1.0"] > counts["0.8"]
            else "tie"
        )
        strength_round_wins[winner] += 1
        round_results.append(
            {
                "round": round_index,
                "decoded_votes": decoded,
                "counts": counts,
                "winner": winner,
            }
        )

    contact_sheet_severe_count = int(
        observations["contact_sheet_severe_findings"][
            "confirmed_new_severe_count"
        ]
    )
    full_resolution_severe_count = int(
        config["full_resolution_review"]["confirmed_new_severe_count"]
    )
    severe_count = (
        contact_sheet_severe_count + full_resolution_severe_count
    )
    required = int(
        experiment["comparison"]["preference_gate"][
            "minimum_round_wins_for_0_80"
        ]
    )
    preference_pass = strength_round_wins["0.8"] >= required
    severe_pass = severe_count == 0
    decision = (
        "promote_0.8_as_global_development_challenger"
        if preference_pass and severe_pass
        else "retain_1.0_and_close_0.8_global_preference_challenge"
    )
    stable_payload = {
        "experiment_sha256": config["experiment"]["sha256"],
        "automatic_report_sha256": config["automatic_report"]["sha256"],
        "blind_observations_sha256": config["blind_observations"]["sha256"],
        "blind_mapping_sha256": [
            binding["sha256"] for binding in config["blind_mappings"]
        ],
        "round_results": round_results,
        "strength_round_wins": strength_round_wins,
        "contact_sheet_confirmed_new_severe_count": (
            contact_sheet_severe_count
        ),
        "full_resolution_confirmed_new_severe_count": (
            full_resolution_severe_count
        ),
        "confirmed_new_severe_count": severe_count,
        "decision": decision,
    }
    stable_id = hashlib.sha256(
        json.dumps(
            stable_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "neuro_film.u6_p8bn_fresh_strength_adjudication_result.v1",
        **stable_payload,
        "per_scene_vote_counts": per_scene,
        "required_0.8_round_wins": required,
        "preference_gate_pass": preference_pass,
        "severe_artifact_gate_pass": severe_pass,
        "automatic_gate_pass": True,
        "production_default_changed": False,
        "decision": decision,
        "stable_evidence_id": stable_id,
        "claim_ceiling": config["claim_ceiling"],
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
