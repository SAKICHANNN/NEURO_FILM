"""Hash-bound adjudication for the fresh B0/AO6/native blind comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.fresh_native_standard_confirmation import ARMS


SCHEMA = "neuro_film.u6_p8bp_fixed_arm_adjudication_result.v1"


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


def _stable_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def adjudicate_fresh_native_standard(
    *,
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if config.get("schema") != (
        "neuro_film.u6_p8bp_fixed_arm_adjudication.v1"
    ):
        raise ValueError("unsupported U6.P8BP adjudication schema")

    experiment = _load_bound_json(root, config["experiment"])
    automatic_reports = [
        _load_bound_json(root, binding)
        for binding in config["automatic_reports"]
    ]
    observations = _load_bound_json(root, config["blind_observations"])
    mappings = [
        _load_bound_json(root, binding)
        for binding in config["blind_mappings"]
    ]

    if tuple(experiment["comparison"]["arms"]) != ARMS:
        raise ValueError("unexpected fixed comparison arms")
    if len(automatic_reports) != 2:
        raise ValueError("exactly two automatic runs are required")
    if any(
        not report["automatic_gate_pass"]
        or not report["blind_review_allowed"]
        for report in automatic_reports
    ):
        raise ValueError("automatic gate did not open visual adjudication")
    stable_ids = {
        report["stable_evidence_id"] for report in automatic_reports
    }
    if len(stable_ids) != 1:
        raise ValueError("automatic runs are not stable")
    output_identities = [
        [
            (
                row["source_id"],
                row["arm_id"],
                row["output_sha256"],
            )
            for row in report["rows"]
        ]
        for report in automatic_reports
    ]
    if output_identities[0] != output_identities[1]:
        raise ValueError("automatic output inventories are not exact")
    if not observations["mapping_unread_when_recorded"]:
        raise ValueError("observations were not recorded blind")

    rounds = observations["rounds"]
    if len(rounds) != 3 or len(mappings) != 3:
        raise ValueError("exactly three blind rounds are required")
    expected_ids = {
        row["source_id"] for row in mappings[0]
    }
    if not expected_ids or len(expected_ids) != len(mappings[0]):
        raise ValueError("blind population is empty")
    allow_ties = bool(
        experiment["comparison"]["preference_gate"]["ties_allowed"]
    )
    per_scene = {
        source_id: {arm: 0 for arm in (*ARMS, "tie")}
        for source_id in expected_ids
    }
    arm_total_choices = {arm: 0 for arm in (*ARMS, "tie")}
    pairwise_round_wins = {
        ARMS[1]: 0,
        ARMS[2]: 0,
        "tie": 0,
    }
    round_results: list[dict[str, Any]] = []

    for round_index, (round_observation, mapping) in enumerate(
        zip(rounds, mappings, strict=True),
        start=1,
    ):
        if round_observation["round"] != round_index:
            raise ValueError("round order mismatch")
        by_id = {row["source_id"]: row for row in mapping}
        votes = round_observation["votes"]
        if (
            set(by_id) != expected_ids
            or len(by_id) != len(mapping)
            or set(votes) != expected_ids
            or any(
                {row["A"], row["B"], row["C"]} != set(ARMS)
                for row in mapping
            )
        ):
            raise ValueError("blind row identity mismatch")
        counts = {arm: 0 for arm in (*ARMS, "tie")}
        decoded: dict[str, str] = {}
        for source_id in sorted(expected_ids):
            side = votes[source_id]
            if side == "tie":
                if not allow_ties:
                    raise ValueError("tie is forbidden by the frozen gate")
                arm = "tie"
            else:
                if side not in {"A", "B", "C"}:
                    raise ValueError("vote must be A, B, C, or tie")
                arm = by_id[source_id][side]
                if arm not in ARMS:
                    raise ValueError("mapping contains an unfrozen arm")
            decoded[source_id] = arm
            counts[arm] += 1
            per_scene[source_id][arm] += 1
            arm_total_choices[arm] += 1
        pairwise_winner = (
            ARMS[2]
            if counts[ARMS[2]] > counts[ARMS[1]]
            else ARMS[1]
            if counts[ARMS[1]] > counts[ARMS[2]]
            else "tie"
        )
        pairwise_round_wins[pairwise_winner] += 1
        round_results.append(
            {
                "round": round_index,
                "decoded_votes": decoded,
                "counts": counts,
                "native_vs_ao6_winner": pairwise_winner,
            }
        )

    contact_severe = int(
        observations["contact_sheet_severe_findings"][
            "confirmed_new_severe_count"
        ]
    )
    full_severe = int(
        config["full_resolution_review"][
            "confirmed_new_severe_count"
        ]
    )
    severe_count = contact_severe + full_severe
    preference = experiment["comparison"]["preference_gate"]
    required_round_wins = int(
        preference["minimum_native_standard_round_wins_vs_ao6"]
    )
    pairwise_choices = (
        arm_total_choices[ARMS[1]] + arm_total_choices[ARMS[2]]
    )
    native_share = (
        arm_total_choices[ARMS[2]] / pairwise_choices
        if pairwise_choices
        else 0.0
    )
    required_share = float(
        preference["minimum_native_standard_total_choices_vs_ao6"]
    )
    preference_pass = (
        pairwise_round_wins[ARMS[2]] >= required_round_wins
        and native_share >= required_share
    )
    severe_pass = severe_count == 0
    if not severe_pass:
        decision = "reject_native_standard_severe_failure"
    elif preference_pass:
        decision = "retain_native_standard_as_product_challenger"
    else:
        decision = (
            "keep_ao6_colour_only_and_close_current_physical_"
            "product_challenger"
        )

    stable_payload = {
        "experiment_sha256": config["experiment"]["sha256"],
        "automatic_report_sha256": [
            binding["sha256"]
            for binding in config["automatic_reports"]
        ],
        "automatic_stable_evidence_id": next(iter(stable_ids)),
        "blind_observations_sha256": (
            config["blind_observations"]["sha256"]
        ),
        "blind_mapping_sha256": [
            binding["sha256"] for binding in config["blind_mappings"]
        ],
        "round_results": round_results,
        "arm_total_choices": arm_total_choices,
        "native_vs_ao6_round_wins": pairwise_round_wins,
        "native_share_of_native_or_ao6_choices": native_share,
        "contact_sheet_confirmed_new_severe_count": contact_severe,
        "full_resolution_confirmed_new_severe_count": full_severe,
        "confirmed_new_severe_count": severe_count,
        "decision": decision,
    }
    return {
        "schema": SCHEMA,
        **stable_payload,
        "per_scene_vote_counts": per_scene,
        "required_native_round_wins": required_round_wins,
        "required_native_share": required_share,
        "preference_gate_pass": preference_pass,
        "severe_artifact_gate_pass": severe_pass,
        "automatic_gate_pass": True,
        "production_default_changed": False,
        "stable_evidence_id": _stable_sha256(stable_payload),
        "claim_ceiling": config["claim_ceiling"],
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA",
    "adjudicate_fresh_native_standard",
    "write_report",
]
