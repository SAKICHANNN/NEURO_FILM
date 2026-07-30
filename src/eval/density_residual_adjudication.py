"""Adjudicate the frozen AZ0 optical-density versus AO6 blind comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def adjudicate_density_residual_visual(
    *,
    root: Path,
    config: dict[str, Any],
    observations_path: Path,
    build_dir: Path,
) -> dict[str, Any]:
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if observations.get("mapping_unread_when_recorded") is not True:
        raise ValueError("observations were not recorded blind")
    if observations.get("experiment_id") != "U5.R2AZ0V":
        raise ValueError("observation experiment identity drift")

    build_path = build_dir / "build_report.json"
    mapping_path = build_dir / "private_mapping.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if build.get("experiment_id") != config["experiment_id"]:
        raise ValueError("build experiment identity drift")
    if _sha256(mapping_path) != build["private_mapping_sha256"]:
        raise ValueError("private mapping identity drift")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    expected_sheets = {
        int(row["round"]): row["sha256"] for row in build["blind_sheets"]
    }
    rounds = []
    passing_rounds = 0
    for observed in observations["rounds"]:
        round_index = int(observed["round"])
        if expected_sheets.get(round_index) != observed["sheet_sha256"]:
            raise ValueError("blind sheet identity drift")
        round_mapping = mapping.get(f"round_{round_index}")
        choices = observed["choices"]
        if round_mapping is None or set(round_mapping) != set(choices):
            raise ValueError("blind population identity drift")
        decoded = {}
        candidate_preferences = 0
        for source_id, label in choices.items():
            if label not in {"A", "B"}:
                raise ValueError("invalid blind choice")
            arm = round_mapping[source_id][label]
            if arm not in {"candidate", "comparator"}:
                raise ValueError("invalid mapped arm")
            decoded[source_id] = arm
            candidate_preferences += arm == "candidate"
        passed = candidate_preferences >= int(
            config["blind"][
                "minimum_candidate_preferences_per_passing_round"
            ]
        )
        passing_rounds += passed
        rounds.append(
            {
                "round": round_index,
                "candidate_preferences": candidate_preferences,
                "comparator_preferences": len(choices)
                - candidate_preferences,
                "total_pairs": len(choices),
                "pass": bool(passed),
                "decoded_choices": decoded,
            }
        )

    severe_count = int(
        observations["direct_severe_review"]["confirmed_severe_count"]
    )
    blind_pass = (
        passing_rounds >= int(config["blind"]["minimum_passing_rounds"])
        and severe_count
        <= int(config["blind"]["maximum_confirmed_severe_artifacts"])
    )
    core = {
        "schema": "neuro-film.u5-r2az0v-density-residual-adjudication.v1",
        "experiment_id": config["experiment_id"],
        "candidate_id": config["candidate"]["candidate_id"],
        "comparator_id": config["comparator"]["candidate_id"],
        "build_report_sha256": _sha256(build_path),
        "private_mapping_sha256": _sha256(mapping_path),
        "observations_sha256": _sha256(observations_path),
        "rounds": rounds,
        "passing_rounds": passing_rounds,
        "confirmed_severe_artifact_count": severe_count,
        "blind_gate_passed": blind_pass,
        "production_integration_opened": False,
        "branch": (
            "retain_development_champion_and_run_fresh_independent_confirmation"
            if blind_pass
            else "close_density_factorization_without_retuning"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _canonical_sha256(core)}


def write_json(payload: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = ["adjudicate_density_residual_visual", "write_json"]
