"""Audit historical blind-choice connectivity before any preference learning.

The audit deliberately collapses repeated blind rounds into one effective
source-by-experiment unit.  It measures whether the retained evidence can
support a future group-held evaluator pilot without pretending that repeated
views of the same source are independent labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "neuro_film.u5_r2bn8_historical_blind_preference_feasibility_report.v1"


class HistoricalPreferenceFeasibilityError(ValueError):
    """Raised when pinned blind evidence is malformed or drifts."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HistoricalPreferenceFeasibilityError(f"expected object: {path}")
    return payload


def _severe_count(payload: Mapping[str, Any]) -> int:
    for key in ("confirmed_severe_artifact_count",):
        if key in payload:
            return int(payload[key])
    visual = payload.get("visual")
    if isinstance(visual, Mapping) and "confirmed_severe_artifact_count" in visual:
        return int(visual["confirmed_severe_artifact_count"])
    gates = payload.get("gates")
    if isinstance(gates, Mapping):
        gate = gates.get("maximum_confirmed_severe_artifact_count")
        if isinstance(gate, Mapping) and "value" in gate:
            return int(gate["value"])
    raise HistoricalPreferenceFeasibilityError(
        f"missing severe-artifact count for {payload.get('experiment_id')}"
    )


def _append_round(
    rows: list[dict[str, Any]],
    *,
    experiment_id: str,
    round_index: int,
    selected_by_source: Mapping[str, str],
    arms: set[str],
) -> None:
    if len(arms) < 2:
        raise HistoricalPreferenceFeasibilityError(
            f"{experiment_id} has fewer than two candidate arms: {sorted(arms)}"
        )
    for source_id, selected_arm in selected_by_source.items():
        if selected_arm not in arms:
            raise HistoricalPreferenceFeasibilityError(
                f"unknown selected arm in {experiment_id}: {selected_arm}"
            )
        rows.append(
            {
                "experiment_id": experiment_id,
                "round": round_index,
                "source_id": str(source_id),
                "selected_arm": str(selected_arm),
                "arms": sorted(arms),
            }
        )


def _extract_rows(payload: Mapping[str, Any], parser: str) -> list[dict[str, Any]]:
    experiment_id = str(payload.get("experiment_id", ""))
    if not experiment_id:
        raise HistoricalPreferenceFeasibilityError("missing experiment_id")
    rows: list[dict[str, Any]] = []

    if parser == "resolved_rounds":
        rounds = payload.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise HistoricalPreferenceFeasibilityError("missing resolved rounds")
        all_arms = {
            str(arm)
            for round_payload in rounds
            for arm in round_payload.get("counts", {}).keys()
        }
        for round_payload in rounds:
            selected = round_payload.get("resolved_choices")
            if not isinstance(selected, Mapping):
                raise HistoricalPreferenceFeasibilityError("missing resolved choices")
            _append_round(
                rows,
                experiment_id=experiment_id,
                round_index=int(round_payload["round"]),
                selected_by_source={str(k): str(v) for k, v in selected.items()},
                arms=all_arms,
            )
        return rows

    if parser == "decoded_rounds":
        rounds = payload.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise HistoricalPreferenceFeasibilityError("missing decoded rounds")
        all_arms = {
            str(row["selected_arm"])
            for round_payload in rounds
            for row in round_payload.get("decoded", [])
        }
        for round_payload in rounds:
            decoded = round_payload.get("decoded")
            if not isinstance(decoded, list):
                raise HistoricalPreferenceFeasibilityError("missing decoded rows")
            selected = {str(row["source_id"]): str(row["selected_arm"]) for row in decoded}
            if len(selected) != len(decoded):
                raise HistoricalPreferenceFeasibilityError("duplicate decoded source")
            _append_round(
                rows,
                experiment_id=experiment_id,
                round_index=int(round_payload["round"]),
                selected_by_source=selected,
                arms=all_arms,
            )
        return rows

    if parser == "decoded_choices":
        decoded = payload.get("decoded_choices")
        if not isinstance(decoded, list) or not decoded:
            raise HistoricalPreferenceFeasibilityError("missing decoded choices")
        selected = {str(row["source_id"]): str(row["selected_arm"]) for row in decoded}
        if len(selected) != len(decoded):
            raise HistoricalPreferenceFeasibilityError("duplicate decoded source")
        arms = {str(row["selected_arm"]) for row in decoded}
        candidate = payload.get("candidate")
        reference = payload.get("reference")
        if candidate is not None:
            arms.add(str(candidate))
        if reference is not None:
            arms.add(str(reference))
        _append_round(
            rows,
            experiment_id=experiment_id,
            round_index=1,
            selected_by_source=selected,
            arms=arms,
        )
        return rows

    if parser == "nested_decoded_rounds":
        rounds = payload.get("decoded_rounds")
        if not isinstance(rounds, list) or not rounds:
            raise HistoricalPreferenceFeasibilityError("missing nested decoded rounds")
        all_arms = {
            str(row["selected_arm"])
            for round_payload in rounds
            for row in round_payload.get("decoded_choices", [])
        }
        for round_payload in rounds:
            decoded = round_payload.get("decoded_choices")
            if not isinstance(decoded, list):
                raise HistoricalPreferenceFeasibilityError("missing nested decoded choices")
            selected = {str(row["source_id"]): str(row["selected_arm"]) for row in decoded}
            if len(selected) != len(decoded):
                raise HistoricalPreferenceFeasibilityError("duplicate decoded source")
            _append_round(
                rows,
                experiment_id=experiment_id,
                round_index=int(round_payload["round"]),
                selected_by_source=selected,
                arms=all_arms,
            )
        return rows

    if parser == "decoded_vote_rounds":
        rounds = payload.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise HistoricalPreferenceFeasibilityError("missing decoded vote rounds")
        all_arms = {
            str(arm)
            for round_payload in rounds
            for arm in round_payload.get("counts", {}).keys()
            if arm != "tie"
        }
        for round_payload in rounds:
            decoded = round_payload.get("decoded_votes")
            if not isinstance(decoded, Mapping):
                raise HistoricalPreferenceFeasibilityError("missing decoded votes")
            _append_round(
                rows,
                experiment_id=experiment_id,
                round_index=int(round_payload["round"]),
                selected_by_source={str(k): str(v) for k, v in decoded.items()},
                arms=all_arms,
            )
        return rows

    raise HistoricalPreferenceFeasibilityError(f"unsupported parser: {parser}")


def _largest_component_share(edges: Iterable[tuple[str, str]]) -> float:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    if not adjacency:
        return 0.0
    unseen = set(adjacency)
    largest = 0
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        largest = max(largest, size)
    return largest / len(adjacency)


def audit_files(*, root: Path, config_path: Path) -> dict[str, Any]:
    """Return a deterministic feasibility report from exact pinned decisions."""

    config = _read_json(config_path)
    evidence_rows: list[dict[str, Any]] = []
    severe_total = 0
    evidence_identities: list[dict[str, str]] = []
    experiment_arms: dict[str, list[str]] = {}

    aliases = {str(k): str(v) for k, v in config.get("arm_aliases", {}).items()}
    for entry in config["evidence_inputs"]:
        path = root / entry["path"]
        actual_sha = _sha256_file(path)
        if actual_sha != entry["sha256"]:
            raise HistoricalPreferenceFeasibilityError(
                f"evidence hash drift: {entry['path']}"
            )
        payload = _read_json(path)
        rows = _extract_rows(payload, str(entry["parser"]))
        for row in rows:
            row["selected_arm"] = aliases.get(row["selected_arm"], row["selected_arm"])
            row["arms"] = sorted({aliases.get(arm, arm) for arm in row["arms"]})
        experiment_id = str(payload["experiment_id"])
        arms = sorted({arm for row in rows for arm in row["arms"]})
        if len(arms) < 2:
            raise HistoricalPreferenceFeasibilityError(
                f"single-arm experiment after extraction: {experiment_id}"
            )
        experiment_arms[experiment_id] = arms
        evidence_rows.extend(rows)
        severe_total += _severe_count(payload)
        evidence_identities.append(
            {"path": str(entry["path"]), "sha256": actual_sha}
        )

    by_unit: dict[tuple[str, str], list[str]] = defaultdict(list)
    sources_by_experiment: dict[str, set[str]] = defaultdict(set)
    for row in evidence_rows:
        key = (row["experiment_id"], row["source_id"])
        by_unit[key].append(row["selected_arm"])
        sources_by_experiment[row["experiment_id"]].add(row["source_id"])

    units: list[dict[str, Any]] = []
    unassigned = 0
    for (experiment_id, source_id), votes in sorted(by_unit.items()):
        counts = Counter(votes)
        ordered = counts.most_common()
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            majority = None
            unassigned += 1
        else:
            majority = ordered[0][0]
        units.append(
            {
                "experiment_id": experiment_id,
                "source_id": source_id,
                "arms": experiment_arms[experiment_id],
                "vote_counts": dict(sorted(counts.items())),
                "vote_count": len(votes),
                "majority_arm": majority,
                "agreement": ordered[0][1] / len(votes),
            }
        )

    cohort_members: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for experiment_id, source_ids in sources_by_experiment.items():
        cohort_members[tuple(sorted(source_ids))].append(experiment_id)
    repeated_experiments = {
        experiment_id
        for experiment_ids in cohort_members.values()
        if len(experiment_ids) >= 2
        for experiment_id in experiment_ids
    }
    repeated_units = sum(
        unit["experiment_id"] in repeated_experiments for unit in units
    )

    assigned_units = [unit for unit in units if unit["majority_arm"] is not None]
    global_correct = 0
    experiment_global: dict[str, dict[str, Any]] = {}
    for experiment_id in sorted(sources_by_experiment):
        experiment_units = [
            unit for unit in assigned_units if unit["experiment_id"] == experiment_id
        ]
        counts = Counter(unit["majority_arm"] for unit in experiment_units)
        winner, correct = counts.most_common(1)[0]
        global_correct += correct
        experiment_global[experiment_id] = {
            "selected_arm": winner,
            "correct_units": correct,
            "unit_count": len(experiment_units),
            "accuracy": correct / len(experiment_units),
        }
    baseline_accuracy = global_correct / len(assigned_units)
    oracle_gain = 1.0 - baseline_accuracy

    arm_edges = [
        (left, right)
        for arms in experiment_arms.values()
        for index, left in enumerate(arms)
        for right in arms[index + 1 :]
    ]
    unique_sources = sorted({unit["source_id"] for unit in units})
    measurements = {
        "experiment_count": len(experiment_arms),
        "unique_source_count": len(unique_sources),
        "effective_source_experiment_units": len(units),
        "raw_blind_votes": len(evidence_rows),
        "minimum_sources_per_experiment": min(map(len, sources_by_experiment.values())),
        "candidate_arm_count": len({arm for edge in arm_edges for arm in edge}),
        "candidate_graph_largest_component_share": _largest_component_share(arm_edges),
        "source_cohort_count": len(cohort_members),
        "repeated_source_cohort_count": sum(
            len(experiment_ids) >= 2 for experiment_ids in cohort_members.values()
        ),
        "repeated_source_cohort_units": repeated_units,
        "units_in_repeated_source_cohorts_share": repeated_units / len(units),
        "mean_within_unit_vote_agreement": sum(unit["agreement"] for unit in units)
        / len(units),
        "experiment_global_accuracy": baseline_accuracy,
        "oracle_gain_over_experiment_global": oracle_gain,
        "unassigned_units": unassigned,
        "confirmed_severe_artifact_count": severe_total,
    }
    thresholds = config["gates"]
    gates = {
        "minimum_experiment_count": measurements["experiment_count"]
        >= thresholds["minimum_experiment_count"],
        "minimum_unique_source_count": measurements["unique_source_count"]
        >= thresholds["minimum_unique_source_count"],
        "minimum_effective_source_experiment_units": measurements[
            "effective_source_experiment_units"
        ]
        >= thresholds["minimum_effective_source_experiment_units"],
        "minimum_raw_blind_votes": measurements["raw_blind_votes"]
        >= thresholds["minimum_raw_blind_votes"],
        "minimum_sources_per_experiment": measurements[
            "minimum_sources_per_experiment"
        ]
        >= thresholds["minimum_sources_per_experiment"],
        "minimum_candidate_graph_largest_component_share": measurements[
            "candidate_graph_largest_component_share"
        ]
        >= thresholds["minimum_candidate_graph_largest_component_share"],
        "minimum_source_cohort_count": measurements["source_cohort_count"]
        >= thresholds["minimum_source_cohort_count"],
        "minimum_repeated_source_cohort_units": measurements[
            "repeated_source_cohort_units"
        ]
        >= thresholds["minimum_repeated_source_cohort_units"],
        "minimum_mean_within_unit_vote_agreement": measurements[
            "mean_within_unit_vote_agreement"
        ]
        >= thresholds["minimum_mean_within_unit_vote_agreement"],
        "minimum_oracle_gain_over_experiment_global": measurements[
            "oracle_gain_over_experiment_global"
        ]
        >= thresholds["minimum_oracle_gain_over_experiment_global"],
        "maximum_unassigned_units": measurements["unassigned_units"]
        <= thresholds["maximum_unassigned_units"],
        "maximum_confirmed_severe_artifact_count": measurements[
            "confirmed_severe_artifact_count"
        ]
        <= thresholds["maximum_confirmed_severe_artifact_count"],
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": (
            "feasible_open_preregistered_evaluator_pilot"
            if passed
            else "insufficient_close_preference_learning"
        ),
        "pass": passed,
        "inputs": {
            "config_sha256": _sha256_file(config_path),
            "evidence": evidence_identities,
        },
        "measurements": measurements,
        "thresholds": thresholds,
        "gates": gates,
        "experiment_arms": experiment_arms,
        "experiment_global_baselines": experiment_global,
        "source_cohorts": [
            {
                "source_count": len(source_ids),
                "experiment_ids": sorted(experiment_ids),
                "source_ids": list(source_ids),
            }
            for source_ids, experiment_ids in sorted(cohort_members.items())
        ],
        "effective_units": units,
        "allowed_if_pass": config["allowed_if_pass"],
        "forbidden_even_if_pass": config["forbidden_even_if_pass"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(report)
    return report


__all__ = [
    "HistoricalPreferenceFeasibilityError",
    "SCHEMA",
    "audit_files",
]
