"""Fail-closed local tooling for the U5.R2C empirical-ceiling contract.

This module calculates annotation workload and validates panel/manifest/policy
completeness. It does not acquire images, recruit raters, infer labels, select
a model, or turn autonomous reviews into human evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .filmstylesafe import validate_split_manifest


EXPERIMENT_ID = "u5.r2c-fixed-bank-empirical-ceiling-v1"
IDENTITY_ID = "identity"
OUTCOMES = frozenset({"win", "tie", "loss"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EmpiricalCeilingError(ValueError):
    """Raised when an R2C contract or evidence bundle fails closed."""


def load_empirical_ceiling_contract(path: Path) -> dict[str, Any]:
    """Load and validate the frozen U5.R2C config."""

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmpiricalCeilingError(f"cannot read empirical-ceiling config: {exc}") from exc
    if not isinstance(config, dict) or config.get("experiment_id") != EXPERIMENT_ID:
        raise EmpiricalCeilingError("unexpected empirical-ceiling experiment id")
    candidates = config.get("candidate_ids")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) != len(set(candidates))
        or any(not isinstance(value, str) or not value for value in candidates)
    ):
        raise EmpiricalCeilingError("candidate_ids must be a non-empty unique string list")
    if config.get("candidate_budget_k") != len(candidates) or len(candidates) != 7:
        raise EmpiricalCeilingError("frozen candidate budget must equal the exact K=7 bank")
    if config.get("identity_fallback") is not True or config.get("identity_counts_toward_k") is not False:
        raise EmpiricalCeilingError("identity must be a fallback outside K")
    if config.get("global_comparator") not in candidates:
        raise EmpiricalCeilingError("global comparator must be in the frozen bank")
    selection = _mapping(config.get("selection_panel"), "selection_panel")
    evaluation = _mapping(config.get("evaluation_panel"), "evaluation_panel")
    if selection.get("panel_id") == evaluation.get("panel_id"):
        raise EmpiricalCeilingError("selection and evaluation panel ids must differ")
    if evaluation.get("must_be_disjoint_from_selection_panel") is not True:
        raise EmpiricalCeilingError("cross-rater panel isolation must be mandatory")
    if evaluation.get("all_scenes_remain_in_every_denominator") is not True:
        raise EmpiricalCeilingError("all scenes must remain in every denominator")
    endpoint = _mapping(config.get("planning_endpoint"), "planning_endpoint")
    if endpoint.get("binding_estimator") is not None or endpoint.get("binding_sample_size") is not None:
        raise EmpiricalCeilingError("R2C cannot invent a binding estimator or sample size")
    for key in (
        "external_human_recruitment_authorized",
        "pixel_acquisition_authorized",
        "model_training_authorized",
        "operator_fitting_authorized",
        "adaptive_routing_authorized",
        "production_integration_authorized",
    ):
        if config.get(key) is not False:
            raise EmpiricalCeilingError(f"{key} must remain false")
    scenarios = config.get("workload_scenarios_n")
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in scenarios)
        or scenarios != sorted(set(scenarios))
    ):
        raise EmpiricalCeilingError("workload scenarios must be sorted unique positive integers")
    return config


def annotation_workload(
    config: Mapping[str, Any],
    *,
    scene_count: int,
    candidate_severity_escalation_fraction: float = 0.0,
    deployed_severity_escalation_fraction: float = 0.0,
) -> dict[str, Any]:
    """Calculate transparent non-binding judgement counts for one scenario."""

    if not isinstance(scene_count, int) or isinstance(scene_count, bool) or scene_count <= 0:
        raise EmpiricalCeilingError("scene_count must be a positive integer")
    for value in (candidate_severity_escalation_fraction, deployed_severity_escalation_fraction):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
            raise EmpiricalCeilingError("escalation fractions must be in [0, 1]")
    candidates = config["candidate_ids"]
    k = int(config["candidate_budget_k"])
    if k != len(candidates):
        raise EmpiricalCeilingError("candidate budget and bank length differ")
    selection = _mapping(config["selection_panel"], "selection_panel")
    evaluation = _mapping(config["evaluation_panel"], "evaluation_panel")
    initial_candidate_severity = scene_count * k * _positive_int(
        selection, "initial_severity_raters_per_candidate"
    )
    candidate_look = scene_count * k * _positive_int(selection, "look_raters_per_candidate")
    selection_rank = scene_count * _positive_int(selection, "ranking_raters_per_scene")
    evaluation_preference = scene_count * _positive_int(
        evaluation, "preference_raters_per_scene"
    )
    evaluation_severity = scene_count * _positive_int(
        evaluation, "deployed_severity_raters_per_scene"
    )
    evaluation_look = scene_count * _positive_int(
        evaluation, "deployed_look_raters_per_scene"
    )
    candidate_escalation_units = math.ceil(
        scene_count * k * float(candidate_severity_escalation_fraction)
    )
    deployed_escalation_units = math.ceil(
        scene_count * float(deployed_severity_escalation_fraction)
    )
    senior_candidate_severity = candidate_escalation_units * _positive_int(
        selection, "senior_severity_raters_on_escalation"
    )
    senior_deployed_severity = deployed_escalation_units * _positive_int(
        evaluation, "senior_deployed_severity_raters_on_escalation"
    )
    base = (
        initial_candidate_severity
        + candidate_look
        + selection_rank
        + evaluation_preference
        + evaluation_severity
        + evaluation_look
    )
    total = base + senior_candidate_severity + senior_deployed_severity
    return {
        "scene_count": scene_count,
        "candidate_budget_k": k,
        "identity_fallback_outside_k": True,
        "counts": {
            "initial_candidate_severity": initial_candidate_severity,
            "candidate_look": candidate_look,
            "selection_rank": selection_rank,
            "evaluation_preference": evaluation_preference,
            "evaluation_deployed_severity": evaluation_severity,
            "evaluation_deployed_look": evaluation_look,
            "senior_candidate_severity": senior_candidate_severity,
            "senior_deployed_severity": senior_deployed_severity,
            "base_total": base,
            "total_with_escalation": total,
        },
        "candidate_severity_escalation_fraction": float(
            candidate_severity_escalation_fraction
        ),
        "deployed_severity_escalation_fraction": float(
            deployed_severity_escalation_fraction
        ),
        "binding_sample_size": None,
        "status": "planning_workload_only_not_statistical_power",
    }


def validate_panel_isolation(
    selection_reviewer_hashes: Sequence[str],
    evaluation_reviewer_hashes: Sequence[str],
) -> dict[str, Any]:
    """Require non-empty, duplicate-free and disjoint reviewer panels."""

    selection = _reviewer_hashes(selection_reviewer_hashes, "selection")
    evaluation = _reviewer_hashes(evaluation_reviewer_hashes, "evaluation")
    overlap = sorted(set(selection) & set(evaluation))
    if overlap:
        raise EmpiricalCeilingError("selection and evaluation reviewer panels overlap")
    return {
        "selection_reviewer_count": len(selection),
        "evaluation_reviewer_count": len(evaluation),
        "overlap_count": 0,
        "panels_disjoint": True,
    }


def validate_b1_manifest(
    proposed_rows: Iterable[Mapping[str, Any]],
    *,
    existing_rows: Iterable[Mapping[str, Any]],
    forbidden_origins: Sequence[str],
) -> dict[str, Any]:
    """Reject seen evidence and lineage overlap before a proposed B1 manifest."""

    proposed = list(proposed_rows)
    existing = list(existing_rows)
    if not proposed or not existing:
        raise EmpiricalCeilingError("proposed and existing lineage manifests are required")
    forbidden = set(forbidden_origins)
    for row in proposed:
        if row.get("split") != "B1":
            raise EmpiricalCeilingError("every proposed empirical-ceiling row must be B1")
        if row.get("evidence_origin") in forbidden:
            raise EmpiricalCeilingError("seen A0/B0 evidence origin is forbidden in B1")
    for row in existing:
        if row.get("split") not in {"A0", "A1", "B0"}:
            raise EmpiricalCeilingError("existing comparison rows must be A0, A1 or B0")
    try:
        report = validate_split_manifest([*existing, *proposed])
    except ValueError as exc:
        raise EmpiricalCeilingError(str(exc)) from exc
    return {
        "proposed_b1_rows": len(proposed),
        "existing_reference_rows": len(existing),
        "lineage_leaks": 0,
        "forbidden_origin_rows": 0,
        "b1_manifest_eligible": report["hidden_split_eligible"],
    }


def validate_complete_policy(
    expected_scene_ids: Sequence[str],
    selections: Iterable[Mapping[str, Any]],
    *,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """Require exactly one valid selection or justified identity per scene."""

    expected = _unique_nonempty_strings(expected_scene_ids, "expected scene ids")
    candidates = set(_unique_nonempty_strings(candidate_ids, "candidate ids"))
    by_scene: dict[str, Mapping[str, Any]] = {}
    for row in selections:
        if set(row) != {"parent_scene_id", "eligible_candidate_ids", "selected_candidate_id"}:
            raise EmpiricalCeilingError("selection record keys mismatch")
        scene_id = row["parent_scene_id"]
        if scene_id not in expected or scene_id in by_scene:
            raise EmpiricalCeilingError("selection scene is unexpected or duplicated")
        eligible = _unique_strings_allow_empty(
            row["eligible_candidate_ids"], "eligible candidate ids"
        )
        if not set(eligible).issubset(candidates):
            raise EmpiricalCeilingError("selection contains a candidate outside the frozen bank")
        selected = row["selected_candidate_id"]
        if eligible:
            if selected not in eligible:
                raise EmpiricalCeilingError("selected candidate must be eligible")
        elif selected != IDENTITY_ID:
            raise EmpiricalCeilingError("no eligible candidate must fall back to identity")
        by_scene[scene_id] = row
    missing = sorted(set(expected) - set(by_scene))
    if missing:
        raise EmpiricalCeilingError(f"complete policy is missing scenes: {missing[:3]}")
    identity_count = sum(
        row["selected_candidate_id"] == IDENTITY_ID for row in by_scene.values()
    )
    return {
        "scene_count": len(expected),
        "selection_count": len(by_scene),
        "identity_fallback_count": identity_count,
        "all_scenes_in_denominator": True,
        "complete_policy_valid": True,
    }


def all_scene_tie_score(
    expected_scene_ids: Sequence[str],
    outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the descriptive all-scene score without inventing a confidence bound."""

    expected = _unique_nonempty_strings(expected_scene_ids, "expected scene ids")
    by_scene: dict[str, str] = {}
    for row in outcomes:
        if set(row) != {"parent_scene_id", "outcome"}:
            raise EmpiricalCeilingError("outcome record keys mismatch")
        scene_id = row["parent_scene_id"]
        outcome = row["outcome"]
        if scene_id not in expected or scene_id in by_scene or outcome not in OUTCOMES:
            raise EmpiricalCeilingError("outcome scene is unexpected/duplicated or label is invalid")
        by_scene[scene_id] = outcome
    missing = sorted(set(expected) - set(by_scene))
    if missing:
        raise EmpiricalCeilingError(f"tie score is missing scenes: {missing[:3]}")
    wins = sum(value == "win" for value in by_scene.values())
    ties = sum(value == "tie" for value in by_scene.values())
    losses = len(by_scene) - wins - ties
    return {
        "scene_count": len(by_scene),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "all_scene_tie_score": (wins + 0.5 * ties) / len(by_scene),
        "one_sided_lower_bound": None,
        "binding_decision_allowed": False,
        "status": "descriptive_only_until_tie_model_and_cluster_pilot",
    }


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmpiricalCeilingError(f"{context} must be an object")
    return value


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise EmpiricalCeilingError(f"{key} must be a positive integer")
    return result


def _reviewer_hashes(values: Sequence[str], panel: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise EmpiricalCeilingError(f"{panel} panel must be non-empty")
    result = tuple(values)
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in result):
        raise EmpiricalCeilingError(f"{panel} reviewer ids must be lowercase SHA-256")
    if len(result) != len(set(result)):
        raise EmpiricalCeilingError(f"{panel} panel contains duplicate reviewer ids")
    return result


def _unique_nonempty_strings(values: Sequence[str], context: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise EmpiricalCeilingError(f"{context} must be a non-empty sequence")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise EmpiricalCeilingError(f"{context} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise EmpiricalCeilingError(f"{context} must be unique")
    return result


def _unique_strings_allow_empty(values: Any, context: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise EmpiricalCeilingError(f"{context} must be a list")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise EmpiricalCeilingError(f"{context} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise EmpiricalCeilingError(f"{context} must be unique")
    return result
