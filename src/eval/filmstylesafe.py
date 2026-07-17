"""Fail-closed FilmStyleSafe R1A annotation and split-contract primitives.

This module validates metadata and aggregates protocol-adjudicated labels. It
does not inspect pixels, infer artifacts, recruit raters, or turn autonomous
vision reviews into human evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


SCHEMA_ID = "kmcfm.filmstylesafe-annotation.v1"
CONTRACT_ID = "kmcfm.filmstylesafe-r1a.v1"
SPLITS = frozenset({"A0", "A1", "B0", "B1", "B2", "B3", "B4"})
SEVERITIES = frozenset({"safe", "minor", "severe", "uncertain"})
HUMAN_REVIEWER_KINDS = frozenset({"human_initial", "human_senior"})
REVIEWER_KINDS = HUMAN_REVIEWER_KINDS | {"autonomous_vlm_secondary"}
LOOK_ADHERENCE = frozenset({"pass", "fail", "uncertain"})
KEEP_PREFERENCES = frozenset({"left", "tie", "right"})
CHROMATIC_CATEGORIES = frozenset(
    {
        "large_smooth_region_cast",
        "skin_or_neutral_contamination",
        "neon_chroma_island_or_highlight_speckle",
        "banding_posterization_or_colour_shelf",
        "clipping_or_gamut_hard_boundary",
        "local_hue_jump_grid_seam_or_halo",
        "benign_perturbation_chromatic_instability",
    }
)
CURRENT_A0_ONLY_ORIGINS = frozenset(
    {
        "u41_provisional_gold",
        "u42_owner_anchor_replays",
        "rf2_c0_spektrafilm_outputs",
        "known_id11_red_speckle",
    }
)
GROUP_KEYS = (
    "parent_scene_id",
    "source_id",
    "uploader_or_creator_group",
    "camera_group",
    "roll_group",
    "exact_hash",
    "perceptual_hash",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class FilmStyleSafeContractError(ValueError):
    """Raised when an R1A record or split contract fails closed."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise FilmStyleSafeContractError(f"{context} keys mismatch; missing={missing}, extra={extra}")


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise FilmStyleSafeContractError(f"{context} must be a bounded identifier")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FilmStyleSafeContractError(f"{context} must be lowercase SHA-256")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FilmStyleSafeContractError(f"cannot read contract {path}: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("contract_id") != CONTRACT_ID:
        raise FilmStyleSafeContractError("unexpected FilmStyleSafe contract id")
    if frozenset(contract.get("severity_labels", [])) != SEVERITIES:
        raise FilmStyleSafeContractError("contract severity labels drifted from implementation")
    if frozenset(contract.get("chromatic_categories", [])) != CHROMATIC_CATEGORIES:
        raise FilmStyleSafeContractError("contract chromatic ontology drifted from implementation")
    if set(contract.get("splits", {})) != SPLITS:
        raise FilmStyleSafeContractError("contract split set drifted from implementation")
    if contract.get("preregistration_planning_values", {}).get("binding_sample_size") is not None:
        raise FilmStyleSafeContractError("R1A cannot invent a binding sample size before pilots")
    return contract


def validate_annotation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable severity or deployed-output style rating."""

    if not isinstance(record, Mapping):
        raise FilmStyleSafeContractError("annotation must be an object")
    expected = {
        "schema_id", "record_id", "record_kind", "split", "evidence_origin",
        "parent_scene_id", "candidate_id", "input_sha256", "output_sha256",
        "transform_family", "failure_family", "reviewer", "viewing",
        "severity", "style",
    }
    _exact_keys(record, expected, "annotation")
    if record["schema_id"] != SCHEMA_ID:
        raise FilmStyleSafeContractError("unexpected annotation schema id")
    _identifier(record["record_id"], "record_id")
    _identifier(record["parent_scene_id"], "parent_scene_id")
    _identifier(record["candidate_id"], "candidate_id")
    _identifier(record["transform_family"], "transform_family")
    _identifier(record["failure_family"], "failure_family")
    _sha256(record["input_sha256"], "input_sha256")
    _sha256(record["output_sha256"], "output_sha256")
    if record["split"] not in SPLITS:
        raise FilmStyleSafeContractError("annotation split is invalid")
    if record["evidence_origin"] in CURRENT_A0_ONLY_ORIGINS and record["split"] != "A0":
        raise FilmStyleSafeContractError("seen project evidence is irrevocably A0-only")

    reviewer = record["reviewer"]
    if not isinstance(reviewer, Mapping):
        raise FilmStyleSafeContractError("reviewer must be an object")
    _exact_keys(
        reviewer,
        {"reviewer_id_hash", "reviewer_kind", "panel_id", "blind", "qualification_version", "primary_evidence"},
        "reviewer",
    )
    _sha256(reviewer["reviewer_id_hash"], "reviewer_id_hash")
    _identifier(reviewer["panel_id"], "panel_id")
    _identifier(reviewer["qualification_version"], "qualification_version")
    if reviewer["reviewer_kind"] not in REVIEWER_KINDS or reviewer["blind"] is not True:
        raise FilmStyleSafeContractError("reviewer kind is invalid or review is not blind")
    if not isinstance(reviewer["primary_evidence"], bool):
        raise FilmStyleSafeContractError("primary_evidence must be boolean")
    if reviewer["reviewer_kind"] == "autonomous_vlm_secondary" and reviewer["primary_evidence"]:
        raise FilmStyleSafeContractError("autonomous VLM evidence cannot be primary human evidence")
    if reviewer["reviewer_kind"] in HUMAN_REVIEWER_KINDS and not reviewer["primary_evidence"]:
        raise FilmStyleSafeContractError("human protocol ratings must be marked primary evidence")

    viewing = record["viewing"]
    if not isinstance(viewing, Mapping):
        raise FilmStyleSafeContractError("viewing must be an object")
    _exact_keys(
        viewing,
        {"full_resolution", "zoom_percent", "input_visible", "effects_disabled", "display_protocol_id", "colour_state"},
        "viewing",
    )
    if viewing["full_resolution"] is not True or viewing["input_visible"] is not True:
        raise FilmStyleSafeContractError("severity/style protocol requires full resolution and visible input")
    if viewing["effects_disabled"] is not True:
        raise FilmStyleSafeContractError("colour study requires optical effects disabled")
    if not isinstance(viewing["zoom_percent"], int) or viewing["zoom_percent"] < 100:
        raise FilmStyleSafeContractError("viewing zoom must include at least 100 percent")
    _identifier(viewing["display_protocol_id"], "display_protocol_id")
    if viewing["colour_state"] not in {"display_srgb", "declared_display", "unknown_look_approximation"}:
        raise FilmStyleSafeContractError("viewing colour state is invalid")

    kind = record["record_kind"]
    if kind == "severity_rating":
        if record["style"] is not None:
            raise FilmStyleSafeContractError("severity rating cannot contain a style rating")
        _validate_severity(record["severity"])
        if reviewer["reviewer_kind"] == "human_senior" and record["severity"]["label"] == "uncertain":
            # Allowed as a rating, but aggregation will conservatively fail it.
            pass
    elif kind == "style_rating":
        if record["severity"] is not None:
            raise FilmStyleSafeContractError("style rating cannot contain a severity rating")
        _validate_style(record["style"])
    else:
        raise FilmStyleSafeContractError("record_kind must be severity_rating or style_rating")
    return dict(record)


def _validate_severity(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise FilmStyleSafeContractError("severity must be an object")
    _exact_keys(value, {"label", "categories", "regions", "notes"}, "severity")
    if value["label"] not in SEVERITIES:
        raise FilmStyleSafeContractError("severity label is invalid")
    if not isinstance(value["categories"], list) or len(value["categories"]) != len(set(value["categories"])):
        raise FilmStyleSafeContractError("severity categories must be a unique list")
    if not set(value["categories"]).issubset(CHROMATIC_CATEGORIES):
        raise FilmStyleSafeContractError("severity contains an unknown chromatic category")
    if not isinstance(value["regions"], list):
        raise FilmStyleSafeContractError("severity regions must be a list")
    for region in value["regions"]:
        if not isinstance(region, Mapping):
            raise FilmStyleSafeContractError("region must be an object")
        _exact_keys(region, {"region_id", "kind", "artifact_category", "asset_path", "asset_sha256"}, "region")
        _identifier(region["region_id"], "region_id")
        if region["kind"] not in {"mask", "polygon", "whole_frame"}:
            raise FilmStyleSafeContractError("region kind is invalid")
        if region["artifact_category"] not in CHROMATIC_CATEGORIES:
            raise FilmStyleSafeContractError("region artifact category is invalid")
        if not isinstance(region["asset_path"], str) or not region["asset_path"]:
            raise FilmStyleSafeContractError("region asset path is required")
        _sha256(region["asset_sha256"], "region asset_sha256")
    if value["label"] == "severe" and (not value["categories"] or not value["regions"]):
        raise FilmStyleSafeContractError("severe ratings require category and region evidence")
    if not isinstance(value["notes"], str):
        raise FilmStyleSafeContractError("severity notes must be text")


def _validate_style(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise FilmStyleSafeContractError("style must be an object")
    _exact_keys(
        value,
        {"look_id", "reference_board_sha256", "film_origin_traceable", "look_adherence", "style_strength", "keep_preference", "comparator_id"},
        "style",
    )
    _identifier(value["look_id"], "look_id")
    _sha256(value["reference_board_sha256"], "reference_board_sha256")
    if not isinstance(value["film_origin_traceable"], bool):
        raise FilmStyleSafeContractError("film_origin_traceable must be boolean")
    if value["look_adherence"] not in LOOK_ADHERENCE:
        raise FilmStyleSafeContractError("look_adherence is invalid")
    if value["style_strength"] not in {1, 2, 3, 4, 5}:
        raise FilmStyleSafeContractError("style_strength must be 1 through 5")
    if value["keep_preference"] not in KEEP_PREFERENCES:
        raise FilmStyleSafeContractError("keep_preference is invalid")
    _identifier(value["comparator_id"], "comparator_id")
    if not value["film_origin_traceable"] and value["look_id"].startswith("film_stock:"):
        raise FilmStyleSafeContractError("untraceable reference boards cannot name a film stock")


def aggregate_scene_severity(records: Iterable[Mapping[str, Any]], *, parent_scene_id: str, candidate_id: str) -> dict[str, Any]:
    """Apply the frozen 3+3 human escalation rule to one scene/candidate."""

    validated = [validate_annotation(record) for record in records]
    ratings = [
        record for record in validated
        if record["record_kind"] == "severity_rating"
        and record["parent_scene_id"] == parent_scene_id
        and record["candidate_id"] == candidate_id
        and record["reviewer"]["reviewer_kind"] in HUMAN_REVIEWER_KINDS
    ]
    identities = [record["reviewer"]["reviewer_id_hash"] for record in ratings]
    if len(identities) != len(set(identities)):
        raise FilmStyleSafeContractError("duplicate human reviewer for scene/candidate")
    initial = [record for record in ratings if record["reviewer"]["reviewer_kind"] == "human_initial"]
    senior = [record for record in ratings if record["reviewer"]["reviewer_kind"] == "human_senior"]
    if len(initial) > 3 or len(senior) > 3:
        raise FilmStyleSafeContractError("scene exceeds frozen 3+3 panel size")
    initial_labels = [record["severity"]["label"] for record in initial]
    escalate = len(initial) != 3 or any(label in {"severe", "uncertain"} for label in initial_labels)
    if not escalate:
        if senior:
            raise FilmStyleSafeContractError("senior ratings are forbidden without an escalation trigger")
        return {
            "parent_scene_id": parent_scene_id,
            "candidate_id": candidate_id,
            "protocol_adjudicated_severe": 0,
            "status": "initial_unanimous_nonsevere",
            "initial_count": 3,
            "senior_count": 0,
            "autonomous_records_ignored": sum(
                record["reviewer"]["reviewer_kind"] == "autonomous_vlm_secondary"
                for record in validated
            ),
        }
    if len(senior) != 3:
        label, status = 1, "missing_or_incomplete_senior_panel"
    else:
        senior_labels = [record["severity"]["label"] for record in senior]
        severe_votes = sum(label == "severe" for label in senior_labels)
        nonsevere_votes = sum(label in {"safe", "minor"} for label in senior_labels)
        if severe_votes >= 2:
            label, status = 1, "senior_majority_severe"
        elif nonsevere_votes >= 2:
            label, status = 0, "senior_majority_nonsevere"
        else:
            label, status = 1, "senior_unresolved_conservative_severe"
    return {
        "parent_scene_id": parent_scene_id,
        "candidate_id": candidate_id,
        "protocol_adjudicated_severe": label,
        "status": status,
        "initial_count": len(initial),
        "senior_count": len(senior),
        "autonomous_records_ignored": sum(
            record["reviewer"]["reviewer_kind"] == "autonomous_vlm_secondary"
            for record in validated
        ),
    }


def validate_split_manifest(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reject lineage/hash leakage and hidden reuse in A0/A1/B0-B4."""

    materialized = list(rows)
    if not materialized:
        raise FilmStyleSafeContractError("split manifest cannot be empty")
    expected = {
        "record_id", "split", "evidence_origin", "parent_scene_id", "source_id",
        "uploader_or_creator_group", "camera_group", "roll_group", "exact_hash",
        "perceptual_hash", "transform_family", "failure_family",
    }
    seen_ids: set[str] = set()
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    family_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)
    for row in materialized:
        if not isinstance(row, Mapping):
            raise FilmStyleSafeContractError("split row must be an object")
        _exact_keys(row, expected, "split row")
        record_id = _identifier(row["record_id"], "split record_id")
        if record_id in seen_ids:
            raise FilmStyleSafeContractError(f"duplicate split record id: {record_id}")
        seen_ids.add(record_id)
        split = row["split"]
        if split not in SPLITS:
            raise FilmStyleSafeContractError("split row has invalid split")
        if row["evidence_origin"] in CURRENT_A0_ONLY_ORIGINS and split != "A0":
            raise FilmStyleSafeContractError("seen project evidence cannot enter hidden splits")
        for key in GROUP_KEYS:
            value = row[key]
            if key == "exact_hash":
                _sha256(value, "exact_hash")
            elif not isinstance(value, str) or not value:
                raise FilmStyleSafeContractError(f"{key} is required")
            group_splits[(key, value)].add(split)
        for key in ("transform_family", "failure_family"):
            value = _identifier(row[key], key)
            family_splits[(key, value)].add(split)
        split_counts[split] += 1
    leaks = [
        {"key": key, "value": value, "splits": sorted(splits)}
        for (key, value), splits in group_splits.items()
        if len(splits) > 1
    ]
    if leaks:
        first = leaks[0]
        raise FilmStyleSafeContractError(
            f"cross-split lineage leakage: {first['key']}={first['value']} in {first['splits']}"
        )
    a_family_leaks = [
        (key, value, splits)
        for (key, value), splits in family_splits.items()
        if "A0" in splits and "A1" in splits
    ]
    if a_family_leaks:
        key, value, splits = a_family_leaks[0]
        raise FilmStyleSafeContractError(f"A1 unseen-family violation: {key}={value} in {sorted(splits)}")
    return {
        "schema_version": 1,
        "row_count": len(materialized),
        "split_counts": dict(sorted(split_counts.items())),
        "cross_split_lineage_leaks": 0,
        "a0_a1_transform_or_failure_family_leaks": 0,
        "hidden_split_eligible": True,
    }


def zero_event_power_worksheet(*, epsilon: float, alpha: float = 0.05, expected_coverage: float = 0.5) -> dict[str, Any]:
    """Return planning counts for a zero-event one-sided binomial bound.

    The result is explicitly non-binding until clustering, label error and the
    tie model are separately piloted.
    """

    if not 0.0 < epsilon < 1.0 or not 0.0 < alpha < 1.0 or not 0.0 < expected_coverage <= 1.0:
        raise FilmStyleSafeContractError("power worksheet inputs must be probabilities")
    accepted = math.ceil(math.log(alpha) / math.log(1.0 - epsilon))
    total = math.ceil(accepted / expected_coverage)
    return {
        "epsilon": epsilon,
        "one_sided_confidence": 1.0 - alpha,
        "expected_coverage": expected_coverage,
        "zero_event_independent_accepted_scenes": accepted,
        "total_independent_scenes_before_adjustments": total,
        "binding_sample_size": None,
        "status": "planning_only_unknown_until_cluster_label_error_and_tie_model_pilots",
    }
