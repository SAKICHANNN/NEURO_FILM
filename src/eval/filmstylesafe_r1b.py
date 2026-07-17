"""FilmStyleSafe R1B failure-suite membership validation.

Validates design-time suite cards only. Does not generate pixels, recruit
raters, train models, or populate hidden splits.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONTRACT_ID = "kmcfm.filmstylesafe-r1b.v1"
ROLES = frozenset(
    {
        "source_scene",
        "synthetic_failure",
        "real_failure",
        "legitimate_local_hard_negative",
        "strength_control",
        "external_style_control",
        "regression_case",
    }
)
SPLITS = frozenset({"A0", "A1", "B0", "B1", "B2", "B3", "B4"})
A0_ONLY_ORIGINS = frozenset(
    {
        "u41_provisional_gold",
        "u42_owner_anchor_replays",
        "rf2_c0_spektrafilm_outputs",
        "known_id11_red_speckle",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class FilmStyleSafeR1BError(ValueError):
    """Raised when an R1B suite card fails closed."""


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise FilmStyleSafeR1BError(f"{context} must be a bounded identifier")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FilmStyleSafeR1BError(f"{context} must be lowercase SHA-256")
    return value


def load_r1b_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FilmStyleSafeR1BError(f"cannot read R1B contract {path}: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("contract_id") != CONTRACT_ID:
        raise FilmStyleSafeR1BError("unexpected FilmStyleSafe R1B contract id")
    if frozenset(contract.get("suite_membership_roles", [])) != ROLES:
        raise FilmStyleSafeR1BError("suite membership roles drifted")
    if contract.get("authorization", {}).get("hidden_split_population") is not False:
        raise FilmStyleSafeR1BError("R1B must keep hidden_split_population false")
    if contract.get("authorization", {}).get("model_training") is not False:
        raise FilmStyleSafeR1BError("R1B must keep model_training false")
    if contract.get("authorization", {}).get("external_human_recruitment") is not False:
        raise FilmStyleSafeR1BError("R1B must keep external_human_recruitment false")
    return contract


def validate_suite_member(record: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one failure-suite membership card."""
    required = {
        "member_id",
        "role",
        "split",
        "parent_scene_id",
        "source_id",
        "transform_family",
        "failure_family",
        "origin",
        "exact_hash",
        "perceptual_hash",
    }
    if set(record) < required:
        raise FilmStyleSafeR1BError(f"suite member missing keys: {sorted(required - set(record))}")
    member_id = _identifier(record["member_id"], "member_id")
    role = str(record["role"])
    if role not in ROLES:
        raise FilmStyleSafeR1BError(f"invalid role: {role}")
    split = str(record["split"])
    if split not in SPLITS:
        raise FilmStyleSafeR1BError(f"invalid split: {split}")
    origin = _identifier(record["origin"], "origin")
    if origin in A0_ONLY_ORIGINS and split != "A0":
        raise FilmStyleSafeR1BError(f"origin {origin} is A0-only")
    transform_family = _identifier(record["transform_family"], "transform_family")
    failure_family = str(record["failure_family"])
    allowed_failures = set(contract.get("failure_families_a0", [])) | {
        "none",
        "legitimate_local_hard_negative",
    }
    if failure_family not in allowed_failures:
        raise FilmStyleSafeR1BError(f"unknown failure_family: {failure_family}")
    allowed_transforms = set(contract.get("transform_families_a0", []))
    if role == "source_scene":
        if transform_family != "identity_control":
            raise FilmStyleSafeR1BError("source_scene transform_family must be identity_control")
    elif transform_family not in allowed_transforms:
        raise FilmStyleSafeR1BError(f"unknown transform_family: {transform_family}")
    if role == "synthetic_failure":
        params = record.get("operator_card")
        if not isinstance(params, Mapping):
            raise FilmStyleSafeR1BError("synthetic_failure requires operator_card")
        for field in contract["synthetic_explicit_non_generative_failures"][
            "minimum_parameter_card_fields"
        ]:
            if field not in params:
                raise FilmStyleSafeR1BError(f"operator_card missing {field}")
        _identifier(params["operator_id"], "operator_id")
        _sha256(params["parameter_hash"], "parameter_hash")
        _sha256(params["input_hash"], "input_hash")
        if str(params.get("transform_family")) != transform_family:
            raise FilmStyleSafeR1BError("operator_card transform_family mismatch")
        if str(params.get("failure_family")) != failure_family:
            raise FilmStyleSafeR1BError("operator_card failure_family mismatch")
    if role == "legitimate_local_hard_negative":
        if failure_family != "legitimate_local_hard_negative":
            raise FilmStyleSafeR1BError("hard-negative failure_family must be dedicated")
        label = str(record.get("hard_negative_label") or "")
        if label not in set(contract.get("legitimate_local_hard_negatives", [])):
            raise FilmStyleSafeR1BError(f"unknown hard_negative_label: {label}")
    if role == "strength_control":
        scheme = str(record.get("owner_scheme_id") or "")
        allowed = set(contract["owner_strength_controls"]["full_anchors"]) | set(
            contract["owner_strength_controls"]["smoke_cues"]
        )
        if scheme not in allowed:
            raise FilmStyleSafeR1BError(f"unknown owner_scheme_id: {scheme}")
    if role == "external_style_control":
        if transform_family != "external_spektrafilm_look_approximation_control":
            raise FilmStyleSafeR1BError(
                "external_style_control requires external_spektrafilm_look_approximation_control"
            )
        if failure_family != "none":
            raise FilmStyleSafeR1BError("external_style_control failure_family must be none")
        if origin != "rf2_c0_spektrafilm_outputs":
            raise FilmStyleSafeR1BError("external_style_control currently limited to RF2.C0 origin")
        control_id = record.get("external_control_id")
        if not isinstance(control_id, str) or not control_id:
            raise FilmStyleSafeR1BError("external_style_control requires external_control_id")
        _identifier(control_id, "external_control_id")
    if role == "regression_case":
        if origin != "known_id11_red_speckle":
            raise FilmStyleSafeR1BError("regression_case currently limited to ID11 origin")
    return {
        "member_id": member_id,
        "role": role,
        "split": split,
        "parent_scene_id": _identifier(record["parent_scene_id"], "parent_scene_id"),
        "source_id": _identifier(record["source_id"], "source_id"),
        "transform_family": transform_family,
        "failure_family": failure_family,
        "origin": origin,
        "exact_hash": _sha256(record["exact_hash"], "exact_hash"),
        "perceptual_hash": _sha256(record["perceptual_hash"], "perceptual_hash"),
        "passed": True,
    }


def audit_suite_leakage(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reject parent-scene / hash / family leakage across A0 and A1."""
    by_split: dict[str, list[Mapping[str, Any]]] = {"A0": [], "A1": []}
    for row in members:
        split = str(row["split"])
        if split in by_split:
            by_split[split].append(row)
    violations: list[str] = []

    def collect(key: str, split: str) -> set[str]:
        return {str(row[key]) for row in by_split[split]}

    for key in ("parent_scene_id", "source_id", "exact_hash", "perceptual_hash"):
        overlap = collect(key, "A0") & collect(key, "A1")
        if overlap:
            violations.append(f"{key} overlap: {sorted(overlap)[:3]}")
    a0_families = {
        (str(row["transform_family"]), str(row["failure_family"])) for row in by_split["A0"]
    }
    for row in by_split["A1"]:
        pair = (str(row["transform_family"]), str(row["failure_family"]))
        if pair in a0_families:
            violations.append(f"A1 reuses A0 family pair {pair}")
    if violations:
        raise FilmStyleSafeR1BError("; ".join(violations))
    return {
        "passed": True,
        "a0_members": len(by_split["A0"]),
        "a1_members": len(by_split["A1"]),
        "violations": [],
    }
