"""Versioned contracts for deterministic uploaded-reference colour matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping

import numpy as np

from .strict_json import strict_json_loads
from .canonical import canonical_sha256


REFERENCE_LOOK_RECIPE_SCHEMA_ID = "neuro-film.reference-look-recipe.v1"
REFERENCE_LOOK_ALGORITHM_ID = "safe-lab-reference-statistics.v1"
SUPPORTED_WORKING_SPACES = frozenset({"linear_srgb", "linear_rec2020"})
_CLAIM_CEILING = "reference-look"
_EVIDENCE_GRADE = "deterministic-statistical-baseline"
_EXPECTED_POLICY_KEYS = {
    "strength",
    "luma_strength",
    "tone_rolloff",
    "shadow_floor_l",
    "highlight_ceiling_l",
    "preserve_luma_detail_strength",
    "chroma_curve_strength",
    "neutral_protect",
    "skin_protect",
    "max_chroma_gain",
    "max_chroma_boost",
    "max_chroma_absolute",
    "gamut_mode",
    "gamut_iterations",
}
_EXPECTED_RECIPE_KEYS = {
    "schema_id",
    "algorithm_id",
    "recipe_id",
    "claim_ceiling",
    "evidence_grade",
    "reference_working_space",
    "reference_transfer_state",
    "reference_shape",
    "reference_pixel_sha256",
    "destination_lab_mean",
    "destination_lab_std",
    "policy",
}


class ReferenceMatchContractError(ValueError):
    """Raised when a reference-look recipe or policy is not safe to execute."""


@dataclass(frozen=True)
class ReferenceLookPolicy:
    """Explicit safe-Lab policy frozen into a replayable look recipe."""

    strength: float = 1.0
    luma_strength: float = 0.85
    tone_rolloff: float = 0.0
    shadow_floor_l: float = 1.0
    highlight_ceiling_l: float = 99.0
    preserve_luma_detail_strength: float = 0.35
    chroma_curve_strength: float = 0.2
    neutral_protect: float = 0.35
    skin_protect: float = 0.25
    max_chroma_gain: float | None = 1.8
    max_chroma_boost: float | None = 18.0
    max_chroma_absolute: float | None = 110.0
    gamut_mode: str = "source"
    gamut_iterations: int = 24


@dataclass(frozen=True)
class ReferenceLookRecipe:
    """Immutable product recipe fitted from exactly one uploaded reference."""

    schema_id: str
    algorithm_id: str
    recipe_id: str
    claim_ceiling: str
    evidence_grade: str
    reference_working_space: str
    reference_transfer_state: str
    reference_shape: tuple[int, int, int]
    reference_pixel_sha256: str
    destination_lab_mean: tuple[float, float, float]
    destination_lab_std: tuple[float, float, float]
    policy: ReferenceLookPolicy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceMatchContractError(f"{label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ReferenceMatchContractError(f"{label} must be a finite number")
    return number


def _bounded(value: Any, label: str, low: float, high: float) -> float:
    number = _finite_number(value, label)
    if number < low or number > high:
        raise ReferenceMatchContractError(f"{label} must be within [{low}, {high}]")
    return number


def _optional_positive(value: Any, label: str) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, label)
    if number <= 0:
        raise ReferenceMatchContractError(f"{label} must be positive when present")
    return number


def validate_policy(policy: ReferenceLookPolicy) -> None:
    if not isinstance(policy, ReferenceLookPolicy):
        raise ReferenceMatchContractError("policy must be ReferenceLookPolicy")
    _bounded(policy.strength, "policy.strength", 0.0, 1.0)
    _bounded(policy.luma_strength, "policy.luma_strength", 0.0, 1.0)
    _bounded(policy.tone_rolloff, "policy.tone_rolloff", 0.0, 1.0)
    _bounded(policy.shadow_floor_l, "policy.shadow_floor_l", 0.0, 25.0)
    _bounded(policy.highlight_ceiling_l, "policy.highlight_ceiling_l", 75.0, 100.0)
    if policy.shadow_floor_l >= policy.highlight_ceiling_l:
        raise ReferenceMatchContractError(
            "policy shadow floor must be below highlight ceiling"
        )
    _bounded(
        policy.preserve_luma_detail_strength,
        "policy.preserve_luma_detail_strength",
        0.0,
        1.0,
    )
    _bounded(
        policy.chroma_curve_strength,
        "policy.chroma_curve_strength",
        0.0,
        1.0,
    )
    _bounded(policy.neutral_protect, "policy.neutral_protect", 0.0, 1.0)
    _bounded(policy.skin_protect, "policy.skin_protect", 0.0, 1.0)
    _optional_positive(policy.max_chroma_gain, "policy.max_chroma_gain")
    _optional_positive(policy.max_chroma_boost, "policy.max_chroma_boost")
    _optional_positive(policy.max_chroma_absolute, "policy.max_chroma_absolute")
    if policy.gamut_mode not in {"source", "chroma"}:
        raise ReferenceMatchContractError("policy.gamut_mode must be source or chroma")
    if (
        isinstance(policy.gamut_iterations, bool)
        or not isinstance(policy.gamut_iterations, int)
        or policy.gamut_iterations < 1
        or policy.gamut_iterations > 64
    ):
        raise ReferenceMatchContractError(
            "policy.gamut_iterations must be an integer within [1, 64]"
        )


def _three_finite(values: Any, label: str, *, positive: bool) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ReferenceMatchContractError(f"{label} must contain three values")
    result = tuple(_finite_number(value, f"{label}[{index}]") for index, value in enumerate(values))
    if positive and any(value <= 0 for value in result):
        raise ReferenceMatchContractError(f"{label} values must be positive")
    return result


def _canonical_payload(recipe: ReferenceLookRecipe) -> dict[str, Any]:
    payload = recipe.to_dict()
    payload.pop("recipe_id", None)
    return payload


def compute_recipe_id(recipe: ReferenceLookRecipe) -> str:
    return canonical_sha256(_canonical_payload(recipe))


def validate_recipe(recipe: ReferenceLookRecipe) -> None:
    if not isinstance(recipe, ReferenceLookRecipe):
        raise ReferenceMatchContractError("recipe must be ReferenceLookRecipe")
    if recipe.schema_id != REFERENCE_LOOK_RECIPE_SCHEMA_ID:
        raise ReferenceMatchContractError("unsupported reference-look recipe schema")
    if recipe.algorithm_id != REFERENCE_LOOK_ALGORITHM_ID:
        raise ReferenceMatchContractError("unsupported reference-look algorithm")
    if recipe.claim_ceiling != _CLAIM_CEILING:
        raise ReferenceMatchContractError("reference-look recipe claim ceiling mismatch")
    if recipe.evidence_grade != _EVIDENCE_GRADE:
        raise ReferenceMatchContractError("reference-look recipe evidence grade mismatch")
    if recipe.reference_working_space not in SUPPORTED_WORKING_SPACES:
        raise ReferenceMatchContractError("unsupported reference working space")
    if recipe.reference_transfer_state != "display_linear":
        raise ReferenceMatchContractError("reference recipe must be display-linear")
    if (
        not isinstance(recipe.reference_shape, tuple)
        or len(recipe.reference_shape) != 3
        or recipe.reference_shape[2] != 3
        or recipe.reference_shape[0] <= 0
        or recipe.reference_shape[1] <= 0
        or any(isinstance(value, bool) or not isinstance(value, int) for value in recipe.reference_shape)
    ):
        raise ReferenceMatchContractError("reference_shape must be positive HxWx3")
    if len(recipe.reference_pixel_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in recipe.reference_pixel_sha256
    ):
        raise ReferenceMatchContractError("reference_pixel_sha256 must be lowercase SHA-256")
    _three_finite(recipe.destination_lab_mean, "destination_lab_mean", positive=False)
    _three_finite(recipe.destination_lab_std, "destination_lab_std", positive=True)
    validate_policy(recipe.policy)
    if len(recipe.recipe_id) != 64 or any(char not in "0123456789abcdef" for char in recipe.recipe_id):
        raise ReferenceMatchContractError("recipe_id must be lowercase SHA-256")
    if recipe.recipe_id != compute_recipe_id(recipe):
        raise ReferenceMatchContractError("recipe_id does not match canonical payload")


def recipe_to_json(recipe: ReferenceLookRecipe) -> str:
    validate_recipe(recipe)
    return json.dumps(
        recipe.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReferenceMatchContractError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def recipe_from_dict(payload: Mapping[str, Any]) -> ReferenceLookRecipe:
    if not isinstance(payload, Mapping):
        raise ReferenceMatchContractError("recipe payload must be an object")
    _strict_keys(payload, _EXPECTED_RECIPE_KEYS, "recipe")
    policy_payload = payload["policy"]
    if not isinstance(policy_payload, Mapping):
        raise ReferenceMatchContractError("recipe.policy must be an object")
    _strict_keys(policy_payload, _EXPECTED_POLICY_KEYS, "recipe.policy")
    try:
        policy = ReferenceLookPolicy(**dict(policy_payload))
        recipe = ReferenceLookRecipe(
            schema_id=str(payload["schema_id"]),
            algorithm_id=str(payload["algorithm_id"]),
            recipe_id=str(payload["recipe_id"]),
            claim_ceiling=str(payload["claim_ceiling"]),
            evidence_grade=str(payload["evidence_grade"]),
            reference_working_space=str(payload["reference_working_space"]),
            reference_transfer_state=str(payload["reference_transfer_state"]),
            reference_shape=tuple(payload["reference_shape"]),
            reference_pixel_sha256=str(payload["reference_pixel_sha256"]),
            destination_lab_mean=tuple(payload["destination_lab_mean"]),
            destination_lab_std=tuple(payload["destination_lab_std"]),
            policy=policy,
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError("recipe payload types are invalid") from exc
    validate_recipe(recipe)
    return recipe


def recipe_from_json(encoded: str) -> ReferenceLookRecipe:
    if not isinstance(encoded, str):
        raise ReferenceMatchContractError("encoded recipe must be a string")
    try:
        payload = strict_json_loads(encoded)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReferenceMatchContractError("encoded recipe is not valid JSON") from exc
    return recipe_from_dict(payload)


__all__ = [
    "REFERENCE_LOOK_ALGORITHM_ID",
    "REFERENCE_LOOK_RECIPE_SCHEMA_ID",
    "SUPPORTED_WORKING_SPACES",
    "ReferenceLookPolicy",
    "ReferenceLookRecipe",
    "ReferenceMatchContractError",
    "compute_recipe_id",
    "recipe_from_dict",
    "recipe_from_json",
    "recipe_to_json",
    "validate_policy",
    "validate_recipe",
]
