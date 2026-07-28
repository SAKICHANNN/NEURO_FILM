"""Explicit product boundary between reference colour and film effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping

import numpy as np

from src.inference import validate_render_profile

from .strict_json import strict_json_loads
from .canonical import canonical_sha256
from .contracts import (
    ReferenceLookRecipe,
    ReferenceMatchContractError,
    validate_recipe,
)


REFERENCE_COMPOSITION_SCHEMA_ID = "neuro-film.reference-composition.v1"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_PLAN_KEYS = {
    "schema_id",
    "plan_id",
    "color_owner",
    "reference_color_status",
    "research_baseline_override",
    "reference_recipe_id",
    "claim_ceiling",
    "output_label",
    "execution_order",
    "film_color_profile_id",
    "film_stock_identity_claimed",
    "film_effects",
}
_EFFECT_KEYS = {
    "profile_id",
    "profile_version",
    "profile_sha256",
    "film_stock_id",
    "interpretation",
    "grain",
    "halation",
    "dust",
    "halation_model",
}


@dataclass(frozen=True)
class FilmEffectBinding:
    """Film-profile provenance used only for procedural effects."""

    profile_id: str
    profile_version: str
    profile_sha256: str
    film_stock_id: str | None
    interpretation: str
    grain: float
    halation: float
    dust: float
    halation_model: str


@dataclass(frozen=True)
class ReferenceCompositionPlan:
    """Replayable colour/effect ownership plan for the reference-match mode."""

    schema_id: str
    plan_id: str
    color_owner: str
    reference_color_status: str
    research_baseline_override: bool
    reference_recipe_id: str
    claim_ceiling: str
    output_label: str
    execution_order: tuple[str, ...]
    film_color_profile_id: None
    film_stock_identity_claimed: bool
    film_effects: FilmEffectBinding | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_payload(plan: ReferenceCompositionPlan) -> dict[str, Any]:
    payload = plan.to_dict()
    payload.pop("plan_id", None)
    return payload


def _plan_id(plan: ReferenceCompositionPlan) -> str:
    return canonical_sha256(_canonical_payload(plan))


def build_film_effect_binding(
    profile: Mapping[str, Any],
    *,
    profile_sha256: str,
) -> FilmEffectBinding:
    try:
        validate_render_profile(profile)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "film effect profile violates the render-profile contract"
        ) from exc
    if not isinstance(profile_sha256, str) or not _HASH.fullmatch(profile_sha256):
        raise ReferenceMatchContractError(
            "film effect profile SHA-256 must be lowercase hexadecimal"
        )
    identity = profile["identity"]
    effects = profile["effect_defaults"]
    return FilmEffectBinding(
        profile_id=str(profile["profile_id"]),
        profile_version=str(profile["profile_version"]),
        profile_sha256=profile_sha256,
        film_stock_id=identity["film_stock_id"],
        interpretation=str(identity["interpretation"]),
        grain=float(effects["grain"]),
        halation=float(effects["halation"]),
        dust=float(effects["dust"]),
        halation_model=str(effects["halation_model"]),
    )


def validate_film_effect_binding(binding: FilmEffectBinding) -> None:
    if not isinstance(binding, FilmEffectBinding):
        raise ReferenceMatchContractError(
            "film effect binding type is invalid"
        )
    if (
        not isinstance(binding.profile_id, str)
        or not _IDENTIFIER.fullmatch(binding.profile_id)
    ):
        raise ReferenceMatchContractError(
            "film effect binding profile_id is invalid"
        )
    if (
        not isinstance(binding.profile_version, str)
        or not _VERSION.fullmatch(binding.profile_version)
    ):
        raise ReferenceMatchContractError(
            "film effect binding profile_version is invalid"
        )
    if (
        binding.film_stock_id is not None
        and (
            not isinstance(binding.film_stock_id, str)
            or not _IDENTIFIER.fullmatch(binding.film_stock_id)
        )
    ):
        raise ReferenceMatchContractError(
            "film effect binding film_stock_id is invalid"
        )
    if (
        not isinstance(binding.interpretation, str)
        or not binding.interpretation
    ):
        raise ReferenceMatchContractError(
            "film effect binding interpretation is invalid"
        )
    if (
        not isinstance(binding.profile_sha256, str)
        or not _HASH.fullmatch(binding.profile_sha256)
    ):
        raise ReferenceMatchContractError(
            "film effect binding profile hash is invalid"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or float(value) < 0.0
        or float(value) > 1.0
        for value in (binding.grain, binding.halation, binding.dust)
    ):
        raise ReferenceMatchContractError(
            "film effect strengths must be within [0, 1]"
        )
    if binding.halation_model not in {"simple", "physical"}:
        raise ReferenceMatchContractError(
            "film effect halation model is invalid"
        )


def validate_reference_composition(plan: ReferenceCompositionPlan) -> None:
    """Reject hidden film-colour stacking or claim escalation."""

    if not isinstance(plan, ReferenceCompositionPlan):
        raise ReferenceMatchContractError(
            "composition plan must be ReferenceCompositionPlan"
        )
    if plan.schema_id != REFERENCE_COMPOSITION_SCHEMA_ID:
        raise ReferenceMatchContractError("unsupported reference composition schema")
    if not isinstance(plan.research_baseline_override, bool):
        raise ReferenceMatchContractError(
            "composition research_baseline_override must be boolean"
        )
    expected_status = (
        "research-baseline"
        if plan.research_baseline_override
        else "identity-fallback"
    )
    if plan.reference_color_status != expected_status:
        raise ReferenceMatchContractError(
            "composition reference colour status mismatch"
        )
    expected_owner = (
        "reference-look"
        if plan.research_baseline_override
        else "identity"
    )
    if plan.color_owner != expected_owner:
        raise ReferenceMatchContractError(
            "composition colour owner mismatch"
        )
    expected_claim = (
        "reference-look"
        if plan.research_baseline_override
        else "identity"
    )
    if plan.claim_ceiling != expected_claim:
        raise ReferenceMatchContractError("composition claim ceiling mismatch")
    if plan.film_color_profile_id is not None:
        raise ReferenceMatchContractError(
            "film and reference colour operators cannot be silently stacked"
        )
    if plan.film_stock_identity_claimed:
        raise ReferenceMatchContractError(
            "film effects do not establish a film-stock identity"
        )
    if (
        not isinstance(plan.reference_recipe_id, str)
        or not _HASH.fullmatch(plan.reference_recipe_id)
    ):
        raise ReferenceMatchContractError(
            "reference_recipe_id must be lowercase SHA-256"
        )
    if plan.film_effects is not None and not plan.research_baseline_override:
        raise ReferenceMatchContractError(
            "film effects require available reference colour or a separate "
            "film-simulation mode"
        )
    expected_order = (
        ("reference_color",)
        if plan.research_baseline_override and plan.film_effects is None
        else (
            ("reference_color", "film_effects")
            if plan.research_baseline_override
            else ("identity_color",)
        )
    )
    if plan.execution_order != expected_order:
        raise ReferenceMatchContractError("composition execution order mismatch")
    expected_label = (
        (
            "reference-look"
            if plan.film_effects is None
            else "reference-look+film-effects"
        )
        if plan.research_baseline_override
        else "identity"
    )
    if plan.output_label != expected_label:
        raise ReferenceMatchContractError("composition output label mismatch")
    if plan.film_effects is not None:
        validate_film_effect_binding(plan.film_effects)
    if (
        not isinstance(plan.plan_id, str)
        or not _HASH.fullmatch(plan.plan_id)
        or plan.plan_id != _plan_id(plan)
    ):
        raise ReferenceMatchContractError(
            "composition plan_id does not match canonical payload"
        )


def build_reference_composition(
    recipe: ReferenceLookRecipe,
    *,
    include_film_effects: bool = False,
    film_profile: Mapping[str, Any] | None = None,
    film_profile_sha256: str | None = None,
    allow_research_baseline: bool = False,
) -> ReferenceCompositionPlan:
    """Build reference-colour mode with optional film-derived effects only."""

    validate_recipe(recipe)
    if not isinstance(include_film_effects, bool):
        raise ReferenceMatchContractError("include_film_effects must be boolean")
    if not isinstance(allow_research_baseline, bool):
        raise ReferenceMatchContractError(
            "allow_research_baseline must be boolean"
        )
    if include_film_effects and not allow_research_baseline:
        raise ReferenceMatchContractError(
            "film effects cannot compose with an unavailable reference colour; "
            "use the separate film-simulation mode"
        )
    if include_film_effects:
        if film_profile is None or film_profile_sha256 is None:
            raise ReferenceMatchContractError(
                "film profile and hash are required when film effects are enabled"
            )
        effects = build_film_effect_binding(
            film_profile,
            profile_sha256=film_profile_sha256,
        )
    else:
        if film_profile is not None or film_profile_sha256 is not None:
            raise ReferenceMatchContractError(
                "film profile inputs require include_film_effects=True"
            )
        effects = None

    provisional = ReferenceCompositionPlan(
        schema_id=REFERENCE_COMPOSITION_SCHEMA_ID,
        plan_id="0" * 64,
        color_owner=(
            "reference-look" if allow_research_baseline else "identity"
        ),
        reference_color_status=(
            "research-baseline"
            if allow_research_baseline
            else "identity-fallback"
        ),
        research_baseline_override=allow_research_baseline,
        reference_recipe_id=recipe.recipe_id,
        claim_ceiling=(
            "reference-look" if allow_research_baseline else "identity"
        ),
        output_label=(
            (
                "reference-look"
                if effects is None
                else "reference-look+film-effects"
            )
            if allow_research_baseline
            else "identity"
        ),
        execution_order=(
            (
                ("reference_color",)
                if effects is None
                else ("reference_color", "film_effects")
            )
            if allow_research_baseline
            else ("identity_color",)
        ),
        film_color_profile_id=None,
        film_stock_identity_claimed=False,
        film_effects=effects,
    )
    plan = replace(provisional, plan_id=_plan_id(provisional))
    validate_reference_composition(plan)
    return plan


def composition_plan_to_json(plan: ReferenceCompositionPlan) -> str:
    validate_reference_composition(plan)
    return json.dumps(
        plan.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def composition_plan_from_dict(
    payload: Mapping[str, Any],
) -> ReferenceCompositionPlan:
    if not isinstance(payload, Mapping) or set(payload) != _PLAN_KEYS:
        raise ReferenceMatchContractError(
            "composition plan keys do not match the v1 contract"
        )
    raw_effects = payload["film_effects"]
    effects: FilmEffectBinding | None
    if raw_effects is None:
        effects = None
    elif isinstance(raw_effects, Mapping) and set(raw_effects) == _EFFECT_KEYS:
        try:
            effects = FilmEffectBinding(**dict(raw_effects))
        except TypeError as exc:
            raise ReferenceMatchContractError(
                "composition film_effects types are invalid"
            ) from exc
    else:
        raise ReferenceMatchContractError(
            "composition film_effects keys do not match the v1 contract"
        )
    order = payload["execution_order"]
    if not isinstance(order, (list, tuple)):
        raise ReferenceMatchContractError(
            "composition execution_order must be an array"
        )
    try:
        plan = ReferenceCompositionPlan(
            schema_id=payload["schema_id"],
            plan_id=payload["plan_id"],
            color_owner=payload["color_owner"],
            reference_color_status=payload["reference_color_status"],
            research_baseline_override=payload[
                "research_baseline_override"
            ],
            reference_recipe_id=payload["reference_recipe_id"],
            claim_ceiling=payload["claim_ceiling"],
            output_label=payload["output_label"],
            execution_order=tuple(order),
            film_color_profile_id=payload["film_color_profile_id"],
            film_stock_identity_claimed=payload[
                "film_stock_identity_claimed"
            ],
            film_effects=effects,
        )
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "composition plan types are invalid"
        ) from exc
    validate_reference_composition(plan)
    return plan


def composition_plan_from_json(encoded: str) -> ReferenceCompositionPlan:
    if not isinstance(encoded, str):
        raise ReferenceMatchContractError(
            "encoded composition plan must be a string"
        )
    try:
        payload = strict_json_loads(encoded)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "encoded composition plan is not valid JSON"
        ) from exc
    return composition_plan_from_dict(payload)


__all__ = [
    "REFERENCE_COMPOSITION_SCHEMA_ID",
    "FilmEffectBinding",
    "ReferenceCompositionPlan",
    "build_film_effect_binding",
    "build_reference_composition",
    "composition_plan_from_dict",
    "composition_plan_from_json",
    "composition_plan_to_json",
    "validate_reference_composition",
    "validate_film_effect_binding",
]
