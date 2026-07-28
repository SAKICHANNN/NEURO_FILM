"""Composition plan for a verified, genuinely promoted external reference look."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .composition import (
    FilmEffectBinding,
    build_film_effect_binding,
    validate_film_effect_binding,
)
from .contracts import ReferenceMatchContractError
from .core_staging_verification import (
    ExternalCoreStagingVerificationV1,
    validate_external_core_staging_verification_v1,
)


EXTERNAL_REFERENCE_COMPOSITION_SCHEMA_ID = (
    "neuro-film.external-reference-composition.v1"
)
EXTERNAL_REFERENCE_COMPOSITION_STATE = (
    "composition-ready-not-rendered"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "plan_id",
    "staging_verification_id",
    "staging_run_id",
    "authorization_id",
    "reference_intent_id",
    "source_count",
    "state",
    "color_owner",
    "reference_color_status",
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
class ExternalReferenceCompositionV1:
    """Replayable ownership plan; it does not render or deliver files."""

    schema_id: str
    plan_id: str
    staging_verification_id: str
    staging_run_id: str
    authorization_id: str
    reference_intent_id: str
    source_count: int
    state: str
    color_owner: str
    reference_color_status: str
    claim_ceiling: str
    output_label: str
    execution_order: tuple[str, ...]
    film_color_profile_id: None
    film_stock_identity_claimed: bool
    film_effects: FilmEffectBinding | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict(
    value: Any,
    expected: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceMatchContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ReferenceMatchContractError(
            f"{label} keys mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _identity_payload(
    value: ExternalReferenceCompositionV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("plan_id")
    return payload


def build_external_reference_composition_v1(
    verification: ExternalCoreStagingVerificationV1,
    *,
    include_film_effects: bool = False,
    film_profile: Mapping[str, Any] | None = None,
    film_profile_sha256: str | None = None,
) -> ExternalReferenceCompositionV1:
    """Bind verified external colour to optional procedural FilmFX."""

    validate_external_core_staging_verification_v1(verification)
    if not isinstance(include_film_effects, bool):
        raise ReferenceMatchContractError(
            "include_film_effects must be boolean"
        )
    if include_film_effects:
        if film_profile is None or film_profile_sha256 is None:
            raise ReferenceMatchContractError(
                "film profile and hash are required for FilmFX"
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
    provisional = ExternalReferenceCompositionV1(
        schema_id=EXTERNAL_REFERENCE_COMPOSITION_SCHEMA_ID,
        plan_id="0" * 64,
        staging_verification_id=verification.verification_id,
        staging_run_id=verification.run_id,
        authorization_id=verification.authorization_id,
        reference_intent_id=verification.reference_intent_id,
        source_count=verification.source_count,
        state=EXTERNAL_REFERENCE_COMPOSITION_STATE,
        color_owner="external-reference-look",
        reference_color_status="verified-staging",
        claim_ceiling="reference-look",
        output_label=(
            "reference-look"
            if effects is None
            else "reference-look+film-effects"
        ),
        execution_order=(
            ("verified_reference_color",)
            if effects is None
            else ("verified_reference_color", "film_effects")
        ),
        film_color_profile_id=None,
        film_stock_identity_claimed=False,
        film_effects=effects,
    )
    result = replace(
        provisional,
        plan_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_external_reference_composition_v1(result)
    return result


def validate_external_reference_composition_v1(
    value: ExternalReferenceCompositionV1,
) -> None:
    if not isinstance(value, ExternalReferenceCompositionV1):
        raise ReferenceMatchContractError(
            "external reference composition type is invalid"
        )
    if value.schema_id != EXTERNAL_REFERENCE_COMPOSITION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported external reference composition schema"
        )
    _sha256(value.plan_id, "plan_id")
    _sha256(
        value.staging_verification_id,
        "staging_verification_id",
    )
    _sha256(value.staging_run_id, "staging_run_id")
    _sha256(value.authorization_id, "authorization_id")
    _sha256(value.reference_intent_id, "reference_intent_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
    ):
        raise ReferenceMatchContractError(
            "external composition source_count is invalid"
        )
    if value.state != EXTERNAL_REFERENCE_COMPOSITION_STATE:
        raise ReferenceMatchContractError(
            "external composition state is unsupported"
        )
    if (
        value.color_owner != "external-reference-look"
        or value.reference_color_status != "verified-staging"
        or value.claim_ceiling != "reference-look"
    ):
        raise ReferenceMatchContractError(
            "external composition colour claim is inconsistent"
        )
    if value.film_color_profile_id is not None:
        raise ReferenceMatchContractError(
            "external composition cannot stack film colour"
        )
    if value.film_stock_identity_claimed is not False:
        raise ReferenceMatchContractError(
            "FilmFX cannot claim a film stock identity"
        )
    if value.film_effects is None:
        expected_label = "reference-look"
        expected_order = ("verified_reference_color",)
    else:
        validate_film_effect_binding(value.film_effects)
        expected_label = "reference-look+film-effects"
        expected_order = (
            "verified_reference_color",
            "film_effects",
        )
    if (
        value.output_label != expected_label
        or value.execution_order != expected_order
    ):
        raise ReferenceMatchContractError(
            "external composition execution plan is inconsistent"
        )
    if value.plan_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "external composition plan_id mismatch"
        )


def external_reference_composition_to_json(
    value: ExternalReferenceCompositionV1,
) -> str:
    validate_external_reference_composition_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def external_reference_composition_from_json(
    encoded: str,
) -> ExternalReferenceCompositionV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "external composition is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "external composition")
    raw_effects = payload["film_effects"]
    if raw_effects is None:
        effects = None
    else:
        raw_effects = _strict(
            raw_effects,
            _EFFECT_KEYS,
            "external composition film_effects",
        )
        try:
            effects = FilmEffectBinding(**raw_effects)
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "external composition FilmFX fields are invalid"
            ) from exc
    converted = dict(payload)
    raw_order = converted["execution_order"]
    if not isinstance(raw_order, list):
        raise ReferenceMatchContractError(
            "external composition execution_order must be an array"
        )
    converted["execution_order"] = tuple(raw_order)
    converted["film_effects"] = effects
    try:
        value = ExternalReferenceCompositionV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "external composition fields are invalid"
        ) from exc
    validate_external_reference_composition_v1(value)
    return value


__all__ = [
    "EXTERNAL_REFERENCE_COMPOSITION_SCHEMA_ID",
    "EXTERNAL_REFERENCE_COMPOSITION_STATE",
    "ExternalReferenceCompositionV1",
    "build_external_reference_composition_v1",
    "external_reference_composition_from_json",
    "external_reference_composition_to_json",
    "validate_external_reference_composition_v1",
]
