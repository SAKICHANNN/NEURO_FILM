"""Composition plan for a verified shared reference-operator staging run."""

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
from .shared_staging_verification import (
    ExternalSharedStagingVerificationV1,
    validate_external_shared_staging_verification_v1,
)


SHARED_REFERENCE_COMPOSITION_SCHEMA_ID = (
    "neuro-film.shared-reference-composition.v1"
)
SHARED_REFERENCE_COMPOSITION_STATE = (
    "composition-ready-not-rendered"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "plan_id",
    "staging_verification_id",
    "staging_run_id",
    "authorization_id",
    "numeric_guard_batch_id",
    "operator_id",
    "reference_view_id",
    "source_count",
    "state",
    "color_owner",
    "reference_color_status",
    "claim_ceiling",
    "output_label",
    "execution_order",
    "film_color_profile_id",
    "film_stock_identity_claimed",
    "calibrated_reference_claimed",
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
class SharedReferenceCompositionV1:
    """Replayable ownership plan; it does not render or deliver files."""

    schema_id: str
    plan_id: str
    staging_verification_id: str
    staging_run_id: str
    authorization_id: str
    numeric_guard_batch_id: str
    operator_id: str
    reference_view_id: str
    source_count: int
    state: str
    color_owner: str
    reference_color_status: str
    claim_ceiling: str
    output_label: str
    execution_order: tuple[str, ...]
    film_color_profile_id: None
    film_stock_identity_claimed: bool
    calibrated_reference_claimed: bool
    film_effects: FilmEffectBinding | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from shared composition"
        )
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _identity_payload(value: SharedReferenceCompositionV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("plan_id")
    return payload


def build_shared_reference_composition_v1(
    verification: ExternalSharedStagingVerificationV1,
    *,
    include_film_effects: bool = False,
    film_profile: Mapping[str, Any] | None = None,
    film_profile_sha256: str | None = None,
) -> SharedReferenceCompositionV1:
    """Bind verified shared colour to optional procedural FilmFX."""

    validate_external_shared_staging_verification_v1(verification)
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
    provisional = SharedReferenceCompositionV1(
        schema_id=SHARED_REFERENCE_COMPOSITION_SCHEMA_ID,
        plan_id="0" * 64,
        staging_verification_id=verification.verification_id,
        staging_run_id=verification.run_id,
        authorization_id=verification.authorization_id,
        numeric_guard_batch_id=verification.numeric_guard_batch_id,
        operator_id=verification.operator_id,
        reference_view_id=verification.reference_view_id,
        source_count=verification.source_count,
        state=SHARED_REFERENCE_COMPOSITION_STATE,
        color_owner="external-shared-reference-look",
        reference_color_status="verified-shared-staging",
        claim_ceiling="reference-look",
        output_label=(
            "reference-look"
            if effects is None
            else "reference-look+film-effects"
        ),
        execution_order=(
            ("verified_shared_reference_color",)
            if effects is None
            else (
                "verified_shared_reference_color",
                "film_effects",
            )
        ),
        film_color_profile_id=None,
        film_stock_identity_claimed=False,
        calibrated_reference_claimed=False,
        film_effects=effects,
    )
    result = replace(
        provisional,
        plan_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_shared_reference_composition_v1(result)
    return result


def validate_shared_reference_composition_v1(
    value: SharedReferenceCompositionV1,
) -> None:
    if not isinstance(value, SharedReferenceCompositionV1):
        raise ReferenceMatchContractError(
            "shared reference composition type is invalid"
        )
    if value.schema_id != SHARED_REFERENCE_COMPOSITION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared reference composition schema is invalid"
        )
    for field in (
        "plan_id",
        "staging_verification_id",
        "staging_run_id",
        "authorization_id",
        "numeric_guard_batch_id",
        "operator_id",
        "reference_view_id",
    ):
        _hash(getattr(value, field), field)
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
    ):
        raise ReferenceMatchContractError(
            "shared composition source_count is invalid"
        )
    if value.state != SHARED_REFERENCE_COMPOSITION_STATE:
        raise ReferenceMatchContractError(
            "shared composition state is invalid"
        )
    if (
        value.color_owner != "external-shared-reference-look"
        or value.reference_color_status != "verified-shared-staging"
        or value.claim_ceiling != "reference-look"
    ):
        raise ReferenceMatchContractError(
            "shared composition colour claim is inconsistent"
        )
    if value.film_color_profile_id is not None:
        raise ReferenceMatchContractError(
            "shared composition cannot stack film colour"
        )
    if value.film_stock_identity_claimed is not False:
        raise ReferenceMatchContractError(
            "shared FilmFX cannot claim a film stock identity"
        )
    if value.calibrated_reference_claimed is not False:
        raise ReferenceMatchContractError(
            "shared composition cannot claim calibrated reference"
        )
    if value.film_effects is None:
        expected_label = "reference-look"
        expected_order = ("verified_shared_reference_color",)
    else:
        validate_film_effect_binding(value.film_effects)
        expected_label = "reference-look+film-effects"
        expected_order = (
            "verified_shared_reference_color",
            "film_effects",
        )
    if (
        value.output_label != expected_label
        or value.execution_order != expected_order
    ):
        raise ReferenceMatchContractError(
            "shared composition execution plan is inconsistent"
        )
    if value.plan_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "shared composition plan identity mismatch"
        )


def shared_reference_composition_to_json(
    value: SharedReferenceCompositionV1,
) -> str:
    validate_shared_reference_composition_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_reference_composition_from_json(
    encoded: str,
) -> SharedReferenceCompositionV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared composition is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "shared composition")
    raw_effects = payload["film_effects"]
    if raw_effects is None:
        effects = None
    else:
        raw_effects = _strict(
            raw_effects,
            _EFFECT_KEYS,
            "shared composition film_effects",
        )
        try:
            effects = FilmEffectBinding(**dict(raw_effects))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "shared composition FilmFX fields are invalid"
            ) from exc
    raw_order = payload["execution_order"]
    if not isinstance(raw_order, list):
        raise ReferenceMatchContractError(
            "shared composition execution_order must be an array"
        )
    converted = dict(payload)
    converted["execution_order"] = tuple(raw_order)
    converted["film_effects"] = effects
    try:
        result = SharedReferenceCompositionV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "shared composition fields are invalid"
        ) from exc
    validate_shared_reference_composition_v1(result)
    return result


__all__ = [
    "SHARED_REFERENCE_COMPOSITION_SCHEMA_ID",
    "SHARED_REFERENCE_COMPOSITION_STATE",
    "SharedReferenceCompositionV1",
    "build_shared_reference_composition_v1",
    "shared_reference_composition_from_json",
    "shared_reference_composition_to_json",
    "validate_shared_reference_composition_v1",
]
