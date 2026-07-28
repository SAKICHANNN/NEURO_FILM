"""No-write product authorization for verified local reference-look export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping

from .strict_json import strict_json_loads
from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_product_authorization import (
    CoreProductStagingAuthorizationV1,
    validate_core_product_staging_authorization_v1,
)
from .core_staging_verification import (
    ExternalCoreStagingVerificationV1,
    validate_external_core_staging_verification_v1,
)
from .external_composition import (
    ExternalReferenceCompositionV1,
    validate_external_reference_composition_v1,
)
from .external_filmfx_verification import (
    ExternalFilmFxStagingVerificationV1,
    validate_external_filmfx_staging_verification_v1,
    verify_external_filmfx_staging_v1,
)


EXTERNAL_DELIVERY_AUTHORIZATION_SCHEMA_ID = (
    "neuro-film.external-local-delivery-authorization.v1"
)
EXTERNAL_DELIVERY_AUTHORIZATION_CLAIM_CEILING = (
    "authorized-local-delivery-not-committed"
)
_STATE = "authorized-for-local-delivery"
_SCOPE = "local-user-export"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "authorization_id",
    "filmfx_verification_id",
    "filmfx_run_id",
    "composition_plan_id",
    "core_staging_verification_id",
    "product_staging_authorization_id",
    "reference_intent_id",
    "source_count",
    "delivery_scope",
    "output_label",
    "state",
    "claim_ceiling",
}


@dataclass(frozen=True)
class ExternalLocalDeliveryAuthorizationV1:
    """Immutable permission boundary; it does not write or deliver files."""

    schema_id: str
    authorization_id: str
    filmfx_verification_id: str
    filmfx_run_id: str
    composition_plan_id: str
    core_staging_verification_id: str
    product_staging_authorization_id: str
    reference_intent_id: str
    source_count: int
    delivery_scope: str
    output_label: str
    state: str
    claim_ceiling: str

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
    value: ExternalLocalDeliveryAuthorizationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("authorization_id")
    return payload


def authorize_external_local_delivery_v1(
    *,
    filmfx_verification: ExternalFilmFxStagingVerificationV1,
    composition: ExternalReferenceCompositionV1,
    core_verification: ExternalCoreStagingVerificationV1,
    product_authorization: CoreProductStagingAuthorizationV1,
) -> ExternalLocalDeliveryAuthorizationV1:
    """Authorize a future local export without writing any destination."""

    validate_external_filmfx_staging_verification_v1(
        filmfx_verification
    )
    validate_external_reference_composition_v1(composition)
    validate_external_core_staging_verification_v1(core_verification)
    validate_core_product_staging_authorization_v1(product_authorization)
    refreshed = verify_external_filmfx_staging_v1(
        report_path=filmfx_verification.report_path,
        expected_report_sha256=(
            filmfx_verification.report_file_sha256
        ),
        expected_run_id=filmfx_verification.run_id,
    )
    if refreshed != filmfx_verification:
        raise ReferenceMatchContractError(
            "local delivery FilmFX verification changed on refresh"
        )
    if (
        filmfx_verification.composition_plan_id != composition.plan_id
        or filmfx_verification.staging_verification_id
        != core_verification.verification_id
        or composition.staging_verification_id
        != core_verification.verification_id
        or composition.staging_run_id != core_verification.run_id
        or composition.authorization_id
        != core_verification.authorization_id
        or composition.authorization_id
        != product_authorization.authorization_id
        or composition.reference_intent_id
        != core_verification.reference_intent_id
        or filmfx_verification.source_count
        != composition.source_count
        or filmfx_verification.source_count
        != core_verification.source_count
        or filmfx_verification.source_count
        != product_authorization.source_count
    ):
        raise ReferenceMatchContractError(
            "local delivery authorization chain identity mismatch"
        )
    if (
        product_authorization.state != "authorized-for-staging"
        or any(
            source.action != "authorized-for-staging"
            or source.accepted_for_product_staging is not True
            for source in product_authorization.sources
        )
    ):
        raise ReferenceMatchContractError(
            "local delivery requires fully product-authorized sources"
        )
    if (
        composition.film_effects is None
        or composition.output_label != "reference-look+film-effects"
    ):
        raise ReferenceMatchContractError(
            "local delivery FilmFX chain requires rendered effects"
        )
    provisional = ExternalLocalDeliveryAuthorizationV1(
        schema_id=EXTERNAL_DELIVERY_AUTHORIZATION_SCHEMA_ID,
        authorization_id="0" * 64,
        filmfx_verification_id=filmfx_verification.verification_id,
        filmfx_run_id=filmfx_verification.run_id,
        composition_plan_id=composition.plan_id,
        core_staging_verification_id=core_verification.verification_id,
        product_staging_authorization_id=(
            product_authorization.authorization_id
        ),
        reference_intent_id=composition.reference_intent_id,
        source_count=composition.source_count,
        delivery_scope=_SCOPE,
        output_label=composition.output_label,
        state=_STATE,
        claim_ceiling=EXTERNAL_DELIVERY_AUTHORIZATION_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        authorization_id=canonical_sha256(
            _identity_payload(provisional)
        ),
    )
    validate_external_local_delivery_authorization_v1(result)
    return result


def validate_external_local_delivery_authorization_v1(
    value: ExternalLocalDeliveryAuthorizationV1,
) -> None:
    if not isinstance(value, ExternalLocalDeliveryAuthorizationV1):
        raise ReferenceMatchContractError(
            "local delivery authorization type is invalid"
        )
    if value.schema_id != EXTERNAL_DELIVERY_AUTHORIZATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported local delivery authorization schema"
        )
    _sha256(value.authorization_id, "authorization_id")
    _sha256(value.filmfx_verification_id, "filmfx_verification_id")
    _sha256(value.filmfx_run_id, "filmfx_run_id")
    _sha256(value.composition_plan_id, "composition_plan_id")
    _sha256(
        value.core_staging_verification_id,
        "core_staging_verification_id",
    )
    _sha256(
        value.product_staging_authorization_id,
        "product_staging_authorization_id",
    )
    _sha256(value.reference_intent_id, "reference_intent_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
    ):
        raise ReferenceMatchContractError(
            "local delivery authorization source_count is invalid"
        )
    if value.delivery_scope != _SCOPE:
        raise ReferenceMatchContractError(
            "local delivery authorization scope is unsupported"
        )
    if value.output_label != "reference-look+film-effects":
        raise ReferenceMatchContractError(
            "local delivery authorization output label is unsupported"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "local delivery authorization state is unsupported"
        )
    if (
        value.claim_ceiling
        != EXTERNAL_DELIVERY_AUTHORIZATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "local delivery authorization claim ceiling mismatch"
        )
    if value.authorization_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "local delivery authorization_id mismatch"
        )


def external_delivery_authorization_to_json(
    value: ExternalLocalDeliveryAuthorizationV1,
) -> str:
    validate_external_local_delivery_authorization_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def external_delivery_authorization_from_json(
    encoded: str,
) -> ExternalLocalDeliveryAuthorizationV1:
    try:
        payload = strict_json_loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "local delivery authorization is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "local delivery authorization")
    try:
        value = ExternalLocalDeliveryAuthorizationV1(**payload)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "local delivery authorization fields are invalid"
        ) from exc
    validate_external_local_delivery_authorization_v1(value)
    return value


__all__ = [
    "EXTERNAL_DELIVERY_AUTHORIZATION_CLAIM_CEILING",
    "EXTERNAL_DELIVERY_AUTHORIZATION_SCHEMA_ID",
    "ExternalLocalDeliveryAuthorizationV1",
    "authorize_external_local_delivery_v1",
    "external_delivery_authorization_from_json",
    "external_delivery_authorization_to_json",
    "validate_external_local_delivery_authorization_v1",
]
