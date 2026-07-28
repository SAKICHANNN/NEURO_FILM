"""No-write local-delivery authorization for a verified shared look."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .shared_composition import (
    SharedReferenceCompositionV1,
    validate_shared_reference_composition_v1,
)
from .shared_filmfx_verification import (
    SharedFilmFxStagingVerificationV1,
    validate_shared_filmfx_staging_verification_v1,
    verify_shared_filmfx_staging_v1,
)
from .shared_product_authorization import (
    SharedProductStagingAuthorizationV1,
    validate_shared_product_staging_authorization_v1,
)
from .shared_staging_verification import (
    ExternalSharedStagingVerificationV1,
    validate_external_shared_staging_verification_v1,
)


SHARED_DELIVERY_AUTHORIZATION_SCHEMA_ID = (
    "neuro-film.shared-local-delivery-authorization.v1"
)
SHARED_DELIVERY_AUTHORIZATION_CLAIM_CEILING = (
    "authorized-shared-local-delivery-not-committed"
)
_STATE = "authorized-for-shared-local-delivery"
_SCOPE = "local-user-export"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "authorization_id",
    "filmfx_verification_id",
    "filmfx_run_id",
    "composition_plan_id",
    "staging_verification_id",
    "staging_run_id",
    "product_staging_authorization_id",
    "numeric_guard_batch_id",
    "operator_id",
    "reference_view_id",
    "source_count",
    "delivery_scope",
    "output_label",
    "state",
    "claim_ceiling",
}


@dataclass(frozen=True)
class SharedLocalDeliveryAuthorizationV1:
    schema_id: str
    authorization_id: str
    filmfx_verification_id: str
    filmfx_run_id: str
    composition_plan_id: str
    staging_verification_id: str
    staging_run_id: str
    product_staging_authorization_id: str
    numeric_guard_batch_id: str
    operator_id: str
    reference_view_id: str
    source_count: int
    delivery_scope: str
    output_label: str
    state: str
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from shared delivery authorization"
        )
    return value


def _identity_payload(
    value: SharedLocalDeliveryAuthorizationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("authorization_id")
    return payload


def authorize_shared_local_delivery_v1(
    *,
    filmfx_verification: SharedFilmFxStagingVerificationV1,
    composition: SharedReferenceCompositionV1,
    staging_verification: ExternalSharedStagingVerificationV1,
    product_authorization: SharedProductStagingAuthorizationV1,
) -> SharedLocalDeliveryAuthorizationV1:
    """Authorize a future local export without creating destinations."""

    validate_shared_filmfx_staging_verification_v1(filmfx_verification)
    validate_shared_reference_composition_v1(composition)
    validate_external_shared_staging_verification_v1(staging_verification)
    validate_shared_product_staging_authorization_v1(
        product_authorization
    )
    refreshed = verify_shared_filmfx_staging_v1(
        report_path=filmfx_verification.report_path,
        expected_report_sha256=filmfx_verification.report_file_sha256,
        expected_run_id=filmfx_verification.run_id,
    )
    if refreshed != filmfx_verification:
        raise ReferenceMatchContractError(
            "shared delivery FilmFX verification changed on refresh"
        )
    if (
        filmfx_verification.composition_plan_id != composition.plan_id
        or filmfx_verification.staging_verification_id
        != staging_verification.verification_id
        or filmfx_verification.staging_run_id
        != staging_verification.run_id
        or filmfx_verification.authorization_id
        != product_authorization.authorization_id
        or composition.staging_verification_id
        != staging_verification.verification_id
        or composition.staging_run_id != staging_verification.run_id
        or composition.authorization_id
        != product_authorization.authorization_id
        or composition.authorization_id
        != staging_verification.authorization_id
        or composition.numeric_guard_batch_id
        != product_authorization.numeric_guard_batch_id
        or composition.numeric_guard_batch_id
        != staging_verification.numeric_guard_batch_id
        or composition.operator_id != product_authorization.operator_id
        or composition.operator_id != staging_verification.operator_id
        or composition.reference_view_id
        != staging_verification.reference_view_id
        or composition.source_count != filmfx_verification.source_count
        or composition.source_count != staging_verification.source_count
        or composition.source_count != product_authorization.source_count
    ):
        raise ReferenceMatchContractError(
            "shared delivery authorization chain identity mismatch"
        )
    if (
        product_authorization.state != "authorized-for-staging"
        or product_authorization.admission_evaluation_ready is not True
        or product_authorization.admission_product_ready is not True
        or any(
            source.action != "authorized-for-staging"
            or source.numeric_accepted_for_transaction is not True
            for source in product_authorization.sources
        )
    ):
        raise ReferenceMatchContractError(
            "shared delivery requires fully product-authorized sources"
        )
    if (
        composition.film_effects is None
        or composition.output_label != "reference-look+film-effects"
        or composition.film_stock_identity_claimed
        or composition.calibrated_reference_claimed
    ):
        raise ReferenceMatchContractError(
            "shared delivery requires a non-calibrated FilmFX composition"
        )
    provisional = SharedLocalDeliveryAuthorizationV1(
        schema_id=SHARED_DELIVERY_AUTHORIZATION_SCHEMA_ID,
        authorization_id="0" * 64,
        filmfx_verification_id=filmfx_verification.verification_id,
        filmfx_run_id=filmfx_verification.run_id,
        composition_plan_id=composition.plan_id,
        staging_verification_id=staging_verification.verification_id,
        staging_run_id=staging_verification.run_id,
        product_staging_authorization_id=(
            product_authorization.authorization_id
        ),
        numeric_guard_batch_id=composition.numeric_guard_batch_id,
        operator_id=composition.operator_id,
        reference_view_id=composition.reference_view_id,
        source_count=composition.source_count,
        delivery_scope=_SCOPE,
        output_label=composition.output_label,
        state=_STATE,
        claim_ceiling=SHARED_DELIVERY_AUTHORIZATION_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        authorization_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_shared_local_delivery_authorization_v1(result)
    return result


def validate_shared_local_delivery_authorization_v1(
    value: SharedLocalDeliveryAuthorizationV1,
) -> None:
    if not isinstance(value, SharedLocalDeliveryAuthorizationV1):
        raise ReferenceMatchContractError(
            "shared delivery authorization type is invalid"
        )
    if value.schema_id != SHARED_DELIVERY_AUTHORIZATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared delivery authorization schema is invalid"
        )
    for field in (
        "authorization_id",
        "filmfx_verification_id",
        "filmfx_run_id",
        "composition_plan_id",
        "staging_verification_id",
        "staging_run_id",
        "product_staging_authorization_id",
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
            "shared delivery authorization source_count is invalid"
        )
    if value.delivery_scope != _SCOPE:
        raise ReferenceMatchContractError(
            "shared delivery authorization scope is invalid"
        )
    if value.output_label != "reference-look+film-effects":
        raise ReferenceMatchContractError(
            "shared delivery authorization output label is invalid"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "shared delivery authorization state is invalid"
        )
    if value.claim_ceiling != SHARED_DELIVERY_AUTHORIZATION_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "shared delivery authorization claim ceiling mismatch"
        )
    if value.authorization_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "shared delivery authorization identity mismatch"
        )


def shared_delivery_authorization_to_json(
    value: SharedLocalDeliveryAuthorizationV1,
) -> str:
    validate_shared_local_delivery_authorization_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_delivery_authorization_from_json(
    encoded: str,
) -> SharedLocalDeliveryAuthorizationV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared delivery authorization is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "shared delivery authorization")
    try:
        result = SharedLocalDeliveryAuthorizationV1(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "shared delivery authorization fields are invalid"
        ) from exc
    validate_shared_local_delivery_authorization_v1(result)
    return result


__all__ = [
    "SHARED_DELIVERY_AUTHORIZATION_CLAIM_CEILING",
    "SHARED_DELIVERY_AUTHORIZATION_SCHEMA_ID",
    "SharedLocalDeliveryAuthorizationV1",
    "authorize_shared_local_delivery_v1",
    "shared_delivery_authorization_from_json",
    "shared_delivery_authorization_to_json",
    "validate_shared_local_delivery_authorization_v1",
]
