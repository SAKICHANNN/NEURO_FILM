"""Product staging authorization over verified external-core batch state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping, Sequence

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_acceptance import (
    CoreAcceptanceDecisionV1,
    validate_core_acceptance_decision,
)
from .core_numeric_batch_guard import (
    CoreNumericBatchGuardV1,
    validate_core_numeric_batch_guard_v1,
)
from .dpct_batch import (
    DpctBatchResolutionV1,
    validate_dpct_batch_resolution_v1,
)


CORE_PRODUCT_AUTHORIZATION_SCHEMA_ID = (
    "neuro-film.core-product-staging-authorization.v1"
)
CORE_PRODUCT_AUTHORIZATION_CLAIM_CEILING = (
    "staging-only-not-committed"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_AUTHORIZATION_KEYS = {
    "schema_id",
    "authorization_id",
    "upstream_batch_id",
    "numeric_guard_batch_id",
    "source_count",
    "state",
    "sources",
    "claim_ceiling",
}
_SOURCE_KEYS = {
    "source_index",
    "source_view_id",
    "acceptance_decision_id",
    "admission_id",
    "numeric_decision_id",
    "acceptance_accepted_for_product_guard",
    "core_status",
    "promotion_status",
    "research_baseline_override",
    "accepted_for_product_staging",
    "action",
    "reasons",
}


@dataclass(frozen=True)
class CoreProductAuthorizationSourceV1:
    source_index: int
    source_view_id: str
    acceptance_decision_id: str | None
    admission_id: str | None
    numeric_decision_id: str | None
    acceptance_accepted_for_product_guard: bool | None
    core_status: str | None
    promotion_status: str | None
    research_baseline_override: bool | None
    accepted_for_product_staging: bool | None
    action: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CoreProductStagingAuthorizationV1:
    """Immutable authorization to stage, never to commit or deliver."""

    schema_id: str
    authorization_id: str
    upstream_batch_id: str
    numeric_guard_batch_id: str
    source_count: int
    state: str
    sources: tuple[CoreProductAuthorizationSourceV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = []
        for source in self.sources:
            row = asdict(source)
            row["reasons"] = list(source.reasons)
            payload["sources"].append(row)
        return payload


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


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _optional_hash(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _hash(value, label)


def _identity_payload(
    value: CoreProductStagingAuthorizationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("authorization_id")
    return payload


def authorize_core_product_staging_v1(
    *,
    batch: DpctBatchResolutionV1,
    numeric_guard: CoreNumericBatchGuardV1,
    acceptances: Sequence[CoreAcceptanceDecisionV1] | None = None,
) -> CoreProductStagingAuthorizationV1:
    """Authorize all sources for staging or preserve full fallback."""

    validate_dpct_batch_resolution_v1(batch)
    validate_core_numeric_batch_guard_v1(numeric_guard)
    if (
        numeric_guard.upstream_batch_id != batch.batch_id
        or numeric_guard.source_count != batch.source_count
    ):
        raise ReferenceMatchContractError(
            "numeric guard does not bind the D-PCT batch"
        )
    for index, (batch_row, guard_row) in enumerate(
        zip(batch.sources, numeric_guard.sources, strict=True)
    ):
        if (
            batch_row.source_index != index
            or guard_row.source_index != index
            or guard_row.source_view_id != batch_row.source_view_id
            or guard_row.apply_receipt_id != batch_row.apply_receipt_id
            or guard_row.admission_id != batch_row.admission_id
        ):
            raise ReferenceMatchContractError(
                f"numeric guard source {index} does not bind batch row"
            )

    if numeric_guard.atomic_state == "identity-fallback":
        if acceptances is not None:
            raise ReferenceMatchContractError(
                "numeric fallback cannot authorize acceptances"
            )
        rows = tuple(
            CoreProductAuthorizationSourceV1(
                source_index=row.source_index,
                source_view_id=row.source_view_id,
                acceptance_decision_id=row.acceptance_decision_id,
                admission_id=row.admission_id,
                numeric_decision_id=guard.numeric_decision_id,
                acceptance_accepted_for_product_guard=None,
                core_status=None,
                promotion_status=None,
                research_baseline_override=None,
                accepted_for_product_staging=None,
                action="not-evaluated",
                reasons=("upstream-numeric-fallback",),
            )
            for row, guard in zip(
                batch.sources,
                numeric_guard.sources,
                strict=True,
            )
        )
        state = "identity-fallback"
    else:
        if (
            acceptances is None
            or not isinstance(acceptances, Sequence)
            or isinstance(acceptances, (str, bytes))
            or len(acceptances) != batch.source_count
        ):
            raise ReferenceMatchContractError(
                "numeric-eligible batch requires one acceptance per source"
            )
        built: list[CoreProductAuthorizationSourceV1] = []
        for index, (batch_row, guard_row, acceptance) in enumerate(
            zip(
                batch.sources,
                numeric_guard.sources,
                acceptances,
                strict=True,
            )
        ):
            validate_core_acceptance_decision(acceptance)
            if (
                acceptance.decision_id
                != batch_row.acceptance_decision_id
                or acceptance.transform_id
                != batch_row.consumer_transform_id
                or guard_row.numeric_decision_id is None
                or guard_row.numeric_accepted_for_transaction is not True
            ):
                raise ReferenceMatchContractError(
                    f"acceptance {index} does not bind eligible batch row"
                )
            reasons: list[str] = []
            if not acceptance.accepted_for_product_guard:
                reasons.append("acceptance-not-product-accepted")
            if acceptance.core_status != "ok":
                reasons.append("core-status-not-ok")
            if acceptance.promotion_status != "promoted":
                reasons.append("algorithm-not-promoted")
            if acceptance.research_baseline_override:
                reasons.append("research-baseline-override")
            normalized = tuple(sorted(set(reasons)))
            accepted = not normalized
            built.append(
                CoreProductAuthorizationSourceV1(
                    source_index=index,
                    source_view_id=batch_row.source_view_id,
                    acceptance_decision_id=acceptance.decision_id,
                    admission_id=batch_row.admission_id,
                    numeric_decision_id=guard_row.numeric_decision_id,
                    acceptance_accepted_for_product_guard=(
                        acceptance.accepted_for_product_guard
                    ),
                    core_status=acceptance.core_status,
                    promotion_status=acceptance.promotion_status,
                    research_baseline_override=(
                        acceptance.research_baseline_override
                    ),
                    accepted_for_product_staging=accepted,
                    action=(
                        "authorized-for-staging"
                        if accepted
                        else "identity-fallback"
                    ),
                    reasons=normalized,
                )
            )
        rows = tuple(built)
        state = (
            "authorized-for-staging"
            if all(
                row.accepted_for_product_staging is True for row in rows
            )
            else "identity-fallback"
        )

    provisional = CoreProductStagingAuthorizationV1(
        schema_id=CORE_PRODUCT_AUTHORIZATION_SCHEMA_ID,
        authorization_id="0" * 64,
        upstream_batch_id=batch.batch_id,
        numeric_guard_batch_id=numeric_guard.guard_batch_id,
        source_count=batch.source_count,
        state=state,
        sources=rows,
        claim_ceiling=CORE_PRODUCT_AUTHORIZATION_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        authorization_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_core_product_staging_authorization_v1(result)
    return result


def validate_core_product_staging_authorization_v1(
    value: CoreProductStagingAuthorizationV1,
) -> None:
    if not isinstance(value, CoreProductStagingAuthorizationV1):
        raise ReferenceMatchContractError(
            "product authorization type is invalid"
        )
    if value.schema_id != CORE_PRODUCT_AUTHORIZATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported product authorization schema"
        )
    _hash(value.authorization_id, "authorization_id")
    _hash(value.upstream_batch_id, "upstream_batch_id")
    _hash(value.numeric_guard_batch_id, "numeric_guard_batch_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.sources)
    ):
        raise ReferenceMatchContractError(
            "product authorization source_count mismatch"
        )
    if value.state not in {
        "authorized-for-staging",
        "identity-fallback",
    }:
        raise ReferenceMatchContractError(
            "product authorization state is unsupported"
        )
    if value.claim_ceiling != CORE_PRODUCT_AUTHORIZATION_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "product authorization claim ceiling mismatch"
        )
    accepted: list[bool] = []
    evaluated = False
    for index, source in enumerate(value.sources):
        if not isinstance(source, CoreProductAuthorizationSourceV1):
            raise ReferenceMatchContractError(
                "product authorization source row type is invalid"
            )
        if source.source_index != index:
            raise ReferenceMatchContractError(
                "product authorization source indices must be contiguous"
            )
        _hash(source.source_view_id, "source.source_view_id")
        _optional_hash(
            source.acceptance_decision_id,
            "source.acceptance_decision_id",
        )
        _optional_hash(source.admission_id, "source.admission_id")
        _optional_hash(
            source.numeric_decision_id,
            "source.numeric_decision_id",
        )
        if (
            not isinstance(source.reasons, tuple)
            or source.reasons != tuple(sorted(set(source.reasons)))
            or any(
                not isinstance(reason, str) or not reason
                for reason in source.reasons
            )
        ):
            raise ReferenceMatchContractError(
                "product authorization reasons must be sorted unique strings"
            )
        if source.action == "not-evaluated":
            if (
                source.core_status is not None
                or source.promotion_status is not None
                or source.research_baseline_override is not None
                or source.acceptance_accepted_for_product_guard is not None
                or source.accepted_for_product_staging is not None
                or source.reasons != ("upstream-numeric-fallback",)
            ):
                raise ReferenceMatchContractError(
                    "not-evaluated authorization source is inconsistent"
                )
            continue
        evaluated = True
        if source.action not in {
            "authorized-for-staging",
            "identity-fallback",
        }:
            raise ReferenceMatchContractError(
                "product authorization source action is unsupported"
            )
        if source.acceptance_decision_id is None:
            raise ReferenceMatchContractError(
                "evaluated source must bind an acceptance"
            )
        if not isinstance(
            source.acceptance_accepted_for_product_guard, bool
        ):
            raise ReferenceMatchContractError(
                "product authorization acceptance state must be boolean"
            )
        if source.core_status not in {
            "ok",
            "identity-fallback",
            "unsupported",
            "invalid",
        }:
            raise ReferenceMatchContractError(
                "product authorization core status is unsupported"
            )
        if source.promotion_status not in {
            "rejected",
            "eligible-for-visual-review",
            "promoted",
        }:
            raise ReferenceMatchContractError(
                "product authorization promotion status is unsupported"
            )
        if not isinstance(source.research_baseline_override, bool):
            raise ReferenceMatchContractError(
                "product authorization research override must be boolean"
            )
        if not isinstance(source.accepted_for_product_staging, bool):
            raise ReferenceMatchContractError(
                "product staging acceptance must be boolean"
            )
        expected_reasons: list[str] = []
        if not source.acceptance_accepted_for_product_guard:
            expected_reasons.append("acceptance-not-product-accepted")
        if source.core_status != "ok":
            expected_reasons.append("core-status-not-ok")
        if source.promotion_status != "promoted":
            expected_reasons.append("algorithm-not-promoted")
        if source.research_baseline_override:
            expected_reasons.append("research-baseline-override")
        expected_accepted = (
            source.acceptance_accepted_for_product_guard
            and source.core_status == "ok"
            and source.promotion_status == "promoted"
            and not source.research_baseline_override
            and not source.reasons
        )
        expected_action = (
            "authorized-for-staging"
            if expected_accepted
            else "identity-fallback"
        )
        if (
            source.accepted_for_product_staging != expected_accepted
            or source.action != expected_action
        ):
            raise ReferenceMatchContractError(
                "product authorization source state is inconsistent"
            )
        if source.reasons != tuple(sorted(expected_reasons)):
            raise ReferenceMatchContractError(
                "product authorization source reasons are inconsistent"
            )
        accepted.append(expected_accepted)
    if evaluated and any(
        source.action == "not-evaluated" for source in value.sources
    ):
        raise ReferenceMatchContractError(
            "product authorization cannot mix evaluated states"
        )
    expected_state = (
        "authorized-for-staging"
        if evaluated and all(accepted)
        else "identity-fallback"
    )
    if value.state != expected_state:
        raise ReferenceMatchContractError(
            "product authorization atomic state is inconsistent"
        )
    if value.authorization_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "product authorization_id mismatch"
        )


def core_product_authorization_to_json(
    value: CoreProductStagingAuthorizationV1,
) -> str:
    validate_core_product_staging_authorization_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def core_product_authorization_from_json(
    encoded: str,
) -> CoreProductStagingAuthorizationV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "product authorization is not valid JSON"
        ) from exc
    payload = _strict(payload, _AUTHORIZATION_KEYS, "product authorization")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ReferenceMatchContractError(
            "product authorization sources must be non-empty"
        )
    sources: list[CoreProductAuthorizationSourceV1] = []
    for index, raw in enumerate(raw_sources):
        raw = _strict(raw, _SOURCE_KEYS, f"authorization source {index}")
        converted = dict(raw)
        converted["reasons"] = tuple(converted["reasons"])
        try:
            sources.append(CoreProductAuthorizationSourceV1(**converted))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "product authorization source fields are invalid"
            ) from exc
    try:
        value = CoreProductStagingAuthorizationV1(
            schema_id=payload["schema_id"],
            authorization_id=payload["authorization_id"],
            upstream_batch_id=payload["upstream_batch_id"],
            numeric_guard_batch_id=payload["numeric_guard_batch_id"],
            source_count=payload["source_count"],
            state=payload["state"],
            sources=tuple(sources),
            claim_ceiling=payload["claim_ceiling"],
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "product authorization fields are invalid"
        ) from exc
    validate_core_product_staging_authorization_v1(value)
    return value


__all__ = [
    "CORE_PRODUCT_AUTHORIZATION_CLAIM_CEILING",
    "CORE_PRODUCT_AUTHORIZATION_SCHEMA_ID",
    "CoreProductAuthorizationSourceV1",
    "CoreProductStagingAuthorizationV1",
    "authorize_core_product_staging_v1",
    "core_product_authorization_from_json",
    "core_product_authorization_to_json",
    "validate_core_product_staging_authorization_v1",
]
