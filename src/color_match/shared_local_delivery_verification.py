"""Restart-safe verification of one shared local delivery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .shared_local_delivery import shared_local_delivery_from_json
from .verification_io import read_hashed_utf8_report, verify_hashed_file


SHARED_LOCAL_DELIVERY_VERIFICATION_SCHEMA_ID = (
    "neuro-film.shared-local-delivery-verification.v1"
)
SHARED_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING = (
    "verified-local-files-shared-reference-look"
)
_STATE = "verified-shared-local-delivery"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "verification_id",
    "delivery_id",
    "report_path",
    "report_file_sha256",
    "authorization_id",
    "filmfx_verification_id",
    "source_count",
    "state",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "apply_receipt_id",
    "producer_apply_result_id",
    "staging_output_path",
    "staging_output_file_sha256",
    "delivered_path",
    "delivered_file_sha256",
    "output_format",
    "output_bit_depth",
}


@dataclass(frozen=True)
class VerifiedSharedLocalDeliveryOutputV1:
    source_index: int
    apply_receipt_id: str
    producer_apply_result_id: str
    staging_output_path: str
    staging_output_file_sha256: str
    delivered_path: str
    delivered_file_sha256: str
    output_format: str
    output_bit_depth: int


@dataclass(frozen=True)
class SharedLocalDeliveryVerificationV1:
    schema_id: str
    verification_id: str
    delivery_id: str
    report_path: str
    report_file_sha256: str
    authorization_id: str
    filmfx_verification_id: str
    source_count: int
    state: str
    outputs: tuple[VerifiedSharedLocalDeliveryOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _producer_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _PRODUCER_HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be lowercase sha256:<hex>"
        )
    return value


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from shared delivery verification"
        )
    return value


def _identity_payload(
    value: SharedLocalDeliveryVerificationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("verification_id")
    return payload


def verify_shared_local_delivery_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_delivery_id: str,
) -> SharedLocalDeliveryVerificationV1:
    """Reread and verify a P56 report, staging sources and delivered files."""

    _hash(expected_delivery_id, "expected_delivery_id")
    encoded, report_digest, resolved_report = read_hashed_utf8_report(
        Path(report_path),
        expected_sha256=expected_report_sha256,
        label="shared local delivery report",
    )
    delivery = shared_local_delivery_from_json(encoded)
    if delivery.delivery_id != expected_delivery_id:
        raise ReferenceMatchContractError(
            "shared local delivery identity mismatch"
        )
    if (
        str(Path(delivery.report_path).resolve(strict=False)).casefold()
        != resolved_report.casefold()
    ):
        raise ReferenceMatchContractError(
            "shared local delivery report path mismatch"
        )
    outputs: list[VerifiedSharedLocalDeliveryOutputV1] = []
    for row in delivery.outputs:
        staging_digest, staging_path = verify_hashed_file(
            Path(row.staging_output_path),
            expected_sha256=row.staging_output_file_sha256,
            label=(
                f"shared local delivery staging source {row.source_index}"
            ),
        )
        delivered_digest, delivered_path = verify_hashed_file(
            Path(row.delivered_path),
            expected_sha256=row.delivered_file_sha256,
            label=f"shared local delivered output {row.source_index}",
        )
        outputs.append(
            VerifiedSharedLocalDeliveryOutputV1(
                source_index=row.source_index,
                apply_receipt_id=row.apply_receipt_id,
                producer_apply_result_id=row.producer_apply_result_id,
                staging_output_path=staging_path,
                staging_output_file_sha256=staging_digest,
                delivered_path=delivered_path,
                delivered_file_sha256=delivered_digest,
                output_format=row.output_format,
                output_bit_depth=row.output_bit_depth,
            )
        )
    provisional = SharedLocalDeliveryVerificationV1(
        schema_id=SHARED_LOCAL_DELIVERY_VERIFICATION_SCHEMA_ID,
        verification_id="0" * 64,
        delivery_id=delivery.delivery_id,
        report_path=resolved_report,
        report_file_sha256=report_digest,
        authorization_id=delivery.authorization_id,
        filmfx_verification_id=delivery.filmfx_verification_id,
        source_count=delivery.source_count,
        state=_STATE,
        outputs=tuple(outputs),
        claim_ceiling=SHARED_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        verification_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_shared_local_delivery_verification_v1(result)
    return result


def validate_shared_local_delivery_verification_v1(
    value: SharedLocalDeliveryVerificationV1,
) -> None:
    if not isinstance(value, SharedLocalDeliveryVerificationV1):
        raise ReferenceMatchContractError(
            "shared local delivery verification type is invalid"
        )
    if value.schema_id != SHARED_LOCAL_DELIVERY_VERIFICATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared local delivery verification schema is invalid"
        )
    for field in (
        "verification_id",
        "delivery_id",
        "report_file_sha256",
        "authorization_id",
        "filmfx_verification_id",
    ):
        _hash(getattr(value, field), field)
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "shared local delivery verification source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "shared local delivery verification state is invalid"
        )
    if (
        value.claim_ceiling
        != SHARED_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "shared local delivery verification claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "shared local delivery verification report path is invalid"
        )
    staging_paths: set[str] = set()
    delivered_paths: set[str] = set()
    receipts: set[str] = set()
    results: set[str] = set()
    for index, row in enumerate(value.outputs):
        if not isinstance(row, VerifiedSharedLocalDeliveryOutputV1):
            raise ReferenceMatchContractError(
                "shared local delivery verification output type is invalid"
            )
        if row.source_index != index:
            raise ReferenceMatchContractError(
                "shared local delivery verification source order is invalid"
            )
        _hash(row.apply_receipt_id, "output.apply_receipt_id")
        _producer_hash(
            row.producer_apply_result_id,
            "output.producer_apply_result_id",
        )
        _hash(
            row.staging_output_file_sha256,
            "output.staging_output_file_sha256",
        )
        _hash(
            row.delivered_file_sha256,
            "output.delivered_file_sha256",
        )
        if row.staging_output_file_sha256 != row.delivered_file_sha256:
            raise ReferenceMatchContractError(
                "shared local delivery verification byte identity mismatch"
            )
        if (
            row.output_format not in {"PNG", "JPEG", "TIFF"}
            or row.output_bit_depth not in {8, 16}
            or not row.staging_output_path
            or not row.delivered_path
        ):
            raise ReferenceMatchContractError(
                "shared local delivery verification output is invalid"
            )
        staging_key = str(
            Path(row.staging_output_path).resolve(strict=False)
        ).casefold()
        delivered_key = str(
            Path(row.delivered_path).resolve(strict=False)
        ).casefold()
        if (
            staging_key in staging_paths
            or delivered_key in delivered_paths
            or row.apply_receipt_id in receipts
            or row.producer_apply_result_id in results
        ):
            raise ReferenceMatchContractError(
                "shared local delivery verification identities must be distinct"
            )
        staging_paths.add(staging_key)
        delivered_paths.add(delivered_key)
        receipts.add(row.apply_receipt_id)
        results.add(row.producer_apply_result_id)
    if staging_paths & delivered_paths:
        raise ReferenceMatchContractError(
            "shared local delivery verification path collision"
        )
    report_key = str(
        Path(value.report_path).resolve(strict=False)
    ).casefold()
    if report_key in staging_paths | delivered_paths:
        raise ReferenceMatchContractError(
            "shared local delivery verification report/file collision"
        )
    if value.verification_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "shared local delivery verification identity mismatch"
        )


def shared_local_delivery_verification_to_json(
    value: SharedLocalDeliveryVerificationV1,
) -> str:
    validate_shared_local_delivery_verification_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_local_delivery_verification_from_json(
    encoded: str,
) -> SharedLocalDeliveryVerificationV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared local delivery verification is not valid JSON"
        ) from exc
    payload = _strict(
        payload,
        _KEYS,
        "shared local delivery verification",
    )
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "shared local delivery verification outputs must be non-empty"
        )
    rows: list[VerifiedSharedLocalDeliveryOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(
            raw,
            _OUTPUT_KEYS,
            f"shared local delivery verification output {index}",
        )
        try:
            rows.append(VerifiedSharedLocalDeliveryOutputV1(**dict(raw)))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "shared local delivery verification output fields invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(rows)
    try:
        result = SharedLocalDeliveryVerificationV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "shared local delivery verification fields invalid"
        ) from exc
    validate_shared_local_delivery_verification_v1(result)
    return result


__all__ = [
    "SHARED_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING",
    "SHARED_LOCAL_DELIVERY_VERIFICATION_SCHEMA_ID",
    "SharedLocalDeliveryVerificationV1",
    "VerifiedSharedLocalDeliveryOutputV1",
    "shared_local_delivery_verification_from_json",
    "shared_local_delivery_verification_to_json",
    "validate_shared_local_delivery_verification_v1",
    "verify_shared_local_delivery_v1",
]
