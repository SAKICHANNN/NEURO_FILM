"""Restart-safe verification of one committed local delivery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from src.inference import sha256_file

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .external_local_delivery import (
    ExternalLocalDeliveryV1,
    external_local_delivery_from_json,
)


EXTERNAL_LOCAL_DELIVERY_VERIFICATION_SCHEMA_ID = (
    "neuro-film.external-local-delivery-verification.v1"
)
EXTERNAL_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING = (
    "verified-local-files-reference-look"
)
_STATE = "verified-local-delivery"
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_HASH = re.compile(r"^[0-9a-f]{64}$")
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
    "staging_output_path",
    "staging_output_file_sha256",
    "delivered_path",
    "delivered_file_sha256",
    "output_format",
    "output_bit_depth",
}


@dataclass(frozen=True)
class VerifiedExternalLocalDeliveryOutputV1:
    source_index: int
    staging_output_path: str
    staging_output_file_sha256: str
    delivered_path: str
    delivered_file_sha256: str
    output_format: str
    output_bit_depth: int


@dataclass(frozen=True)
class ExternalLocalDeliveryVerificationV1:
    schema_id: str
    verification_id: str
    delivery_id: str
    report_path: str
    report_file_sha256: str
    authorization_id: str
    filmfx_verification_id: str
    source_count: int
    state: str
    outputs: tuple[VerifiedExternalLocalDeliveryOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


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
    value: ExternalLocalDeliveryVerificationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("verification_id")
    return payload


def _read_report(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[ExternalLocalDeliveryV1, str]:
    _sha256(expected_sha256, "expected_report_sha256")
    if not path.is_file():
        raise ReferenceMatchContractError(
            "local delivery report must be an existing file"
        )
    try:
        size = path.stat().st_size
        raw = path.read_bytes()
    except OSError as exc:
        raise ReferenceMatchContractError(
            "local delivery report is unreadable"
        ) from exc
    if size <= 0 or size > _MAX_REPORT_BYTES or len(raw) != size:
        raise ReferenceMatchContractError(
            "local delivery report violates the bounded size contract"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ReferenceMatchContractError(
            "local delivery report hash mismatch"
        )
    try:
        encoded = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ReferenceMatchContractError(
            "local delivery report must be UTF-8"
        ) from exc
    return external_local_delivery_from_json(encoded), digest


def _verified_file(
    path_value: str,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[str, str]:
    path = Path(path_value)
    if not path.is_file():
        raise ReferenceMatchContractError(f"{label} is missing")
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise ReferenceMatchContractError(f"{label} is unreadable") from exc
    if digest != expected_sha256:
        raise ReferenceMatchContractError(f"{label} hash mismatch")
    return str(path.resolve(strict=True)), digest


def verify_external_local_delivery_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_delivery_id: str,
) -> ExternalLocalDeliveryVerificationV1:
    """Reread and verify a P39 report plus source and delivered files."""

    _sha256(expected_delivery_id, "expected_delivery_id")
    report = Path(report_path)
    delivery, report_digest = _read_report(
        report,
        expected_sha256=expected_report_sha256,
    )
    if delivery.delivery_id != expected_delivery_id:
        raise ReferenceMatchContractError(
            "local delivery identity mismatch"
        )
    resolved_report = str(report.resolve(strict=True))
    if (
        str(Path(delivery.report_path).resolve(strict=False)).casefold()
        != resolved_report.casefold()
    ):
        raise ReferenceMatchContractError(
            "local delivery report path mismatch"
        )
    outputs: list[VerifiedExternalLocalDeliveryOutputV1] = []
    for row in delivery.outputs:
        staging_path, staging_digest = _verified_file(
            row.staging_output_path,
            row.staging_output_file_sha256,
            label=f"local delivery staging source {row.source_index}",
        )
        delivered_path, delivered_digest = _verified_file(
            row.delivered_path,
            row.delivered_file_sha256,
            label=f"local delivered output {row.source_index}",
        )
        outputs.append(
            VerifiedExternalLocalDeliveryOutputV1(
                source_index=row.source_index,
                staging_output_path=staging_path,
                staging_output_file_sha256=staging_digest,
                delivered_path=delivered_path,
                delivered_file_sha256=delivered_digest,
                output_format=row.output_format,
                output_bit_depth=row.output_bit_depth,
            )
        )
    provisional = ExternalLocalDeliveryVerificationV1(
        schema_id=EXTERNAL_LOCAL_DELIVERY_VERIFICATION_SCHEMA_ID,
        verification_id="0" * 64,
        delivery_id=delivery.delivery_id,
        report_path=resolved_report,
        report_file_sha256=report_digest,
        authorization_id=delivery.authorization_id,
        filmfx_verification_id=delivery.filmfx_verification_id,
        source_count=delivery.source_count,
        state=_STATE,
        outputs=tuple(outputs),
        claim_ceiling=(
            EXTERNAL_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING
        ),
    )
    result = replace(
        provisional,
        verification_id=canonical_sha256(
            _identity_payload(provisional)
        ),
    )
    validate_external_local_delivery_verification_v1(result)
    return result


def validate_external_local_delivery_verification_v1(
    value: ExternalLocalDeliveryVerificationV1,
) -> None:
    if not isinstance(value, ExternalLocalDeliveryVerificationV1):
        raise ReferenceMatchContractError(
            "local delivery verification type is invalid"
        )
    if value.schema_id != EXTERNAL_LOCAL_DELIVERY_VERIFICATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported local delivery verification schema"
        )
    _sha256(value.verification_id, "verification_id")
    _sha256(value.delivery_id, "delivery_id")
    _sha256(value.report_file_sha256, "report_file_sha256")
    _sha256(value.authorization_id, "authorization_id")
    _sha256(value.filmfx_verification_id, "filmfx_verification_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "local delivery verification source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "local delivery verification state is unsupported"
        )
    if (
        value.claim_ceiling
        != EXTERNAL_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "local delivery verification claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "local delivery verification report path is invalid"
        )
    staging_paths: list[str] = []
    delivered_paths: list[str] = []
    for index, output in enumerate(value.outputs):
        if not isinstance(output, VerifiedExternalLocalDeliveryOutputV1):
            raise ReferenceMatchContractError(
                "local delivery verification output type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "local delivery verification indices must be contiguous"
            )
        _sha256(
            output.staging_output_file_sha256,
            "output.staging_output_file_sha256",
        )
        _sha256(
            output.delivered_file_sha256,
            "output.delivered_file_sha256",
        )
        if (
            output.staging_output_file_sha256
            != output.delivered_file_sha256
        ):
            raise ReferenceMatchContractError(
                "local delivery verification byte identity mismatch"
            )
        if output.output_format not in {"PNG", "JPEG", "TIFF"}:
            raise ReferenceMatchContractError(
                "local delivery verification format is unsupported"
            )
        if output.output_bit_depth not in {8, 16}:
            raise ReferenceMatchContractError(
                "local delivery verification bit depth is unsupported"
            )
        if (
            not isinstance(output.staging_output_path, str)
            or not output.staging_output_path
            or not isinstance(output.delivered_path, str)
            or not output.delivered_path
        ):
            raise ReferenceMatchContractError(
                "local delivery verification paths are invalid"
            )
        staging_paths.append(
            str(
                Path(output.staging_output_path).resolve(strict=False)
            ).casefold()
        )
        delivered_paths.append(
            str(Path(output.delivered_path).resolve(strict=False)).casefold()
        )
    if len(set(staging_paths)) != len(staging_paths):
        raise ReferenceMatchContractError(
            "local delivery verification staging paths must be unique"
        )
    if len(set(delivered_paths)) != len(delivered_paths):
        raise ReferenceMatchContractError(
            "local delivery verification delivered paths must be unique"
        )
    if set(staging_paths) & set(delivered_paths):
        raise ReferenceMatchContractError(
            "local delivery verification staging/delivery collision"
        )
    report_key = str(
        Path(value.report_path).resolve(strict=False)
    ).casefold()
    if report_key in set(staging_paths) | set(delivered_paths):
        raise ReferenceMatchContractError(
            "local delivery verification report/file collision"
        )
    if value.verification_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "local delivery verification_id mismatch"
        )


def external_local_delivery_verification_to_json(
    value: ExternalLocalDeliveryVerificationV1,
) -> str:
    validate_external_local_delivery_verification_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def external_local_delivery_verification_from_json(
    encoded: str,
) -> ExternalLocalDeliveryVerificationV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "local delivery verification is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "local delivery verification")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "local delivery verification outputs must be non-empty"
        )
    outputs: list[VerifiedExternalLocalDeliveryOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(
            raw,
            _OUTPUT_KEYS,
            f"local delivery verification output {index}",
        )
        try:
            outputs.append(VerifiedExternalLocalDeliveryOutputV1(**raw))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "local delivery verification output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        value = ExternalLocalDeliveryVerificationV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "local delivery verification fields are invalid"
        ) from exc
    validate_external_local_delivery_verification_v1(value)
    return value


__all__ = [
    "EXTERNAL_LOCAL_DELIVERY_VERIFICATION_CLAIM_CEILING",
    "EXTERNAL_LOCAL_DELIVERY_VERIFICATION_SCHEMA_ID",
    "ExternalLocalDeliveryVerificationV1",
    "VerifiedExternalLocalDeliveryOutputV1",
    "external_local_delivery_verification_from_json",
    "external_local_delivery_verification_to_json",
    "validate_external_local_delivery_verification_v1",
    "verify_external_local_delivery_v1",
]
