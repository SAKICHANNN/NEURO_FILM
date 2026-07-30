"""Atomic local export of an exactly authorized shared FilmFX run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping
import uuid

from src.inference import atomic_write_json, sha256_file

from .strict_json import strict_json_loads
from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .files import _commit_staged_batch, _stage_path
from .shared_composition import SharedReferenceCompositionV1
from .shared_delivery_authorization import (
    SharedLocalDeliveryAuthorizationV1,
    authorize_shared_local_delivery_v1,
    validate_shared_local_delivery_authorization_v1,
)
from .shared_filmfx_verification import SharedFilmFxStagingVerificationV1
from .shared_product_authorization import (
    SharedProductStagingAuthorizationV1,
)
from .shared_staging_verification import (
    ExternalSharedStagingVerificationV1,
)
from .staging_io import (
    staging_output_paths,
    validate_sdr_staging_destinations,
)


SHARED_LOCAL_DELIVERY_SCHEMA_ID = "neuro-film.shared-local-delivery.v1"
SHARED_LOCAL_DELIVERY_CLAIM_CEILING = (
    "local-files-delivered-shared-reference-look"
)
_STATE = "committed-shared-local-delivery"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_FORMAT_EXTENSIONS = {
    "PNG": {".png"},
    "JPEG": {".jpg", ".jpeg"},
    "TIFF": {".tif", ".tiff"},
}
_KEYS = {
    "schema_id",
    "delivery_id",
    "authorization_id",
    "filmfx_verification_id",
    "source_count",
    "state",
    "outputs",
    "report_path",
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
class SharedLocalDeliveryOutputV1:
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
class SharedLocalDeliveryV1:
    schema_id: str
    delivery_id: str
    authorization_id: str
    filmfx_verification_id: str
    source_count: int
    state: str
    outputs: tuple[SharedLocalDeliveryOutputV1, ...]
    report_path: str
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class CommittedSharedLocalDeliveryV1:
    delivery: SharedLocalDeliveryV1
    report_file_sha256: str


def _hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _HASH.fullmatch(value.removeprefix("sha256:")) is None
    ):
        raise ReferenceMatchContractError(f"{label} is not a SHA-256")
    return value


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from shared local delivery"
        )
    return value


def _identity_payload(value: SharedLocalDeliveryV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("delivery_id")
    return payload


def _copy_exact(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle:
        with destination.open("wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())


def _protect_sources(
    outputs: tuple[Path, ...],
    report: Path,
    *,
    filmfx: SharedFilmFxStagingVerificationV1,
    staging: ExternalSharedStagingVerificationV1,
) -> None:
    protected = {
        str(Path(filmfx.report_path).resolve(strict=True)).casefold(),
        str(Path(staging.report_path).resolve(strict=True)).casefold(),
    }
    for row in filmfx.outputs:
        protected.add(str(Path(row.input_path).resolve(strict=True)).casefold())
        protected.add(
            str(Path(row.output_path).resolve(strict=True)).casefold()
        )
    destinations = {
        str(path.resolve(strict=False)).casefold() for path in outputs
    }
    destinations.add(str(report.resolve(strict=False)).casefold())
    if protected & destinations:
        raise ReferenceMatchContractError(
            "shared local delivery destinations must not overwrite staging"
        )


def commit_shared_local_delivery_v1(
    *,
    authorization: SharedLocalDeliveryAuthorizationV1,
    filmfx_verification: SharedFilmFxStagingVerificationV1,
    composition: SharedReferenceCompositionV1,
    staging_verification: ExternalSharedStagingVerificationV1,
    product_authorization: SharedProductStagingAuthorizationV1,
    output_paths: Iterable[Path | str],
    report_path: Path | str,
) -> CommittedSharedLocalDeliveryV1:
    """Commit all exact shared-look files and report atomically."""

    validate_shared_local_delivery_authorization_v1(authorization)
    refreshed = authorize_shared_local_delivery_v1(
        filmfx_verification=filmfx_verification,
        composition=composition,
        staging_verification=staging_verification,
        product_authorization=product_authorization,
    )
    if refreshed != authorization:
        raise ReferenceMatchContractError(
            "shared local delivery authorization changed on refresh"
        )
    outputs = staging_output_paths(
        output_paths,
        count=filmfx_verification.source_count,
    )
    report = Path(report_path)
    depths = {row.output_bit_depth for row in filmfx_verification.outputs}
    if len(depths) != 1:
        raise ReferenceMatchContractError(
            "shared local delivery requires one output bit depth"
        )
    depth = next(iter(depths))
    for path, row in zip(outputs, filmfx_verification.outputs, strict=True):
        if path.suffix.casefold() not in _FORMAT_EXTENSIONS[row.output_format]:
            raise ReferenceMatchContractError(
                "shared local delivery extension must match staged format"
            )
    validate_sdr_staging_destinations(
        outputs,
        report,
        depth,
        label="shared local delivery",
    )
    _protect_sources(
        outputs,
        report,
        filmfx=filmfx_verification,
        staging=staging_verification,
    )
    token = uuid.uuid4().hex
    staged: list[Path] = []
    prepared: list[SharedLocalDeliveryOutputV1] = []
    try:
        for row, output in zip(
            filmfx_verification.outputs,
            outputs,
            strict=True,
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            stage = _stage_path(output, token)
            staged.append(stage)
            _copy_exact(Path(row.output_path), stage)
            digest = sha256_file(stage)
            if digest != row.output_file_sha256:
                raise ReferenceMatchContractError(
                    f"shared local delivery source {row.source_index} changed"
                )
            prepared.append(
                SharedLocalDeliveryOutputV1(
                    source_index=row.source_index,
                    apply_receipt_id=row.apply_receipt_id,
                    producer_apply_result_id=row.producer_apply_result_id,
                    staging_output_path=row.output_path,
                    staging_output_file_sha256=row.output_file_sha256,
                    delivered_path=str(output.resolve(strict=False)),
                    delivered_file_sha256=digest,
                    output_format=row.output_format,
                    output_bit_depth=row.output_bit_depth,
                )
            )
        provisional = SharedLocalDeliveryV1(
            schema_id=SHARED_LOCAL_DELIVERY_SCHEMA_ID,
            delivery_id="0" * 64,
            authorization_id=authorization.authorization_id,
            filmfx_verification_id=filmfx_verification.verification_id,
            source_count=filmfx_verification.source_count,
            state=_STATE,
            outputs=tuple(prepared),
            report_path=str(report.resolve(strict=False)),
            claim_ceiling=SHARED_LOCAL_DELIVERY_CLAIM_CEILING,
        )
        delivery = replace(
            provisional,
            delivery_id=canonical_sha256(_identity_payload(provisional)),
        )
        validate_shared_local_delivery_v1(delivery)
        report.parent.mkdir(parents=True, exist_ok=True)
        staged_report = _stage_path(report, token)
        staged.append(staged_report)
        atomic_write_json(staged_report, delivery.to_dict())
        report_digest = sha256_file(staged_report)
        pairs = tuple(
            (_stage_path(output, token), output) for output in outputs
        ) + ((staged_report, report),)
        _commit_staged_batch(pairs, token=token, cleanup=staged)
        return CommittedSharedLocalDeliveryV1(
            delivery=delivery,
            report_file_sha256=report_digest,
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def validate_shared_local_delivery_v1(value: SharedLocalDeliveryV1) -> None:
    if not isinstance(value, SharedLocalDeliveryV1):
        raise ReferenceMatchContractError("shared local delivery type invalid")
    if value.schema_id != SHARED_LOCAL_DELIVERY_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared local delivery schema invalid"
        )
    _hash(value.delivery_id, "delivery_id")
    _hash(value.authorization_id, "authorization_id")
    _hash(value.filmfx_verification_id, "filmfx_verification_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "shared local delivery source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "shared local delivery state is invalid"
        )
    if value.claim_ceiling != SHARED_LOCAL_DELIVERY_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "shared local delivery claim ceiling mismatch"
        )
    staged: set[str] = set()
    delivered: set[str] = set()
    receipts: set[str] = set()
    results: set[str] = set()
    for index, row in enumerate(value.outputs):
        if not isinstance(row, SharedLocalDeliveryOutputV1):
            raise ReferenceMatchContractError(
                "shared local delivery output type invalid"
            )
        if row.source_index != index:
            raise ReferenceMatchContractError(
                "shared local delivery source order invalid"
            )
        _hash(row.apply_receipt_id, "apply_receipt_id")
        _hash(row.producer_apply_result_id, "producer_apply_result_id")
        _hash(row.staging_output_file_sha256, "staging hash")
        _hash(row.delivered_file_sha256, "delivered hash")
        if row.delivered_file_sha256 != row.staging_output_file_sha256:
            raise ReferenceMatchContractError(
                "shared local delivery must preserve exact bytes"
            )
        if (
            row.output_format not in _FORMAT_EXTENSIONS
            or row.output_bit_depth not in {8, 16}
            or Path(row.delivered_path).suffix.casefold()
            not in _FORMAT_EXTENSIONS[row.output_format]
        ):
            raise ReferenceMatchContractError(
                "shared local delivery output format invalid"
            )
        source_key = str(
            Path(row.staging_output_path).resolve(strict=False)
        ).casefold()
        target_key = str(
            Path(row.delivered_path).resolve(strict=False)
        ).casefold()
        if (
            source_key in staged
            or target_key in delivered
            or row.apply_receipt_id in receipts
            or row.producer_apply_result_id in results
        ):
            raise ReferenceMatchContractError(
                "shared local delivery identities must be distinct"
            )
        staged.add(source_key)
        delivered.add(target_key)
        receipts.add(row.apply_receipt_id)
        results.add(row.producer_apply_result_id)
    if staged & delivered:
        raise ReferenceMatchContractError(
            "shared local delivery cannot overwrite staging"
        )
    report_key = str(
        Path(value.report_path).resolve(strict=False)
    ).casefold()
    if report_key in staged | delivered:
        raise ReferenceMatchContractError(
            "shared local delivery report/file collision"
        )
    if value.delivery_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "shared local delivery identity mismatch"
        )


def shared_local_delivery_to_json(value: SharedLocalDeliveryV1) -> str:
    validate_shared_local_delivery_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_local_delivery_from_json(encoded: str) -> SharedLocalDeliveryV1:
    try:
        payload = strict_json_loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared local delivery is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "shared local delivery")
    raw = payload["outputs"]
    if not isinstance(raw, list) or not raw:
        raise ReferenceMatchContractError(
            "shared local delivery outputs must be non-empty"
        )
    rows = []
    for index, item in enumerate(raw):
        item = _strict(
            item,
            _OUTPUT_KEYS,
            f"shared local delivery output {index}",
        )
        rows.append(SharedLocalDeliveryOutputV1(**dict(item)))
    converted = dict(payload)
    converted["outputs"] = tuple(rows)
    result = SharedLocalDeliveryV1(**converted)
    validate_shared_local_delivery_v1(result)
    return result


__all__ = [
    "SHARED_LOCAL_DELIVERY_CLAIM_CEILING",
    "SHARED_LOCAL_DELIVERY_SCHEMA_ID",
    "CommittedSharedLocalDeliveryV1",
    "SharedLocalDeliveryOutputV1",
    "SharedLocalDeliveryV1",
    "commit_shared_local_delivery_v1",
    "shared_local_delivery_from_json",
    "shared_local_delivery_to_json",
    "validate_shared_local_delivery_v1",
]
