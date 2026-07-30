"""Atomic local-user export of an exactly authorized FilmFX staging run."""

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
from .core_product_authorization import CoreProductStagingAuthorizationV1
from .staging_io import (
    staging_output_paths,
    validate_sdr_staging_destinations,
)
from .core_staging_verification import ExternalCoreStagingVerificationV1
from .external_composition import ExternalReferenceCompositionV1
from .external_delivery_authorization import (
    ExternalLocalDeliveryAuthorizationV1,
    authorize_external_local_delivery_v1,
    validate_external_local_delivery_authorization_v1,
)
from .external_filmfx_verification import (
    ExternalFilmFxStagingVerificationV1,
)
from .files import _commit_staged_batch, _stage_path


EXTERNAL_LOCAL_DELIVERY_SCHEMA_ID = (
    "neuro-film.external-local-delivery.v1"
)
EXTERNAL_LOCAL_DELIVERY_CLAIM_CEILING = (
    "local-files-delivered-reference-look"
)
_STATE = "committed-local-delivery"
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
    "staging_output_path",
    "staging_output_file_sha256",
    "delivered_path",
    "delivered_file_sha256",
    "output_format",
    "output_bit_depth",
}


@dataclass(frozen=True)
class ExternalLocalDeliveryOutputV1:
    source_index: int
    staging_output_path: str
    staging_output_file_sha256: str
    delivered_path: str
    delivered_file_sha256: str
    output_format: str
    output_bit_depth: int


@dataclass(frozen=True)
class ExternalLocalDeliveryV1:
    schema_id: str
    delivery_id: str
    authorization_id: str
    filmfx_verification_id: str
    source_count: int
    state: str
    outputs: tuple[ExternalLocalDeliveryOutputV1, ...]
    report_path: str
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class CommittedExternalLocalDeliveryV1:
    delivery: ExternalLocalDeliveryV1
    report_file_sha256: str


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


def _identity_payload(value: ExternalLocalDeliveryV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("delivery_id")
    return payload


def _protect_staging(
    outputs: tuple[Path, ...],
    report: Path,
    *,
    filmfx: ExternalFilmFxStagingVerificationV1,
    core: ExternalCoreStagingVerificationV1,
) -> None:
    protected = {
        str(Path(filmfx.report_path).resolve(strict=True)).casefold(),
        str(Path(core.report_path).resolve(strict=True)).casefold(),
    }
    for row in filmfx.outputs:
        protected.add(
            str(Path(row.input_path).resolve(strict=True)).casefold()
        )
        protected.add(
            str(Path(row.output_path).resolve(strict=True)).casefold()
        )
    destinations = {
        str(path.resolve(strict=False)).casefold() for path in outputs
    }
    destinations.add(str(report.resolve(strict=False)).casefold())
    if protected & destinations:
        raise ReferenceMatchContractError(
            "local delivery destinations must not overwrite staging"
        )


def _copy_exact(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle:
        with destination.open("wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())


def commit_external_local_delivery_v1(
    *,
    authorization: ExternalLocalDeliveryAuthorizationV1,
    filmfx_verification: ExternalFilmFxStagingVerificationV1,
    composition: ExternalReferenceCompositionV1,
    core_verification: ExternalCoreStagingVerificationV1,
    product_authorization: CoreProductStagingAuthorizationV1,
    output_paths: Iterable[Path | str],
    report_path: Path | str,
) -> CommittedExternalLocalDeliveryV1:
    """Atomically copy all authorized files and one report to local targets."""

    validate_external_local_delivery_authorization_v1(authorization)
    refreshed = authorize_external_local_delivery_v1(
        filmfx_verification=filmfx_verification,
        composition=composition,
        core_verification=core_verification,
        product_authorization=product_authorization,
    )
    if refreshed != authorization:
        raise ReferenceMatchContractError(
            "local delivery authorization changed on refresh"
        )
    outputs = staging_output_paths(
        output_paths,
        count=filmfx_verification.source_count,
    )
    report = Path(report_path)
    bit_depths = {
        row.output_bit_depth for row in filmfx_verification.outputs
    }
    if len(bit_depths) != 1:
        raise ReferenceMatchContractError(
            "local delivery requires one output bit depth"
        )
    output_bit_depth = next(iter(bit_depths))
    for path, row in zip(
        outputs,
        filmfx_verification.outputs,
        strict=True,
    ):
        if path.suffix.casefold() not in _FORMAT_EXTENSIONS[row.output_format]:
            raise ReferenceMatchContractError(
                "local delivery extension must match staged format"
            )
    validate_sdr_staging_destinations(
        outputs,
        report,
        output_bit_depth,
        label="external staging",
    )
    _protect_staging(
        outputs,
        report,
        filmfx=filmfx_verification,
        core=core_verification,
    )

    token = uuid.uuid4().hex
    staged: list[Path] = []
    prepared: list[ExternalLocalDeliveryOutputV1] = []
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
                    f"local delivery source {row.source_index} changed"
                )
            prepared.append(
                ExternalLocalDeliveryOutputV1(
                    source_index=row.source_index,
                    staging_output_path=row.output_path,
                    staging_output_file_sha256=row.output_file_sha256,
                    delivered_path=str(output.resolve(strict=False)),
                    delivered_file_sha256=digest,
                    output_format=row.output_format,
                    output_bit_depth=row.output_bit_depth,
                )
            )
        provisional = ExternalLocalDeliveryV1(
            schema_id=EXTERNAL_LOCAL_DELIVERY_SCHEMA_ID,
            delivery_id="0" * 64,
            authorization_id=authorization.authorization_id,
            filmfx_verification_id=filmfx_verification.verification_id,
            source_count=filmfx_verification.source_count,
            state=_STATE,
            outputs=tuple(prepared),
            report_path=str(report.resolve(strict=False)),
            claim_ceiling=EXTERNAL_LOCAL_DELIVERY_CLAIM_CEILING,
        )
        delivery = replace(
            provisional,
            delivery_id=canonical_sha256(
                _identity_payload(provisional)
            ),
        )
        validate_external_local_delivery_v1(delivery)
        report.parent.mkdir(parents=True, exist_ok=True)
        staged_report = _stage_path(report, token)
        staged.append(staged_report)
        atomic_write_json(staged_report, delivery.to_dict())
        report_digest = sha256_file(staged_report)
        pairs = tuple(
            (_stage_path(output, token), output) for output in outputs
        ) + ((staged_report, report),)
        _commit_staged_batch(pairs, token=token, cleanup=staged)
        return CommittedExternalLocalDeliveryV1(
            delivery=delivery,
            report_file_sha256=report_digest,
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def validate_external_local_delivery_v1(
    value: ExternalLocalDeliveryV1,
) -> None:
    if not isinstance(value, ExternalLocalDeliveryV1):
        raise ReferenceMatchContractError(
            "local delivery type is invalid"
        )
    if value.schema_id != EXTERNAL_LOCAL_DELIVERY_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported local delivery schema"
        )
    _sha256(value.delivery_id, "delivery_id")
    _sha256(value.authorization_id, "authorization_id")
    _sha256(value.filmfx_verification_id, "filmfx_verification_id")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "local delivery source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "local delivery state is unsupported"
        )
    if value.claim_ceiling != EXTERNAL_LOCAL_DELIVERY_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "local delivery claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "local delivery report path is invalid"
        )
    staging_paths: list[str] = []
    delivered_paths: list[str] = []
    for index, output in enumerate(value.outputs):
        if not isinstance(output, ExternalLocalDeliveryOutputV1):
            raise ReferenceMatchContractError(
                "local delivery output type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "local delivery indices must be contiguous"
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
            output.delivered_file_sha256
            != output.staging_output_file_sha256
        ):
            raise ReferenceMatchContractError(
                "local delivery must preserve exact file bytes"
            )
        if output.output_format not in _FORMAT_EXTENSIONS:
            raise ReferenceMatchContractError(
                "local delivery format is unsupported"
            )
        if output.output_bit_depth not in {8, 16}:
            raise ReferenceMatchContractError(
                "local delivery bit depth is unsupported"
            )
        if (
            not isinstance(output.staging_output_path, str)
            or not output.staging_output_path
            or not isinstance(output.delivered_path, str)
            or not output.delivered_path
            or Path(output.delivered_path).suffix.casefold()
            not in _FORMAT_EXTENSIONS[output.output_format]
        ):
            raise ReferenceMatchContractError(
                "local delivery paths are invalid"
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
            "local delivery staging paths must be unique"
        )
    if len(set(delivered_paths)) != len(delivered_paths):
        raise ReferenceMatchContractError(
            "local delivery destinations must be unique"
        )
    if set(staging_paths) & set(delivered_paths):
        raise ReferenceMatchContractError(
            "local delivery cannot overwrite staging"
        )
    report_key = str(
        Path(value.report_path).resolve(strict=False)
    ).casefold()
    if report_key in set(staging_paths) | set(delivered_paths):
        raise ReferenceMatchContractError(
            "local delivery report/file collision"
        )
    if value.delivery_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "local delivery_id mismatch"
        )


def external_local_delivery_to_json(
    value: ExternalLocalDeliveryV1,
) -> str:
    validate_external_local_delivery_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def external_local_delivery_from_json(
    encoded: str,
) -> ExternalLocalDeliveryV1:
    try:
        payload = strict_json_loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "local delivery is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "local delivery")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "local delivery outputs must be non-empty"
        )
    outputs: list[ExternalLocalDeliveryOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"local delivery output {index}")
        try:
            outputs.append(ExternalLocalDeliveryOutputV1(**raw))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "local delivery output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        value = ExternalLocalDeliveryV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "local delivery fields are invalid"
        ) from exc
    validate_external_local_delivery_v1(value)
    return value


__all__ = [
    "EXTERNAL_LOCAL_DELIVERY_CLAIM_CEILING",
    "EXTERNAL_LOCAL_DELIVERY_SCHEMA_ID",
    "CommittedExternalLocalDeliveryV1",
    "ExternalLocalDeliveryOutputV1",
    "ExternalLocalDeliveryV1",
    "commit_external_local_delivery_v1",
    "external_local_delivery_from_json",
    "external_local_delivery_to_json",
    "validate_external_local_delivery_v1",
]
