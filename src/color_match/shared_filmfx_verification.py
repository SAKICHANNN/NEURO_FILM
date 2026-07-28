"""Restart-safe verification of shared procedural FilmFX staging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .filmfx_staging_io import signed_seed
from .shared_filmfx_transaction import shared_filmfx_run_from_json
from .verification_io import read_hashed_utf8_report, verify_hashed_file


SHARED_FILMFX_VERIFICATION_SCHEMA_ID = (
    "neuro-film.shared-filmfx-staging-verification.v1"
)
SHARED_FILMFX_VERIFICATION_CLAIM_CEILING = (
    "verified-shared-filmfx-staging-not-delivered"
)
_STATE = "verified-shared-filmfx-staging"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "verification_id",
    "run_id",
    "report_path",
    "report_file_sha256",
    "composition_plan_id",
    "staging_verification_id",
    "staging_run_id",
    "authorization_id",
    "numeric_guard_batch_id",
    "operator_id",
    "source_count",
    "seed",
    "state",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "apply_receipt_id",
    "producer_apply_result_id",
    "input_path",
    "input_file_sha256",
    "output_path",
    "output_file_sha256",
    "output_format",
    "output_bit_depth",
    "grain_seed",
    "dust_seed",
}


@dataclass(frozen=True)
class VerifiedSharedFilmFxOutputV1:
    source_index: int
    apply_receipt_id: str
    producer_apply_result_id: str
    input_path: str
    input_file_sha256: str
    output_path: str
    output_file_sha256: str
    output_format: str
    output_bit_depth: int
    grain_seed: int
    dust_seed: int


@dataclass(frozen=True)
class SharedFilmFxStagingVerificationV1:
    schema_id: str
    verification_id: str
    run_id: str
    report_path: str
    report_file_sha256: str
    composition_plan_id: str
    staging_verification_id: str
    staging_run_id: str
    authorization_id: str
    numeric_guard_batch_id: str
    operator_id: str
    source_count: int
    seed: int
    state: str
    outputs: tuple[VerifiedSharedFilmFxOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from shared FilmFX verification"
        )
    return value


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


def _identity_payload(
    value: SharedFilmFxStagingVerificationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("verification_id")
    return payload


def verify_shared_filmfx_staging_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_run_id: str,
) -> SharedFilmFxStagingVerificationV1:
    """Reread and bind one complete P53 run without writing."""

    _hash(expected_run_id, "expected_run_id")
    encoded, report_digest, resolved_report = read_hashed_utf8_report(
        Path(report_path),
        expected_sha256=expected_report_sha256,
        label="shared FilmFX report",
    )
    run = shared_filmfx_run_from_json(encoded)
    if run.run_id != expected_run_id:
        raise ReferenceMatchContractError(
            "shared FilmFX run identity mismatch"
        )
    if (
        str(Path(run.report_path).resolve(strict=False)).casefold()
        != resolved_report.casefold()
    ):
        raise ReferenceMatchContractError(
            "shared FilmFX report path mismatch"
        )
    outputs: list[VerifiedSharedFilmFxOutputV1] = []
    for row in run.outputs:
        input_digest, input_path = verify_hashed_file(
            Path(row.input_path),
            expected_sha256=row.input_file_sha256,
            label=f"shared FilmFX input {row.source_index}",
        )
        output_digest, output_path = verify_hashed_file(
            Path(row.output_path),
            expected_sha256=row.output_file_sha256,
            label=f"shared FilmFX output {row.source_index}",
        )
        outputs.append(
            VerifiedSharedFilmFxOutputV1(
                source_index=row.source_index,
                apply_receipt_id=row.apply_receipt_id,
                producer_apply_result_id=row.producer_apply_result_id,
                input_path=input_path,
                input_file_sha256=input_digest,
                output_path=output_path,
                output_file_sha256=output_digest,
                output_format=row.output_format,
                output_bit_depth=row.output_bit_depth,
                grain_seed=row.grain_seed,
                dust_seed=row.dust_seed,
            )
        )
    provisional = SharedFilmFxStagingVerificationV1(
        schema_id=SHARED_FILMFX_VERIFICATION_SCHEMA_ID,
        verification_id="0" * 64,
        run_id=run.run_id,
        report_path=resolved_report,
        report_file_sha256=report_digest,
        composition_plan_id=run.composition_plan_id,
        staging_verification_id=run.staging_verification_id,
        staging_run_id=run.staging_run_id,
        authorization_id=run.authorization_id,
        numeric_guard_batch_id=run.numeric_guard_batch_id,
        operator_id=run.operator_id,
        source_count=run.source_count,
        seed=run.seed,
        state=_STATE,
        outputs=tuple(outputs),
        claim_ceiling=SHARED_FILMFX_VERIFICATION_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        verification_id=canonical_sha256(_identity_payload(provisional)),
    )
    validate_shared_filmfx_staging_verification_v1(result)
    return result


def validate_shared_filmfx_staging_verification_v1(
    value: SharedFilmFxStagingVerificationV1,
) -> None:
    if not isinstance(value, SharedFilmFxStagingVerificationV1):
        raise ReferenceMatchContractError(
            "shared FilmFX verification type is invalid"
        )
    if value.schema_id != SHARED_FILMFX_VERIFICATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared FilmFX verification schema is invalid"
        )
    for field in (
        "verification_id",
        "run_id",
        "report_file_sha256",
        "composition_plan_id",
        "staging_verification_id",
        "staging_run_id",
        "authorization_id",
        "numeric_guard_batch_id",
        "operator_id",
    ):
        _hash(getattr(value, field), field)
    signed_seed(value.seed, "seed")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "shared FilmFX verification source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "shared FilmFX verification state is invalid"
        )
    if value.claim_ceiling != SHARED_FILMFX_VERIFICATION_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "shared FilmFX verification claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "shared FilmFX verification report path is invalid"
        )
    inputs: set[str] = set()
    outputs: set[str] = set()
    receipts: set[str] = set()
    results: set[str] = set()
    depths: set[int] = set()
    for index, row in enumerate(value.outputs):
        if not isinstance(row, VerifiedSharedFilmFxOutputV1):
            raise ReferenceMatchContractError(
                "shared FilmFX verification output type is invalid"
            )
        if row.source_index != index:
            raise ReferenceMatchContractError(
                "shared FilmFX verification source order is invalid"
            )
        _hash(row.apply_receipt_id, "output.apply_receipt_id")
        _producer_hash(
            row.producer_apply_result_id,
            "output.producer_apply_result_id",
        )
        _hash(row.input_file_sha256, "output.input_file_sha256")
        _hash(row.output_file_sha256, "output.output_file_sha256")
        signed_seed(row.grain_seed, "output.grain_seed")
        signed_seed(row.dust_seed, "output.dust_seed")
        if row.grain_seed != value.seed + index * 1009:
            raise ReferenceMatchContractError(
                "shared FilmFX verification grain seed mismatch"
            )
        if row.dust_seed != row.grain_seed + 17:
            raise ReferenceMatchContractError(
                "shared FilmFX verification dust seed mismatch"
            )
        if (
            not isinstance(row.input_path, str)
            or not row.input_path
            or not isinstance(row.output_path, str)
            or not row.output_path
        ):
            raise ReferenceMatchContractError(
                "shared FilmFX verification paths are invalid"
            )
        if row.output_format not in {"PNG", "JPEG", "TIFF"}:
            raise ReferenceMatchContractError(
                "shared FilmFX verification format is invalid"
            )
        if row.output_bit_depth not in {8, 16}:
            raise ReferenceMatchContractError(
                "shared FilmFX verification depth is invalid"
            )
        input_key = str(Path(row.input_path).resolve(strict=False)).casefold()
        output_key = str(
            Path(row.output_path).resolve(strict=False)
        ).casefold()
        if (
            input_key in inputs
            or output_key in outputs
            or row.apply_receipt_id in receipts
            or row.producer_apply_result_id in results
        ):
            raise ReferenceMatchContractError(
                "shared FilmFX verification identities must be distinct"
            )
        inputs.add(input_key)
        outputs.add(output_key)
        receipts.add(row.apply_receipt_id)
        results.add(row.producer_apply_result_id)
        depths.add(row.output_bit_depth)
    if inputs & outputs:
        raise ReferenceMatchContractError(
            "shared FilmFX verification input/output collision"
        )
    if len(depths) != 1:
        raise ReferenceMatchContractError(
            "shared FilmFX verification depth must be uniform"
        )
    report_key = str(
        Path(value.report_path).resolve(strict=False)
    ).casefold()
    if report_key in inputs | outputs:
        raise ReferenceMatchContractError(
            "shared FilmFX verification report/file collision"
        )
    if value.verification_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "shared FilmFX verification identity mismatch"
        )


def shared_filmfx_verification_to_json(
    value: SharedFilmFxStagingVerificationV1,
) -> str:
    validate_shared_filmfx_staging_verification_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_filmfx_verification_from_json(
    encoded: str,
) -> SharedFilmFxStagingVerificationV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared FilmFX verification is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "shared FilmFX verification")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "shared FilmFX verification outputs must be non-empty"
        )
    rows: list[VerifiedSharedFilmFxOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(
            raw,
            _OUTPUT_KEYS,
            f"shared FilmFX verification output {index}",
        )
        try:
            rows.append(VerifiedSharedFilmFxOutputV1(**dict(raw)))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "shared FilmFX verification output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(rows)
    try:
        result = SharedFilmFxStagingVerificationV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "shared FilmFX verification fields are invalid"
        ) from exc
    validate_shared_filmfx_staging_verification_v1(result)
    return result


__all__ = [
    "SHARED_FILMFX_VERIFICATION_CLAIM_CEILING",
    "SHARED_FILMFX_VERIFICATION_SCHEMA_ID",
    "SharedFilmFxStagingVerificationV1",
    "VerifiedSharedFilmFxOutputV1",
    "shared_filmfx_verification_from_json",
    "shared_filmfx_verification_to_json",
    "validate_shared_filmfx_staging_verification_v1",
    "verify_shared_filmfx_staging_v1",
]
