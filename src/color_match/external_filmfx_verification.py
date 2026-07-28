"""Restart-safe verification of a committed external FilmFX staging run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from src.inference import sha256_file

from .strict_json import strict_json_loads
from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .external_filmfx_transaction import (
    ExternalFilmFxRunV1,
    external_filmfx_run_from_json,
)


EXTERNAL_FILMFX_VERIFICATION_SCHEMA_ID = (
    "neuro-film.external-filmfx-staging-verification.v1"
)
EXTERNAL_FILMFX_VERIFICATION_CLAIM_CEILING = (
    "verified-filmfx-staging-not-delivered"
)
_STATE = "verified-filmfx-staging"
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "verification_id",
    "run_id",
    "report_path",
    "report_file_sha256",
    "composition_plan_id",
    "staging_verification_id",
    "source_count",
    "seed",
    "state",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
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
class VerifiedExternalFilmFxOutputV1:
    source_index: int
    input_path: str
    input_file_sha256: str
    output_path: str
    output_file_sha256: str
    output_format: str
    output_bit_depth: int
    grain_seed: int
    dust_seed: int


@dataclass(frozen=True)
class ExternalFilmFxStagingVerificationV1:
    """Canonical proof that a P36 report and its complete lineage still match."""

    schema_id: str
    verification_id: str
    run_id: str
    report_path: str
    report_file_sha256: str
    composition_plan_id: str
    staging_verification_id: str
    source_count: int
    seed: int
    state: str
    outputs: tuple[VerifiedExternalFilmFxOutputV1, ...]
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


def _seed(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < -(2**31)
        or value > 2**31 - 1
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a signed 32-bit integer"
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
    value: ExternalFilmFxStagingVerificationV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("verification_id")
    return payload


def _read_report(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[ExternalFilmFxRunV1, str]:
    _sha256(expected_sha256, "expected_report_sha256")
    if not path.is_file():
        raise ReferenceMatchContractError(
            "external FilmFX report must be an existing file"
        )
    try:
        size = path.stat().st_size
        raw = path.read_bytes()
    except OSError as exc:
        raise ReferenceMatchContractError(
            "external FilmFX report is unreadable"
        ) from exc
    if size <= 0 or size > _MAX_REPORT_BYTES or len(raw) != size:
        raise ReferenceMatchContractError(
            "external FilmFX report violates the bounded size contract"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ReferenceMatchContractError(
            "external FilmFX report hash mismatch"
        )
    try:
        encoded = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ReferenceMatchContractError(
            "external FilmFX report must be UTF-8"
        ) from exc
    return external_filmfx_run_from_json(encoded), digest


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


def verify_external_filmfx_staging_v1(
    *,
    report_path: Path | str,
    expected_report_sha256: str,
    expected_run_id: str,
) -> ExternalFilmFxStagingVerificationV1:
    """Reread and bind a complete P36 FilmFX staging run without writing."""

    _sha256(expected_run_id, "expected_run_id")
    report = Path(report_path)
    run, report_digest = _read_report(
        report,
        expected_sha256=expected_report_sha256,
    )
    if run.run_id != expected_run_id:
        raise ReferenceMatchContractError(
            "external FilmFX run identity mismatch"
        )
    resolved_report = str(report.resolve(strict=True))
    if (
        str(Path(run.report_path).resolve(strict=False)).casefold()
        != resolved_report.casefold()
    ):
        raise ReferenceMatchContractError(
            "external FilmFX report path mismatch"
        )
    outputs: list[VerifiedExternalFilmFxOutputV1] = []
    for row in run.outputs:
        input_path, input_digest = _verified_file(
            row.input_path,
            row.input_file_sha256,
            label=f"external FilmFX input {row.source_index}",
        )
        output_path, output_digest = _verified_file(
            row.output_path,
            row.output_file_sha256,
            label=f"external FilmFX output {row.source_index}",
        )
        outputs.append(
            VerifiedExternalFilmFxOutputV1(
                source_index=row.source_index,
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
    provisional = ExternalFilmFxStagingVerificationV1(
        schema_id=EXTERNAL_FILMFX_VERIFICATION_SCHEMA_ID,
        verification_id="0" * 64,
        run_id=run.run_id,
        report_path=resolved_report,
        report_file_sha256=report_digest,
        composition_plan_id=run.composition_plan_id,
        staging_verification_id=run.staging_verification_id,
        source_count=run.source_count,
        seed=run.seed,
        state=_STATE,
        outputs=tuple(outputs),
        claim_ceiling=EXTERNAL_FILMFX_VERIFICATION_CLAIM_CEILING,
    )
    result = replace(
        provisional,
        verification_id=canonical_sha256(
            _identity_payload(provisional)
        ),
    )
    validate_external_filmfx_staging_verification_v1(result)
    return result


def validate_external_filmfx_staging_verification_v1(
    value: ExternalFilmFxStagingVerificationV1,
) -> None:
    if not isinstance(value, ExternalFilmFxStagingVerificationV1):
        raise ReferenceMatchContractError(
            "external FilmFX verification type is invalid"
        )
    if value.schema_id != EXTERNAL_FILMFX_VERIFICATION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported external FilmFX verification schema"
        )
    _sha256(value.verification_id, "verification_id")
    _sha256(value.run_id, "run_id")
    _sha256(value.report_file_sha256, "report_file_sha256")
    _sha256(value.composition_plan_id, "composition_plan_id")
    _sha256(value.staging_verification_id, "staging_verification_id")
    _seed(value.seed, "seed")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "external FilmFX verification source_count mismatch"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "external FilmFX verification state is unsupported"
        )
    if (
        value.claim_ceiling
        != EXTERNAL_FILMFX_VERIFICATION_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "external FilmFX verification claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "external FilmFX verification report_path is invalid"
        )
    input_paths: list[str] = []
    output_paths: list[str] = []
    bit_depths: list[int] = []
    for index, output in enumerate(value.outputs):
        if not isinstance(output, VerifiedExternalFilmFxOutputV1):
            raise ReferenceMatchContractError(
                "external FilmFX verification output type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "external FilmFX verification indices must be contiguous"
            )
        _sha256(output.input_file_sha256, "output.input_file_sha256")
        _sha256(output.output_file_sha256, "output.output_file_sha256")
        _seed(output.grain_seed, "output.grain_seed")
        _seed(output.dust_seed, "output.dust_seed")
        if output.grain_seed != value.seed + index * 1009:
            raise ReferenceMatchContractError(
                "external FilmFX verification grain seed mismatch"
            )
        if output.dust_seed != output.grain_seed + 17:
            raise ReferenceMatchContractError(
                "external FilmFX verification dust seed mismatch"
            )
        if (
            not isinstance(output.input_path, str)
            or not output.input_path
            or not isinstance(output.output_path, str)
            or not output.output_path
        ):
            raise ReferenceMatchContractError(
                "external FilmFX verification paths are invalid"
            )
        if output.output_format not in {"PNG", "JPEG", "TIFF"}:
            raise ReferenceMatchContractError(
                "external FilmFX verification format is unsupported"
            )
        if output.output_bit_depth not in {8, 16}:
            raise ReferenceMatchContractError(
                "external FilmFX verification bit depth is unsupported"
            )
        input_paths.append(
            str(Path(output.input_path).resolve(strict=False)).casefold()
        )
        output_paths.append(
            str(Path(output.output_path).resolve(strict=False)).casefold()
        )
        bit_depths.append(output.output_bit_depth)
    if len(set(input_paths)) != len(input_paths):
        raise ReferenceMatchContractError(
            "external FilmFX verification input paths must be unique"
        )
    if len(set(output_paths)) != len(output_paths):
        raise ReferenceMatchContractError(
            "external FilmFX verification output paths must be unique"
        )
    if set(input_paths) & set(output_paths):
        raise ReferenceMatchContractError(
            "external FilmFX verification input/output collision"
        )
    if len(set(bit_depths)) != 1:
        raise ReferenceMatchContractError(
            "external FilmFX verification bit depth must be uniform"
        )
    report_key = str(
        Path(value.report_path).resolve(strict=False)
    ).casefold()
    if report_key in set(input_paths) | set(output_paths):
        raise ReferenceMatchContractError(
            "external FilmFX verification report/file collision"
        )
    if value.verification_id != canonical_sha256(
        _identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "external FilmFX verification_id mismatch"
        )


def external_filmfx_verification_to_json(
    value: ExternalFilmFxStagingVerificationV1,
) -> str:
    validate_external_filmfx_staging_verification_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def external_filmfx_verification_from_json(
    encoded: str,
) -> ExternalFilmFxStagingVerificationV1:
    try:
        payload = strict_json_loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "external FilmFX verification is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "external FilmFX verification")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "external FilmFX verification outputs must be non-empty"
        )
    outputs: list[VerifiedExternalFilmFxOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(
            raw,
            _OUTPUT_KEYS,
            f"external FilmFX verification output {index}",
        )
        try:
            outputs.append(VerifiedExternalFilmFxOutputV1(**raw))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "external FilmFX verification output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        value = ExternalFilmFxStagingVerificationV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "external FilmFX verification fields are invalid"
        ) from exc
    validate_external_filmfx_staging_verification_v1(value)
    return value


__all__ = [
    "EXTERNAL_FILMFX_VERIFICATION_CLAIM_CEILING",
    "EXTERNAL_FILMFX_VERIFICATION_SCHEMA_ID",
    "ExternalFilmFxStagingVerificationV1",
    "VerifiedExternalFilmFxOutputV1",
    "external_filmfx_verification_from_json",
    "external_filmfx_verification_to_json",
    "validate_external_filmfx_staging_verification_v1",
    "verify_external_filmfx_staging_v1",
]
