"""Atomic procedural FilmFX staging for an exact shared composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import uuid

from src.inference import atomic_write_json, sha256_file

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .files import _commit_staged_batch, _stage_path
from .filmfx_staging_io import (
    protect_staging_inputs,
    render_procedural_filmfx,
    signed_seed,
    validate_procedural_filmfx,
)
from .shared_composition import (
    SharedReferenceCompositionV1,
    validate_shared_reference_composition_v1,
)
from .shared_staging_verification import (
    ExternalSharedStagingVerificationV1,
    validate_external_shared_staging_verification_v1,
    verify_external_shared_staging_v1,
)
from .staging_io import (
    staging_output_paths,
    validate_sdr_staging_destinations,
)


SHARED_FILMFX_RUN_SCHEMA_ID = "neuro-film.shared-filmfx-run.v1"
SHARED_FILMFX_CLAIM_CEILING = "shared-filmfx-staging-not-delivered"
_STATE = "shared-filmfx-rendered-to-staging"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "run_id",
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
    "report_path",
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
class SharedFilmFxOutputV1:
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
class SharedFilmFxRunV1:
    schema_id: str
    run_id: str
    composition_plan_id: str
    staging_verification_id: str
    staging_run_id: str
    authorization_id: str
    numeric_guard_batch_id: str
    operator_id: str
    source_count: int
    seed: int
    state: str
    outputs: tuple[SharedFilmFxOutputV1, ...]
    report_path: str
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class CommittedSharedFilmFxRunV1:
    run: SharedFilmFxRunV1
    report_file_sha256: str


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from shared FilmFX staging"
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


def _identity_payload(value: SharedFilmFxRunV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("run_id")
    return payload


def _validate_bindings(
    plan: SharedReferenceCompositionV1,
    verification: ExternalSharedStagingVerificationV1,
) -> ExternalSharedStagingVerificationV1:
    validate_shared_reference_composition_v1(plan)
    validate_external_shared_staging_verification_v1(verification)
    if (
        plan.staging_verification_id != verification.verification_id
        or plan.staging_run_id != verification.run_id
        or plan.authorization_id != verification.authorization_id
        or plan.numeric_guard_batch_id
        != verification.numeric_guard_batch_id
        or plan.operator_id != verification.operator_id
        or plan.reference_view_id != verification.reference_view_id
        or plan.source_count != verification.source_count
    ):
        raise ReferenceMatchContractError(
            "shared FilmFX plan does not bind staging verification"
        )
    if plan.film_effects is None:
        raise ReferenceMatchContractError(
            "shared FilmFX execution requires a plan with effects"
        )
    validate_procedural_filmfx(plan.film_effects)
    refreshed = verify_external_shared_staging_v1(
        report_path=verification.report_path,
        expected_report_sha256=verification.report_file_sha256,
        expected_run_id=verification.run_id,
        expected_authorization_id=verification.authorization_id,
        expected_numeric_guard_batch_id=(
            verification.numeric_guard_batch_id
        ),
        expected_operator_id=verification.operator_id,
    )
    if refreshed != verification:
        raise ReferenceMatchContractError(
            "shared FilmFX staging verification changed on refresh"
        )
    return refreshed


def commit_shared_filmfx_staging_v1(
    *,
    plan: SharedReferenceCompositionV1,
    verification: ExternalSharedStagingVerificationV1,
    output_paths: Iterable[Path | str],
    report_path: Path | str,
    seed: int,
    output_bit_depth: int = 16,
) -> CommittedSharedFilmFxRunV1:
    """Render shared-base procedural FilmFX as one atomic transaction."""

    refreshed = _validate_bindings(plan, verification)
    base_seed = signed_seed(seed, "seed")
    outputs = staging_output_paths(
        output_paths, count=refreshed.source_count
    )
    report = Path(report_path)
    validate_sdr_staging_destinations(
        outputs,
        report,
        output_bit_depth,
        label="shared FilmFX staging",
    )
    protect_staging_inputs(
        outputs,
        report,
        input_paths=tuple(
            Path(row.output_path) for row in refreshed.outputs
        ),
        input_report=Path(refreshed.report_path),
        protected_label="P50",
    )
    assert plan.film_effects is not None
    effects = plan.film_effects

    token = uuid.uuid4().hex
    staged: list[Path] = []
    prepared: list[SharedFilmFxOutputV1] = []
    try:
        for index, (source_row, output) in enumerate(
            zip(refreshed.outputs, outputs, strict=True)
        ):
            source_path = Path(source_row.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            stage = _stage_path(output, token)
            staged.append(stage)
            output_format, row_seed, dust_seed = render_procedural_filmfx(
                source_path=source_path,
                destination=stage,
                effects=effects,
                base_seed=base_seed,
                source_index=index,
                output_bit_depth=output_bit_depth,
            )
            prepared.append(
                SharedFilmFxOutputV1(
                    source_index=index,
                    apply_receipt_id=source_row.apply_receipt_id,
                    producer_apply_result_id=(
                        source_row.producer_apply_result_id
                    ),
                    input_path=str(source_path.resolve(strict=True)),
                    input_file_sha256=source_row.output_file_sha256,
                    output_path=str(output.resolve(strict=False)),
                    output_file_sha256=sha256_file(stage),
                    output_format=output_format,
                    output_bit_depth=output_bit_depth,
                    grain_seed=row_seed,
                    dust_seed=dust_seed,
                )
            )
        provisional = SharedFilmFxRunV1(
            schema_id=SHARED_FILMFX_RUN_SCHEMA_ID,
            run_id="0" * 64,
            composition_plan_id=plan.plan_id,
            staging_verification_id=refreshed.verification_id,
            staging_run_id=refreshed.run_id,
            authorization_id=refreshed.authorization_id,
            numeric_guard_batch_id=refreshed.numeric_guard_batch_id,
            operator_id=refreshed.operator_id,
            source_count=refreshed.source_count,
            seed=base_seed,
            state=_STATE,
            outputs=tuple(prepared),
            report_path=str(report.resolve(strict=False)),
            claim_ceiling=SHARED_FILMFX_CLAIM_CEILING,
        )
        run = replace(
            provisional,
            run_id=canonical_sha256(_identity_payload(provisional)),
        )
        validate_shared_filmfx_run_v1(run)
        report.parent.mkdir(parents=True, exist_ok=True)
        staged_report = _stage_path(report, token)
        staged.append(staged_report)
        atomic_write_json(staged_report, run.to_dict())
        report_sha256 = sha256_file(staged_report)
        pairs = tuple(
            (_stage_path(output, token), output) for output in outputs
        ) + ((staged_report, report),)
        _commit_staged_batch(pairs, token=token, cleanup=staged)
        return CommittedSharedFilmFxRunV1(
            run=run,
            report_file_sha256=report_sha256,
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def validate_shared_filmfx_run_v1(value: SharedFilmFxRunV1) -> None:
    if not isinstance(value, SharedFilmFxRunV1):
        raise ReferenceMatchContractError(
            "shared FilmFX run type is invalid"
        )
    if value.schema_id != SHARED_FILMFX_RUN_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared FilmFX run schema is invalid"
        )
    for field in (
        "run_id",
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
            "shared FilmFX source_count is invalid"
        )
    if value.state != _STATE:
        raise ReferenceMatchContractError(
            "shared FilmFX run state is invalid"
        )
    if value.claim_ceiling != SHARED_FILMFX_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "shared FilmFX claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError(
            "shared FilmFX report path is invalid"
        )
    paths: set[str] = set()
    receipts: set[str] = set()
    results: set[str] = set()
    bit_depths: set[int] = set()
    for index, output in enumerate(value.outputs):
        if not isinstance(output, SharedFilmFxOutputV1):
            raise ReferenceMatchContractError(
                "shared FilmFX output type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "shared FilmFX source order is invalid"
            )
        _hash(output.apply_receipt_id, "output.apply_receipt_id")
        _producer_hash(
            output.producer_apply_result_id,
            "output.producer_apply_result_id",
        )
        _hash(output.input_file_sha256, "output.input_file_sha256")
        _hash(output.output_file_sha256, "output.output_file_sha256")
        signed_seed(output.grain_seed, "output.grain_seed")
        signed_seed(output.dust_seed, "output.dust_seed")
        if output.grain_seed != value.seed + index * 1009:
            raise ReferenceMatchContractError(
                "shared FilmFX grain seed mismatch"
            )
        if output.dust_seed != output.grain_seed + 17:
            raise ReferenceMatchContractError(
                "shared FilmFX dust seed mismatch"
            )
        if (
            not isinstance(output.input_path, str)
            or not output.input_path
            or not isinstance(output.output_path, str)
            or not output.output_path
        ):
            raise ReferenceMatchContractError(
                "shared FilmFX paths are invalid"
            )
        if output.output_format not in {"PNG", "JPEG", "TIFF"}:
            raise ReferenceMatchContractError(
                "shared FilmFX output format is invalid"
            )
        if output.output_bit_depth not in {8, 16}:
            raise ReferenceMatchContractError(
                "shared FilmFX output depth is invalid"
            )
        path_key = str(
            Path(output.output_path).resolve(strict=False)
        ).casefold()
        if (
            path_key in paths
            or output.apply_receipt_id in receipts
            or output.producer_apply_result_id in results
        ):
            raise ReferenceMatchContractError(
                "shared FilmFX output identities must be distinct"
            )
        paths.add(path_key)
        receipts.add(output.apply_receipt_id)
        results.add(output.producer_apply_result_id)
        bit_depths.add(output.output_bit_depth)
    if len(bit_depths) != 1:
        raise ReferenceMatchContractError(
            "shared FilmFX output depth must be uniform"
        )
    if str(
        Path(value.report_path).resolve(strict=False)
    ).casefold() in paths:
        raise ReferenceMatchContractError(
            "shared FilmFX report/output collision"
        )
    if value.run_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "shared FilmFX run identity mismatch"
        )


def shared_filmfx_run_to_json(value: SharedFilmFxRunV1) -> str:
    validate_shared_filmfx_run_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_filmfx_run_from_json(encoded: str) -> SharedFilmFxRunV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared FilmFX run is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "shared FilmFX run")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "shared FilmFX outputs must be non-empty"
        )
    outputs: list[SharedFilmFxOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"shared FilmFX output {index}")
        try:
            outputs.append(SharedFilmFxOutputV1(**dict(raw)))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "shared FilmFX output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = SharedFilmFxRunV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "shared FilmFX run fields are invalid"
        ) from exc
    validate_shared_filmfx_run_v1(result)
    return result


__all__ = [
    "SHARED_FILMFX_CLAIM_CEILING",
    "SHARED_FILMFX_RUN_SCHEMA_ID",
    "CommittedSharedFilmFxRunV1",
    "SharedFilmFxOutputV1",
    "SharedFilmFxRunV1",
    "commit_shared_filmfx_staging_v1",
    "shared_filmfx_run_from_json",
    "shared_filmfx_run_to_json",
    "validate_shared_filmfx_run_v1",
]
