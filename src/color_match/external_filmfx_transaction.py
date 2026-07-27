"""Atomic procedural FilmFX execution for an exact P35 composition plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import uuid

from src.filmfx import (
    composite_layers,
    dust_scratch_layer,
    grain_residual_layer,
    halation_layer,
)
from src.inference import atomic_write_json, sha256_file
from src.preprocess import load_working_image, working_image_to_srgb_float

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .staging_io import (
    staging_output_paths,
    validate_sdr_staging_destinations,
)
from .core_staging_verification import (
    ExternalCoreStagingVerificationV1,
    validate_external_core_staging_verification_v1,
    verify_external_core_staging_v1,
)
from .external_composition import (
    ExternalReferenceCompositionV1,
    validate_external_reference_composition_v1,
)
from .files import _commit_staged_batch, _encode_srgb, _stage_path


EXTERNAL_FILMFX_RUN_SCHEMA_ID = "neuro-film.external-filmfx-run.v1"
EXTERNAL_FILMFX_CLAIM_CEILING = "filmfx-staging-not-delivered"
_STATE = "filmfx-rendered-to-staging"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "run_id",
    "composition_plan_id",
    "staging_verification_id",
    "source_count",
    "seed",
    "state",
    "outputs",
    "report_path",
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
class ExternalFilmFxOutputV1:
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
class ExternalFilmFxRunV1:
    schema_id: str
    run_id: str
    composition_plan_id: str
    staging_verification_id: str
    source_count: int
    seed: int
    state: str
    outputs: tuple[ExternalFilmFxOutputV1, ...]
    report_path: str
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class CommittedExternalFilmFxRunV1:
    run: ExternalFilmFxRunV1
    report_file_sha256: str


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


def _identity_payload(value: ExternalFilmFxRunV1) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("run_id")
    return payload


def _validate_bindings(
    plan: ExternalReferenceCompositionV1,
    verification: ExternalCoreStagingVerificationV1,
) -> ExternalCoreStagingVerificationV1:
    validate_external_reference_composition_v1(plan)
    validate_external_core_staging_verification_v1(verification)
    if (
        plan.staging_verification_id != verification.verification_id
        or plan.staging_run_id != verification.run_id
        or plan.authorization_id != verification.authorization_id
        or plan.reference_intent_id != verification.reference_intent_id
        or plan.source_count != verification.source_count
    ):
        raise ReferenceMatchContractError(
            "FilmFX composition does not bind staging verification"
        )
    if plan.film_effects is None:
        raise ReferenceMatchContractError(
            "FilmFX execution requires a plan with effects"
        )
    effects = plan.film_effects
    if effects.halation_model != "simple":
        raise ReferenceMatchContractError(
            "physical FilmFX requires separately bound resolved controls"
        )
    if effects.grain <= 0.0 and effects.halation <= 0.0 and effects.dust <= 0.0:
        raise ReferenceMatchContractError(
            "FilmFX execution requires at least one active effect"
        )
    refreshed = verify_external_core_staging_v1(
        report_path=verification.report_path,
        expected_report_sha256=verification.report_file_sha256,
        expected_run_id=verification.run_id,
    )
    if refreshed != verification:
        raise ReferenceMatchContractError(
            "FilmFX staging verification changed on refresh"
        )
    return refreshed


def _validate_paths(
    outputs: tuple[Path, ...],
    report: Path,
    verification: ExternalCoreStagingVerificationV1,
) -> None:
    protected = {
        str(Path(row.output_path).resolve(strict=True)).casefold()
        for row in verification.outputs
    }
    protected.add(
        str(Path(verification.report_path).resolve(strict=True)).casefold()
    )
    destinations = {
        str(path.resolve(strict=False)).casefold() for path in outputs
    }
    destinations.add(str(report.resolve(strict=False)).casefold())
    if protected & destinations:
        raise ReferenceMatchContractError(
            "FilmFX destinations must not overwrite P33 staging artifacts"
        )


def commit_external_filmfx_staging_v1(
    *,
    plan: ExternalReferenceCompositionV1,
    verification: ExternalCoreStagingVerificationV1,
    output_paths: Iterable[Path | str],
    report_path: Path | str,
    seed: int,
    output_bit_depth: int = 16,
) -> CommittedExternalFilmFxRunV1:
    """Render procedural FilmFX to a new atomic staging transaction."""

    refreshed = _validate_bindings(plan, verification)
    base_seed = _seed(seed, "seed")
    outputs = staging_output_paths(
        output_paths, count=refreshed.source_count
    )
    report = Path(report_path)
    validate_sdr_staging_destinations(
        outputs,
        report,
        output_bit_depth,
        label="external staging",
    )
    _validate_paths(outputs, report, refreshed)
    assert plan.film_effects is not None
    effects = plan.film_effects

    token = uuid.uuid4().hex
    staged: list[Path] = []
    prepared: list[ExternalFilmFxOutputV1] = []
    try:
        for index, (source_row, output) in enumerate(
            zip(refreshed.outputs, outputs, strict=True)
        ):
            source_path = Path(source_row.output_path)
            working = load_working_image(source_path)
            if (
                working.working_space != "linear_srgb"
                or working.transfer_state != "display_linear"
                or working.alpha_policy != "absent"
            ):
                raise ReferenceMatchContractError(
                    "FilmFX execution requires alpha-free display-linear "
                    "linear-sRGB staging input"
                )
            base = working_image_to_srgb_float(working)
            row_seed = _seed(base_seed + index * 1009, "row_seed")
            dust_seed = _seed(row_seed + 17, "dust_seed")
            layers = []
            if effects.grain > 0.0:
                layers.append(
                    grain_residual_layer(
                        base,
                        strength=effects.grain,
                        seed=row_seed,
                        color=effects.interpretation != "bw_developer_scan",
                    )
                )
            if effects.halation > 0.0:
                layers.append(
                    halation_layer(base, strength=effects.halation)
                )
            if effects.dust > 0.0:
                layers.append(
                    dust_scratch_layer(
                        base.shape,
                        strength=effects.dust,
                        seed=dust_seed,
                    )
                )
            rendered = composite_layers(base, layers, output_margin=4)
            output.parent.mkdir(parents=True, exist_ok=True)
            stage = _stage_path(output, token)
            staged.append(stage)
            output_format, _clipped_fraction = _encode_srgb(
                rendered,
                stage,
                output_bit_depth=output_bit_depth,
            )
            prepared.append(
                ExternalFilmFxOutputV1(
                    source_index=index,
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
        provisional = ExternalFilmFxRunV1(
            schema_id=EXTERNAL_FILMFX_RUN_SCHEMA_ID,
            run_id="0" * 64,
            composition_plan_id=plan.plan_id,
            staging_verification_id=refreshed.verification_id,
            source_count=refreshed.source_count,
            seed=base_seed,
            state=_STATE,
            outputs=tuple(prepared),
            report_path=str(report.resolve(strict=False)),
            claim_ceiling=EXTERNAL_FILMFX_CLAIM_CEILING,
        )
        run = replace(
            provisional,
            run_id=canonical_sha256(_identity_payload(provisional)),
        )
        validate_external_filmfx_run_v1(run)
        report.parent.mkdir(parents=True, exist_ok=True)
        staged_report = _stage_path(report, token)
        staged.append(staged_report)
        atomic_write_json(staged_report, run.to_dict())
        report_sha256 = sha256_file(staged_report)
        pairs = tuple(
            (_stage_path(output, token), output) for output in outputs
        ) + ((staged_report, report),)
        _commit_staged_batch(pairs, token=token, cleanup=staged)
        return CommittedExternalFilmFxRunV1(
            run=run,
            report_file_sha256=report_sha256,
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def validate_external_filmfx_run_v1(value: ExternalFilmFxRunV1) -> None:
    if not isinstance(value, ExternalFilmFxRunV1):
        raise ReferenceMatchContractError("FilmFX run type is invalid")
    if value.schema_id != EXTERNAL_FILMFX_RUN_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported external FilmFX run schema"
        )
    _sha256(value.run_id, "run_id")
    _sha256(value.composition_plan_id, "composition_plan_id")
    _sha256(value.staging_verification_id, "staging_verification_id")
    _seed(value.seed, "seed")
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError("FilmFX source_count mismatch")
    if value.state != _STATE:
        raise ReferenceMatchContractError("FilmFX run state is unsupported")
    if value.claim_ceiling != EXTERNAL_FILMFX_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "FilmFX run claim ceiling mismatch"
        )
    if (
        not isinstance(value.report_path, str)
        or not value.report_path
        or Path(value.report_path).suffix.casefold() != ".json"
    ):
        raise ReferenceMatchContractError("FilmFX report_path is invalid")
    paths: list[str] = []
    bit_depths: list[int] = []
    for index, output in enumerate(value.outputs):
        if not isinstance(output, ExternalFilmFxOutputV1):
            raise ReferenceMatchContractError(
                "FilmFX output row type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "FilmFX source indices must be contiguous"
            )
        _sha256(output.input_file_sha256, "output.input_file_sha256")
        _sha256(output.output_file_sha256, "output.output_file_sha256")
        _seed(output.grain_seed, "output.grain_seed")
        _seed(output.dust_seed, "output.dust_seed")
        if output.grain_seed != value.seed + index * 1009:
            raise ReferenceMatchContractError("FilmFX grain seed mismatch")
        if output.dust_seed != output.grain_seed + 17:
            raise ReferenceMatchContractError("FilmFX dust seed mismatch")
        if (
            not isinstance(output.input_path, str)
            or not output.input_path
            or not isinstance(output.output_path, str)
            or not output.output_path
        ):
            raise ReferenceMatchContractError("FilmFX output paths are invalid")
        if output.output_format not in {"PNG", "JPEG", "TIFF"}:
            raise ReferenceMatchContractError(
                "FilmFX output format is unsupported"
            )
        if output.output_bit_depth not in {8, 16}:
            raise ReferenceMatchContractError(
                "FilmFX output bit depth is unsupported"
            )
        bit_depths.append(output.output_bit_depth)
        paths.append(
            str(Path(output.output_path).resolve(strict=False)).casefold()
        )
    if len(set(paths)) != len(paths):
        raise ReferenceMatchContractError(
            "FilmFX output paths must be unique"
        )
    if len(set(bit_depths)) != 1:
        raise ReferenceMatchContractError(
            "FilmFX output bit depth must be uniform"
        )
    if str(
        Path(value.report_path).resolve(strict=False)
    ).casefold() in set(paths):
        raise ReferenceMatchContractError(
            "FilmFX report/output path collision"
        )
    if value.run_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError("FilmFX run_id mismatch")


def external_filmfx_run_to_json(value: ExternalFilmFxRunV1) -> str:
    validate_external_filmfx_run_v1(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def external_filmfx_run_from_json(encoded: str) -> ExternalFilmFxRunV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "external FilmFX run is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "external FilmFX run")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "external FilmFX outputs must be non-empty"
        )
    outputs: list[ExternalFilmFxOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"FilmFX output {index}")
        try:
            outputs.append(ExternalFilmFxOutputV1(**raw))
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "external FilmFX output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        value = ExternalFilmFxRunV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "external FilmFX run fields are invalid"
        ) from exc
    validate_external_filmfx_run_v1(value)
    return value


__all__ = [
    "EXTERNAL_FILMFX_CLAIM_CEILING",
    "EXTERNAL_FILMFX_RUN_SCHEMA_ID",
    "CommittedExternalFilmFxRunV1",
    "ExternalFilmFxOutputV1",
    "ExternalFilmFxRunV1",
    "commit_external_filmfx_staging_v1",
    "external_filmfx_run_from_json",
    "external_filmfx_run_to_json",
    "validate_external_filmfx_run_v1",
]
