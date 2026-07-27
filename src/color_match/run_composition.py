"""Bind a composition plan to one verified file-match run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from src.inference import sha256_file

from .canonical import canonical_sha256
from .composition import (
    ReferenceCompositionPlan,
    composition_plan_from_dict,
    validate_reference_composition,
)
from .contracts import ReferenceMatchContractError
from .files import FileReferenceMatchResult, FileReferenceReplayResult
from .reporting import (
    REFERENCE_MATCH_REPLAY_REPORT_SCHEMA_ID,
    REFERENCE_MATCH_REPORT_SCHEMA_ID,
)


REFERENCE_RUN_COMPOSITION_SCHEMA_ID = (
    "neuro-film.reference-run-composition-binding.v1"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_BINDING_KEYS = {
    "schema_id",
    "binding_id",
    "run_report_schema_id",
    "run_report_sha256",
    "batch_safety_status",
    "research_baseline_requested",
    "composition_plan",
    "outputs",
}
_OUTPUT_KEYS = {
    "source_index",
    "source_sha256",
    "output_sha256",
    "safety_action",
}


@dataclass(frozen=True)
class RunCompositionOutputBinding:
    """One output identity and the delivered colour action."""

    source_index: int
    source_sha256: str
    output_sha256: str
    safety_action: str


@dataclass(frozen=True)
class ReferenceRunCompositionBinding:
    """Replayable proof that a composition plan matches one complete run."""

    schema_id: str
    binding_id: str
    run_report_schema_id: str
    run_report_sha256: str
    batch_safety_status: str
    research_baseline_requested: bool
    composition_plan: ReferenceCompositionPlan
    outputs: tuple[RunCompositionOutputBinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_payload(
    binding: ReferenceRunCompositionBinding,
) -> dict[str, Any]:
    payload = binding.to_dict()
    payload.pop("binding_id", None)
    return payload


def _binding_id(binding: ReferenceRunCompositionBinding) -> str:
    return canonical_sha256(_canonical_payload(binding))


def _strict_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ReferenceMatchContractError(
            f"{label} keys mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def validate_reference_run_composition(
    binding: ReferenceRunCompositionBinding,
) -> None:
    """Reject a run binding that could hide fallback or colour stacking."""

    if not isinstance(binding, ReferenceRunCompositionBinding):
        raise ReferenceMatchContractError(
            "run composition must be ReferenceRunCompositionBinding"
        )
    if binding.schema_id != REFERENCE_RUN_COMPOSITION_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported reference run-composition schema"
        )
    if binding.run_report_schema_id not in {
        REFERENCE_MATCH_REPORT_SCHEMA_ID,
        REFERENCE_MATCH_REPLAY_REPORT_SCHEMA_ID,
    }:
        raise ReferenceMatchContractError(
            "run composition report schema is unsupported"
        )
    _sha256(binding.run_report_sha256, "run_report_sha256")
    if not isinstance(binding.research_baseline_requested, bool):
        raise ReferenceMatchContractError(
            "research_baseline_requested must be boolean"
        )
    validate_reference_composition(binding.composition_plan)
    if (
        binding.research_baseline_requested
        != binding.composition_plan.research_baseline_override
    ):
        raise ReferenceMatchContractError(
            "run composition research-baseline request mismatch"
        )
    if not isinstance(binding.outputs, tuple) or not binding.outputs:
        raise ReferenceMatchContractError(
            "run composition outputs must be a non-empty tuple"
        )
    for index, output in enumerate(binding.outputs):
        if not isinstance(output, RunCompositionOutputBinding):
            raise ReferenceMatchContractError(
                "run composition output type is invalid"
            )
        if output.source_index != index:
            raise ReferenceMatchContractError(
                "run composition source indices must be contiguous"
            )
        _sha256(output.source_sha256, "output.source_sha256")
        _sha256(output.output_sha256, "output.output_sha256")
        if output.safety_action not in {"applied", "identity-fallback"}:
            raise ReferenceMatchContractError(
                "run composition safety action is invalid"
            )

    actions = {output.safety_action for output in binding.outputs}
    if len(actions) != 1:
        raise ReferenceMatchContractError(
            "mixed delivery cannot form one batch composition"
        )
    expected_status = (
        "reference-color-applied"
        if actions == {"applied"}
        else "identity-fallback"
    )
    if binding.batch_safety_status != expected_status:
        raise ReferenceMatchContractError(
            "run composition batch safety status mismatch"
        )
    plan = binding.composition_plan
    if expected_status == "reference-color-applied":
        if (
            not plan.research_baseline_override
            or plan.reference_color_status != "research-baseline"
            or plan.color_owner != "reference-look"
        ):
            raise ReferenceMatchContractError(
                "applied run requires a reference-colour composition plan"
            )
    elif (
        plan.research_baseline_override
        or plan.reference_color_status != "identity-fallback"
        or plan.color_owner != "identity"
        or plan.film_effects is not None
    ):
        raise ReferenceMatchContractError(
            "fallback run requires an identity composition without effects"
        )
    if (
        not isinstance(binding.binding_id, str)
        or not _HASH.fullmatch(binding.binding_id)
        or binding.binding_id != _binding_id(binding)
    ):
        raise ReferenceMatchContractError(
            "run composition binding_id does not match canonical payload"
        )


def _read_bound_report(
    result: FileReferenceMatchResult | FileReferenceReplayResult,
) -> tuple[Mapping[str, Any], str]:
    if result.report_path is None or result.report_file_sha256 is None:
        raise ReferenceMatchContractError(
            "run composition requires a transaction-bound report"
        )
    report_path = Path(result.report_path)
    if not report_path.is_file():
        raise ReferenceMatchContractError(
            "run composition report path must be an existing file"
        )
    try:
        size = report_path.stat().st_size
        raw = report_path.read_bytes()
    except OSError as exc:
        raise ReferenceMatchContractError(
            "run composition report is unreadable"
        ) from exc
    if (
        size <= 0
        or size > _MAX_REPORT_BYTES
        or len(raw) != size
    ):
        raise ReferenceMatchContractError(
            "run composition report violates the bounded size contract"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != result.report_file_sha256:
        raise ReferenceMatchContractError(
            "run composition report hash mismatch"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "run composition report must be finite UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReferenceMatchContractError(
            "run composition report must be an object"
        )
    expected_schema = (
        REFERENCE_MATCH_REPORT_SCHEMA_ID
        if isinstance(result, FileReferenceMatchResult)
        else REFERENCE_MATCH_REPLAY_REPORT_SCHEMA_ID
    )
    if (
        payload.get("schema_id") != expected_schema
        or payload.get("recipe_id") != result.recipe.recipe_id
        or payload.get("algorithm_id") != result.recipe.algorithm_id
        or payload.get("claim_ceiling") != result.recipe.claim_ceiling
        or payload.get("evidence_grade") != result.recipe.evidence_grade
        or not isinstance(payload.get("outputs"), list)
        or len(payload["outputs"]) != len(result.outputs)
    ):
        raise ReferenceMatchContractError(
            "run composition report payload does not match run result"
        )
    if isinstance(result, FileReferenceMatchResult):
        reference = payload.get("reference")
        if (
            not isinstance(reference, Mapping)
            or reference.get("pixel_sha256")
            != result.recipe.reference_pixel_sha256
        ):
            raise ReferenceMatchContractError(
                "run composition reference identity mismatch"
            )
    elif (
        payload.get("reference_pixel_sha256")
        != result.recipe.reference_pixel_sha256
    ):
        raise ReferenceMatchContractError(
            "run composition reference identity mismatch"
        )
    for index, (report_row, output) in enumerate(
        zip(payload["outputs"], result.outputs, strict=True)
    ):
        if not isinstance(report_row, Mapping):
            raise ReferenceMatchContractError(
                "run composition report output must be an object"
            )
        safety = report_row.get("safety")
        diagnostics = report_row.get("candidate_diagnostics")
        if (
            report_row.get("output_sha256") != output.output_sha256
            or report_row.get("output_path")
            != str(output.output_path.resolve())
            or report_row.get("source_path")
            != str(output.source_path.resolve())
            or not isinstance(safety, Mapping)
            or safety.get("action") != output.safety.action
            or safety.get("accepted") != output.safety.accepted
            or safety.get("research_baseline_override")
            != output.safety.research_baseline_override
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("recipe_id") != result.recipe.recipe_id
            or diagnostics.get("source_index") != index
        ):
            raise ReferenceMatchContractError(
                "run composition report output does not match run result"
            )
        if (
            not output.output_path.is_file()
            or sha256_file(output.output_path) != output.output_sha256
        ):
            raise ReferenceMatchContractError(
                "run composition output file hash mismatch"
            )
    return payload, digest


def build_reference_run_composition(
    plan: ReferenceCompositionPlan,
    result: FileReferenceMatchResult | FileReferenceReplayResult,
) -> ReferenceRunCompositionBinding:
    """Bind one static composition plan to actual uniform run delivery."""

    validate_reference_composition(plan)
    if not isinstance(
        result,
        (FileReferenceMatchResult, FileReferenceReplayResult),
    ):
        raise ReferenceMatchContractError(
            "run result type is unsupported"
        )
    if plan.reference_recipe_id != result.recipe.recipe_id:
        raise ReferenceMatchContractError(
            "composition recipe does not match the run recipe"
        )
    report, report_sha256 = _read_bound_report(result)
    if len(report["outputs"]) != len(result.outputs):
        raise ReferenceMatchContractError(
            "composition report output count mismatch"
        )
    actions = {row.safety.action for row in result.outputs}
    if len(actions) != 1:
        raise ReferenceMatchContractError(
            "mixed delivery cannot form one batch composition"
        )
    overrides = {
        row.safety.research_baseline_override for row in result.outputs
    }
    if len(overrides) != 1:
        raise ReferenceMatchContractError(
            "run outputs disagree on research-baseline request"
        )
    action = next(iter(actions))
    if action == "applied":
        if not plan.research_baseline_override:
            raise ReferenceMatchContractError(
                "applied run requires a reference-colour composition plan"
            )
        batch_status = "reference-color-applied"
    else:
        if plan.research_baseline_override or plan.film_effects is not None:
            raise ReferenceMatchContractError(
                "fallback run cannot bind reference colour or film effects"
            )
        batch_status = "identity-fallback"

    output_bindings = tuple(
        RunCompositionOutputBinding(
            source_index=index,
            source_sha256=str(report_row["source_sha256"]),
            output_sha256=str(report_row["output_sha256"]),
            safety_action=row.safety.action,
        )
        for index, (row, report_row) in enumerate(
            zip(result.outputs, report["outputs"], strict=True)
        )
    )
    provisional = ReferenceRunCompositionBinding(
        schema_id=REFERENCE_RUN_COMPOSITION_SCHEMA_ID,
        binding_id="0" * 64,
        run_report_schema_id=str(report["schema_id"]),
        run_report_sha256=report_sha256,
        batch_safety_status=batch_status,
        research_baseline_requested=next(iter(overrides)),
        composition_plan=plan,
        outputs=output_bindings,
    )
    binding = replace(
        provisional,
        binding_id=_binding_id(provisional),
    )
    validate_reference_run_composition(binding)
    return binding


def reference_run_composition_to_json(
    binding: ReferenceRunCompositionBinding,
) -> str:
    validate_reference_run_composition(binding)
    return json.dumps(
        binding.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def reference_run_composition_from_dict(
    payload: Mapping[str, Any],
) -> ReferenceRunCompositionBinding:
    if not isinstance(payload, Mapping):
        raise ReferenceMatchContractError(
            "run composition payload must be an object"
        )
    _strict_keys(payload, _BINDING_KEYS, "run composition")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "run composition outputs must be a non-empty array"
        )
    outputs: list[RunCompositionOutputBinding] = []
    for index, raw in enumerate(raw_outputs):
        if not isinstance(raw, Mapping):
            raise ReferenceMatchContractError(
                f"run composition outputs[{index}] must be an object"
            )
        _strict_keys(
            raw,
            _OUTPUT_KEYS,
            f"run composition outputs[{index}]",
        )
        try:
            outputs.append(RunCompositionOutputBinding(**dict(raw)))
        except TypeError as exc:
            raise ReferenceMatchContractError(
                "run composition output types are invalid"
            ) from exc
    raw_plan = payload["composition_plan"]
    if not isinstance(raw_plan, Mapping):
        raise ReferenceMatchContractError(
            "run composition plan must be an object"
        )
    try:
        binding = ReferenceRunCompositionBinding(
            schema_id=payload["schema_id"],
            binding_id=payload["binding_id"],
            run_report_schema_id=payload["run_report_schema_id"],
            run_report_sha256=payload["run_report_sha256"],
            batch_safety_status=payload["batch_safety_status"],
            research_baseline_requested=payload[
                "research_baseline_requested"
            ],
            composition_plan=composition_plan_from_dict(raw_plan),
            outputs=tuple(outputs),
        )
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "run composition types are invalid"
        ) from exc
    validate_reference_run_composition(binding)
    return binding


def reference_run_composition_from_json(
    encoded: str,
) -> ReferenceRunCompositionBinding:
    if not isinstance(encoded, str):
        raise ReferenceMatchContractError(
            "encoded run composition must be a string"
        )
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "encoded run composition is not valid JSON"
        ) from exc
    return reference_run_composition_from_dict(payload)


__all__ = [
    "REFERENCE_RUN_COMPOSITION_SCHEMA_ID",
    "ReferenceRunCompositionBinding",
    "RunCompositionOutputBinding",
    "build_reference_run_composition",
    "reference_run_composition_from_dict",
    "reference_run_composition_from_json",
    "reference_run_composition_to_json",
    "validate_reference_run_composition",
]
