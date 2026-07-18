"""Fail-closed interpretation-plugin boundary without physical claims."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


INTERPRETATION_CONTRACT_VERSION = "interpretation-plugin-boundary-v1"
INTERPRETATION_IDS = {
    "color_negative_neutral_scan",
    "color_negative_print",
    "slide_direct_scan",
    "bw_developer_scan",
}
COLOR_DOMAINS = {"negative_scan_linear_rgb", "display_linear_rgb"}
EVIDENCE_SCOPES = {"synthetic_test_only", "measured", "paired", "held_out"}
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class InterpretationContractError(ValueError):
    """Raised when an interpretation request or plugin violates the boundary."""


@dataclass(frozen=True)
class InterpretationRequest:
    interpretation_id: str
    operator_id: str
    input_domain: str
    output_domain: str
    production: bool = False


@dataclass(frozen=True)
class InterpretationPlugin:
    interpretation_id: str
    operator_id: str
    input_domain: str
    output_domain: str
    evidence_scope: str
    production_eligible: bool
    claim_ceiling: str
    apply: Callable[[np.ndarray], tuple[np.ndarray, object]]


@dataclass(frozen=True)
class InterpretationMetadata:
    contract_version: str
    interpretation_id: str
    operator_id: str
    input_domain: str
    output_domain: str
    evidence_scope: str
    production_eligible: bool
    claim_ceiling: str
    plugin_metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class InterpretationResult:
    rgb: np.ndarray
    metadata: InterpretationMetadata


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise InterpretationContractError(f"{label} is not a safe identifier")
    return value


def _validate_plugin(plugin: InterpretationPlugin) -> None:
    if plugin.interpretation_id not in INTERPRETATION_IDS:
        raise InterpretationContractError("plugin interpretation is unsupported")
    _identifier(plugin.operator_id, "plugin operator_id")
    if plugin.input_domain not in COLOR_DOMAINS or plugin.output_domain not in COLOR_DOMAINS:
        raise InterpretationContractError("plugin colour domain is unsupported")
    if plugin.evidence_scope not in EVIDENCE_SCOPES:
        raise InterpretationContractError("plugin evidence_scope is unsupported")
    if not isinstance(plugin.production_eligible, bool):
        raise InterpretationContractError("plugin production_eligible must be boolean")
    if not isinstance(plugin.claim_ceiling, str) or not plugin.claim_ceiling.strip():
        raise InterpretationContractError("plugin claim_ceiling must be non-empty")
    if plugin.evidence_scope == "synthetic_test_only" and plugin.production_eligible:
        raise InterpretationContractError("synthetic plugins cannot be production eligible")
    if not callable(plugin.apply):
        raise InterpretationContractError("plugin apply must be callable")


def _registry(plugins: Sequence[InterpretationPlugin]) -> dict[tuple[str, str], InterpretationPlugin]:
    result: dict[tuple[str, str], InterpretationPlugin] = {}
    for plugin in plugins:
        if not isinstance(plugin, InterpretationPlugin):
            raise InterpretationContractError("registry entries must be InterpretationPlugin")
        _validate_plugin(plugin)
        key = (plugin.interpretation_id, plugin.operator_id)
        if key in result:
            raise InterpretationContractError("duplicate interpretation/operator registration")
        result[key] = plugin
    return result


def _validate_rgb(rgb: object, label: str) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32:
        raise InterpretationContractError(f"{label} must be a float32 ndarray")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise InterpretationContractError(f"{label} must have non-empty HxWx3 shape")
    if not np.isfinite(rgb).all():
        raise InterpretationContractError(f"{label} must be finite")
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise InterpretationContractError(f"{label} must be bounded in [0, 1]")
    return rgb


def _metadata(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise InterpretationContractError("plugin metadata must be an immutable tuple")
    rows: list[tuple[str, str]] = []
    keys: set[str] = set()
    for row in value:
        if not isinstance(row, tuple) or len(row) != 2:
            raise InterpretationContractError("plugin metadata rows must be key/value tuples")
        key = _identifier(row[0], "plugin metadata key")
        if key in keys or not isinstance(row[1], str):
            raise InterpretationContractError("plugin metadata must have unique string values")
        keys.add(key)
        rows.append((key, row[1]))
    return tuple(rows)


def execute_interpretation(
    rgb: np.ndarray,
    request: InterpretationRequest,
    plugins: Sequence[InterpretationPlugin],
) -> InterpretationResult:
    """Execute one exactly matched plugin under numeric and claim guardrails."""
    source = _validate_rgb(rgb, "interpretation input")
    if not isinstance(request, InterpretationRequest):
        raise InterpretationContractError("request must be InterpretationRequest")
    if request.interpretation_id not in INTERPRETATION_IDS:
        raise InterpretationContractError("request interpretation is unsupported")
    _identifier(request.operator_id, "request operator_id")
    if request.input_domain not in COLOR_DOMAINS or request.output_domain not in COLOR_DOMAINS:
        raise InterpretationContractError("request colour domain is unsupported")
    if not isinstance(request.production, bool):
        raise InterpretationContractError("request production must be boolean")

    plugin = _registry(plugins).get((request.interpretation_id, request.operator_id))
    if plugin is None:
        raise InterpretationContractError("interpretation/operator is not registered")
    if request.input_domain != plugin.input_domain or request.output_domain != plugin.output_domain:
        raise InterpretationContractError("request/plugin colour-domain mismatch")
    if request.production and not plugin.production_eligible:
        raise InterpretationContractError("plugin is not production eligible")

    before = source.tobytes()
    working = source.copy()
    working.setflags(write=False)
    try:
        output, plugin_metadata = plugin.apply(working)
    except Exception as error:
        raise InterpretationContractError(f"plugin execution failed: {error}") from error
    if source.tobytes() != before:
        raise InterpretationContractError("plugin mutated the caller input")
    result = _validate_rgb(output, "interpretation output")
    if result.shape != source.shape:
        raise InterpretationContractError("interpretation output shape changed")
    if np.shares_memory(result, working) or np.shares_memory(result, source):
        raise InterpretationContractError("interpretation output aliases input memory")
    metadata = InterpretationMetadata(
        contract_version=INTERPRETATION_CONTRACT_VERSION,
        interpretation_id=plugin.interpretation_id,
        operator_id=plugin.operator_id,
        input_domain=plugin.input_domain,
        output_domain=plugin.output_domain,
        evidence_scope=plugin.evidence_scope,
        production_eligible=plugin.production_eligible,
        claim_ceiling=plugin.claim_ceiling,
        plugin_metadata=_metadata(plugin_metadata),
    )
    return InterpretationResult(rgb=result, metadata=metadata)

