"""Strict consumer contracts for a future external colour-match core.

These records bind metadata and opaque payload identities only.  They do not
define D-PCT algorithm parameters, media decoding, file I/O, or product
promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError


MATCH_VIEW_SCHEMA_ID = "neuro-film.reference-core-match-view.v1"
TRANSFORM_BUNDLE_SCHEMA_ID = (
    "neuro-film.reference-core-transform-bundle.v1"
)
DIAGNOSTICS_SCHEMA_ID = "neuro-film.reference-core-diagnostics.v1"
CAPABILITIES_SCHEMA_ID = "neuro-film.reference-core-capabilities.v1"

MATCH_PROFILE_DISPLAY_SRGB = (
    "neuro-film.display-relative-linear-srgb-d65.v1"
)
MATCH_PROFILE_DISPLAY_REC2020 = (
    "neuro-film.display-relative-linear-rec2020-d65.v1"
)
MATCH_PROFILE_SCENE_ACESCG = "scene-relative-linear-acescg-d60.v1"
MATCH_PROFILE_ABSOLUTE_XYZ = "display-absolute-linear-xyz-d65.v1"

_PROFILE_SEMANTICS = {
    MATCH_PROFILE_DISPLAY_SRGB: (
        "display-relative-linear",
        "srgb-rec709",
        "D65",
        False,
    ),
    MATCH_PROFILE_DISPLAY_REC2020: (
        "display-relative-linear",
        "rec2020",
        "D65",
        False,
    ),
    MATCH_PROFILE_SCENE_ACESCG: (
        "scene-relative-linear",
        "aces-ap1",
        "D60",
        False,
    ),
    MATCH_PROFILE_ABSOLUTE_XYZ: (
        "display-absolute-linear",
        "xyz",
        "D65",
        True,
    ),
}
_STATUSES = frozenset(
    {"ok", "identity-fallback", "unsupported", "invalid"}
)
_DETERMINISM_LEVELS = frozenset({"bit-exact", "bounded-numeric"})

_MATCH_VIEW_KEYS = {
    "schema_id",
    "view_id",
    "profile_id",
    "pixel_sha256",
    "shape",
    "strides_bytes",
    "channel_layout",
    "alpha_mode",
    "domain",
    "primaries",
    "white_point",
    "transfer",
    "range",
    "reference_white_nits",
    "render_bridge_id",
    "provenance_fingerprint",
}
_TRANSFORM_KEYS = {
    "schema_id",
    "transform_id",
    "producer_id",
    "producer_version",
    "producer_build_sha256",
    "contract_schema_id",
    "algorithm_id",
    "algorithm_version",
    "binding_scope",
    "intent_id",
    "source_view_id",
    "reference_view_id",
    "payload_schema_id",
    "payload_sha256",
    "capability_requirements",
    "determinism",
}
_DIAGNOSTICS_KEYS = {
    "schema_id",
    "transform_id",
    "status",
    "fallback_reason",
    "finite",
    "out_of_gamut_fraction",
    "clipping_fraction",
    "projected_fraction",
    "confidence",
    "timing_ms",
    "backend_id",
    "backend_fingerprint",
    "warnings",
}
_CAPABILITIES_KEYS = {
    "schema_id",
    "capability_id",
    "producer_id",
    "producer_version",
    "producer_build_sha256",
    "contract_schema_ids",
    "supported_profile_ids",
    "supported_algorithm_ids",
    "feature_flags",
    "determinism",
}


@dataclass(frozen=True)
class MatchViewV1:
    """Serialized metadata for a read-only dense float32 RGB view."""

    schema_id: str
    view_id: str
    profile_id: str
    pixel_sha256: str
    shape: tuple[int, int, int]
    strides_bytes: tuple[int, int, int]
    channel_layout: str
    alpha_mode: str
    domain: str
    primaries: str
    white_point: str
    transfer: str
    range: str
    reference_white_nits: float | None
    render_bridge_id: str
    provenance_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["strides_bytes"] = list(self.strides_bytes)
        return payload


@dataclass(frozen=True)
class TransformBundleV1:
    """Binding to an opaque, source-specific external-core payload."""

    schema_id: str
    transform_id: str
    producer_id: str
    producer_version: str
    producer_build_sha256: str
    contract_schema_id: str
    algorithm_id: str
    algorithm_version: str
    binding_scope: str
    intent_id: str
    source_view_id: str
    reference_view_id: str
    payload_schema_id: str
    payload_sha256: str
    capability_requirements: tuple[str, ...]
    determinism: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capability_requirements"] = list(
            self.capability_requirements
        )
        return payload


@dataclass(frozen=True)
class DiagnosticsV1:
    """Core execution facts consumed by Neuro-Film's product guard."""

    schema_id: str
    transform_id: str
    status: str
    fallback_reason: str | None
    finite: bool
    out_of_gamut_fraction: float
    clipping_fraction: float
    projected_fraction: float
    confidence: float | None
    timing_ms: float
    backend_id: str
    backend_fingerprint: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True)
class CapabilitiesV1:
    """Pinned producer/build capabilities accepted by a consumer."""

    schema_id: str
    capability_id: str
    producer_id: str
    producer_version: str
    producer_build_sha256: str
    contract_schema_ids: tuple[str, ...]
    supported_profile_ids: tuple[str, ...]
    supported_algorithm_ids: tuple[str, ...]
    feature_flags: tuple[str, ...]
    determinism: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_schema_ids"] = list(self.contract_schema_ids)
        payload["supported_profile_ids"] = list(
            self.supported_profile_ids
        )
        payload["supported_algorithm_ids"] = list(
            self.supported_algorithm_ids
        )
        payload["feature_flags"] = list(self.feature_flags)
        return payload


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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceMatchContractError(f"{label} must be an object")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceMatchContractError(
            f"{label} must be a non-empty string"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _finite(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceMatchContractError(
            f"{label} must be a finite number"
        )
    result = float(value)
    if not np.isfinite(result):
        raise ReferenceMatchContractError(
            f"{label} must be a finite number"
        )
    if minimum is not None and result < minimum:
        raise ReferenceMatchContractError(
            f"{label} must be at least {minimum}"
        )
    if maximum is not None and result > maximum:
        raise ReferenceMatchContractError(
            f"{label} must be at most {maximum}"
        )
    return result


def _string_tuple(
    value: Any,
    label: str,
    *,
    nonempty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReferenceMatchContractError(f"{label} must be an array")
    result = tuple(_nonempty(item, f"{label}[]") for item in value)
    if nonempty and not result:
        raise ReferenceMatchContractError(f"{label} must not be empty")
    if result != tuple(sorted(set(result))):
        raise ReferenceMatchContractError(
            f"{label} must be sorted and unique"
        )
    return result


def _shape(value: Any, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
        or value[0] <= 0
        or value[1] <= 0
        or value[2] != 3
    ):
        raise ReferenceMatchContractError(
            f"{label} must be positive HxWx3"
        )
    return int(value[0]), int(value[1]), 3


def _dense_strides(
    value: Any,
    shape: tuple[int, int, int],
    label: str,
) -> tuple[int, int, int]:
    expected = (shape[1] * 12, 12, 4)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or tuple(value) != expected
    ):
        raise ReferenceMatchContractError(
            f"{label} must equal dense float32 RGB strides {expected}"
        )
    return expected


def _identity_payload(value: Any, identity_key: str) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop(identity_key)
    return payload


def make_match_view(
    *,
    profile_id: str,
    pixel_sha256: str,
    shape: tuple[int, int, int],
    render_bridge_id: str,
    provenance_fingerprint: str,
    reference_white_nits: float | None = None,
    alpha_mode: str = "absent",
) -> MatchViewV1:
    if profile_id not in _PROFILE_SEMANTICS:
        raise ReferenceMatchContractError("unsupported match profile")
    domain, primaries, white_point, requires_white = _PROFILE_SEMANTICS[
        profile_id
    ]
    if requires_white:
        reference_white = _finite(
            reference_white_nits,
            "reference_white_nits",
            minimum=np.finfo(np.float64).tiny,
        )
    elif reference_white_nits is not None:
        raise ReferenceMatchContractError(
            "relative/scene match profiles must not declare "
            "reference_white_nits"
        )
    else:
        reference_white = None
    validated_shape = _shape(shape, "shape")
    provisional = MatchViewV1(
        schema_id=MATCH_VIEW_SCHEMA_ID,
        view_id="0" * 64,
        profile_id=profile_id,
        pixel_sha256=_sha256(pixel_sha256, "pixel_sha256"),
        shape=validated_shape,
        strides_bytes=(validated_shape[1] * 12, 12, 4),
        channel_layout="rgb",
        alpha_mode=alpha_mode,
        domain=domain,
        primaries=primaries,
        white_point=white_point,
        transfer="linear",
        range="extended-float",
        reference_white_nits=reference_white,
        render_bridge_id=_nonempty(render_bridge_id, "render_bridge_id"),
        provenance_fingerprint=_sha256(
            provenance_fingerprint,
            "provenance_fingerprint",
        ),
    )
    result = replace(
        provisional,
        view_id=canonical_sha256(
            _identity_payload(provisional, "view_id")
        ),
    )
    validate_match_view(result)
    return result


def validate_match_view(value: MatchViewV1) -> None:
    if not isinstance(value, MatchViewV1):
        raise ReferenceMatchContractError("match view must be MatchViewV1")
    if value.schema_id != MATCH_VIEW_SCHEMA_ID:
        raise ReferenceMatchContractError("unsupported match-view schema")
    _sha256(value.view_id, "view_id")
    _sha256(value.pixel_sha256, "pixel_sha256")
    shape = _shape(value.shape, "shape")
    _dense_strides(value.strides_bytes, shape, "strides_bytes")
    if value.profile_id not in _PROFILE_SEMANTICS:
        raise ReferenceMatchContractError("unsupported match profile")
    expected = _PROFILE_SEMANTICS[value.profile_id]
    if (
        value.domain,
        value.primaries,
        value.white_point,
    ) != expected[:3]:
        raise ReferenceMatchContractError(
            "match profile colour semantics mismatch"
        )
    if value.channel_layout != "rgb":
        raise ReferenceMatchContractError("channel_layout must be rgb")
    if value.alpha_mode not in {"absent", "straight", "premultiplied"}:
        raise ReferenceMatchContractError("unsupported alpha_mode")
    if value.transfer != "linear" or value.range != "extended-float":
        raise ReferenceMatchContractError(
            "match view must be linear extended-float"
        )
    if expected[3]:
        _finite(
            value.reference_white_nits,
            "reference_white_nits",
            minimum=np.finfo(np.float64).tiny,
        )
    elif value.reference_white_nits is not None:
        raise ReferenceMatchContractError(
            "relative/scene match profiles must not declare "
            "reference_white_nits"
        )
    _nonempty(value.render_bridge_id, "render_bridge_id")
    _sha256(value.provenance_fingerprint, "provenance_fingerprint")
    if value.view_id != canonical_sha256(
        _identity_payload(value, "view_id")
    ):
        raise ReferenceMatchContractError(
            "view_id does not match canonical payload"
        )


def make_transform_bundle(
    *,
    producer_id: str,
    producer_version: str,
    producer_build_sha256: str,
    contract_schema_id: str,
    algorithm_id: str,
    algorithm_version: str,
    intent_id: str,
    source_view_id: str,
    reference_view_id: str,
    payload_schema_id: str,
    payload_sha256: str,
    capability_requirements: Sequence[str] = (),
    determinism: str = "bounded-numeric",
) -> TransformBundleV1:
    provisional = TransformBundleV1(
        schema_id=TRANSFORM_BUNDLE_SCHEMA_ID,
        transform_id="0" * 64,
        producer_id=_nonempty(producer_id, "producer_id"),
        producer_version=_nonempty(
            producer_version,
            "producer_version",
        ),
        producer_build_sha256=_sha256(
            producer_build_sha256,
            "producer_build_sha256",
        ),
        contract_schema_id=_nonempty(
            contract_schema_id,
            "contract_schema_id",
        ),
        algorithm_id=_nonempty(algorithm_id, "algorithm_id"),
        algorithm_version=_nonempty(
            algorithm_version,
            "algorithm_version",
        ),
        binding_scope="source-bound",
        intent_id=_sha256(intent_id, "intent_id"),
        source_view_id=_sha256(source_view_id, "source_view_id"),
        reference_view_id=_sha256(
            reference_view_id,
            "reference_view_id",
        ),
        payload_schema_id=_nonempty(
            payload_schema_id,
            "payload_schema_id",
        ),
        payload_sha256=_sha256(payload_sha256, "payload_sha256"),
        capability_requirements=tuple(
            sorted(set(capability_requirements))
        ),
        determinism=determinism,
    )
    result = replace(
        provisional,
        transform_id=canonical_sha256(
            _identity_payload(provisional, "transform_id")
        ),
    )
    validate_transform_bundle(result)
    return result


def validate_transform_bundle(value: TransformBundleV1) -> None:
    if not isinstance(value, TransformBundleV1):
        raise ReferenceMatchContractError(
            "transform bundle must be TransformBundleV1"
        )
    if value.schema_id != TRANSFORM_BUNDLE_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported transform-bundle schema"
        )
    _sha256(value.transform_id, "transform_id")
    _nonempty(value.producer_id, "producer_id")
    _nonempty(value.producer_version, "producer_version")
    _sha256(value.producer_build_sha256, "producer_build_sha256")
    _nonempty(value.contract_schema_id, "contract_schema_id")
    _nonempty(value.algorithm_id, "algorithm_id")
    _nonempty(value.algorithm_version, "algorithm_version")
    if value.binding_scope != "source-bound":
        raise ReferenceMatchContractError(
            "v1 transform bundles must be source-bound"
        )
    _sha256(value.intent_id, "intent_id")
    _sha256(value.source_view_id, "source_view_id")
    _sha256(value.reference_view_id, "reference_view_id")
    _nonempty(value.payload_schema_id, "payload_schema_id")
    _sha256(value.payload_sha256, "payload_sha256")
    _string_tuple(
        value.capability_requirements,
        "capability_requirements",
        nonempty=False,
    )
    if value.determinism not in _DETERMINISM_LEVELS:
        raise ReferenceMatchContractError("unsupported determinism level")
    if value.transform_id != canonical_sha256(
        _identity_payload(value, "transform_id")
    ):
        raise ReferenceMatchContractError(
            "transform_id does not match canonical payload"
        )


def make_capabilities(
    *,
    producer_id: str,
    producer_version: str,
    producer_build_sha256: str,
    contract_schema_ids: Sequence[str],
    supported_profile_ids: Sequence[str],
    supported_algorithm_ids: Sequence[str],
    feature_flags: Sequence[str] = (),
    determinism: str = "bounded-numeric",
) -> CapabilitiesV1:
    provisional = CapabilitiesV1(
        schema_id=CAPABILITIES_SCHEMA_ID,
        capability_id="0" * 64,
        producer_id=_nonempty(producer_id, "producer_id"),
        producer_version=_nonempty(
            producer_version,
            "producer_version",
        ),
        producer_build_sha256=_sha256(
            producer_build_sha256,
            "producer_build_sha256",
        ),
        contract_schema_ids=tuple(sorted(set(contract_schema_ids))),
        supported_profile_ids=tuple(sorted(set(supported_profile_ids))),
        supported_algorithm_ids=tuple(
            sorted(set(supported_algorithm_ids))
        ),
        feature_flags=tuple(sorted(set(feature_flags))),
        determinism=determinism,
    )
    result = replace(
        provisional,
        capability_id=canonical_sha256(
            _identity_payload(provisional, "capability_id")
        ),
    )
    validate_capabilities(result)
    return result


def validate_capabilities(value: CapabilitiesV1) -> None:
    if not isinstance(value, CapabilitiesV1):
        raise ReferenceMatchContractError(
            "capabilities must be CapabilitiesV1"
        )
    if value.schema_id != CAPABILITIES_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported capabilities schema"
        )
    _sha256(value.capability_id, "capability_id")
    _nonempty(value.producer_id, "producer_id")
    _nonempty(value.producer_version, "producer_version")
    _sha256(value.producer_build_sha256, "producer_build_sha256")
    _string_tuple(
        value.contract_schema_ids,
        "contract_schema_ids",
        nonempty=True,
    )
    profiles = _string_tuple(
        value.supported_profile_ids,
        "supported_profile_ids",
        nonempty=True,
    )
    if any(profile not in _PROFILE_SEMANTICS for profile in profiles):
        raise ReferenceMatchContractError(
            "capabilities contain an unsupported match profile"
        )
    _string_tuple(
        value.supported_algorithm_ids,
        "supported_algorithm_ids",
        nonempty=True,
    )
    _string_tuple(
        value.feature_flags,
        "feature_flags",
        nonempty=False,
    )
    if value.determinism not in _DETERMINISM_LEVELS:
        raise ReferenceMatchContractError("unsupported determinism level")
    if value.capability_id != canonical_sha256(
        _identity_payload(value, "capability_id")
    ):
        raise ReferenceMatchContractError(
            "capability_id does not match canonical payload"
        )


def validate_diagnostics(value: DiagnosticsV1) -> None:
    if not isinstance(value, DiagnosticsV1):
        raise ReferenceMatchContractError(
            "diagnostics must be DiagnosticsV1"
        )
    if value.schema_id != DIAGNOSTICS_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported diagnostics schema"
        )
    _sha256(value.transform_id, "transform_id")
    if value.status not in _STATUSES:
        raise ReferenceMatchContractError("unsupported core status")
    if value.status == "ok":
        if value.fallback_reason is not None:
            raise ReferenceMatchContractError(
                "ok diagnostics must not have a fallback reason"
            )
        if value.finite is not True:
            raise ReferenceMatchContractError(
                "ok diagnostics must report finite output"
            )
    else:
        _nonempty(value.fallback_reason, "fallback_reason")
    if not isinstance(value.finite, bool):
        raise ReferenceMatchContractError("finite must be boolean")
    for label in (
        "out_of_gamut_fraction",
        "clipping_fraction",
        "projected_fraction",
    ):
        _finite(getattr(value, label), label, minimum=0.0, maximum=1.0)
    if value.confidence is not None:
        _finite(
            value.confidence,
            "confidence",
            minimum=0.0,
            maximum=1.0,
        )
    _finite(value.timing_ms, "timing_ms", minimum=0.0)
    _nonempty(value.backend_id, "backend_id")
    _sha256(value.backend_fingerprint, "backend_fingerprint")
    _string_tuple(value.warnings, "warnings", nonempty=False)


def validate_core_binding(
    *,
    source: MatchViewV1,
    reference: MatchViewV1,
    transform: TransformBundleV1,
    capabilities: CapabilitiesV1,
    diagnostics: DiagnosticsV1 | None = None,
) -> None:
    """Validate producer/build/view bindings without promoting the result."""

    validate_match_view(source)
    validate_match_view(reference)
    validate_transform_bundle(transform)
    validate_capabilities(capabilities)
    if transform.source_view_id != source.view_id:
        raise ReferenceMatchContractError(
            "transform source_view_id mismatch"
        )
    if transform.reference_view_id != reference.view_id:
        raise ReferenceMatchContractError(
            "transform reference_view_id mismatch"
        )
    producer = (
        transform.producer_id,
        transform.producer_version,
        transform.producer_build_sha256,
    )
    advertised = (
        capabilities.producer_id,
        capabilities.producer_version,
        capabilities.producer_build_sha256,
    )
    if producer != advertised:
        raise ReferenceMatchContractError(
            "transform producer/build does not match capabilities"
        )
    if transform.contract_schema_id not in capabilities.contract_schema_ids:
        raise ReferenceMatchContractError(
            "transform contract schema is not advertised"
        )
    if transform.algorithm_id not in capabilities.supported_algorithm_ids:
        raise ReferenceMatchContractError(
            "transform algorithm is not advertised"
        )
    for view in (source, reference):
        if view.profile_id not in capabilities.supported_profile_ids:
            raise ReferenceMatchContractError(
                "match-view profile is not advertised"
            )
    missing = set(transform.capability_requirements) - set(
        capabilities.feature_flags
    )
    if missing:
        raise ReferenceMatchContractError(
            f"missing required core capabilities: {sorted(missing)}"
        )
    if transform.determinism != capabilities.determinism:
        raise ReferenceMatchContractError(
            "transform determinism does not match capabilities"
        )
    if diagnostics is not None:
        validate_diagnostics(diagnostics)
        if diagnostics.transform_id != transform.transform_id:
            raise ReferenceMatchContractError(
                "diagnostics transform_id mismatch"
            )


def _to_json(value: Any, validator: Any) -> str:
    validator(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def match_view_to_json(value: MatchViewV1) -> str:
    return _to_json(value, validate_match_view)


def transform_bundle_to_json(value: TransformBundleV1) -> str:
    return _to_json(value, validate_transform_bundle)


def diagnostics_to_json(value: DiagnosticsV1) -> str:
    return _to_json(value, validate_diagnostics)


def capabilities_to_json(value: CapabilitiesV1) -> str:
    return _to_json(value, validate_capabilities)


def _from_json(
    encoded: str,
    *,
    expected_keys: set[str],
    cls: Any,
    validator: Any,
    label: str,
) -> Any:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            f"{label} is not valid JSON"
        ) from exc
    value = _mapping(payload, label)
    _strict_keys(value, expected_keys, label)
    converted = dict(value)
    for key in (
        "shape",
        "strides_bytes",
        "capability_requirements",
        "contract_schema_ids",
        "supported_profile_ids",
        "supported_algorithm_ids",
        "feature_flags",
        "warnings",
    ):
        if key in converted:
            converted[key] = tuple(converted[key])
    try:
        result = cls(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            f"{label} contains invalid fields"
        ) from exc
    validator(result)
    return result


def match_view_from_json(encoded: str) -> MatchViewV1:
    return _from_json(
        encoded,
        expected_keys=_MATCH_VIEW_KEYS,
        cls=MatchViewV1,
        validator=validate_match_view,
        label="match view",
    )


def transform_bundle_from_json(encoded: str) -> TransformBundleV1:
    return _from_json(
        encoded,
        expected_keys=_TRANSFORM_KEYS,
        cls=TransformBundleV1,
        validator=validate_transform_bundle,
        label="transform bundle",
    )


def diagnostics_from_json(encoded: str) -> DiagnosticsV1:
    return _from_json(
        encoded,
        expected_keys=_DIAGNOSTICS_KEYS,
        cls=DiagnosticsV1,
        validator=validate_diagnostics,
        label="diagnostics",
    )


def capabilities_from_json(encoded: str) -> CapabilitiesV1:
    return _from_json(
        encoded,
        expected_keys=_CAPABILITIES_KEYS,
        cls=CapabilitiesV1,
        validator=validate_capabilities,
        label="capabilities",
    )


__all__ = [
    "CAPABILITIES_SCHEMA_ID",
    "DIAGNOSTICS_SCHEMA_ID",
    "MATCH_PROFILE_ABSOLUTE_XYZ",
    "MATCH_PROFILE_DISPLAY_REC2020",
    "MATCH_PROFILE_DISPLAY_SRGB",
    "MATCH_PROFILE_SCENE_ACESCG",
    "MATCH_VIEW_SCHEMA_ID",
    "TRANSFORM_BUNDLE_SCHEMA_ID",
    "CapabilitiesV1",
    "DiagnosticsV1",
    "MatchViewV1",
    "TransformBundleV1",
    "capabilities_from_json",
    "capabilities_to_json",
    "diagnostics_from_json",
    "diagnostics_to_json",
    "make_capabilities",
    "make_match_view",
    "make_transform_bundle",
    "match_view_from_json",
    "match_view_to_json",
    "transform_bundle_from_json",
    "transform_bundle_to_json",
    "validate_capabilities",
    "validate_core_binding",
    "validate_diagnostics",
    "validate_match_view",
    "validate_transform_bundle",
]
