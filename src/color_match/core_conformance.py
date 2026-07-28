"""Frozen cross-language conformance for the external-core consumer boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.preprocess import DecodeWarning, SourceProfile, WorkingImage

from .strict_json import strict_json_loads
from .canonical import canonical_sha256
from .conformance import decode_f32be_hex
from .contracts import ReferenceMatchContractError
from .core_adapter import (
    prepare_working_image_match_view,
    validate_prepared_view_support,
)
from .core_contracts import (
    CapabilitiesV1,
    capabilities_from_json,
)


CORE_CONSUMER_CONFORMANCE_SCHEMA_ID = (
    "neuro-film.reference-core-consumer-conformance.v1"
)
CORE_CONSUMER_CONFORMANCE_RESULT_SCHEMA_ID = (
    "neuro-film.reference-core-consumer-conformance-result.v1"
)
_FLOAT_ENCODING = "ieee754-binary32-big-endian-hex"
_PRODUCER_ROLE = "synthetic-consumer-fixture"
_BUNDLE_KEYS = {
    "schema_id",
    "fixture_id",
    "float_encoding",
    "producer_role",
    "capabilities",
    "algorithm_id",
    "contract_schema_id",
    "capability_requirements",
    "cases",
}
_CASE_KEYS = {
    "case_id",
    "working_space",
    "source_transfer_state",
    "source_profile",
    "hdr_metadata",
    "orientation_applied",
    "alpha_policy",
    "bit_depth_in",
    "warnings",
    "shape",
    "pixels_f32be_hex",
    "expected_match_view",
}
_SOURCE_PROFILE_KEYS = {"kind", "description", "bytes_length"}
_WARNING_KEYS = {"code", "message"}


@dataclass(frozen=True)
class CoreConsumerConformanceCaseResultV1:
    case_id: str
    passed: bool
    reasons: tuple[str, ...]
    expected_view_id: str
    actual_view_id: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class CoreConsumerConformanceResultV1:
    schema_id: str
    fixture_id: str
    passed: bool
    case_results: tuple[CoreConsumerConformanceCaseResultV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "fixture_id": self.fixture_id,
            "passed": self.passed,
            "case_results": [
                case.to_dict() for case in self.case_results
            ],
        }


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


def _shape(value: Any, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
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
    return value[0], value[1], value[2]


def _string_array(
    value: Any,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ReferenceMatchContractError(f"{label} must be an array")
    result = tuple(_nonempty(item, f"{label}[]") for item in value)
    if result != tuple(sorted(set(result))):
        raise ReferenceMatchContractError(
            f"{label} must be sorted and unique"
        )
    return result


def _fixture_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(bundle)
    payload.pop("fixture_id", None)
    return payload


def _source_profile(
    value: Any,
    label: str,
) -> SourceProfile:
    payload = _mapping(value, label)
    _strict_keys(payload, _SOURCE_PROFILE_KEYS, label)
    kind = payload["kind"]
    if kind not in {
        "icc",
        "cicp",
        "nclx",
        "raw_metadata",
        "assumed_srgb",
        "unknown",
    }:
        raise ReferenceMatchContractError(
            f"{label}.kind is unsupported"
        )
    bytes_length = payload["bytes_length"]
    if (
        isinstance(bytes_length, bool)
        or not isinstance(bytes_length, int)
        or bytes_length < 0
    ):
        raise ReferenceMatchContractError(
            f"{label}.bytes_length must be a non-negative integer"
        )
    return SourceProfile(
        kind,
        _nonempty(payload["description"], f"{label}.description"),
        bytes_length,
    )


def _warnings(value: Any, label: str) -> list[DecodeWarning]:
    if not isinstance(value, list):
        raise ReferenceMatchContractError(f"{label} must be an array")
    result: list[DecodeWarning] = []
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        payload = _mapping(item, item_label)
        _strict_keys(payload, _WARNING_KEYS, item_label)
        result.append(
            DecodeWarning(
                _nonempty(payload["code"], f"{item_label}.code"),
                _nonempty(
                    payload["message"],
                    f"{item_label}.message",
                ),
            )
        )
    return result


def _working_image(case: Mapping[str, Any]) -> WorkingImage:
    shape = _shape(case["shape"], "case.shape")
    pixels = decode_f32be_hex(
        case["pixels_f32be_hex"],
        shape,
        label="case.pixels_f32be_hex",
    )
    if case["working_space"] not in {"linear_srgb", "linear_rec2020"}:
        raise ReferenceMatchContractError(
            "case.working_space is unsupported"
        )
    source_transfer = case["source_transfer_state"]
    if source_transfer not in {
        "scene_linear",
        "display_linear",
        "display_referred",
        "unknown",
    }:
        raise ReferenceMatchContractError(
            "case.source_transfer_state is unsupported"
        )
    if case["orientation_applied"] is not True:
        raise ReferenceMatchContractError(
            "consumer conformance requires applied orientation"
        )
    if case["alpha_policy"] != "absent":
        raise ReferenceMatchContractError(
            "consumer conformance requires absent alpha"
        )
    bit_depth = case["bit_depth_in"]
    if (
        isinstance(bit_depth, bool)
        or not isinstance(bit_depth, int)
        or bit_depth < 1
    ):
        raise ReferenceMatchContractError(
            "case.bit_depth_in must be a positive integer"
        )
    hdr_metadata = case["hdr_metadata"]
    if not isinstance(hdr_metadata, Mapping):
        raise ReferenceMatchContractError(
            "case.hdr_metadata must be an object"
        )
    return WorkingImage(
        pixels=pixels,
        working_space=case["working_space"],
        transfer_state="display_linear",
        source_transfer_state=source_transfer,
        source_profile=_source_profile(
            case["source_profile"],
            "case.source_profile",
        ),
        hdr_metadata=dict(hdr_metadata),
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=bit_depth,
        source_path=Path("fixture-a.bin"),
        warnings=_warnings(case["warnings"], "case.warnings"),
    )


def _verify_case(
    case: Mapping[str, Any],
    *,
    capabilities: CapabilitiesV1,
    algorithm_id: str,
    contract_schema_id: str,
    capability_requirements: tuple[str, ...],
) -> CoreConsumerConformanceCaseResultV1:
    expected = _mapping(
        case["expected_match_view"],
        "case.expected_match_view",
    )
    expected_view_id = _sha256(
        expected.get("view_id"),
        "case.expected_match_view.view_id",
    )
    reasons: list[str] = []
    actual_view_id = "0" * 64
    try:
        working = _working_image(case)
        prepared = prepare_working_image_match_view(working)
        actual_view_id = prepared.descriptor.view_id
        validate_prepared_view_support(
            prepared,
            capabilities,
            algorithm_id=algorithm_id,
            contract_schema_id=contract_schema_id,
            capability_requirements=capability_requirements,
        )
        if prepared.descriptor.to_dict() != dict(expected):
            reasons.append("match-view-payload")
        moved = WorkingImage(
            pixels=working.pixels.copy(),
            working_space=working.working_space,
            transfer_state=working.transfer_state,
            source_transfer_state=working.source_transfer_state,
            source_profile=working.source_profile,
            hdr_metadata=dict(working.hdr_metadata),
            orientation_applied=working.orientation_applied,
            alpha_policy=working.alpha_policy,
            bit_depth_in=working.bit_depth_in,
            source_path=Path("fixture-b.bin"),
            warnings=list(working.warnings),
        )
        if (
            prepare_working_image_match_view(moved).descriptor
            != prepared.descriptor
        ):
            reasons.append("path-dependent-identity")
    except ReferenceMatchContractError as exc:
        reasons.append(f"contract:{exc}")
    normalized = tuple(sorted(set(reasons)))
    return CoreConsumerConformanceCaseResultV1(
        case_id=_nonempty(case["case_id"], "case.case_id"),
        passed=not normalized,
        reasons=normalized,
        expected_view_id=expected_view_id,
        actual_view_id=actual_view_id,
    )


def verify_core_consumer_conformance_bundle(
    value: Mapping[str, Any] | Path | str,
) -> CoreConsumerConformanceResultV1:
    if isinstance(value, (Path, str)):
        try:
            bundle = strict_json_loads(
                Path(value).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceMatchContractError(
                "consumer conformance bundle is unreadable"
            ) from exc
    else:
        bundle = value
    payload = _mapping(bundle, "consumer conformance bundle")
    _strict_keys(payload, _BUNDLE_KEYS, "consumer conformance bundle")
    if payload["schema_id"] != CORE_CONSUMER_CONFORMANCE_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported consumer conformance schema"
        )
    fixture_id = _sha256(payload["fixture_id"], "fixture_id")
    if fixture_id != canonical_sha256(_fixture_payload(payload)):
        raise ReferenceMatchContractError(
            "fixture_id does not match canonical payload"
        )
    if payload["float_encoding"] != _FLOAT_ENCODING:
        raise ReferenceMatchContractError(
            "unsupported consumer conformance float encoding"
        )
    if payload["producer_role"] != _PRODUCER_ROLE:
        raise ReferenceMatchContractError(
            "consumer conformance producer role must be synthetic"
        )
    capabilities = capabilities_from_json(
        json.dumps(payload["capabilities"])
    )
    algorithm_id = _nonempty(
        payload["algorithm_id"],
        "algorithm_id",
    )
    contract_schema_id = _nonempty(
        payload["contract_schema_id"],
        "contract_schema_id",
    )
    requirements = _string_array(
        payload["capability_requirements"],
        "capability_requirements",
    )
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) < 2:
        raise ReferenceMatchContractError(
            "consumer conformance requires at least two cases"
        )
    identifiers: set[str] = set()
    results: list[CoreConsumerConformanceCaseResultV1] = []
    for index, item in enumerate(cases):
        case = _mapping(item, f"cases[{index}]")
        _strict_keys(case, _CASE_KEYS, f"cases[{index}]")
        case_id = _nonempty(case["case_id"], f"cases[{index}].case_id")
        if case_id in identifiers:
            raise ReferenceMatchContractError(
                "consumer conformance case IDs must be unique"
            )
        identifiers.add(case_id)
        results.append(
            _verify_case(
                case,
                capabilities=capabilities,
                algorithm_id=algorithm_id,
                contract_schema_id=contract_schema_id,
                capability_requirements=requirements,
            )
        )
    frozen_results = tuple(results)
    return CoreConsumerConformanceResultV1(
        schema_id=CORE_CONSUMER_CONFORMANCE_RESULT_SCHEMA_ID,
        fixture_id=fixture_id,
        passed=all(result.passed for result in frozen_results),
        case_results=frozen_results,
    )


def core_consumer_conformance_result_to_json(
    value: CoreConsumerConformanceResultV1,
) -> str:
    if not isinstance(value, CoreConsumerConformanceResultV1):
        raise ReferenceMatchContractError(
            "consumer conformance result has invalid type"
        )
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


__all__ = [
    "CORE_CONSUMER_CONFORMANCE_RESULT_SCHEMA_ID",
    "CORE_CONSUMER_CONFORMANCE_SCHEMA_ID",
    "CoreConsumerConformanceCaseResultV1",
    "CoreConsumerConformanceResultV1",
    "core_consumer_conformance_result_to_json",
    "verify_core_consumer_conformance_bundle",
]
