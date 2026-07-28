"""Language-neutral conformance vectors for reference-match engine ports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.color_engine import linear_rgb_to_lab
from src.preprocess.types import SourceProfile, WorkingImage

from .strict_json import strict_json_loads
from .canonical import canonical_sha256
from .contracts import (
    REFERENCE_LOOK_ALGORITHM_ID,
    SUPPORTED_WORKING_SPACES,
    ReferenceLookPolicy,
    ReferenceMatchContractError,
    validate_policy,
)
from .fit import fit_reference_look
from .render import render_reference_look


PORTABLE_CONFORMANCE_SCHEMA_ID = (
    "neuro-film.reference-match-portable-conformance.v1"
)
PORTABLE_CONFORMANCE_RESULT_SCHEMA_ID = (
    "neuro-film.reference-match-portable-conformance-result.v1"
)
_FLOAT_ENCODING = "ieee754-binary32-big-endian-hex"
_BUNDLE_KEYS = {
    "schema_id",
    "algorithm_id",
    "fixture_id",
    "float_encoding",
    "tolerances",
    "cases",
}
_TOLERANCE_KEYS = {
    "linear_rgb_max_abs",
    "lab_delta_e76_p95",
    "lab_delta_e76_max",
    "diagnostics_max_abs",
}
_CASE_KEYS = {"case_id", "reference", "source", "policy", "expected"}
_IMAGE_KEYS = {"shape", "working_space", "pixels_f32be_hex"}
_EXPECTED_KEYS = {
    "recipe_id",
    "reference_pixel_sha256",
    "output_f32be_hex",
    "gamut_adjusted_fraction",
    "output_min",
    "output_max",
}
_POLICY_KEYS = {field.name for field in fields(ReferenceLookPolicy)}


@dataclass(frozen=True)
class PortableConformanceCaseResult:
    """One executable cross-language conformance decision."""

    case_id: str
    passed: bool
    recipe_identity_match: bool
    reference_pixel_identity_match: bool
    linear_rgb_max_abs: float
    lab_delta_e76_p95: float
    lab_delta_e76_max: float
    diagnostics_max_abs: float


@dataclass(frozen=True)
class PortableConformanceResult:
    """Deterministic result for one frozen conformance bundle."""

    schema_id: str
    fixture_id: str
    algorithm_id: str
    passed: bool
    case_results: tuple[PortableConformanceCaseResult, ...]


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


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceMatchContractError(f"{label} must be a positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ReferenceMatchContractError(f"{label} must be a positive number")
    return result


def _lower_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def encode_f32be_hex(value: np.ndarray) -> str:
    """Encode a finite float32 array as shape-independent network-order bits."""

    array = np.asarray(value)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ReferenceMatchContractError(
            "portable float payload must be finite float32"
        )
    return array.astype(">f4", copy=False).tobytes(order="C").hex()


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
        raise ReferenceMatchContractError(f"{label} must be positive HxWx3")
    return value[0], value[1], value[2]


def decode_f32be_hex(
    encoded: Any,
    shape: tuple[int, int, int],
    *,
    label: str,
) -> np.ndarray:
    """Decode exact IEEE-754 binary32 bits into a native float32 image."""

    expected_characters = int(np.prod(shape, dtype=np.int64)) * 8
    if (
        not isinstance(encoded, str)
        or len(encoded) != expected_characters
        or any(char not in "0123456789abcdef" for char in encoded)
    ):
        raise ReferenceMatchContractError(
            f"{label} must contain exactly {expected_characters} "
            "lowercase hexadecimal characters"
        )
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as exc:
        raise ReferenceMatchContractError(
            f"{label} is not valid hexadecimal"
        ) from exc
    result = np.frombuffer(raw, dtype=">f4").astype(np.float32).reshape(shape)
    if not np.isfinite(result).all():
        raise ReferenceMatchContractError(f"{label} must decode to finite values")
    return result


def _working_image(payload: Any, label: str) -> WorkingImage:
    image = _mapping(payload, label)
    _strict_keys(image, _IMAGE_KEYS, label)
    shape = _shape(image["shape"], f"{label}.shape")
    working_space = image["working_space"]
    if working_space not in SUPPORTED_WORKING_SPACES:
        raise ReferenceMatchContractError(
            f"{label}.working_space is unsupported"
        )
    pixels = decode_f32be_hex(
        image["pixels_f32be_hex"],
        shape,
        label=f"{label}.pixels_f32be_hex",
    )
    if float(np.min(pixels)) < 0.0 or float(np.max(pixels)) > 1.0:
        raise ReferenceMatchContractError(
            f"{label} pixels must lie within the declared working gamut"
        )
    return WorkingImage(
        pixels=pixels,
        working_space=working_space,
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile(
            "unknown",
            "portable-conformance-fixture",
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path(f"{label}.f32"),
        warnings=[],
    )


def _policy(payload: Any, label: str) -> ReferenceLookPolicy:
    value = _mapping(payload, label)
    _strict_keys(value, _POLICY_KEYS, label)
    try:
        policy = ReferenceLookPolicy(**dict(value))
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            f"{label} contains invalid policy values"
        ) from exc
    validate_policy(policy)
    return policy


def _validated_bundle(
    payload: Any,
) -> tuple[
    Mapping[str, Any],
    dict[str, float],
    tuple[Mapping[str, Any], ...],
]:
    bundle = _mapping(payload, "conformance bundle")
    _strict_keys(bundle, _BUNDLE_KEYS, "conformance bundle")
    if bundle["schema_id"] != PORTABLE_CONFORMANCE_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported portable conformance schema"
        )
    if bundle["algorithm_id"] != REFERENCE_LOOK_ALGORITHM_ID:
        raise ReferenceMatchContractError(
            "portable conformance algorithm mismatch"
        )
    if bundle["float_encoding"] != _FLOAT_ENCODING:
        raise ReferenceMatchContractError(
            "unsupported portable conformance float encoding"
        )
    fixture_id = _lower_sha256(bundle["fixture_id"], "fixture_id")
    identity_payload = dict(bundle)
    identity_payload.pop("fixture_id")
    if fixture_id != canonical_sha256(identity_payload):
        raise ReferenceMatchContractError(
            "fixture_id does not match canonical bundle payload"
        )

    tolerances_payload = _mapping(bundle["tolerances"], "tolerances")
    _strict_keys(tolerances_payload, _TOLERANCE_KEYS, "tolerances")
    tolerances = {
        key: _finite_positive(tolerances_payload[key], f"tolerances.{key}")
        for key in sorted(_TOLERANCE_KEYS)
    }

    cases_payload = bundle["cases"]
    if not isinstance(cases_payload, list) or not cases_payload:
        raise ReferenceMatchContractError(
            "conformance cases must be a non-empty array"
        )
    cases: list[Mapping[str, Any]] = []
    case_ids: set[str] = set()
    for index, raw_case in enumerate(cases_payload):
        case = _mapping(raw_case, f"cases[{index}]")
        _strict_keys(case, _CASE_KEYS, f"cases[{index}]")
        case_id = case["case_id"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
        ):
            raise ReferenceMatchContractError(
                "case_id must be a non-empty unique string"
            )
        case_ids.add(case_id)
        cases.append(case)
    return bundle, tolerances, tuple(cases)


def load_portable_conformance_bundle(path: Path) -> Mapping[str, Any]:
    """Load one strict finite JSON conformance bundle."""

    try:
        payload = strict_json_loads(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "portable conformance bundle is not valid finite JSON"
        ) from exc
    _validated_bundle(payload)
    return payload


def verify_portable_conformance_bundle(
    payload: Mapping[str, Any],
) -> PortableConformanceResult:
    """Execute every vector and apply its exact-ID and numeric gates."""

    bundle, tolerances, cases = _validated_bundle(payload)
    results: list[PortableConformanceCaseResult] = []
    for index, case in enumerate(cases):
        label = f"cases[{index}]"
        reference = _working_image(case["reference"], f"{label}.reference")
        source = _working_image(case["source"], f"{label}.source")
        policy = _policy(case["policy"], f"{label}.policy")
        expected = _mapping(case["expected"], f"{label}.expected")
        _strict_keys(expected, _EXPECTED_KEYS, f"{label}.expected")
        expected_recipe_id = _lower_sha256(
            expected["recipe_id"],
            f"{label}.expected.recipe_id",
        )
        expected_reference_id = _lower_sha256(
            expected["reference_pixel_sha256"],
            f"{label}.expected.reference_pixel_sha256",
        )
        expected_output = decode_f32be_hex(
            expected["output_f32be_hex"],
            tuple(int(value) for value in source.pixels.shape),
            label=f"{label}.expected.output_f32be_hex",
        )
        expected_diagnostics = np.asarray(
            [
                expected["gamut_adjusted_fraction"],
                expected["output_min"],
                expected["output_max"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(expected_diagnostics).all():
            raise ReferenceMatchContractError(
                f"{label}.expected diagnostics must be finite"
            )

        recipe = fit_reference_look(reference, policy=policy)
        rendered = render_reference_look(recipe, source)
        actual_output = rendered.image.pixels
        rgb_max_abs = float(
            np.max(
                np.abs(
                    actual_output.astype(np.float64)
                    - expected_output.astype(np.float64)
                )
            )
        )
        actual_lab = linear_rgb_to_lab(
            actual_output,
            working_space=source.working_space,
        )
        expected_lab = linear_rgb_to_lab(
            expected_output,
            working_space=source.working_space,
        )
        delta_e = np.linalg.norm(
            actual_lab.astype(np.float64) - expected_lab.astype(np.float64),
            axis=2,
        )
        delta_p95 = float(np.percentile(delta_e, 95.0))
        delta_max = float(np.max(delta_e))
        actual_diagnostics = np.asarray(
            [
                rendered.diagnostics.gamut_adjusted_fraction,
                rendered.diagnostics.output_min,
                rendered.diagnostics.output_max,
            ],
            dtype=np.float64,
        )
        diagnostics_max_abs = float(
            np.max(np.abs(actual_diagnostics - expected_diagnostics))
        )
        recipe_match = recipe.recipe_id == expected_recipe_id
        reference_match = (
            recipe.reference_pixel_sha256 == expected_reference_id
        )
        passed = (
            recipe_match
            and reference_match
            and rgb_max_abs <= tolerances["linear_rgb_max_abs"]
            and delta_p95 <= tolerances["lab_delta_e76_p95"]
            and delta_max <= tolerances["lab_delta_e76_max"]
            and diagnostics_max_abs <= tolerances["diagnostics_max_abs"]
        )
        results.append(
            PortableConformanceCaseResult(
                case_id=str(case["case_id"]),
                passed=passed,
                recipe_identity_match=recipe_match,
                reference_pixel_identity_match=reference_match,
                linear_rgb_max_abs=rgb_max_abs,
                lab_delta_e76_p95=delta_p95,
                lab_delta_e76_max=delta_max,
                diagnostics_max_abs=diagnostics_max_abs,
            )
        )

    return PortableConformanceResult(
        schema_id=PORTABLE_CONFORMANCE_RESULT_SCHEMA_ID,
        fixture_id=str(bundle["fixture_id"]),
        algorithm_id=str(bundle["algorithm_id"]),
        passed=all(result.passed for result in results),
        case_results=tuple(results),
    )


def portable_conformance_result_to_json(
    result: PortableConformanceResult,
) -> str:
    """Serialize a deterministic finite result report."""

    return json.dumps(
        asdict(result),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


__all__ = [
    "PORTABLE_CONFORMANCE_RESULT_SCHEMA_ID",
    "PORTABLE_CONFORMANCE_SCHEMA_ID",
    "PortableConformanceCaseResult",
    "PortableConformanceResult",
    "decode_f32be_hex",
    "encode_f32be_hex",
    "load_portable_conformance_bundle",
    "portable_conformance_result_to_json",
    "verify_portable_conformance_bundle",
]
