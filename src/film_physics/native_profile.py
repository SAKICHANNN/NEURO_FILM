"""Portable native parameter boundary for the physical print sub-chain.

This module compiles only the pointwise scene-linear sensitometry and
density-to-print interpretation parameters.  Spatial response, neutral-axis
gauge, output transfer and the AO6 display-look remain outside this v1 ABI.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from src.roll2film.sensitometry_print import SensitometryPrintOperator

from .profile_consumer import validate_standalone_profile_artifact


NATIVE_PRINT_PROFILE_SCHEMA = (
    "neuro_film.native_sensitometry_print_profile.v1"
)
NATIVE_PRINT_ABI = "nf_physical_print_v1"
NATIVE_PRINT_ABI_VERSION = 1
NATIVE_PRINT_MAX_KNOTS = 16


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def native_print_payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def compile_native_print_profile_payload(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Extract one hash-bound, fixed-layout native profile subset."""

    bundle = validate_standalone_profile_artifact(artifact)
    physical_binding = next(
        item
        for item in bundle.components
        if item.component_id == "physical-chain-4000dpi"
    )
    operator_payload = artifact["component_payloads"][
        "physical-chain-4000dpi"
    ]["print_operator"]
    operator = SensitometryPrintOperator.from_dict(operator_payload)
    if any(
        len(curve.spline.x_knots) > NATIVE_PRINT_MAX_KNOTS
        for curve in operator.sensitometry.curves
    ):
        raise ValueError("native print spline exceeds fixed ABI capacity")

    payload = {
        "schema": NATIVE_PRINT_PROFILE_SCHEMA,
        "abi": NATIVE_PRINT_ABI,
        "abi_version": NATIVE_PRINT_ABI_VERSION,
        "bundle_sha256": artifact["bundle_sha256"],
        "source_component": {
            "component_id": physical_binding.component_id,
            "schema": physical_binding.schema,
            "sha256": physical_binding.sha256,
        },
        "input_domain": physical_binding.input_domain.value,
        "output_domain": physical_binding.output_domain.value,
        "operator": operator.to_dict(),
        "claim_ceiling": (
            "portable pointwise generic physical-inspired sensitometry and "
            "density-to-print interpretation only; excludes spatial response, "
            "neutral gauge, display look, stock calibration and product runtime"
        ),
    }
    validate_native_print_profile_payload(payload, artifact=artifact)
    return payload


def validate_native_print_profile_payload(
    payload: dict[str, Any],
    *,
    artifact: dict[str, Any] | None = None,
) -> SensitometryPrintOperator:
    if set(payload) != {
        "schema",
        "abi",
        "abi_version",
        "bundle_sha256",
        "source_component",
        "input_domain",
        "output_domain",
        "operator",
        "claim_ceiling",
    }:
        raise ValueError("native print profile fields drift")
    if (
        payload["schema"] != NATIVE_PRINT_PROFILE_SCHEMA
        or payload["abi"] != NATIVE_PRINT_ABI
        or payload["abi_version"] != NATIVE_PRINT_ABI_VERSION
        or payload["input_domain"] != "scene-linear-relative-exposure"
        or payload["output_domain"] != "scanner-linear-signal"
    ):
        raise ValueError("native print profile identity or domain drift")
    source = payload["source_component"]
    if (
        not isinstance(source, dict)
        or set(source) != {"component_id", "schema", "sha256"}
        or source["component_id"] != "physical-chain-4000dpi"
        or source["schema"]
        != "neuro_film.compiled_physical_chain_component.v1"
    ):
        raise ValueError("native print source binding drift")
    try:
        raw_curves = payload["operator"]["sensitometry"]["curves"]
        oversized = any(
            len(curve["spline"]["x_knots"]) > NATIVE_PRINT_MAX_KNOTS
            for curve in raw_curves
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("native print operator structure drift") from exc
    if oversized:
        raise ValueError("native print spline exceeds fixed ABI capacity")
    operator = SensitometryPrintOperator.from_dict(payload["operator"])
    if artifact is not None:
        bundle = validate_standalone_profile_artifact(artifact)
        binding = next(
            item
            for item in bundle.components
            if item.component_id == "physical-chain-4000dpi"
        )
        expected = artifact["component_payloads"][
            "physical-chain-4000dpi"
        ]["print_operator"]
        if (
            payload["bundle_sha256"] != artifact["bundle_sha256"]
            or source["sha256"] != binding.sha256
            or payload["operator"] != expected
        ):
            raise ValueError("native print profile provenance drift")
    return operator


def build_native_print_oracle(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a small deterministic oracle spanning tails, knots and colours."""

    operator = validate_native_print_profile_payload(payload)
    inputs = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.18, 0.18, 0.18],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.9, 0.2, 0.1],
            [0.1, 0.8, 0.3],
            [0.15, 0.25, 0.95],
            [1.0e-8, 1.0e-5, 1.0e-3],
            [0.0123456789, 0.234567891, 0.876543219],
        ],
        dtype=np.float64,
    )
    output = operator.apply(inputs)
    input_bytes = np.ascontiguousarray(inputs).tobytes()
    output_bytes = np.ascontiguousarray(output).tobytes()
    core = {
        "schema": "neuro_film.native_sensitometry_print_oracle.v1",
        "abi": NATIVE_PRINT_ABI,
        "profile_payload_sha256": native_print_payload_sha256(payload),
        "input_rgb_f64": inputs.tolist(),
        "expected_scan_linear_f64": output.tolist(),
        "input_array_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "expected_array_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "comparison": {
            "same_binary_repeat": "byte-exact",
            "python_native_max_abs_tolerance": 5.0e-13,
        },
        "claim_ceiling": payload["claim_ceiling"],
    }
    return {
        **core,
        "oracle_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }
