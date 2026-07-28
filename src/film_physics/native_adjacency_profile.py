"""Native bounded developed-density adjacency profile and full-chain oracle."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .profile_consumer import (
    reconstruct_standalone_runtime,
    validate_standalone_profile_artifact,
)
from .spatial_response import (
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
)


NATIVE_ADJACENCY_PROFILE_SCHEMA = (
    "neuro_film.native_bounded_adjacency_profile.v1"
)
NATIVE_ADJACENCY_ABI = "nf_bounded_adjacency_v1"
NATIVE_ADJACENCY_ABI_VERSION = 1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def native_adjacency_payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def compile_native_adjacency_profile_payload(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    bundle = validate_standalone_profile_artifact(artifact)
    binding = next(
        item
        for item in bundle.components
        if item.component_id == "physical-chain-4000dpi"
    )
    physical = artifact["component_payloads"][
        "physical-chain-4000dpi"
    ]
    adjacency = physical["adjacency"]
    spatial = physical["spatial_profile"]
    payload = {
        "schema": NATIVE_ADJACENCY_PROFILE_SCHEMA,
        "abi": NATIVE_ADJACENCY_ABI,
        "abi_version": NATIVE_ADJACENCY_ABI_VERSION,
        "bundle_sha256": artifact["bundle_sha256"],
        "source_component": {
            "component_id": binding.component_id,
            "schema": binding.schema,
            "sha256": binding.sha256,
        },
        "development_adjacency_gain_rgb": spatial[
            "development_adjacency_gain_rgb"
        ],
        "maximum_absolute_transmittance_delta": adjacency[
            "maximum_absolute_transmittance_delta"
        ],
        "maximum_absolute_density_delta": adjacency[
            "maximum_absolute_density_delta"
        ],
        "black_reference_density": adjacency[
            "black_reference_density"
        ],
        "white_reference_density": adjacency[
            "white_reference_density"
        ],
        "claim_ceiling": (
            "portable bounded developed-density adjacency primitive and "
            "small-image ordered native-chain conformance only; not "
            "calibrated development, full-resolution performance, display "
            "look, stock response or product runtime"
        ),
    }
    validate_native_adjacency_profile_payload(payload, artifact=artifact)
    return payload


def validate_native_adjacency_profile_payload(
    payload: dict[str, Any],
    *,
    artifact: dict[str, Any] | None = None,
) -> None:
    if set(payload) != {
        "schema",
        "abi",
        "abi_version",
        "bundle_sha256",
        "source_component",
        "development_adjacency_gain_rgb",
        "maximum_absolute_transmittance_delta",
        "maximum_absolute_density_delta",
        "black_reference_density",
        "white_reference_density",
        "claim_ceiling",
    }:
        raise ValueError("native adjacency fields drift")
    if (
        payload["schema"] != NATIVE_ADJACENCY_PROFILE_SCHEMA
        or payload["abi"] != NATIVE_ADJACENCY_ABI
        or payload["abi_version"] != NATIVE_ADJACENCY_ABI_VERSION
    ):
        raise ValueError("native adjacency identity drift")
    source = payload["source_component"]
    gains = np.asarray(
        payload["development_adjacency_gain_rgb"], dtype=np.float64
    )
    black = np.asarray(
        payload["black_reference_density"], dtype=np.float64
    )
    white = np.asarray(
        payload["white_reference_density"], dtype=np.float64
    )
    if (
        not isinstance(source, dict)
        or set(source) != {"component_id", "schema", "sha256"}
        or source["component_id"] != "physical-chain-4000dpi"
        or source["schema"]
        != "neuro_film.compiled_physical_chain_component.v1"
        or gains.shape != (3,)
        or black.shape != (3,)
        or white.shape != (3,)
        or not np.all(np.isfinite(gains))
        or not np.all(np.isfinite(black))
        or not np.all(np.isfinite(white))
        or np.any(gains < 0.0)
        or np.any(black < 0.0)
        or np.any(white <= black)
        or not 0.0
        < float(payload["maximum_absolute_transmittance_delta"])
        < 1.0
        or float(payload["maximum_absolute_density_delta"]) <= 0.0
    ):
        raise ValueError("native adjacency parameter drift")
    if artifact is not None:
        bundle = validate_standalone_profile_artifact(artifact)
        binding = next(
            item
            for item in bundle.components
            if item.component_id == "physical-chain-4000dpi"
        )
        physical = artifact["component_payloads"][
            "physical-chain-4000dpi"
        ]
        adjacency = physical["adjacency"]
        spatial = physical["spatial_profile"]
        if (
            payload["bundle_sha256"] != artifact["bundle_sha256"]
            or source["sha256"] != binding.sha256
            or payload["development_adjacency_gain_rgb"]
            != spatial["development_adjacency_gain_rgb"]
            or payload["maximum_absolute_transmittance_delta"]
            != adjacency["maximum_absolute_transmittance_delta"]
            or payload["maximum_absolute_density_delta"]
            != adjacency["maximum_absolute_density_delta"]
            or payload["black_reference_density"]
            != adjacency["black_reference_density"]
            or payload["white_reference_density"]
            != adjacency["white_reference_density"]
        ):
            raise ValueError("native adjacency provenance drift")


def build_native_ordered_chain_oracle(
    artifact: dict[str, Any],
    adjacency_payload: dict[str, Any],
) -> dict[str, Any]:
    validate_native_adjacency_profile_payload(
        adjacency_payload, artifact=artifact
    )
    runtime, _ = reconstruct_standalone_runtime(artifact)
    height, width = 11, 13
    y, x = np.mgrid[0:height, 0:width]
    source = np.empty((height, width, 3), dtype=np.float64)
    source[..., 0] = (x + y) / (height + width - 2)
    source[..., 1] = ((x * 5 + y * 3) % 19) / 18.0
    source[..., 2] = np.random.default_rng(20260729).random(
        (height, width)
    )
    source[0, 0] = (1.0, 0.0, 0.5)
    source[-1, -1] = (0.0, 1.0, 0.25)
    source[height // 2, width // 2] = (1.0, 1.0, 1.0)
    forward = apply_forward_scatter(source, runtime.profile)
    density = runtime.print_operator.sensitometry.apply(forward)
    adjacent = runtime.apply_adjacency(density, runtime.profile)
    diffused = apply_dye_diffusion(adjacent, runtime.profile)
    interpreted = runtime.print_operator.interpretation.apply(diffused)
    scanned = apply_scanner_mtf(interpreted, runtime.profile)
    stages = {
        "forward_scatter": forward,
        "developed_density": density,
        "bounded_adjacency": adjacent,
        "dye_diffusion": diffused,
        "interpretation": interpreted,
        "scanner_mtf": scanned,
    }
    core = {
        "schema": "neuro_film.native_ordered_physical_chain_oracle.v1",
        "adjacency_payload_sha256": native_adjacency_payload_sha256(
            adjacency_payload
        ),
        "shape": [height, width, 3],
        "input_scene_linear_f64": source.tolist(),
        "expected_by_stage_f64": {
            name: values.tolist() for name, values in stages.items()
        },
        "comparison": {
            "same_binary_repeat": "byte-exact",
            "stage_max_abs_tolerance": 2.0e-12,
        },
        "claim_ceiling": adjacency_payload["claim_ceiling"],
    }
    return {
        **core,
        "oracle_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }
