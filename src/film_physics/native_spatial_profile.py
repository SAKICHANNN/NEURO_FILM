"""Portable finite-Gaussian subsets for the native spatial-response ABI."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .profile_consumer import validate_standalone_profile_artifact


NATIVE_GAUSSIAN_PROFILE_SCHEMA = (
    "neuro_film.native_gaussian_spatial_profile.v1"
)
NATIVE_GAUSSIAN_ABI = "nf_gaussian_rgb_f64_v1"
NATIVE_GAUSSIAN_ABI_VERSION = 1
NATIVE_GAUSSIAN_MAX_RADIUS = 64
_STAGE_FIELDS = {
    "forward_scatter": "forward_scatter_sigma_um_rgb",
    "development_adjacency": "development_adjacency_sigma_um_rgb",
    "dye_diffusion": "dye_diffusion_sigma_um_rgb",
    "scanner_mtf": "scanner_mtf_sigma_um_rgb",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def native_gaussian_payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def compile_native_gaussian_profile_payload(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    bundle = validate_standalone_profile_artifact(artifact)
    binding = next(
        item
        for item in bundle.components
        if item.component_id == "physical-chain-4000dpi"
    )
    spatial = artifact["component_payloads"][
        "physical-chain-4000dpi"
    ]["spatial_profile"]
    pitch = float(spatial["pixel_pitch_um"])
    truncate = float(spatial["gaussian_truncate"])
    stages = []
    for stage, field in _STAGE_FIELDS.items():
        sigma_um = [float(value) for value in spatial[field]]
        sigma_pixels = [value / pitch for value in sigma_um]
        radii = [
            int(truncate * value + 0.5) if value > 0.0 else 0
            for value in sigma_pixels
        ]
        stages.append(
            {
                "stage": stage,
                "sigma_um_rgb": sigma_um,
                "sigma_pixels_rgb": sigma_pixels,
                "radius_rgb": radii,
                "maximum_radius": max(radii),
                "boundary_mode": "nearest",
                "axis_order": ["vertical", "horizontal"],
            }
        )
    payload = {
        "schema": NATIVE_GAUSSIAN_PROFILE_SCHEMA,
        "abi": NATIVE_GAUSSIAN_ABI,
        "abi_version": NATIVE_GAUSSIAN_ABI_VERSION,
        "bundle_sha256": artifact["bundle_sha256"],
        "source_component": {
            "component_id": binding.component_id,
            "schema": binding.schema,
            "sha256": binding.sha256,
        },
        "pixel_pitch_um": pitch,
        "gaussian_truncate": truncate,
        "stages": stages,
        "claim_ceiling": (
            "portable finite separable Gaussian spatial primitive only; "
            "not a composed physical chain, calibrated MTF, halation, "
            "development model, full renderer or product runtime"
        ),
    }
    validate_native_gaussian_profile_payload(payload, artifact=artifact)
    return payload


def validate_native_gaussian_profile_payload(
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
        "pixel_pitch_um",
        "gaussian_truncate",
        "stages",
        "claim_ceiling",
    }:
        raise ValueError("native Gaussian profile fields drift")
    if (
        payload["schema"] != NATIVE_GAUSSIAN_PROFILE_SCHEMA
        or payload["abi"] != NATIVE_GAUSSIAN_ABI
        or payload["abi_version"] != NATIVE_GAUSSIAN_ABI_VERSION
        or payload["pixel_pitch_um"] <= 0.0
        or payload["gaussian_truncate"] <= 0.0
    ):
        raise ValueError("native Gaussian profile identity drift")
    source = payload["source_component"]
    if (
        not isinstance(source, dict)
        or set(source) != {"component_id", "schema", "sha256"}
        or source["component_id"] != "physical-chain-4000dpi"
        or source["schema"]
        != "neuro_film.compiled_physical_chain_component.v1"
    ):
        raise ValueError("native Gaussian source binding drift")
    stages = payload["stages"]
    if (
        not isinstance(stages, list)
        or [row.get("stage") for row in stages] != list(_STAGE_FIELDS)
    ):
        raise ValueError("native Gaussian stage inventory drift")
    for row in stages:
        if (
            set(row)
            != {
                "stage",
                "sigma_um_rgb",
                "sigma_pixels_rgb",
                "radius_rgb",
                "maximum_radius",
                "boundary_mode",
                "axis_order",
            }
            or row["boundary_mode"] != "nearest"
            or row["axis_order"] != ["vertical", "horizontal"]
            or len(row["sigma_um_rgb"]) != 3
            or len(row["sigma_pixels_rgb"]) != 3
            or len(row["radius_rgb"]) != 3
            or any(
                not np.isfinite(value) or value < 0.0
                for value in row["sigma_pixels_rgb"]
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > NATIVE_GAUSSIAN_MAX_RADIUS
                for value in row["radius_rgb"]
            )
            or row["maximum_radius"] != max(row["radius_rgb"])
        ):
            raise ValueError("native Gaussian stage drift")
    if artifact is not None:
        bundle = validate_standalone_profile_artifact(artifact)
        binding = next(
            item
            for item in bundle.components
            if item.component_id == "physical-chain-4000dpi"
        )
        spatial = artifact["component_payloads"][
            "physical-chain-4000dpi"
        ]["spatial_profile"]
        if (
            payload["bundle_sha256"] != artifact["bundle_sha256"]
            or source["sha256"] != binding.sha256
            or payload["pixel_pitch_um"] != spatial["pixel_pitch_um"]
            or payload["gaussian_truncate"]
            != spatial["gaussian_truncate"]
        ):
            raise ValueError("native Gaussian profile provenance drift")
        for row, (stage, field) in zip(
            stages, _STAGE_FIELDS.items(), strict=True
        ):
            expected_um = [float(value) for value in spatial[field]]
            expected_pixels = [
                value / float(spatial["pixel_pitch_um"])
                for value in expected_um
            ]
            if (
                row["stage"] != stage
                or row["sigma_um_rgb"] != expected_um
                or row["sigma_pixels_rgb"] != expected_pixels
            ):
                raise ValueError(
                    "native Gaussian profile provenance drift"
                )


def build_native_gaussian_oracle(
    payload: dict[str, Any],
) -> dict[str, Any]:
    validate_native_gaussian_profile_payload(payload)
    height, width = 11, 13
    y, x = np.mgrid[0:height, 0:width]
    source = np.empty((height, width, 3), dtype=np.float64)
    source[..., 0] = (x + 2.0 * y) / (
        (width - 1) + 2.0 * (height - 1)
    )
    source[..., 1] = ((x * 7 + y * 11) % 23) / 22.0
    source[..., 2] = np.random.default_rng(20260729).random(
        (height, width)
    )
    source[0, 0] = (1.0, 0.0, 0.5)
    source[-1, -1] = (0.0, 1.0, 0.25)
    source[height // 2, width // 2] = (1.0, 1.0, 1.0)
    outputs = {}
    for stage in payload["stages"]:
        result = np.empty_like(source)
        for channel, sigma in enumerate(stage["sigma_pixels_rgb"]):
            if sigma == 0.0:
                result[..., channel] = source[..., channel]
            else:
                result[..., channel] = gaussian_filter(
                    source[..., channel],
                    sigma=float(sigma),
                    order=0,
                    mode="nearest",
                    truncate=float(payload["gaussian_truncate"]),
                )
        outputs[stage["stage"]] = result.tolist()
    core = {
        "schema": "neuro_film.native_gaussian_spatial_oracle.v1",
        "abi": NATIVE_GAUSSIAN_ABI,
        "profile_payload_sha256": native_gaussian_payload_sha256(
            payload
        ),
        "shape": [height, width, 3],
        "input_rgb_f64": source.tolist(),
        "expected_by_stage_f64": outputs,
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
