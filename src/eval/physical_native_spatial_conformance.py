"""MSVC conformance harness for the finite RGB Gaussian spatial ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll
from src.film_physics.native_abi_layouts import (
    NativeGaussianProfileV1,
    native_gaussian_profile_struct as gaussian_profile_struct,
)
from src.film_physics.native_spatial_profile import (
    build_native_gaussian_oracle,
    compile_native_gaussian_profile_payload,
    native_gaussian_payload_sha256,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _load_exact_json(
    root: Path, relative: str, expected_sha256: str
) -> dict[str, Any]:
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"hash mismatch: {relative}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {relative}")
    return value


def build_msvc_native_gaussian_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_gaussian_rgb_f64_v1.c",
        header_relative="native/film_physics/nf_gaussian_rgb_f64_v1.h",
        basename="nf_gaussian_rgb_f64_v1",
    )


def run_loaded_spatial_conformance(
    *,
    dll_path: Path,
    payload: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    library = ctypes.CDLL(str(dll_path))
    library.nf_gaussian_abi_version_v1.argtypes = []
    library.nf_gaussian_abi_version_v1.restype = ctypes.c_uint32
    library.nf_gaussian_validate_profile_v1.argtypes = [
        ctypes.POINTER(NativeGaussianProfileV1)
    ]
    library.nf_gaussian_validate_profile_v1.restype = ctypes.c_int
    library.nf_gaussian_required_halo_v1.argtypes = [
        ctypes.POINTER(NativeGaussianProfileV1),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.nf_gaussian_required_halo_v1.restype = ctypes.c_int
    library.nf_gaussian_apply_v1.argtypes = [
        ctypes.POINTER(NativeGaussianProfileV1),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_gaussian_apply_v1.restype = ctypes.c_int
    if library.nf_gaussian_abi_version_v1() != 1:
        raise RuntimeError("native Gaussian ABI version drift")

    shape = tuple(oracle["shape"])
    source = np.ascontiguousarray(
        oracle["input_rgb_f64"], dtype=np.float64
    )
    if source.shape != shape:
        raise RuntimeError("native Gaussian oracle shape drift")
    tolerance = float(
        oracle["comparison"]["python_native_max_abs_tolerance"]
    )
    rows = []
    for stage in payload["stages"]:
        profile = gaussian_profile_struct(payload, stage)
        if library.nf_gaussian_validate_profile_v1(
            ctypes.byref(profile)
        ) != 0:
            raise RuntimeError(f"native Gaussian rejected {stage['stage']}")
        halo = ctypes.c_uint32()
        if library.nf_gaussian_required_halo_v1(
            ctypes.byref(profile), ctypes.byref(halo)
        ) != 0 or halo.value != stage["maximum_radius"]:
            raise RuntimeError(f"native Gaussian halo drift: {stage['stage']}")
        workspace = np.full_like(source, -7.0)
        output = np.full_like(source, -9.0)
        status = library.nf_gaussian_apply_v1(
            ctypes.byref(profile),
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            shape[0],
            shape[1],
            workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            workspace.size,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise RuntimeError(
                f"native Gaussian apply failed for {stage['stage']}: {status}"
            )
        expected = np.asarray(
            oracle["expected_by_stage_f64"][stage["stage"]],
            dtype=np.float64,
        )
        maximum_error = float(np.max(np.abs(output - expected)))
        if maximum_error > tolerance:
            raise RuntimeError(
                f"{stage['stage']} error {maximum_error} exceeds {tolerance}"
            )
        repeat_workspace = np.empty_like(source)
        repeat = np.empty_like(source)
        repeat_status = library.nf_gaussian_apply_v1(
            ctypes.byref(profile),
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            shape[0],
            shape[1],
            repeat_workspace.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            repeat_workspace.size,
            repeat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if repeat_status != 0 or repeat.tobytes() != output.tobytes():
            raise RuntimeError(
                f"native Gaussian replay drift: {stage['stage']}"
            )
        rows.append(
            {
                "stage": stage["stage"],
                "halo": halo.value,
                "maximum_absolute_error": maximum_error,
                "tolerance": tolerance,
                "same_binary_repeat_byte_exact": True,
                "output_array_sha256": hashlib.sha256(
                    output.tobytes()
                ).hexdigest(),
            }
        )

    profile = gaussian_profile_struct(payload, payload["stages"][0])
    invalid = source.copy()
    invalid[-1, -1, -1] = np.nan
    workspace = np.full_like(source, -123.0)
    output = np.full_like(source, -456.0)
    workspace_before = workspace.tobytes()
    output_before = output.tobytes()
    invalid_status = library.nf_gaussian_apply_v1(
        ctypes.byref(profile),
        invalid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        shape[0],
        shape[1],
        workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        workspace.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if (
        invalid_status != 3
        or workspace.tobytes() != workspace_before
        or output.tobytes() != output_before
    ):
        raise RuntimeError("native Gaussian invalid-input atomicity failed")
    overlap_status = library.nf_gaussian_apply_v1(
        ctypes.byref(profile),
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        shape[0],
        shape[1],
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        source.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if overlap_status != 1 or output.tobytes() != output_before:
        raise RuntimeError("native Gaussian overlap guard failed")
    return {
        "status": "pass",
        "stages": rows,
        "invalid_input_buffers_unchanged": True,
        "overlap_rejected_before_write": True,
    }


def run_native_spatial_conformance(
    *,
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8Y"):
        raise ValueError("P8Y parent decision drift")
    compiler_config = _load_exact_json(
        root,
        config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(
        root=root, config=compiler_config
    )
    payload = compile_native_gaussian_profile_payload(artifact)
    oracle = build_native_gaussian_oracle(payload)
    if (
        native_gaussian_payload_sha256(payload)
        != config["expected_profile_payload_sha256"]
        or oracle["oracle_sha256"] != config["expected_oracle_sha256"]
    ):
        raise ValueError("P8Y frozen payload or oracle identity drift")
    first_build = build_msvc_native_gaussian_dll(
        root=root, output_dir=output_dir / "build_a"
    )
    second_build = build_msvc_native_gaussian_dll(
        root=root, output_dir=output_dir / "build_b"
    )
    for build in (first_build, second_build):
        if (
            build["source_sha256"] != config["native_source_sha256"]
            or build["header_sha256"] != config["native_header_sha256"]
        ):
            raise ValueError("P8Y native source identity drift")
    if first_build["dll_sha256"] != second_build["dll_sha256"]:
        raise RuntimeError("independent spatial DLL builds are not exact")
    first = run_loaded_spatial_conformance(
        dll_path=Path(first_build["dll_path"]),
        payload=payload,
        oracle=oracle,
    )
    second = run_loaded_spatial_conformance(
        dll_path=Path(second_build["dll_path"]),
        payload=payload,
        oracle=oracle,
    )
    stable_core = {
        "schema": "neuro_film.u6_p8y_native_spatial_conformance.v1",
        "profile_payload_sha256": native_gaussian_payload_sha256(
            payload
        ),
        "oracle_sha256": oracle["oracle_sha256"],
        "native_source_sha256": first_build["source_sha256"],
        "native_header_sha256": first_build["header_sha256"],
        "dll_sha256": first_build["dll_sha256"],
        "independent_build_dll_sha_exact": True,
        "toolchain": first_build["toolchain"],
        "replays": [first, second],
        "decision": (
            "pass finite nearest-edge separable Gaussian C ABI for all "
            "four fixed physical spatial stages; composed physical chain "
            "and full renderer remain open"
        ),
        "claim_ceiling": payload["claim_ceiling"],
        "production_default_changed": False,
    }
    return {
        **stable_core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_core)
        ).hexdigest(),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
