"""MSVC conformance harness for the bounded physical-print C ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.native_profile import (
    build_native_print_oracle,
    compile_native_print_profile_payload,
    native_print_payload_sha256,
)
from src.film_physics.native_abi_layouts import (
    NativePrintProfileV1,
    native_print_profile_struct as profile_struct_from_payload,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)
from src.eval.native_msvc import build_msvc_c11_dll


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
    path = root / relative
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"hash mismatch: {relative}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {relative}")
    return value


def build_msvc_native_print_dll(
    *,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_physical_print_v1.c",
        header_relative="native/film_physics/nf_physical_print_v1.h",
        basename="nf_physical_print_v1",
    )


def run_loaded_conformance(
    *,
    dll_path: Path,
    payload: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    library = ctypes.CDLL(str(dll_path))
    library.nf_physical_print_abi_version_v1.argtypes = []
    library.nf_physical_print_abi_version_v1.restype = ctypes.c_uint32
    library.nf_physical_print_validate_profile_v1.argtypes = [
        ctypes.POINTER(NativePrintProfileV1)
    ]
    library.nf_physical_print_validate_profile_v1.restype = ctypes.c_int
    library.nf_physical_print_apply_v1.argtypes = [
        ctypes.POINTER(NativePrintProfileV1),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_physical_print_apply_v1.restype = ctypes.c_int
    profile = profile_struct_from_payload(payload)
    if library.nf_physical_print_abi_version_v1() != 1:
        raise RuntimeError("native ABI version drift")
    if library.nf_physical_print_validate_profile_v1(
        ctypes.byref(profile)
    ) != 0:
        raise RuntimeError("native profile rejected")

    inputs = np.ascontiguousarray(
        oracle["input_rgb_f64"], dtype=np.float64
    )
    expected = np.ascontiguousarray(
        oracle["expected_scan_linear_f64"], dtype=np.float64
    )
    output = np.full_like(inputs, -7.0)
    status = library.nf_physical_print_apply_v1(
        ctypes.byref(profile),
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(inputs),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise RuntimeError(f"native apply failed with status {status}")
    tolerance = float(
        oracle["comparison"]["python_native_max_abs_tolerance"]
    )
    maximum_error = float(np.max(np.abs(output - expected)))
    if maximum_error > tolerance:
        raise RuntimeError(
            f"native oracle error {maximum_error} exceeds {tolerance}"
        )

    repeat = np.empty_like(inputs)
    repeat_status = library.nf_physical_print_apply_v1(
        ctypes.byref(profile),
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(inputs),
        repeat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if repeat_status != 0 or repeat.tobytes() != output.tobytes():
        raise RuntimeError("native same-binary replay is not byte exact")

    invalid = inputs.copy()
    invalid[-1, -1] = np.nan
    sentinel = np.full_like(inputs, -1234.5)
    before = sentinel.tobytes()
    invalid_status = library.nf_physical_print_apply_v1(
        ctypes.byref(profile),
        invalid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(invalid),
        sentinel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if invalid_status != 3 or sentinel.tobytes() != before:
        raise RuntimeError("native invalid-input atomicity failed")

    inplace = inputs.copy()
    inplace_status = library.nf_physical_print_apply_v1(
        ctypes.byref(profile),
        inplace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(inplace),
        inplace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if inplace_status != 0 or inplace.tobytes() != output.tobytes():
        raise RuntimeError("native in-place execution drift")
    return {
        "status": "pass",
        "rgb_count": len(inputs),
        "maximum_absolute_error": maximum_error,
        "tolerance": tolerance,
        "same_binary_repeat_byte_exact": True,
        "invalid_input_output_unchanged": True,
        "inplace_matches_out_of_place": True,
        "output_array_sha256": hashlib.sha256(
            output.tobytes()
        ).hexdigest(),
    }


def run_native_print_conformance(
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
    if not str(parent.get("next_leaf", "")).startswith("U6.P8X"):
        raise ValueError("P8X parent decision drift")
    compiler_config = _load_exact_json(
        root,
        config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(
        root=root, config=compiler_config
    )
    payload = compile_native_print_profile_payload(artifact)
    payload_sha = native_print_payload_sha256(payload)
    oracle = build_native_print_oracle(payload)
    if (
        payload_sha != config["expected_profile_payload_sha256"]
        or oracle["oracle_sha256"] != config["expected_oracle_sha256"]
    ):
        raise ValueError("P8X frozen payload or oracle identity drift")
    first_build = build_msvc_native_print_dll(
        root=root, output_dir=output_dir / "build_a"
    )
    second_build = build_msvc_native_print_dll(
        root=root, output_dir=output_dir / "build_b"
    )
    if (
        first_build["source_sha256"] != config["native_source_sha256"]
        or first_build["header_sha256"] != config["native_header_sha256"]
        or second_build["source_sha256"] != config["native_source_sha256"]
        or second_build["header_sha256"] != config["native_header_sha256"]
    ):
        raise ValueError("P8X native source identity drift")
    if first_build["dll_sha256"] != second_build["dll_sha256"]:
        raise RuntimeError("independent MSVC builds are not byte exact")
    first = run_loaded_conformance(
        dll_path=Path(first_build["dll_path"]),
        payload=payload,
        oracle=oracle,
    )
    second = run_loaded_conformance(
        dll_path=Path(second_build["dll_path"]),
        payload=payload,
        oracle=oracle,
    )
    stable_core = {
        "schema": "neuro_film.u6_p8x_native_print_conformance.v1",
        "profile_payload_sha256": payload_sha,
        "oracle_sha256": oracle["oracle_sha256"],
        "native_source_sha256": first_build["source_sha256"],
        "native_header_sha256": first_build["header_sha256"],
        "dll_sha256": first_build["dll_sha256"],
        "independent_build_dll_sha_exact": True,
        "toolchain": first_build["toolchain"],
        "replays": [first, second],
        "decision": (
            "pass portable MSVC x64 pointwise physical-print C ABI "
            "conformance; spatial response, gauge, display look, full-frame "
            "renderer and other target runtimes remain open"
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
