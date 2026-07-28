"""MSVC conformance for split sensitometry and interpretation domains."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll
from src.eval.physical_native_print_conformance import (
    NativePrintProfileV1,
    profile_struct_from_payload,
)
from src.film_physics.native_profile import (
    build_native_domains_oracle,
    compile_native_domains_profile_payload,
    native_print_payload_sha256,
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


def build_msvc_native_domains_dll(
    *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_physical_domains_v1.c",
        header_relative="native/film_physics/nf_physical_domains_v1.h",
        basename="nf_physical_domains_v1",
    )


def run_loaded_domains_conformance(
    *,
    dll_path: Path,
    payload: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    library = ctypes.CDLL(str(dll_path))
    library.nf_physical_domains_abi_version_v1.argtypes = []
    library.nf_physical_domains_abi_version_v1.restype = ctypes.c_uint32
    library.nf_physical_domains_validate_profile_v1.argtypes = [
        ctypes.POINTER(NativePrintProfileV1)
    ]
    library.nf_physical_domains_validate_profile_v1.restype = ctypes.c_int
    common_args = [
        ctypes.POINTER(NativePrintProfileV1),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_physical_sensitometry_apply_v1.argtypes = common_args
    library.nf_physical_sensitometry_apply_v1.restype = ctypes.c_int
    library.nf_physical_interpretation_apply_v1.argtypes = common_args
    library.nf_physical_interpretation_apply_v1.restype = ctypes.c_int
    profile = profile_struct_from_payload(payload)
    if library.nf_physical_domains_abi_version_v1() != 1:
        raise RuntimeError("native physical-domains ABI drift")
    if library.nf_physical_domains_validate_profile_v1(
        ctypes.byref(profile)
    ) != 0:
        raise RuntimeError("native physical-domains profile rejected")
    inputs = np.ascontiguousarray(
        oracle["input_scene_linear_f64"], dtype=np.float64
    )
    expected_density = np.ascontiguousarray(
        oracle["expected_developed_density_f64"], dtype=np.float64
    )
    expected_scan = np.ascontiguousarray(
        oracle["expected_scan_linear_f64"], dtype=np.float64
    )
    density = np.empty_like(inputs)
    density_status = library.nf_physical_sensitometry_apply_v1(
        ctypes.byref(profile),
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(inputs),
        density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    scan = np.empty_like(inputs)
    interpretation_status = library.nf_physical_interpretation_apply_v1(
        ctypes.byref(profile),
        density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(density),
        scan.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if density_status != 0 or interpretation_status != 0:
        raise RuntimeError("native split physical-domain apply failed")
    tolerance = float(
        oracle["comparison"]["python_native_max_abs_tolerance"]
    )
    density_error = float(
        np.max(np.abs(density - expected_density))
    )
    scan_error = float(np.max(np.abs(scan - expected_scan)))
    if density_error > tolerance or scan_error > tolerance:
        raise RuntimeError(
            "native split physical-domain error exceeds tolerance"
        )
    repeat_density = np.empty_like(inputs)
    repeat_scan = np.empty_like(inputs)
    if (
        library.nf_physical_sensitometry_apply_v1(
            ctypes.byref(profile),
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs),
            repeat_density.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
        )
        != 0
        or library.nf_physical_interpretation_apply_v1(
            ctypes.byref(profile),
            repeat_density.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            len(repeat_density),
            repeat_scan.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        != 0
        or repeat_density.tobytes() != density.tobytes()
        or repeat_scan.tobytes() != scan.tobytes()
    ):
        raise RuntimeError("native split-domain replay drift")

    inplace_density = inputs.copy()
    if library.nf_physical_sensitometry_apply_v1(
        ctypes.byref(profile),
        inplace_density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(inplace_density),
        inplace_density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ) != 0 or inplace_density.tobytes() != density.tobytes():
        raise RuntimeError("native sensitometry in-place drift")
    inplace_scan = density.copy()
    if library.nf_physical_interpretation_apply_v1(
        ctypes.byref(profile),
        inplace_scan.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(inplace_scan),
        inplace_scan.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ) != 0 or inplace_scan.tobytes() != scan.tobytes():
        raise RuntimeError("native interpretation in-place drift")

    invalid = inputs.copy()
    invalid[-1, -1] = np.nan
    sentinel = np.full_like(inputs, -321.0)
    before = sentinel.tobytes()
    invalid_status = library.nf_physical_sensitometry_apply_v1(
        ctypes.byref(profile),
        invalid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(invalid),
        sentinel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    bad_density = density.copy()
    bad_density[-1, -1] = np.nan
    interpretation_sentinel = np.full_like(inputs, -654.0)
    interpretation_before = interpretation_sentinel.tobytes()
    bad_density_status = library.nf_physical_interpretation_apply_v1(
        ctypes.byref(profile),
        bad_density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(bad_density),
        interpretation_sentinel.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
    )
    if (
        invalid_status != 3
        or sentinel.tobytes() != before
        or bad_density_status != 3
        or interpretation_sentinel.tobytes() != interpretation_before
    ):
        raise RuntimeError("native split-domain failure atomicity drift")
    return {
        "status": "pass",
        "rgb_count": len(inputs),
        "sensitometry_maximum_absolute_error": density_error,
        "interpretation_maximum_absolute_error": scan_error,
        "tolerance": tolerance,
        "same_binary_repeat_byte_exact": True,
        "sensitometry_inplace_exact": True,
        "interpretation_inplace_exact": True,
        "invalid_input_output_unchanged": True,
        "density_array_sha256": hashlib.sha256(
            density.tobytes()
        ).hexdigest(),
        "scan_array_sha256": hashlib.sha256(
            scan.tobytes()
        ).hexdigest(),
    }


def run_native_domains_conformance(
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
    if not str(parent.get("next_leaf", "")).startswith("U6.P8Z"):
        raise ValueError("P8Z parent decision drift")
    compiler_config = _load_exact_json(
        root,
        config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(
        root=root, config=compiler_config
    )
    payload = compile_native_domains_profile_payload(artifact)
    oracle = build_native_domains_oracle(payload)
    if (
        native_print_payload_sha256(payload)
        != config["expected_profile_payload_sha256"]
        or oracle["oracle_sha256"] != config["expected_oracle_sha256"]
    ):
        raise ValueError("P8Z frozen payload or oracle identity drift")
    first_build = build_msvc_native_domains_dll(
        root=root, output_dir=output_dir / "build_a"
    )
    second_build = build_msvc_native_domains_dll(
        root=root, output_dir=output_dir / "build_b"
    )
    for build in (first_build, second_build):
        if (
            build["source_sha256"] != config["native_source_sha256"]
            or build["header_sha256"] != config["native_header_sha256"]
        ):
            raise ValueError("P8Z native source identity drift")
    if first_build["dll_sha256"] != second_build["dll_sha256"]:
        raise RuntimeError("independent physical-domains builds differ")
    first = run_loaded_domains_conformance(
        dll_path=Path(first_build["dll_path"]),
        payload=payload,
        oracle=oracle,
    )
    second = run_loaded_domains_conformance(
        dll_path=Path(second_build["dll_path"]),
        payload=payload,
        oracle=oracle,
    )
    stable_core = {
        "schema": "neuro_film.u6_p8z_native_domains_conformance.v1",
        "profile_payload_sha256": native_print_payload_sha256(payload),
        "oracle_sha256": oracle["oracle_sha256"],
        "native_source_sha256": first_build["source_sha256"],
        "native_header_sha256": first_build["header_sha256"],
        "dll_sha256": first_build["dll_sha256"],
        "independent_build_dll_sha_exact": True,
        "toolchain": first_build["toolchain"],
        "replays": [first, second],
        "decision": (
            "pass separately callable native scene-linear sensitometry and "
            "developed-density interpretation boundaries; full ordered "
            "spatial-chain composition remains open"
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
