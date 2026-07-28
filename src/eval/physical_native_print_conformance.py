"""MSVC conformance harness for the bounded physical-print C ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from src.film_physics.native_profile import (
    NATIVE_PRINT_ABI_VERSION,
    NATIVE_PRINT_MAX_KNOTS,
    build_native_print_oracle,
    compile_native_print_profile_payload,
    native_print_payload_sha256,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


class NativePrintProfileV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_component_sha256", ctypes.c_char * 65),
        ("reference_linear", ctypes.c_double),
        ("black_offset", ctypes.c_double),
        ("knot_count", ctypes.c_uint32 * 3),
        (
            "x_knots",
            (ctypes.c_double * NATIVE_PRINT_MAX_KNOTS) * 3,
        ),
        (
            "y_knots",
            (ctypes.c_double * NATIVE_PRINT_MAX_KNOTS) * 3,
        ),
        (
            "derivatives",
            (ctypes.c_double * NATIVE_PRINT_MAX_KNOTS) * 3,
        ),
        ("dye_absorption_matrix", (ctypes.c_double * 3) * 3),
        ("print_matrix", (ctypes.c_double * 3) * 3),
        ("paper_midpoints", ctypes.c_double * 3),
        ("paper_slopes", ctypes.c_double * 3),
        ("paper_maximum_densities", ctypes.c_double * 3),
        ("black_reference_density", ctypes.c_double * 3),
        ("white_reference_density", ctypes.c_double * 3),
        ("exposure_floor", ctypes.c_double),
        ("matrix_minimum_determinant", ctypes.c_double),
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _copy_matrix(target: Any, values: list[list[float]]) -> None:
    for row in range(3):
        for column in range(3):
            target[row][column] = float(values[row][column])


def profile_struct_from_payload(
    payload: dict[str, Any],
) -> NativePrintProfileV1:
    operator = payload["operator"]
    sensitometry = operator["sensitometry"]
    interpretation = operator["interpretation"]
    result = NativePrintProfileV1()
    result.struct_size = ctypes.sizeof(NativePrintProfileV1)
    result.abi_version = NATIVE_PRINT_ABI_VERSION
    result.source_component_sha256 = payload["source_component"][
        "sha256"
    ].encode("ascii")
    result.reference_linear = float(
        sensitometry["encoder"]["reference_linear"]
    )
    result.black_offset = float(
        sensitometry["encoder"]["black_offset"]
    )
    for channel, curve in enumerate(sensitometry["curves"]):
        spline = curve["spline"]
        count = len(spline["x_knots"])
        result.knot_count[channel] = count
        for index in range(count):
            result.x_knots[channel][index] = float(
                spline["x_knots"][index]
            )
            result.y_knots[channel][index] = float(
                spline["y_knots"][index]
            )
            result.derivatives[channel][index] = float(
                spline["derivatives"][index]
            )
    _copy_matrix(
        result.dye_absorption_matrix,
        interpretation["dye_absorption_matrix"],
    )
    _copy_matrix(result.print_matrix, interpretation["print_matrix"])
    for name in (
        "paper_midpoints",
        "paper_slopes",
        "paper_maximum_densities",
        "black_reference_density",
        "white_reference_density",
    ):
        target = getattr(result, name)
        for index, value in enumerate(interpretation[name]):
            target[index] = float(value)
    result.exposure_floor = float(interpretation["exposure_floor"])
    result.matrix_minimum_determinant = float(
        interpretation["matrix_minimum_determinant"]
    )
    return result


def find_msvc_installation() -> Path:
    vswhere = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio"
        r"\Installer\vswhere.exe"
    )
    if not vswhere.is_file():
        raise RuntimeError("vswhere is unavailable")
    completed = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    lines = [line.strip() for line in completed.stdout.splitlines()]
    if not lines:
        raise RuntimeError("MSVC Build Tools are unavailable")
    installation = Path(lines[-1])
    if not installation.is_dir():
        raise RuntimeError("MSVC installation path is invalid")
    return installation


def build_msvc_native_print_dll(
    *,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7" / "Tools" / "VsDevCmd.bat"
    source = root / "native" / "film_physics" / "nf_physical_print_v1.c"
    header = root / "native" / "film_physics" / "nf_physical_print_v1.h"
    if not vcvars.is_file() or not source.is_file() or not header.is_file():
        raise RuntimeError("native source or MSVC environment is missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = output_dir / "nf_physical_print_v1.dll"
    obj = output_dir / "nf_physical_print_v1.obj"
    import_library = output_dir / "nf_physical_print_v1.lib"
    batch = output_dir / "build_nf_physical_print_v1.bat"
    batch.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f'cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD '
        f'/Fo"{obj}" "{source}" /link /Brepro /OUT:"{dll}" '
        f'/IMPLIB:"{import_library}"\r\n',
        encoding="ascii",
        newline="",
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(batch)],
        capture_output=True,
        timeout=120,
        check=False,
        cwd=output_dir,
    )
    compiler_output = (
        completed.stdout + completed.stderr
    ).decode("utf-8", errors="replace")
    if completed.returncode != 0 or not dll.is_file():
        raise RuntimeError(
            "MSVC native build failed:\n"
            + compiler_output
        )
    return {
        "toolchain": "msvc-x64-c11",
        "source_sha256": _sha256_file(source),
        "header_sha256": _sha256_file(header),
        "dll_sha256": _sha256_file(dll),
        "dll_path": str(dll.resolve()),
        "compiler_output": compiler_output.strip(),
    }


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
