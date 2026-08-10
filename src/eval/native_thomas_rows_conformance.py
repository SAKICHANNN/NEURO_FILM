"""Build and validate the U6.P8BX global-coordinate Thomas row ABI."""

from __future__ import annotations

import ctypes
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import find_msvc_installation, sha256_file
from src.eval.native_thomas_field_conformance import profile_from_contract
from src.film_physics.native_thomas_rows import (
    _pointer,
    _workspace,
    load_native_thomas_rows_library,
)

SCHEMA = "neuro_film.u6_p8bx_native_thomas_row_stream_contract.v1"
SOURCE_PATHS = (
    "native/film_physics/nf_thomas_field_f32_v1.c",
    "native/film_physics/nf_thomas_rows_f32_v1.c",
)
HEADER_PATHS = (
    "native/film_physics/nf_thomas_field_f32_v1.h",
    "native/film_physics/nf_thomas_rows_f32_v1.h",
)


class NativeThomasRowsConformanceError(RuntimeError):
    """Raised when the frozen row-stream contract or native build fails."""


def validate_contract(root: Path, contract: Mapping[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("field_and_profile_unchanged") is not True
        or candidate.get("global_counter_coordinates_required") is not True
        or candidate.get("two_pass_full_field_dc_projection_required") is not True
        or candidate.get("model_or_profile_change_allowed") is not False
        or candidate.get("realized_field_renormalization_allowed") is not False
    ):
        raise NativeThomasRowsConformanceError("P8BX contract drift")
    for binding in contract["parents"].values():
        path = root / str(binding["path"])
        if not path.is_file() or sha256_file(path) != str(binding["sha256"]):
            raise NativeThomasRowsConformanceError("P8BX parent hash drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("decision") != binding["required_decision"]:
            raise NativeThomasRowsConformanceError("P8BX parent decision drift")


def build_msvc(root: Path, output_dir: Path) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7/Tools/VsDevCmd.bat"
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / "nf_thomas_rows_msvc_v1.dll").resolve()
    import_library = (output_dir / "nf_thomas_rows_msvc_v1.lib").resolve()
    batch = output_dir / "build_nf_thomas_rows_msvc_v1.bat"
    sources = " ".join(f'"{(root / path).resolve()}"' for path in SOURCE_PATHS)
    batch.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f"cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD {sources} "
        f'/link /Brepro /OUT:"{dll}" /IMPLIB:"{import_library}"\r\n',
        encoding="ascii",
        newline="",
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(batch.resolve())],
        cwd=output_dir,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not dll.is_file():
        raise NativeThomasRowsConformanceError(
            "MSVC P8BX build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "msvc-x64-c11",
        "source_sha256": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "header_sha256": {path: sha256_file(root / path) for path in HEADER_PATHS},
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll),
    }


def build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / "nf_thomas_rows_llvm_v1.dll").resolve()
    command = [
        str(clang),
        "--target=x86_64-w64-windows-gnu",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffp-model=strict",
        "-shared",
        *(str(root / path) for path in SOURCE_PATHS),
        "-o",
        str(dll),
        "-Wl,--no-insert-timestamp",
    ]
    completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
    if completed.returncode != 0 or not dll.is_file():
        raise NativeThomasRowsConformanceError(
            "LLVM P8BX build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "llvm-mingw-20260616-clang-22.1.8-x64",
        "clang_sha256": sha256_file(clang),
        "source_sha256": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "header_sha256": {path: sha256_file(root / path) for path in HEADER_PATHS},
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll),
    }


def failure_output_atomic(library: ctypes.CDLL, root: Path) -> bool:
    profile = profile_from_contract(
        json.loads(
            (root / "configs/u6_p8bs_native_thomas_field_v1.json").read_text(
                encoding="utf-8"
            )
        )
    ).as_abi()
    base = np.zeros(16, dtype=np.float32)
    base[7] = np.nan
    sigma = np.ones(16, dtype=np.float32)
    workspace = _workspace(library, 4, 4)
    output = np.full(16, np.float32(-13.0))
    status = library.nf_thomas_rows_f32_density_v1(
        ctypes.byref(profile),
        4,
        4,
        0,
        4,
        _pointer(base),
        base.size,
        _pointer(sigma),
        sigma.size,
        0.0,
        _pointer(workspace),
        workspace.size,
        _pointer(output),
        output.size,
    )
    return status != 0 and bool(np.all(output == np.float32(-13.0)))


def build_and_load(
    root: Path, output_dir: Path, clang: Path
) -> tuple[dict[str, Any], dict[str, ctypes.CDLL]]:
    builds = {
        "msvc": build_msvc(root, output_dir / "msvc"),
        "llvm_mingw": build_llvm(root, output_dir / "llvm", clang),
    }
    libraries = {
        name: load_native_thomas_rows_library(Path(row["dll_path"]))
        for name, row in builds.items()
    }
    return builds, libraries


__all__ = [
    "NativeThomasRowsConformanceError",
    "build_and_load",
    "build_llvm",
    "build_msvc",
    "failure_output_atomic",
    "validate_contract",
]
