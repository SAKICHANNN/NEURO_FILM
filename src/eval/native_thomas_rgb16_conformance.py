"""Build and load the U6.P8CJ freestanding Thomas RGB16 program."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Any

from src.eval.native_msvc import find_msvc_installation, sha256_file
from src.film_physics.native_gauge_profile import NativeGaugeProfileF32V1
from src.film_physics.native_granularity_amplitude import (
    NativeGranularityAmplitudeProfileV1,
)
from src.film_physics.native_thomas_field import NativeThomasFieldProfileV1

SOURCE_PATHS = (
    "native/film_physics/nf_granularity_amplitude_f32_v1.c",
    "native/film_physics/nf_thomas_field_f32_v1.c",
    "native/film_physics/nf_thomas_rows_f32_v1.c",
    "native/film_physics/nf_neumaier_f32_v1.c",
    "native/film_physics/nf_neutral_gauge_f32_v1.c",
    "native/reference_srgb_oetf_quantize_v1.c",
    "native/film_physics/nf_thomas_rgb16_f32_v1.c",
)
HEADER_PATHS = (
    "native/film_physics/nf_granularity_amplitude_f32_v1.h",
    "native/film_physics/nf_thomas_field_f32_v1.h",
    "native/film_physics/nf_thomas_rows_f32_v1.h",
    "native/film_physics/nf_neumaier_f32_v1.h",
    "native/film_physics/nf_neutral_gauge_f32_v1.h",
    "native/reference_srgb_oetf_quantize_v1.h",
    "native/film_physics/nf_thomas_rgb16_f32_v1.h",
)

Rgb16Sink = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_size_t,
)


class NativeThomasRgb16ConformanceError(RuntimeError):
    """Raised when the native RGB16 program build or ABI fails."""


def load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_thomas_rgb16_f32_abi_version_v1.argtypes = []
    library.nf_thomas_rgb16_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_thomas_rgb16_f32_workspace_bytes_v1.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.nf_thomas_rgb16_f32_workspace_bytes_v1.restype = ctypes.c_int
    library.nf_thomas_rgb16_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeGranularityAmplitudeProfileV1),
        ctypes.POINTER(NativeThomasFieldProfileV1),
        ctypes.POINTER(NativeGaugeProfileF32V1),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        Rgb16Sink,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_thomas_rgb16_f32_apply_v1.restype = ctypes.c_int
    if library.nf_thomas_rgb16_f32_abi_version_v1() != 1:
        raise NativeThomasRgb16ConformanceError("Thomas RGB16 ABI version mismatch")
    return library


def _report(root: Path, toolchain: str, dll: Path) -> dict[str, Any]:
    return {
        "toolchain": toolchain,
        "source_sha256": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "header_sha256": {path: sha256_file(root / path) for path in HEADER_PATHS},
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll),
    }


def build_msvc(root: Path, output_dir: Path) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7/Tools/VsDevCmd.bat"
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / "nf_thomas_rgb16_msvc_v1.dll").resolve()
    import_library = (output_dir / "nf_thomas_rgb16_msvc_v1.lib").resolve()
    batch = output_dir / "build_nf_thomas_rgb16_msvc_v1.bat"
    sources = " ".join(f'"{(root / path).resolve()}"' for path in SOURCE_PATHS)
    batch.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f'cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD '
        f'/I"{(root / "native").resolve()}" '
        f'/I"{(root / "native/film_physics").resolve()}" {sources} '
        f'/link /Brepro /OUT:"{dll}" /IMPLIB:"{import_library}"\r\n',
        encoding="ascii",
        newline="",
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(batch)],
        cwd=output_dir,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0 or not dll.is_file():
        raise NativeThomasRgb16ConformanceError(
            "MSVC Thomas RGB16 build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return _report(root, "msvc-x64-c11-strict", dll)


def build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / "nf_thomas_rgb16_llvm_v1.dll").resolve()
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
        "-I",
        str(root / "native"),
        "-I",
        str(root / "native/film_physics"),
        *(str(root / path) for path in SOURCE_PATHS),
        "-o",
        str(dll),
        "-Wl,--no-insert-timestamp",
    ]
    completed = subprocess.run(command, capture_output=True, check=False, timeout=180)
    if completed.returncode != 0 or not dll.is_file():
        raise NativeThomasRgb16ConformanceError(
            "LLVM Thomas RGB16 build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    report = _report(root, "llvm-mingw-x64-c11-strict", dll)
    report["clang_sha256"] = sha256_file(clang)
    return report


def build_and_load(
    root: Path, output_dir: Path, clang: Path
) -> tuple[dict[str, Any], dict[str, ctypes.CDLL]]:
    builds = {
        "msvc": build_msvc(root, output_dir / "msvc"),
        "llvm_mingw": build_llvm(root, output_dir / "llvm", clang),
    }
    return builds, {
        name: load_library(Path(report["dll_path"]))
        for name, report in builds.items()
    }


__all__ = [
    "NativeThomasRgb16ConformanceError",
    "Rgb16Sink",
    "build_and_load",
    "build_llvm",
    "build_msvc",
    "load_library",
]
