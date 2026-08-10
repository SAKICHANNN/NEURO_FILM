"""Build the independent exact float32 Neumaier reduction ABI."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Any

from src.eval.native_msvc import find_msvc_installation, sha256_file
from src.film_physics.native_thomas_rows import load_native_neumaier_library

SOURCE = "native/film_physics/nf_neumaier_f32_v1.c"
HEADER = "native/film_physics/nf_neumaier_f32_v1.h"


def build_msvc(root: Path, output_dir: Path) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7/Tools/VsDevCmd.bat"
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / "nf_neumaier_f32_msvc_v1.dll").resolve()
    library = (output_dir / "nf_neumaier_f32_msvc_v1.lib").resolve()
    batch = output_dir / "build_nf_neumaier_f32_msvc_v1.bat"
    batch.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f'cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD "{(root / SOURCE).resolve()}" '
        f'/link /Brepro /OUT:"{dll}" /IMPLIB:"{library}"\r\n',
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
        raise RuntimeError(
            "MSVC native Neumaier build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "msvc-x64-c11-strict",
        "source_sha256": sha256_file(root / SOURCE),
        "header_sha256": sha256_file(root / HEADER),
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll),
    }


def build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / "nf_neumaier_f32_llvm_v1.dll").resolve()
    completed = subprocess.run(
        [
            str(clang),
            "--target=x86_64-w64-windows-gnu",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-model=strict",
            "-shared",
            str(root / SOURCE),
            "-o",
            str(dll),
            "-Wl,--no-insert-timestamp",
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not dll.is_file():
        raise RuntimeError(
            "LLVM native Neumaier build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "llvm-mingw-x64-c11-strict",
        "clang_sha256": sha256_file(clang),
        "source_sha256": sha256_file(root / SOURCE),
        "header_sha256": sha256_file(root / HEADER),
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll),
    }


def build_and_load(
    root: Path, output_dir: Path, clang: Path
) -> tuple[dict[str, Any], dict[str, ctypes.CDLL]]:
    builds = {
        "msvc": build_msvc(root, output_dir / "msvc"),
        "llvm_mingw": build_llvm(root, output_dir / "llvm", clang),
    }
    libraries = {
        name: load_native_neumaier_library(Path(row["dll_path"]))
        for name, row in builds.items()
    }
    return builds, libraries


__all__ = ["build_and_load", "build_llvm", "build_msvc"]
