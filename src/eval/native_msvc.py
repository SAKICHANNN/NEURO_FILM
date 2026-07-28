"""Small deterministic MSVC C11 build helper for local native evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_msvc_c11_dll(
    *,
    root: Path,
    output_dir: Path,
    source_relative: str,
    header_relative: str,
    basename: str,
) -> dict[str, Any]:
    installation = find_msvc_installation()
    vcvars = installation / "Common7" / "Tools" / "VsDevCmd.bat"
    source = (root / source_relative).resolve()
    header = (root / header_relative).resolve()
    if not vcvars.is_file() or not source.is_file() or not header.is_file():
        raise RuntimeError("native source or MSVC environment is missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = (output_dir / f"{basename}.dll").resolve()
    obj = (output_dir / f"{basename}.obj").resolve()
    import_library = (output_dir / f"{basename}.lib").resolve()
    batch = output_dir / f"build_{basename}.bat"
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
        ["cmd.exe", "/d", "/c", str(batch.resolve())],
        capture_output=True,
        timeout=120,
        check=False,
        cwd=output_dir,
    )
    compiler_output = (
        completed.stdout + completed.stderr
    ).decode("utf-8", errors="replace")
    if completed.returncode != 0 or not dll.is_file():
        raise RuntimeError("MSVC native build failed:\n" + compiler_output)
    return {
        "toolchain": "msvc-x64-c11",
        "source_sha256": sha256_file(source),
        "header_sha256": sha256_file(header),
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll),
        "compiler_output": compiler_output.strip(),
    }
