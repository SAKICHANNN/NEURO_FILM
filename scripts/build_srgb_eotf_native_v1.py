"""Build the exact-LUT sRGB EOTF C ABI under factual platform ceilings."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
from typing import Any

from scripts.build_reference_chain_android import (
    TARGETS as ANDROID_TARGETS,
    _validate_ndk,
)
from scripts.build_reference_chain_apple_objects import (
    TARGETS as APPLE_TARGETS,
    _validate_toolchain as _validate_apple_toolchain,
)
from scripts.build_reference_chain_llvm_mingw import (
    _validate_toolchain as _validate_llvm_mingw,
)
from scripts.build_reference_chain_msvc import _developer_environment


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "reference_srgb_eotf_f32_v1.c"
HEADER = ROOT / "native" / "reference_srgb_eotf_f32_v1.h"
ABI_SYMBOLS = {
    "nf_srgb_eotf_f32_apply_v1",
    "nf_srgb_eotf_f32_lut_sha256_v1",
}
DLL_ENTRY_SOURCE = (
    "int DllMainCRTStartup(void *instance, unsigned long reason, "
    "void *reserved) {\n"
    "    (void)instance;\n"
    "    (void)reason;\n"
    "    (void)reserved;\n"
    "    return 1;\n"
    "}\n"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exports(path: Path) -> None:
    path.write_text(
        "EXPORTS\n"
        "nf_srgb_eotf_f32_apply_v1\n"
        "nf_srgb_eotf_f32_lut_sha256_v1\n",
        encoding="ascii",
        newline="\n",
    )


def _base_report() -> dict[str, str]:
    return {
        "source_sha256": _sha256(SOURCE),
        "header_sha256": _sha256(HEADER),
    }


def build_msvc_dll(output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = _developer_environment()
    compiler = shutil.which("cl.exe", path=environment.get("Path"))
    linker = shutil.which("link.exe", path=environment.get("Path"))
    dumpbin = shutil.which("dumpbin.exe", path=environment.get("Path"))
    if compiler is None or linker is None or dumpbin is None:
        raise FileNotFoundError("MSVC DLL tools are unavailable")
    object_path = output.with_suffix(".obj")
    exports_path = output.with_suffix(".def")
    _write_exports(exports_path)
    subprocess.run(
        [
            compiler,
            "/nologo",
            "/TC",
            "/std:c11",
            "/O2",
            "/Brepro",
            "/W4",
            "/WX",
            "/c",
            str(SOURCE),
            f"/Fo:{object_path}",
        ],
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            linker,
            "/NOLOGO",
            "/DLL",
            "/NOENTRY",
            "/Brepro",
            str(object_path),
            f"/DEF:{exports_path}",
            f"/OUT:{output}",
        ],
        env=environment,
        check=True,
    )
    export_output = subprocess.run(
        [dumpbin, "/nologo", "/exports", output],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exports = {
        line.split()[-1]
        for line in export_output.splitlines()
        if line.split()
        and line.split()[-1].startswith("nf_srgb_eotf_")
    }
    if exports != ABI_SYMBOLS:
        raise ValueError("MSVC EOTF DLL exports mismatch")
    return {
        "protocol": "neuro-film.srgb-eotf-msvc-dll-runtime.v1",
        "claim_scope": "Windows x86_64 dynamic EOTF C ABI execution",
        **_base_report(),
        "dll_sha256": _sha256(output),
        "exported_symbols": sorted(exports),
    }


def build_llvm_mingw_dll(
    toolchain: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock, compiler = _validate_llvm_mingw(toolchain.resolve())
    readobj = compiler.with_name("llvm-readobj.exe")
    object_path = output.with_suffix(".o")
    exports_path = output.with_suffix(".def")
    entry_source = output.with_name("srgb_eotf_dll_entry.c")
    entry_object = output.with_name("srgb_eotf_dll_entry.o")
    _write_exports(exports_path)
    entry_source.write_text(
        DLL_ENTRY_SOURCE,
        encoding="ascii",
        newline="\n",
    )
    for source, object_file in (
        (SOURCE, object_path),
        (entry_source, entry_object),
    ):
        subprocess.run(
            [
                compiler,
                "-x",
                "c",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-c",
                source,
                "-o",
                object_file,
            ],
            check=True,
        )
    subprocess.run(
        [
            compiler,
            "-shared",
            "-nostdlib",
            "-Wl,--no-insert-timestamp",
            object_path,
            entry_object,
            exports_path,
            "-o",
            output,
        ],
        check=True,
    )
    export_output = subprocess.run(
        [readobj, "--coff-exports", output],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exports = {
        line.split(":", 1)[1].strip()
        for line in export_output.splitlines()
        if line.strip().startswith("Name:")
        and line.split(":", 1)[1].strip().startswith("nf_srgb_eotf_")
    }
    if exports != ABI_SYMBOLS:
        raise ValueError("LLVM EOTF DLL exports mismatch")
    return {
        "protocol": "neuro-film.srgb-eotf-llvm-mingw-dll-runtime.v1",
        "claim_scope": "independent Windows x86_64 dynamic EOTF execution",
        "llvm_version": lock["llvm_version"],
        **_base_report(),
        "dll_sha256": _sha256(output),
        "exported_symbols": sorted(exports),
    }


def build_android(ndk: Path, output: Path) -> dict[str, Any]:
    ndk = ndk.resolve()
    output = output.resolve()
    lock = _validate_ndk(ndk)
    binary_dir = (
        ndk
        / "toolchains"
        / "llvm"
        / "prebuilt"
        / "windows-x86_64"
        / "bin"
    )
    compiler = binary_dir / "clang.exe"
    readelf = binary_dir / "llvm-readelf.exe"
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for abi, (target, expected_machine) in ANDROID_TARGETS.items():
        library = output / f"libneuro_film_srgb_eotf_{abi}.so"
        subprocess.run(
            [
                compiler,
                f"--target={target}",
                "-std=c11",
                "-O2",
                "-fPIC",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-Wl,--no-undefined",
                SOURCE,
                "-o",
                library,
            ],
            check=True,
        )
        header = subprocess.run(
            [readelf, "-h", library],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        machine = next(
            line.split(":", 1)[1].strip()
            for line in header.splitlines()
            if line.strip().startswith("Machine:")
        )
        if machine != expected_machine:
            raise ValueError(f"{abi} machine mismatch")
        symbols_output = subprocess.run(
            [readelf, "--wide", "--dyn-syms", library],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        symbols = {
            line.split()[-1]
            for line in symbols_output.splitlines()
            if line.split()
            and line.split()[-1].startswith("nf_srgb_eotf_")
        }
        if symbols != ABI_SYMBOLS:
            raise ValueError(f"{abi} exported EOTF symbols mismatch")
        artifacts[abi] = {
            "target": target,
            "elf_machine": machine,
            "sha256": _sha256(library),
            "bytes": library.stat().st_size,
            "exported_symbols": sorted(symbols),
        }
    return {
        "protocol": "neuro-film.srgb-eotf-android-link.v1",
        "claim_scope": "Android cross-compile/link only; no device execution",
        "ndk_revision": lock["revision"],
        **_base_report(),
        "artifacts": artifacts,
    }


def build_apple_objects(toolchain: Path, output: Path) -> dict[str, Any]:
    toolchain = toolchain.resolve()
    output = output.resolve()
    lock, compiler, readobj = _validate_apple_toolchain(toolchain)
    nm = compiler.with_name("llvm-nm.exe")
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for name, target in APPLE_TARGETS.items():
        object_path = output / f"reference_srgb_eotf_{name}.o"
        subprocess.run(
            [
                compiler,
                f"--target={target}",
                "-std=c11",
                "-O2",
                "-ffreestanding",
                "-fno-builtin",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-c",
                SOURCE,
                "-o",
                object_path,
            ],
            check=True,
        )
        header = subprocess.run(
            [readobj, "--file-headers", object_path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "Format: Mach-O arm64" not in header or "Arch: aarch64" not in header:
            raise ValueError(f"{name} is not a Mach-O arm64 object")
        symbols_output = subprocess.run(
            [nm, "--defined-only", "--extern-only", object_path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        symbols = {
            line.split()[-1].removeprefix("_")
            for line in symbols_output.splitlines()
            if line.split()
            and line.split()[-1].startswith("_nf_srgb_eotf_")
        }
        if symbols != ABI_SYMBOLS:
            raise ValueError(f"{name} defined EOTF symbols mismatch")
        artifacts[name] = {
            "target": target,
            "format": "Mach-O arm64",
            "sha256": _sha256(object_path),
            "bytes": object_path.stat().st_size,
            "defined_symbols": sorted(symbols),
        }
    return {
        "protocol": "neuro-film.srgb-eotf-apple-object.v1",
        "claim_scope": "Apple ARM64 object-only; not linked or run",
        "llvm_version": lock["llvm_version"],
        **_base_report(),
        "artifacts": artifacts,
    }
