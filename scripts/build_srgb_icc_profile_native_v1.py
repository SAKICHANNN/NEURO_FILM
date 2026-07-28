"""Build the pinned sRGB ICC C ABI under factual platform ceilings."""

from __future__ import annotations

import hashlib
import json
import os
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
SOURCE = ROOT / "native" / "reference_srgb_icc_profile_v1.c"
HEADER = ROOT / "native" / "reference_srgb_icc_profile_v1.h"
RUNNER = ROOT / "native" / "reference_srgb_icc_profile_conformance.cpp"
CORE = ROOT / "native" / "reference_canonical_core.c"
ABI_SYMBOLS = {
    "nf_srgb_icc_profile_copy_v1",
    "nf_srgb_icc_profile_sha256_v1",
    "nf_srgb_icc_profile_size_v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identities() -> dict[str, str]:
    return {
        "source_sha256": _sha256(SOURCE),
        "header_sha256": _sha256(HEADER),
        "runner_sha256": _sha256(RUNNER),
        "canonical_core_sha256": _sha256(CORE),
    }


def build_msvc(output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = _developer_environment()
    compiler = shutil.which("cl.exe", path=environment.get("Path"))
    if compiler is None:
        raise FileNotFoundError("cl.exe is unavailable")
    objects = {
        "profile": output.with_name("srgb_icc_profile.obj"),
        "core": output.with_name("srgb_icc_core.obj"),
        "runner": output.with_name("srgb_icc_runner.obj"),
    }
    for source, object_path in ((SOURCE, objects["profile"]), (CORE, objects["core"])):
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
                str(source),
                f"/Fo:{object_path}",
            ],
            env=environment,
            check=True,
        )
    subprocess.run(
        [
            compiler,
            "/nologo",
            "/std:c++17",
            "/O2",
            "/Brepro",
            "/EHsc",
            "/W4",
            "/WX",
            "/I",
            str(ROOT / "native"),
            "/c",
            str(RUNNER),
            f"/Fo:{objects['runner']}",
        ],
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            compiler,
            "/nologo",
            "/Brepro",
            *(str(path) for path in objects.values()),
            f"/Fe:{output}",
        ],
        env=environment,
        check=True,
    )
    return {
        "protocol": "neuro-film.srgb-icc-profile-msvc-runtime.v1",
        "claim_scope": "Windows x86_64 host compile and execution",
        **_source_identities(),
        "executable_sha256": _sha256(output),
    }


def build_llvm_mingw(toolchain: Path, output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock, compiler = _validate_llvm_mingw(toolchain.resolve())
    objects = [
        output.with_name("srgb_icc_profile.o"),
        output.with_name("srgb_icc_core.o"),
        output.with_name("srgb_icc_runner.o"),
    ]
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
            SOURCE,
            "-o",
            objects[0],
        ],
        check=True,
    )
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
            CORE,
            "-o",
            objects[1],
        ],
        check=True,
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            ROOT / "native",
            "-c",
            RUNNER,
            "-o",
            objects[2],
        ],
        check=True,
    )
    subprocess.run(
        [
            compiler,
            "-static",
            "-Wl,--no-insert-timestamp",
            *objects,
            "-o",
            output,
        ],
        check=True,
    )
    return {
        "protocol": "neuro-film.srgb-icc-profile-llvm-mingw-runtime.v1",
        "claim_scope": "independent same-host Windows x86_64 execution",
        "llvm_version": lock["llvm_version"],
        **_source_identities(),
        "executable_sha256": _sha256(output),
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
        library = output / f"libneuro_film_srgb_icc_{abi}.so"
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
            and line.split()[-1].startswith("nf_srgb_icc_")
        }
        if symbols != ABI_SYMBOLS:
            raise ValueError(f"{abi} exported ABI symbols mismatch")
        artifacts[abi] = {
            "target": target,
            "elf_machine": machine,
            "sha256": _sha256(library),
            "bytes": library.stat().st_size,
            "exported_symbols": sorted(symbols),
        }
    return {
        "protocol": "neuro-film.srgb-icc-profile-android-link.v1",
        "claim_scope": "Android cross-compile/link only; no device execution",
        "ndk_revision": lock["revision"],
        **_source_identities(),
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
        object_path = output / f"reference_srgb_icc_profile_{name}.o"
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
            and line.split()[-1].startswith("_nf_srgb_icc_")
        }
        if symbols != ABI_SYMBOLS:
            raise ValueError(f"{name} defined ABI symbols mismatch")
        artifacts[name] = {
            "target": target,
            "format": "Mach-O arm64",
            "sha256": _sha256(object_path),
            "bytes": object_path.stat().st_size,
            "defined_symbols": sorted(symbols),
        }
    return {
        "protocol": "neuro-film.srgb-icc-profile-apple-object.v1",
        "claim_scope": "Apple ARM64 object-only; not linked or run",
        "llvm_version": lock["llvm_version"],
        **_source_identities(),
        "artifacts": artifacts,
    }
