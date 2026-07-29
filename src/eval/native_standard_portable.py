"""Cross-toolchain builders for the frozen native Standard C components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


COMPONENT_NAMES = (
    "domains",
    "gaussian",
    "adjacency",
    "gauge",
    "context",
    "display",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(root: Path, contract: dict[str, Any]) -> None:
    if contract.get("schema") != (
        "neuro_film.u6_p8bo_portable_native_standard_contract.v1"
    ):
        raise ValueError("unsupported P8BO contract")
    if tuple(contract["components"]) != COMPONENT_NAMES:
        raise ValueError("P8BO component order drift")
    parent = contract["parent_decision"]
    parent_path = root / parent["path"]
    if sha256_file(parent_path) != parent["sha256"]:
        raise ValueError("P8BO parent identity drift")
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if not str(parent_payload["next_leaf"]).startswith("U6.P8BO"):
        raise ValueError("P8BO parent does not open this leaf")
    for row in contract["components"].values():
        stem = root / "native" / "film_physics" / row["basename"]
        if sha256_file(stem.with_suffix(".h")) != row["header_sha256"]:
            raise ValueError("P8BO native source identity drift")
        sources = row.get("link_sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("P8BO component link closure is missing")
        if f"native/film_physics/{row['basename']}.c" not in sources:
            raise ValueError("P8BO primary component source is missing")
        for relative, expected_sha256 in sources.items():
            if (
                not relative.startswith("native/film_physics/")
                or Path(relative).suffix != ".c"
                or sha256_file(root / relative) != expected_sha256
            ):
                raise ValueError("P8BO native source identity drift")
    if contract["oracle"]["strength"] != 1.0:
        raise ValueError("P8BO must retain full native strength")


def validate_llvm_toolchain(
    toolchain: Path, expected: dict[str, Any]
) -> tuple[Path, Path]:
    clang = toolchain / "bin" / "clang.exe"
    readobj = toolchain / "bin" / "llvm-readobj.exe"
    if (
        sha256_file(clang) != expected["clang_sha256"]
        or sha256_file(readobj) != expected["llvm_readobj_sha256"]
        or sha256_file(toolchain / "LICENSE.TXT")
        != expected["license_sha256"]
    ):
        raise ValueError("P8BO LLVM toolchain identity drift")
    version = subprocess.run(
        [clang, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    if f"clang version {expected['version']}" not in version:
        raise ValueError("P8BO LLVM version drift")
    return clang, readobj


def validate_ndk(ndk: Path, expected: dict[str, Any]) -> tuple[Path, Path]:
    properties = ndk / "source.properties"
    notice = ndk / "NOTICE"
    if (
        sha256_file(properties) != expected["source_properties_sha256"]
        or sha256_file(notice) != expected["notice_sha256"]
        or f"Pkg.Revision = {expected['revision']}"
        not in properties.read_text(encoding="utf-8")
    ):
        raise ValueError("P8BO Android NDK identity drift")
    binary = (
        ndk
        / "toolchains"
        / "llvm"
        / "prebuilt"
        / "windows-x86_64"
        / "bin"
    )
    return binary / "clang.exe", binary / "llvm-readelf.exe"


def _run(command: list[Path | str]) -> None:
    completed = subprocess.run(
        [str(value) for value in command],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "native portable build failed:\n"
            + completed.stdout
            + completed.stderr
        )


def build_windows_dlls(
    *,
    root: Path,
    contract: dict[str, Any],
    clang: Path,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, row in contract["components"].items():
        sources = [root / relative for relative in row["link_sources"]]
        output = output_dir / f"{row['basename']}.dll"
        _run(
            [
                clang,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                *sources,
                "-o",
                output,
                "-lm",
                "-Wl,--no-insert-timestamp",
            ]
        )
        results[name] = {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
    return results


def build_android_libraries(
    *,
    root: Path,
    contract: dict[str, Any],
    clang: Path,
    readelf: Path,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    machines = {
        "arm64-v8a": "AArch64",
        "x86_64": "Advanced Micro Devices X86-64",
    }
    results: dict[str, dict[str, Any]] = {}
    for abi, target in contract["targets"]["android"].items():
        abi_dir = output_dir / abi
        abi_dir.mkdir(parents=True, exist_ok=True)
        components = {}
        for name, row in contract["components"].items():
            sources = [root / relative for relative in row["link_sources"]]
            output = abi_dir / f"lib{row['basename']}.so"
            _run(
                [
                    clang,
                    f"--target={target}",
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-shared",
                    "-fPIC",
                    *sources,
                    "-o",
                    output,
                    "-lm",
                    "-Wl,--build-id=none",
                ]
            )
            header = subprocess.run(
                [readelf, "-h", output],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            machine = next(
                line.split(":", 1)[1].strip()
                for line in header.splitlines()
                if line.strip().startswith("Machine:")
            )
            if machine != machines[abi]:
                raise ValueError("P8BO Android machine drift")
            components[name] = {
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        results[abi] = {
            "target": target,
            "machine": machines[abi],
            "components": components,
        }
    return results


def build_apple_abi_objects(
    *,
    root: Path,
    contract: dict[str, Any],
    clang: Path,
    readobj: Path,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    witness = output_dir / "native_standard_abi_witness.c"
    includes = "\n".join(
        f'#include "{row["basename"]}.h"'
        for row in contract["components"].values()
    )
    witness.write_text(
        includes
        + "\n"
        + "uint32_t nf_native_standard_abi_witness_v1(void) {\n"
        + "  return NF_PHYSICAL_DOMAINS_F32_ABI_VERSION_V1\n"
        + "    + NF_GAUSSIAN_F32_ABI_VERSION_V1\n"
        + "    + NF_BOUNDED_ADJACENCY_F32_ABI_VERSION_V1\n"
        + "    + NF_NEUTRAL_GAUGE_F32_ABI_VERSION_V1\n"
        + "    + NF_AO6_CONTEXT_F32_ABI_VERSION_V2\n"
        + "    + NF_AO6_DISPLAY_F32_ABI_VERSION_V4;\n"
        + "}\n",
        encoding="ascii",
        newline="\n",
    )
    include_dir = root / "native" / "film_physics"
    results = {}
    for name, target in contract["targets"]["apple_object_only"].items():
        output = output_dir / f"native_standard_abi_{name}.o"
        _run(
            [
                clang,
                f"--target={target}",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                include_dir,
                "-c",
                witness,
                "-o",
                output,
            ]
        )
        header = subprocess.run(
            [readobj, "--file-headers", output],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        if (
            "Format: Mach-O arm64" not in header
            or "Arch: aarch64" not in header
        ):
            raise ValueError("P8BO Apple object target drift")
        results[name] = {
            "target": target,
            "format": "Mach-O arm64",
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
    return results


def build_hashes(
    build: dict[str, dict[str, Any]]
) -> dict[str, str]:
    return {name: row["sha256"] for name, row in build.items()}
