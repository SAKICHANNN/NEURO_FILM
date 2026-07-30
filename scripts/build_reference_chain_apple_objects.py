"""Compile the freestanding consumer canonical core to Apple ARM64 objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "reference_match_llvm_mingw_20260616.json"
CORE = ROOT / "native" / "reference_canonical_core.c"
TARGETS = {
    "macos-arm64": "arm64-apple-macosx13.0",
    "ios-arm64": "arm64-apple-ios15.0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_toolchain(toolchain: Path) -> tuple[dict[str, Any], Path, Path]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    compiler_cpp = toolchain / "bin" / "clang++.exe"
    compiler_c = toolchain / "bin" / "clang.exe"
    readobj = toolchain / "bin" / "llvm-readobj.exe"
    license_path = toolchain / "LICENSE.TXT"
    if (
        _sha256(compiler_cpp) != lock["compiler_sha256"]
        or _sha256(license_path) != lock["license_sha256"]
    ):
        raise ValueError("LLVM toolchain identity mismatch")
    version = subprocess.run(
        [compiler_c, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if f"clang version {lock['llvm_version']}" not in version:
        raise ValueError("LLVM version mismatch")
    return lock, compiler_c, readobj


def build(toolchain: Path, output: Path) -> dict[str, Any]:
    toolchain = toolchain.resolve()
    output = output.resolve()
    lock, compiler, readobj = _validate_toolchain(toolchain)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for name, target in TARGETS.items():
        object_path = output / f"reference_canonical_core_{name}.o"
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
                CORE,
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
        artifacts[name] = {
            "target": target,
            "format": "Mach-O arm64",
            "file_type": "relocatable-object",
            "bytes": object_path.stat().st_size,
            "sha256": _sha256(object_path),
        }
    return {
        "protocol": "neuro-film.reference-canonical-apple-object-build.v1",
        "claim_scope": "freestanding Apple ARM64 object compilation only",
        "toolchain": {
            "release": lock["release"],
            "llvm_version": lock["llvm_version"],
            "compiler_cpp_sha256": lock["compiler_sha256"],
            "license_sha256": lock["license_sha256"],
        },
        "canonical_core_sha256": _sha256(CORE),
        "artifacts": artifacts,
        "limitations": [
            "no Apple SDK or platform libc++ was used",
            "no link, load, simulator, device or application execution",
            "no Swift/Objective-C bridge, image I/O, performance or signing evidence",
            "LLVM-MinGW is used only as a pinned Clang distribution",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build(arguments.toolchain, arguments.build_dir)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
