"""Build the consumer-chain verifier with pinned LLVM-MinGW."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "reference_match_llvm_mingw_20260616.json"
SOURCE = ROOT / "native" / "reference_product_chain_conformance.cpp"
CORE = ROOT / "native" / "reference_canonical_core.c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_toolchain(toolchain: Path) -> tuple[dict[str, Any], Path]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    compiler = toolchain / "bin" / "clang++.exe"
    license_path = toolchain / "LICENSE.TXT"
    if (
        _sha256(compiler) != lock["compiler_sha256"]
        or _sha256(license_path) != lock["license_sha256"]
    ):
        raise ValueError("LLVM-MinGW identity mismatch")
    version = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if (
        f"clang version {lock['llvm_version']}" not in version
        or f"Target: {lock['target']}" not in version
    ):
        raise ValueError("LLVM-MinGW version or target mismatch")
    return lock, compiler


def build(toolchain: Path, output: Path) -> dict[str, Any]:
    toolchain = toolchain.resolve()
    output = output.resolve()
    lock, compiler = _validate_toolchain(toolchain)
    output.parent.mkdir(parents=True, exist_ok=True)
    core_object = output.with_name("reference_canonical_core.o")
    runner_object = output.with_name(
        "reference_product_chain_conformance.o"
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
            core_object,
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
            "-c",
            SOURCE,
            "-o",
            runner_object,
        ],
        check=True,
    )
    command = [
        compiler,
        "-static",
        core_object,
        runner_object,
        "-o",
        output,
    ]
    subprocess.run(command, check=True)
    return {
        "protocol": "neuro-film.reference-product-chain-llvm-mingw.v1",
        "claim_scope": "independent Windows x86_64 compile and execution",
        "toolchain": {
            "release": lock["release"],
            "llvm_version": lock["llvm_version"],
            "target": lock["target"],
            "compiler_sha256": lock["compiler_sha256"],
            "license_sha256": lock["license_sha256"],
        },
        "source_sha256": _sha256(SOURCE),
        "canonical_core_sha256": _sha256(CORE),
        "executable": {
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "static_runtime": True,
        },
        "limitations": [
            "same Windows x86_64 host as MSVC evidence",
            "no Android, Apple, image-I/O, quality or performance evidence",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build(arguments.toolchain, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
