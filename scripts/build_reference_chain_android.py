"""Cross-compile the consumer-chain verifier with a pinned Android NDK."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "reference_match_android_ndk_r27d.json"
SOURCE = ROOT / "native" / "reference_product_chain_conformance.cpp"
TARGETS = {
    "arm64-v8a": ("aarch64-linux-android21", "AArch64"),
    "x86_64": ("x86_64-linux-android21", "Advanced Micro Devices X86-64"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_ndk(ndk: Path) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source_properties = ndk / "source.properties"
    notice = ndk / "NOTICE"
    if (
        _sha256(source_properties)
        != lock["source_properties_sha256"]
        or _sha256(notice) != lock["notice_sha256"]
    ):
        raise ValueError("Android NDK identity mismatch")
    if (
        f"Pkg.Revision = {lock['revision']}"
        not in source_properties.read_text(encoding="utf-8")
    ):
        raise ValueError("Android NDK revision mismatch")
    return lock


def build(ndk: Path, output: Path) -> dict[str, Any]:
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
    compiler = binary_dir / "clang++.exe"
    readelf = binary_dir / "llvm-readelf.exe"
    output.mkdir(parents=True, exist_ok=True)
    compiler_version = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    artifacts: dict[str, Any] = {}
    for abi, (target, expected_machine) in TARGETS.items():
        executable = output / f"reference_product_chain_{abi}"
        subprocess.run(
            [
                compiler,
                f"--target={target}",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                SOURCE,
                "-o",
                executable,
            ],
            check=True,
        )
        header = subprocess.run(
            [readelf, "-h", executable],
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
            raise ValueError(
                f"{abi} machine mismatch: {machine} != {expected_machine}"
            )
        artifacts[abi] = {
            "target": target,
            "elf_machine": machine,
            "bytes": executable.stat().st_size,
            "sha256": _sha256(executable),
        }
    return {
        "protocol": "neuro-film.reference-product-chain-android-build.v1",
        "claim_scope": "cross-compile and link; no device execution",
        "ndk": {
            "release": lock["release"],
            "revision": lock["revision"],
            "source_properties_sha256": lock[
                "source_properties_sha256"
            ],
            "notice_sha256": lock["notice_sha256"],
        },
        "compiler": compiler_version,
        "source_sha256": _sha256(SOURCE),
        "artifacts": artifacts,
        "limitations": [
            "no Android device or emulator execution",
            "no JNI, Kotlin, image I/O, performance or thermal evidence",
            "no iOS or macOS compilation evidence",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build(arguments.ndk, arguments.build_dir)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
