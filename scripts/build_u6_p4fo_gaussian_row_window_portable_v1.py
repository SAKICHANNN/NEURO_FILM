from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/film_physics"
SOURCES = tuple(
    NATIVE / name
    for name in (
        "nf_gaussian_rgb_f32_v1.c",
        "nf_gaussian_row_window_f32_v1.c",
    )
)
HEADERS = tuple(
    NATIVE / name
    for name in (
        "nf_gaussian_rgb_f32_v1.h",
        "nf_gaussian_row_window_f32_v1.h",
    )
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
    return result.stdout.decode(errors="replace")


def build(contract_path: Path, ndk: Path, toolchain: Path, output: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    parent = ROOT / contract["parent"]["path"]
    if (
        _sha(parent) != contract["parent"]["sha256"]
        or json.loads(parent.read_text())["decision"]
        != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4FO parent drift")
    output.mkdir(parents=True, exist_ok=True)
    ndk_tools = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    exports = set(contract["required_exports"])
    android = {}
    for abi, target in contract["targets"]["android"].items():
        library = output / f"libnf_gaussian_row_window_{abi}.so"
        _run(
            [
                str(ndk_tools / "clang.exe"), f"--target={target}", "-std=c11",
                "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-model=strict",
                "-shared", "-fPIC", *map(str, SOURCES), "-Wl,--build-id=none",
                "-lm", "-o", str(library),
            ]
        )
        symbols = _run(
            [str(ndk_tools / "llvm-readelf.exe"), "--dyn-symbols", str(library)]
        )
        if not all(name in symbols for name in exports):
            raise RuntimeError("Android P4FO export drift")
        android[abi] = {
            "target": target,
            "sha256": _sha(library),
            "bytes": library.stat().st_size,
        }
    tools = toolchain / "bin"
    apple = {}
    for name, target in contract["targets"]["apple_object_only"].items():
        obj = output / f"nf_gaussian_row_window_{name}.o"
        _run(
            [
                str(tools / "clang.exe"), f"--target={target}", "-std=c11",
                "-O2", "-Wall", "-Wextra", "-Werror", "-ffreestanding",
                "-fno-builtin", "-c", str(SOURCES[1]), "-o", str(obj),
            ]
        )
        symbols = _run(
            [str(tools / "llvm-readobj.exe"), "--file-headers", "--symbols", str(obj)]
        )
        if "Format: Mach-O arm64" not in symbols or not all(
            export in symbols for export in exports
        ):
            raise RuntimeError("Apple P4FO object drift")
        apple[name] = {
            "target": target,
            "sha256": _sha(obj),
            "bytes": obj.stat().st_size,
        }
    stable = {
        "contract_sha256": _sha(contract_path),
        "source_sha256": [_sha(path) for path in SOURCES],
        "header_sha256": [_sha(path) for path in HEADERS],
        "required_exports": sorted(exports),
        "android": android,
        "apple": apple,
        "decision": contract["decision_if_pass"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4fo_gaussian_row_window_portable.v1",
        "automatic_pass": True,
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["build"]
