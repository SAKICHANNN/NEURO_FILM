from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/film_physics/nf_cloud_attenuation_f32_v1.c"
HEADER = ROOT / "native/film_physics/nf_cloud_attenuation_f32_v1.h"
EXPORTS = {
    "nf_cloud_attenuation_f32_abi_version_v1",
    "nf_cloud_attenuation_f32_apply_v1",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
    return result.stdout.decode(errors="replace")


def build_android(ndk: Path, output: Path, targets: dict[str, str]) -> dict:
    tools = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    clang, readelf = tools / "clang.exe", tools / "llvm-readelf.exe"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for abi, target in targets.items():
        library = output / f"libnf_cloud_attenuation_{abi}.so"
        _run(
            [
                str(clang),
                f"--target={target}",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffp-model=strict",
                "-shared",
                "-fPIC",
                str(SOURCE),
                "-Wl,--build-id=none",
                "-lm",
                "-o",
                str(library),
            ]
        )
        symbols = _run([str(readelf), "--dyn-symbols", str(library)])
        if not all(name in symbols for name in EXPORTS):
            raise RuntimeError("Android export drift")
        artifacts[abi] = {
            "target": target,
            "sha256": _sha(library),
            "bytes": library.stat().st_size,
        }
    return artifacts


def build_apple(toolchain: Path, output: Path, targets: dict[str, str]) -> dict:
    clang = toolchain / "bin/clang.exe"
    readobj = toolchain / "bin/llvm-readobj.exe"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, target in targets.items():
        obj = output / f"nf_cloud_attenuation_{name}.o"
        _run(
            [
                str(clang),
                f"--target={target}",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffreestanding",
                "-fno-builtin",
                "-c",
                str(SOURCE),
                "-o",
                str(obj),
            ]
        )
        headers = _run([str(readobj), "--file-headers", "--symbols", str(obj)])
        if "Format: Mach-O arm64" not in headers or not all(
            name in headers for name in EXPORTS
        ):
            raise RuntimeError("Apple object identity drift")
        artifacts[name] = {
            "target": target,
            "sha256": _sha(obj),
            "bytes": obj.stat().st_size,
        }
    return artifacts


def build(contract_path: Path, ndk: Path, toolchain: Path, output: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    result = {
        "schema": "neuro_film.u6_p4ea_native_cloud_attenuation_portable_build.v1",
        "contract_sha256": _sha(contract_path),
        "source_sha256": _sha(SOURCE),
        "header_sha256": _sha(HEADER),
        "android": build_android(
            ndk, output / "android", contract["targets"]["android"]
        ),
        "apple": build_apple(
            toolchain, output / "apple", contract["targets"]["apple_object_only"]
        ),
    }
    result["automatic_pass"] = True
    result["decision"] = contract["decision_if_pass"]
    result["claim_ceiling"] = contract["claim_ceiling"]
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ndk", type=Path, required=True)
    p.add_argument("--toolchain", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = build(
        ROOT / "configs/u6_p4ea_native_cloud_attenuation_portable_build_v1.json",
        a.ndk,
        a.toolchain,
        a.build_dir,
    )
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
