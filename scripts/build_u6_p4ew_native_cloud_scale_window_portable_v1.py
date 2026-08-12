from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/film_physics"
SOURCES = tuple(NATIVE / name for name in (
    "nf_conditioned_cloud_row_chain_f32_v1.c",
    "nf_conditioned_cloud_row_chain_f32_v2.c",
    "nf_density_conditioned_poisson_u16_v3.c",
    "nf_cloud_spatial_response_f32_v2.c",
    "nf_cloud_attenuation_f32_v1.c",
    "nf_deterministic_log10_f32_v1.c",
))
HEADERS = tuple(NATIVE / name for name in (
    "nf_conditioned_cloud_row_chain_f32_v1.h",
    "nf_conditioned_cloud_row_chain_f32_v2.h",
    "nf_density_conditioned_poisson_u16_v3.h",
    "nf_cloud_spatial_response_f32_v2.h",
    "nf_cloud_attenuation_f32_v1.h",
    "nf_deterministic_log10_f32_v1.h",
))


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
    if (_sha(parent) != contract["parent"]["sha256"] or
            json.loads(parent.read_text())["decision"] != contract["parent"]["required_decision"]):
        raise RuntimeError("P4EW parent drift")
    output.mkdir(parents=True, exist_ok=True)
    ndk_tools = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    exports = set(contract["required_exports"])
    android = {}
    for abi, target in contract["targets"]["android"].items():
        library = output / f"libnf_cloud_scale_window_{abi}.so"
        _run([str(ndk_tools / "clang.exe"), f"--target={target}", "-std=c11", "-O2",
              "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", "-shared", "-fPIC",
              *map(str, SOURCES), "-Wl,--build-id=none", "-lm", "-o", str(library)])
        symbols = _run([str(ndk_tools / "llvm-readelf.exe"), "--dyn-symbols", str(library)])
        if not all(name in symbols for name in exports):
            raise RuntimeError("Android P4EW export drift")
        android[abi] = {"target": target, "sha256": _sha(library), "bytes": library.stat().st_size}
    freestanding = output / "freestanding_headers"
    freestanding.mkdir(exist_ok=True)
    (freestanding / "math.h").write_text(
        "#ifndef NF_FREESTANDING_MATH_H\n#define NF_FREESTANDING_MATH_H\n"
        "double ceil(double);\ndouble exp(double);\ndouble fmin(double,double);\n"
        "double fmax(double,double);\ndouble pow(double,double);\nint isfinite(double);\n#endif\n",
        encoding="ascii",
    )
    (freestanding / "string.h").write_text(
        "#ifndef NF_FREESTANDING_STRING_H\n#define NF_FREESTANDING_STRING_H\n"
        "typedef __SIZE_TYPE__ size_t;\nvoid *memcpy(void *, const void *, size_t);\n#endif\n",
        encoding="ascii",
    )
    tools = toolchain / "bin"
    apple = {}
    for name, target in contract["targets"]["apple_object_only"].items():
        objects = []
        for source in SOURCES:
            obj = output / f"{source.stem}_{name}.o"
            _run([str(tools / "clang.exe"), f"--target={target}", "-std=c11", "-O2",
                  "-Wall", "-Wextra", "-Werror", "-ffreestanding", "-fno-builtin",
                  "-isystem", str(freestanding), "-c", str(source), "-o", str(obj)])
            objects.append(obj)
        symbols = "\n".join(_run([str(tools / "llvm-readobj.exe"), "--file-headers", "--symbols", str(obj)]) for obj in objects)
        if "Format: Mach-O arm64" not in symbols or not all(export in symbols for export in exports):
            raise RuntimeError("Apple P4EW object drift")
        apple[name] = {"target": target, "sha256": [_sha(obj) for obj in objects], "bytes": [obj.stat().st_size for obj in objects]}
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
        "schema": "neuro_film.u6_p4ew_native_cloud_scale_window_portable.v1",
        "automatic_pass": True,
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


__all__ = ["build"]
