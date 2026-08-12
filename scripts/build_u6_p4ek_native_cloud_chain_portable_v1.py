from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "native/film_physics/nf_density_conditioned_poisson_u16_v2.c",
    ROOT / "native/film_physics/nf_cloud_spatial_response_f32_v2.c",
)
HEADERS = (
    ROOT / "native/film_physics/nf_density_conditioned_poisson_u16_v2.h",
    ROOT / "native/film_physics/nf_cloud_spatial_response_f32_v2.h",
)
EXPORTS = {
    "nf_density_conditioned_poisson_u16_abi_version_v2",
    "nf_density_conditioned_poisson_u16_sample_region_v2",
    "nf_cloud_spatial_response_f32_abi_version_v2",
    "nf_cloud_spatial_response_f32_apply_v2",
}


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
    evidence = json.loads(parent.read_text())
    if _sha(parent) != contract["parent"]["sha256"] or evidence["decision"] != contract["parent"]["required_decision"]:
        raise RuntimeError("P4EK parent drift")
    ndk_tools = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    output.mkdir(parents=True, exist_ok=True)
    android = {}
    for abi, target in contract["targets"]["android"].items():
        library = output / f"libnf_cloud_chain_{abi}.so"
        _run([str(ndk_tools / "clang.exe"), f"--target={target}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", "-shared", "-fPIC", *map(str, SOURCES), "-Wl,--build-id=none", "-lm", "-o", str(library)])
        symbols = _run([str(ndk_tools / "llvm-readelf.exe"), "--dyn-symbols", str(library)])
        if not all(name in symbols for name in EXPORTS): raise RuntimeError("Android cloud-chain export drift")
        android[abi] = {"target": target, "sha256": _sha(library), "bytes": library.stat().st_size}
    apple = {}
    tools = toolchain / "bin"
    freestanding_headers = output / "freestanding_headers"
    freestanding_headers.mkdir(exist_ok=True)
    (freestanding_headers / "math.h").write_text(
        "#ifndef NF_FREESTANDING_MATH_H\n"
        "#define NF_FREESTANDING_MATH_H\n"
        "double exp(double);\n"
        "double fmin(double, double);\n"
        "double fmax(double, double);\n"
        "int isfinite(double);\n"
        "#endif\n",
        encoding="ascii",
    )
    for name, target in contract["targets"]["apple_object_only"].items():
        objects=[]
        for source in SOURCES:
            obj=output/f"{source.stem}_{name}.o"
            _run([str(tools/"clang.exe"),f"--target={target}","-std=c11","-O2","-Wall","-Wextra","-Werror","-ffreestanding","-fno-builtin","-isystem",str(freestanding_headers),"-c",str(source),"-o",str(obj)])
            objects.append(obj)
        headers="\n".join(_run([str(tools/"llvm-readobj.exe"),"--file-headers","--symbols",str(obj)]) for obj in objects)
        if "Format: Mach-O arm64" not in headers or not all(export in headers for export in EXPORTS): raise RuntimeError("Apple cloud-chain object drift")
        apple[name]={"target":target,"sha256":[_sha(obj) for obj in objects],"bytes":[obj.stat().st_size for obj in objects]}
    stable={"contract_sha256":_sha(contract_path),"source_sha256":[_sha(path) for path in SOURCES],"header_sha256":[_sha(path) for path in HEADERS],"android":android,"apple":apple,"decision":contract["decision_if_pass"],"claim_ceiling":contract["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4ek_native_cloud_chain_portable_build.v1","automatic_pass":True,"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--ndk",type=Path,required=True); parser.add_argument("--toolchain",type=Path,required=True); parser.add_argument("--build-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    report=build(ROOT/"configs/u6_p4ek_native_cloud_chain_portable_build_v1.json",args.ndk,args.toolchain,args.build_dir); args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__": main()
