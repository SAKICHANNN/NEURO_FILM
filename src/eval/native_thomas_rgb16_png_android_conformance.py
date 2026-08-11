"""Android NDK build conformance for the unchanged P8CK PNG program."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_rgb16_png_conformance import HEADER_PATHS, SOURCE_PATHS


class NativeThomasRgb16PngAndroidConformanceError(RuntimeError):
    """Raised when the pinned Android build contract is not satisfied."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _run(command: list[str], *, cwd: Path, timeout: int = 180) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    output = (completed.stdout + completed.stderr).decode(errors="replace")
    if completed.returncode != 0:
        raise NativeThomasRgb16PngAndroidConformanceError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return output


def _ndk_tools(ndk: Path) -> tuple[Path, Path]:
    bin_dir = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    clang = bin_dir / "clang.exe"
    readelf = bin_dir / "llvm-readelf.exe"
    if not clang.is_file() or not readelf.is_file():
        raise NativeThomasRgb16PngAndroidConformanceError(
            "Android NDK clang/readelf is unavailable"
        )
    return clang, readelf


def _validate_ndk(ndk: Path, toolchain: dict[str, Any]) -> dict[str, Any]:
    properties = ndk / "source.properties"
    notice = ndk / "NOTICE"
    if not properties.is_file() or not notice.is_file():
        raise NativeThomasRgb16PngAndroidConformanceError("NDK identity files missing")
    text = properties.read_text(encoding="utf-8")
    expected_revision = str(toolchain["revision"])
    if f"Pkg.Revision = {expected_revision}" not in text:
        raise NativeThomasRgb16PngAndroidConformanceError("NDK revision drift")
    actual_properties = sha256_file(properties)
    actual_notice = sha256_file(notice)
    if actual_properties != toolchain["source_properties_sha256"]:
        raise NativeThomasRgb16PngAndroidConformanceError(
            "NDK source.properties hash drift"
        )
    if actual_notice != toolchain["notice_sha256"]:
        raise NativeThomasRgb16PngAndroidConformanceError("NDK NOTICE hash drift")
    return {
        "release": toolchain["release"],
        "revision": expected_revision,
        "source_properties_sha256": actual_properties,
        "notice_sha256": actual_notice,
    }


def _build_once(
    *,
    root: Path,
    ndk: Path,
    output_dir: Path,
    abi: str,
    target: str,
    expected_machine: str,
    required_symbols: list[str],
) -> dict[str, Any]:
    clang, readelf = _ndk_tools(ndk)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: list[Path] = []
    common = [
        str(clang),
        f"--target={target}",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffp-model=strict",
        "-fPIC",
        "-I",
        str(root / "native"),
        "-I",
        str(root / "native/film_physics"),
    ]
    for index, relative in enumerate(SOURCE_PATHS):
        source = root / relative
        obj = output_dir / f"{index:02d}_{source.stem}.o"
        _run([*common, "-c", str(source), "-o", str(obj)], cwd=root)
        objects.append(obj)
    library = output_dir / "libnf_thomas_rgb16_png_f32_v1.so"
    _run(
        [
            str(clang),
            f"--target={target}",
            "-shared",
            *(str(path) for path in objects),
            "-Wl,--build-id=none",
            "-Wl,--no-undefined",
            "-lm",
            "-o",
            str(library),
        ],
        cwd=root,
    )
    header = _run([str(readelf), "-h", str(library)], cwd=root)
    symbols = _run([str(readelf), "--dyn-syms", "--wide", str(library)], cwd=root)
    machine_lines = [line.strip() for line in header.splitlines() if "Machine:" in line]
    if len(machine_lines) != 1 or expected_machine not in machine_lines[0]:
        raise NativeThomasRgb16PngAndroidConformanceError(
            f"unexpected ELF machine for {abi}: {machine_lines}"
        )
    missing = [name for name in required_symbols if name not in symbols]
    if missing:
        raise NativeThomasRgb16PngAndroidConformanceError(
            f"missing required dynamic symbols for {abi}: {missing}"
        )
    return {
        "abi": abi,
        "target": target,
        "elf_machine": expected_machine,
        "bytes": library.stat().st_size,
        "sha256": sha256_file(library),
        "required_dynamic_symbols": required_symbols,
        "object_sha256": {path.name: sha256_file(path) for path in objects},
        "library_path": str(library),
    }


def evaluate(contract_path: Path, ndk: Path, output_dir: Path) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if contract.get("schema") != (
        "neuro_film.u6_p8cl_android_thomas_rgb16_png_build_contract.v1"
    ):
        raise NativeThomasRgb16PngAndroidConformanceError("P8CL contract drift")
    parent = contract["parent"]
    parent_path = contract_path.parents[1] / parent["path"]
    if sha256_file(parent_path) != parent["sha256"]:
        raise NativeThomasRgb16PngAndroidConformanceError("P8CK evidence hash drift")
    parent_payload = json.loads(parent_path.read_bytes())
    if parent_payload.get("decision") != parent["required_decision"]:
        raise NativeThomasRgb16PngAndroidConformanceError("P8CK decision drift")

    root = contract_path.parents[1]
    ndk_identity = _validate_ndk(ndk, contract["toolchain"])
    required_symbols = list(contract["build"]["required_dynamic_symbols"])
    rows: dict[str, Any] = {}
    all_exact = True
    for abi, entry in contract["toolchain"]["abis"].items():
        compiler_stem = str(entry["compiler"]).removesuffix(".cmd")
        target = compiler_stem.removesuffix("-clang")
        first = _build_once(
            root=root,
            ndk=ndk,
            output_dir=output_dir / abi / "run1",
            abi=abi,
            target=target,
            expected_machine=str(entry["elf_machine"]),
            required_symbols=required_symbols,
        )
        second = _build_once(
            root=root,
            ndk=ndk,
            output_dir=output_dir / abi / "run2",
            abi=abi,
            target=target,
            expected_machine=str(entry["elf_machine"]),
            required_symbols=required_symbols,
        )
        exact = (
            first["sha256"] == second["sha256"]
            and first["object_sha256"] == second["object_sha256"]
        )
        all_exact = all_exact and exact
        rows[abi] = {
            "first": first,
            "second": second,
            "byte_exact_rebuild": exact,
        }
    stable = {
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "ndk": ndk_identity,
        "source_sha256": {path: sha256_file(root / path) for path in SOURCE_PATHS},
        "header_sha256": {path: sha256_file(root / path) for path in HEADER_PATHS},
        "abis": {
            abi: {
                "elf_machine": row["first"]["elf_machine"],
                "bytes": row["first"]["bytes"],
                "sha256": row["first"]["sha256"],
                "required_dynamic_symbols": row["first"][
                    "required_dynamic_symbols"
                ],
                "byte_exact_rebuild": row["byte_exact_rebuild"],
            }
            for abi, row in rows.items()
        },
        "decision": (
            contract["decision_if_pass"] if all_exact else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8cl_android_thomas_rgb16_png_build_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": all_exact,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        **stable,
        "build_rows": rows,
    }


__all__ = [
    "NativeThomasRgb16PngAndroidConformanceError",
    "evaluate",
]
