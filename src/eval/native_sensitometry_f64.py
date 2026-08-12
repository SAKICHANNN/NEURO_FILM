"""P4EX double developed-density sensitometry conformance."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.native_abi_layouts import NativePrintProfileV1


def _build_llvm(root: Path, clang: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    dll = output / "nf_physical_domains_f64_llvm.dll"
    source = root / "native/film_physics/nf_physical_domains_f32_v1.c"
    result = subprocess.run(
        [str(clang), "--target=x86_64-w64-windows-gnu", "-std=c11", "-O2",
         "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", "-shared",
         str(source), "-o", str(dll), "-Wl,--no-insert-timestamp"],
        capture_output=True, check=False, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return dll


def _configure(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    common = [ctypes.POINTER(NativePrintProfileV1), ctypes.POINTER(ctypes.c_float),
              ctypes.c_size_t]
    library.nf_physical_sensitometry_f32_apply_v1.argtypes = [
        *common, ctypes.POINTER(ctypes.c_float)]
    library.nf_physical_sensitometry_f32_apply_v1.restype = ctypes.c_int
    library.nf_physical_sensitometry_f64_apply_v2.argtypes = [
        *common, ctypes.POINTER(ctypes.c_double)]
    library.nf_physical_sensitometry_f64_apply_v2.restype = ctypes.c_int
    return library


def _profile(fixture: dict) -> NativePrintProfileV1:
    profile = NativePrintProfileV1()
    profile.struct_size = ctypes.sizeof(NativePrintProfileV1)
    profile.abi_version = 1
    profile.source_component_sha256 = hashlib.sha256(
        fixture["identity"].encode("ascii")
    ).hexdigest().encode("ascii")
    profile.reference_linear = fixture["reference_linear"]
    profile.black_offset = fixture["black_offset"]
    for channel in range(3):
        profile.knot_count[channel] = 3
        for index in range(3):
            profile.x_knots[channel][index] = fixture["x_knots"][index]
            profile.y_knots[channel][index] = fixture["y_knots"][index]
            profile.derivatives[channel][index] = fixture["derivatives"][index]
        profile.dye_absorption_matrix[channel][channel] = 1.0
        profile.print_matrix[channel][channel] = 1.0
        profile.paper_midpoints[channel] = -1.0
        profile.paper_slopes[channel] = 1.0
        profile.paper_maximum_densities[channel] = 2.0
        profile.black_reference_density[channel] = 0.05
        profile.white_reference_density[channel] = 1.25
    profile.exposure_floor = 1.0 / 65536.0
    profile.matrix_minimum_determinant = 0.01
    return profile


def _run(library: ctypes.CDLL, profile: NativePrintProfileV1,
         source: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    f32 = np.full(source.shape, -77.0, np.float32)
    f64 = np.full(source.shape, -77.0, np.float64)
    fp = ctypes.POINTER(ctypes.c_float)
    if library.nf_physical_sensitometry_f32_apply_v1(
            ctypes.byref(profile), source.ctypes.data_as(fp), source.size // 3,
            f32.ctypes.data_as(fp)) != 0:
        raise RuntimeError("f32 sensitometry failed")
    if library.nf_physical_sensitometry_f64_apply_v2(
            ctypes.byref(profile), source.ctypes.data_as(fp), source.size // 3,
            f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))) != 0:
        raise RuntimeError("f64 sensitometry failed")
    invalid = source.copy(); invalid.reshape(-1)[-1] = np.nan
    sentinel = np.full(source.shape, -1234.5, np.float64); before = sentinel.tobytes()
    status = library.nf_physical_sensitometry_f64_apply_v2(
        ctypes.byref(profile), invalid.ctypes.data_as(fp), invalid.size // 3,
        sentinel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return f32, f64, status != 0 and sentinel.tobytes() == before


def evaluate(root: Path, contract_path: Path, output: Path, clang: Path) -> dict:
    contract = json.loads(contract_path.read_text()); parent = root / contract["parent"]["path"]
    if (sha256_file(parent) != contract["parent"]["sha256"] or
            json.loads(parent.read_text())["decision"] != contract["parent"]["required_decision"]):
        raise RuntimeError("P4EX parent drift")
    profile = _profile(contract["profile_fixture"])
    f = contract["fixture"]
    source = np.random.default_rng(f["seed"]).random((f["height"], f["width"], 3), dtype=np.float32)
    builds = {
        "msvc": Path(build_msvc_c11_dll(
            root=root, output_dir=output / "msvc",
            source_relative="native/film_physics/nf_physical_domains_f32_v1.c",
            header_relative="native/film_physics/nf_physical_domains_f32_v1.h",
            basename="nf_physical_domains_f64_v2")["dll_path"]),
        "llvm": _build_llvm(root, clang, output / "llvm"),
    }
    rows = {}
    try:
        for name, path in builds.items():
            library = _configure(path)
            f32, f64, atomic = _run(library, profile, source)
            repeat32, repeat64, _ = _run(library, profile, source)
            rows[name] = {
                "f32_sha256": hashlib.sha256(f32.tobytes()).hexdigest(),
                "f64_sha256": hashlib.sha256(f64.tobytes()).hexdigest(),
                "rounding_identity": np.array_equal(f64.astype(np.float32), f32),
                "repeat_identity": np.array_equal(f32, repeat32) and np.array_equal(f64, repeat64),
                "failure_atomic": atomic,
                "f64": f64,
            }
            _ctypes.FreeLibrary(library._handle)
    finally:
        pass
    x_knots = np.asarray(contract["profile_fixture"]["x_knots"])
    # Cross-compiler identity is the exact executable oracle for the frozen
    # C expression sequence; the legacy float32 API supplies its rounding
    # identity. The tolerance gate therefore measures cross-compiler drift.
    maximum_error = float(np.max(np.abs(rows["msvc"]["f64"] - rows["llvm"]["f64"])))
    gates = {
        "rounding": all(row["rounding_identity"] for row in rows.values()),
        "compiler": rows["msvc"]["f64_sha256"] == rows["llvm"]["f64_sha256"],
        "repeat": all(row["repeat_identity"] for row in rows.values()),
        "atomic": all(row["failure_atomic"] for row in rows.values()),
        "oracle": maximum_error <= contract["gates"]["maximum_cross_compiler_absolute_error"],
    }
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "source_sha256": sha256_file(root / "native/film_physics/nf_physical_domains_f32_v1.c"),
        "header_sha256": sha256_file(root / "native/film_physics/nf_physical_domains_f32_v1.h"),
        "f32_sha256": rows["msvc"]["f32_sha256"],
        "f64_sha256": rows["msvc"]["f64_sha256"],
        "maximum_cross_compiler_absolute_error": maximum_error,
        "profile_first_channel_knot_span": [float(x_knots[0]), float(x_knots[-1])],
        "gates": gates,
        "decision": contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {"schema": "neuro_film.u6_p4ex_native_sensitometry_f64.v1",
            "automatic_pass": all(gates.values()), "stable": stable,
            "stable_evidence_id": hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


__all__ = ["evaluate"]
