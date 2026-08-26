"""Dual-compiler conformance for the private P220 interior-logit HDR ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file

SOURCE_RELATIVE = "native/color_match/nf_interior_logit_hdr_v1.c"
HEADER_RELATIVE = "native/color_match/nf_interior_logit_hdr_v1.h"
BASENAME = "nf_interior_logit_hdr_v1"
OUTPUT_MAXIMUM = np.float32(10000.0)
INTERIOR_EPSILON = np.float32(1.0 / 65536.0)
KNOT_COUNT = 33


class P220Error(RuntimeError):
    pass


class Diagnostics(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("identity", ctypes.c_uint32),
        ("triplet_count", ctypes.c_uint64),
        ("preserved_boundary_values", ctypes.c_uint64),
        ("strict_interior_values", ctypes.c_uint64),
    ]


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_fixture(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["schema"] != "neuro_film.p220_r1cg_frozen_payloads.v1":
        raise P220Error("unexpected P220 fixture schema")
    payloads = value["payloads"]
    if len(payloads) != 4:
        raise P220Error("P220 requires exactly four R1CG payloads")
    for row in payloads:
        payload = row["payload"]
        if (
            payload["format"] != "zhuise.interior-logit-quantile-transport"
            or payload["version"] != 1
            or payload["identity"] is not False
        ):
            raise P220Error("R1CG payload format drift")
        for name in ("source_knots", "reference_knots"):
            knots = np.asarray(payload[name], dtype=np.float32)
            if knots.shape != (KNOT_COUNT, 3) or np.any(knots[1:] <= knots[:-1]):
                raise P220Error(f"invalid {name}")
    return payloads


def build_probes(payloads: list[dict[str, Any]]) -> np.ndarray:
    values: list[np.float32] = [
        np.float32(0.0),
        np.nextafter(np.float32(0.0), np.float32(np.inf)),
        np.float32(10000.0 / 65536.0),
        np.nextafter(np.float32(10000.0), np.float32(0.0)),
        np.float32(10000.0),
        np.float32(5000.0),
    ]
    for row in sorted(payloads, key=lambda item: item["effect_id"]):
        knots = np.asarray(row["payload"]["source_knots"], dtype=np.float32)
        positive = knots >= 0.0
        sigmoid = np.empty_like(knots)
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-knots[positive]))
        exponent = np.exp(knots[~positive])
        sigmoid[~positive] = exponent / (1.0 + exponent)
        values.extend((OUTPUT_MAXIMUM * sigmoid).reshape(-1).tolist())
    remaining = 257 * 3 - len(values)
    if remaining < 0:
        raise P220Error("frozen probe capacity is insufficient")
    geometric = np.geomspace(1.0e-5, 9999.0, remaining, dtype=np.float32)
    values.extend(geometric.tolist())
    probes = np.asarray(values, dtype=np.float32).reshape(257, 3)
    if probes.shape != (257, 3) or not np.all(np.isfinite(probes)):
        raise P220Error("invalid P220 probes")
    return probes


def _interpolate(
    values: np.ndarray, source_knots: np.ndarray, reference_knots: np.ndarray
) -> np.ndarray:
    flat = values.reshape(-1)
    indices = np.searchsorted(source_knots, flat, side="right") - 1
    interior = np.clip(indices, 0, KNOT_COUNT - 2)
    x0 = source_knots[interior]
    x1 = source_knots[interior + 1]
    y0 = reference_knots[interior]
    y1 = reference_knots[interior + 1]
    bounded = np.minimum(np.maximum(flat, x0), x1)
    result = y0 + ((bounded - x0) / (x1 - x0)) * (y1 - y0)
    below = flat < source_knots[0]
    above = flat > source_knots[-1]
    if np.any(below):
        slope = np.clip(
            (reference_knots[1] - reference_knots[0])
            / (source_knots[1] - source_knots[0]),
            0.0,
            1.0,
        )
        result[below] = reference_knots[0] + slope * (flat[below] - source_knots[0])
    if np.any(above):
        slope = np.clip(
            (reference_knots[-1] - reference_knots[-2])
            / (source_knots[-1] - source_knots[-2]),
            0.0,
            1.0,
        )
        result[above] = reference_knots[-1] + slope * (flat[above] - source_knots[-1])
    return result.reshape(values.shape).astype(np.float32, copy=False)


def python_oracle(payload: dict[str, Any], probes: np.ndarray) -> np.ndarray:
    source = np.asarray(probes, dtype=np.float32)
    if payload["identity"]:
        return source.copy()
    normalized = source / OUTPUT_MAXIMUM
    bounded = np.clip(normalized, INTERIOR_EPSILON, np.float32(1.0) - INTERIOR_EPSILON)
    logits = np.log(bounded) - np.log1p(-bounded)
    source_knots = np.asarray(payload["source_knots"], dtype=np.float32)
    reference_knots = np.asarray(payload["reference_knots"], dtype=np.float32)
    mapped = np.empty_like(logits)
    for channel in range(3):
        mapped[:, channel] = _interpolate(
            logits[:, channel], source_knots[:, channel], reference_knots[:, channel]
        )
    positive = mapped >= 0.0
    sigmoid = np.empty_like(mapped)
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-mapped[positive]))
    exponent = np.exp(mapped[~positive])
    sigmoid[~positive] = exponent / (1.0 + exponent)
    output = OUTPUT_MAXIMUM * sigmoid
    output[source <= 0.0] = np.float32(0.0)
    output[source >= OUTPUT_MAXIMUM] = OUTPUT_MAXIMUM
    return output.astype(np.float32, copy=False)


def build_msvc(root: Path, output_dir: Path, basename: str) -> dict[str, Any]:
    return build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative=SOURCE_RELATIVE,
        header_relative=HEADER_RELATIVE,
        basename=basename,
    )


def build_llvm(
    root: Path, output_dir: Path, clang: Path, basename: str
) -> dict[str, Any]:
    source = root / SOURCE_RELATIVE
    header = root / HEADER_RELATIVE
    dll = output_dir / f"{basename}.dll"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(clang),
            "--target=x86_64-w64-windows-gnu",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-model=strict",
            "-shared",
            str(source),
            "-o",
            str(dll),
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not dll.is_file():
        raise P220Error((completed.stdout + completed.stderr).decode(errors="replace"))
    return {
        "toolchain": "llvm-mingw-x64-c11-strict",
        "clang_sha256": sha256_file(clang),
        "source_sha256": sha256_file(source),
        "header_sha256": sha256_file(header),
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll.resolve()),
    }


def _load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_float)
    library.nf_interior_logit_hdr_abi_version_v1.argtypes = []
    library.nf_interior_logit_hdr_abi_version_v1.restype = ctypes.c_uint32
    library.nf_interior_logit_hdr_apply_v1.argtypes = [
        pointer,
        pointer,
        ctypes.c_uint32,
        pointer,
        ctypes.c_size_t,
        pointer,
        ctypes.c_size_t,
        ctypes.POINTER(Diagnostics),
    ]
    library.nf_interior_logit_hdr_apply_v1.restype = ctypes.c_int
    if library.nf_interior_logit_hdr_abi_version_v1() != 1:
        raise P220Error("P220 ABI version drift")
    return library


def _invoke(
    library: ctypes.CDLL,
    payload: dict[str, Any],
    probes: np.ndarray,
    *,
    identity: int = 0,
    inplace: bool = False,
) -> tuple[int, np.ndarray, Diagnostics]:
    source_knots = np.ascontiguousarray(payload["source_knots"], dtype=np.float32)
    reference_knots = np.ascontiguousarray(payload["reference_knots"], dtype=np.float32)
    input_values = np.ascontiguousarray(probes, dtype=np.float32)
    output = input_values if inplace else np.full_like(input_values, np.float32(-17.0))
    diagnostics = Diagnostics(99, 99, 99, 99, 99)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = library.nf_interior_logit_hdr_apply_v1(
        source_knots.ctypes.data_as(pointer),
        reference_knots.ctypes.data_as(pointer),
        identity,
        input_values.ctypes.data_as(pointer),
        len(input_values),
        output.ctypes.data_as(pointer),
        output.size,
        ctypes.byref(diagnostics),
    )
    return status, output, diagnostics


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=">f4").tobytes()).hexdigest()


def run_loaded(
    dll_path: Path, payloads: list[dict[str, Any]], probes: np.ndarray
) -> dict[str, Any]:
    library = _load_library(dll_path)
    rows: list[dict[str, Any]] = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for row in payloads:
        payload = row["payload"]
        expected = python_oracle(payload, probes)
        status, output, diagnostics = _invoke(library, payload, probes)
        repeat_status, repeat, _ = _invoke(library, payload, probes)
        inplace_status, inplace, _ = _invoke(library, payload, probes, inplace=True)
        identity_status, identity, identity_diagnostics = _invoke(
            library, payload, probes, identity=1
        )
        absolute = np.abs(output.astype(np.float64) - expected.astype(np.float64))
        relative = absolute / np.maximum(np.abs(expected.astype(np.float64)), 1.0)
        maximum_absolute_error = max(maximum_absolute_error, float(np.max(absolute)))
        maximum_relative_error = max(maximum_relative_error, float(np.max(relative)))
        boundary = (probes == 0.0) | (probes == OUTPUT_MAXIMUM)
        strict = ~boundary
        rows.append(
            {
                "effect_id": row["effect_id"],
                "status": status,
                "repeat_status": repeat_status,
                "inplace_status": inplace_status,
                "identity_status": identity_status,
                "output_sha256": _sha256_array(output),
                "repeat_exact": output.tobytes() == repeat.tobytes(),
                "inplace_exact": output.tobytes() == inplace.tobytes(),
                "identity_exact": identity.tobytes() == probes.tobytes(),
                "boundaries_exact": output[boundary].tobytes()
                == probes[boundary].tobytes(),
                "strict_interior_preserved": bool(
                    np.all(output[strict] > 0.0)
                    and np.all(output[strict] < OUTPUT_MAXIMUM)
                ),
                "diagnostics": {
                    "abi_version": diagnostics.abi_version,
                    "identity": diagnostics.identity,
                    "triplet_count": diagnostics.triplet_count,
                    "preserved_boundary_values": diagnostics.preserved_boundary_values,
                    "strict_interior_values": diagnostics.strict_interior_values,
                    "identity_flag": identity_diagnostics.identity,
                },
                "maximum_absolute_error_cdm2": float(np.max(absolute)),
                "maximum_relative_error": float(np.max(relative)),
            }
        )
    return {
        "rows": rows,
        "maximum_absolute_error_cdm2": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


def failure_atomicity(
    dll_path: Path, payload: dict[str, Any], probes: np.ndarray
) -> dict[str, bool]:
    library = _load_library(dll_path)
    source = np.ascontiguousarray(payload["source_knots"], dtype=np.float32)
    reference = np.ascontiguousarray(payload["reference_knots"], dtype=np.float32)
    pointer = ctypes.POINTER(ctypes.c_float)

    def invoke(
        test_source: np.ndarray, test_input: np.ndarray, identity: int, count: int
    ) -> bool:
        output = np.full_like(probes, np.float32(-1234.5))
        diagnostics = Diagnostics(91, 92, 93, 94, 95)
        before_output = output.tobytes()
        before_diagnostics = bytes(diagnostics)
        status = library.nf_interior_logit_hdr_apply_v1(
            test_source.ctypes.data_as(pointer),
            reference.ctypes.data_as(pointer),
            identity,
            test_input.ctypes.data_as(pointer),
            len(probes),
            output.ctypes.data_as(pointer),
            count,
            ctypes.byref(diagnostics),
        )
        return (
            status != 0
            and output.tobytes() == before_output
            and bytes(diagnostics) == before_diagnostics
        )

    nonincreasing = source.copy()
    nonincreasing[1, 0] = nonincreasing[0, 0]
    nonfinite_knots = source.copy()
    nonfinite_knots[1, 1] = np.nan
    nonfinite_input = probes.copy()
    nonfinite_input[-1, -1] = np.inf
    out_of_range = probes.copy()
    out_of_range[-1, -1] = np.float32(10001.0)
    storage = np.arange(probes.size + 1, dtype=np.float32)
    diagnostics = Diagnostics(1, 2, 3, 4, 5)
    before_storage = storage.tobytes()
    before_diagnostics = bytes(diagnostics)
    overlap_status = library.nf_interior_logit_hdr_apply_v1(
        source.ctypes.data_as(pointer),
        reference.ctypes.data_as(pointer),
        0,
        storage.ctypes.data_as(pointer),
        len(probes),
        ctypes.cast(storage.ctypes.data + 4, pointer),
        probes.size,
        ctypes.byref(diagnostics),
    )
    return {
        "nonincreasing_knots": invoke(nonincreasing, probes, 0, probes.size),
        "nonfinite_knots": invoke(nonfinite_knots, probes, 0, probes.size),
        "nonfinite_input": invoke(source, nonfinite_input, 0, probes.size),
        "out_of_range_input": invoke(source, out_of_range, 0, probes.size),
        "invalid_identity": invoke(source, probes, 2, probes.size),
        "undersized_output": invoke(source, probes, 0, probes.size - 1),
        "partial_overlap": (
            overlap_status != 0
            and storage.tobytes() == before_storage
            and bytes(diagnostics) == before_diagnostics
        ),
    }


__all__ = [
    "P220Error",
    "build_llvm",
    "build_msvc",
    "build_probes",
    "canonical_bytes",
    "failure_atomicity",
    "load_fixture",
    "python_oracle",
    "run_loaded",
]
