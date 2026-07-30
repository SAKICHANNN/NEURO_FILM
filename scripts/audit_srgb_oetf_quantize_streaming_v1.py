"""Audit chunked execution of the exact consumer sRGB output quantizer."""

from __future__ import annotations

import argparse
import _ctypes
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_srgb_oetf_quantize_c_v1 import (
    quantize_reference,
    thresholds,
)
from scripts.build_srgb_oetf_quantize_native_v1 import (
    build_llvm_mingw_dll,
    build_msvc_dll,
)


PROTOCOL = "neuro-film.srgb-oetf-quantize-streaming-audit.v1"
DEFAULT_LLVM = (
    Path(__file__).resolve().parents[2]
    / "\u8ffd\u8272"
    / "outputs"
    / "tmp"
    / "tools"
    / "llvm-mingw-20260616-ucrt-x86_64"
)
_FLOAT_ONE_BITS = int(np.float32(1.0).view(np.uint32))


class StreamingQuantizeAuditError(RuntimeError):
    """Raised when the quantizer fails its frozen streaming contract."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _load_apply(path: Path):
    library = ctypes.CDLL(str(path))
    apply = library.nf_srgb_oetf_quantize_apply_v1
    apply.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    apply.restype = ctypes.c_int
    return library, apply


def _close_library(library) -> None:
    handle = library._handle
    if handle:
        _ctypes.FreeLibrary(handle)
        library._handle = 0


def _linear_samples(offset: int, count: int) -> np.ndarray:
    """Generate deterministic valid float32 bit patterns without global state."""
    indices = np.arange(count, dtype=np.uint64)
    indices += np.uint64(offset)
    indices *= np.uint64(1_664_525)
    indices += np.uint64(1_013_904_223)
    bits = (indices % np.uint64(_FLOAT_ONE_BITS + 1)).astype(np.uint32)
    return bits.view(np.float32)


def _run_once(
    apply,
    *,
    sample_count: int,
    chunk_samples: int,
    bit_depth: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    input_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    maximum_live_array_payload = 0
    apply_seconds = 0.0
    oracle_seconds = 0.0
    chunks = 0
    output_dtype = np.uint8 if bit_depth == 8 else np.uint16
    wall_start = time.perf_counter()
    for offset in range(0, sample_count, chunk_samples):
        count = min(chunk_samples, sample_count - offset)
        source = _linear_samples(offset, count)
        oracle_start = time.perf_counter()
        expected = np.ascontiguousarray(
            quantize_reference(source, bit_depth),
            dtype=output_dtype,
        )
        oracle_seconds += time.perf_counter() - oracle_start
        output = np.empty(count, dtype=output_dtype)
        maximum_live_array_payload = max(
            maximum_live_array_payload,
            source.nbytes + expected.nbytes + output.nbytes,
        )
        call_start = time.perf_counter()
        status = apply(
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            count,
            bit_depth,
            output.ctypes.data,
            count,
        )
        apply_seconds += time.perf_counter() - call_start
        if status != 1:
            raise StreamingQuantizeAuditError(
                f"native quantizer rejected depth-{bit_depth} chunk"
            )
        if output.tobytes() != expected.tobytes():
            raise StreamingQuantizeAuditError(
                f"native output differs at depth {bit_depth}, offset {offset}"
            )
        input_hash.update(source.tobytes())
        output_hash.update(output.tobytes())
        chunks += 1
    evidence = {
        "bit_depth": bit_depth,
        "chunks": chunks,
        "input_sha256": input_hash.hexdigest(),
        "max_live_array_payload_bytes": maximum_live_array_payload,
        "output_sha256": output_hash.hexdigest(),
        "sample_count": sample_count,
    }
    timing = {
        "apply_seconds": apply_seconds,
        "oracle_seconds": oracle_seconds,
        "wall_seconds": time.perf_counter() - wall_start,
    }
    return evidence, timing


def audit_streaming_quantize(
    *,
    pixels: int,
    chunk_samples: int,
    llvm_toolchain: Path = DEFAULT_LLVM,
) -> dict[str, Any]:
    if pixels <= 0:
        raise StreamingQuantizeAuditError("pixels must be positive")
    if chunk_samples <= 0:
        raise StreamingQuantizeAuditError("chunk_samples must be positive")
    if pixels > (2**63 - 1) // 3:
        raise StreamingQuantizeAuditError(
            "pixel count overflows the audit contract"
        )
    sample_count = pixels * 3
    _, _, threshold_identity = thresholds()
    with tempfile.TemporaryDirectory(
        prefix="nf-oetf-quantize-streaming-"
    ) as directory:
        root = Path(directory)
        dlls = {
            "llvm-mingw": root / "llvm" / "oetf-quantize.dll",
            "msvc": root / "msvc" / "oetf-quantize.dll",
        }
        builds = {
            "llvm-mingw": build_llvm_mingw_dll(
                llvm_toolchain.resolve(),
                dlls["llvm-mingw"],
            ),
            "msvc": build_msvc_dll(dlls["msvc"]),
        }
        run_evidence: dict[str, Any] = {}
        timing: dict[str, Any] = {}
        for compiler, dll_path in dlls.items():
            library, apply = _load_apply(dll_path)
            try:
                run_evidence[compiler] = {}
                timing[compiler] = {}
                for bit_depth in (8, 16):
                    first, first_timing = _run_once(
                        apply,
                        sample_count=sample_count,
                        chunk_samples=chunk_samples,
                        bit_depth=bit_depth,
                    )
                    second, second_timing = _run_once(
                        apply,
                        sample_count=sample_count,
                        chunk_samples=chunk_samples,
                        bit_depth=bit_depth,
                    )
                    if first != second:
                        raise StreamingQuantizeAuditError(
                            f"{compiler} depth-{bit_depth} replay differs"
                        )
                    run_evidence[compiler][str(bit_depth)] = first
                    timing[compiler][str(bit_depth)] = [
                        first_timing,
                        second_timing,
                    ]
            finally:
                del apply
                _close_library(library)
        for bit_depth in ("8", "16"):
            if (
                run_evidence["llvm-mingw"][bit_depth]
                != run_evidence["msvc"][bit_depth]
            ):
                raise StreamingQuantizeAuditError(
                    f"cross-compiler depth-{bit_depth} evidence differs"
                )
        evidence = {
            "builds": builds,
            "chunk_samples": chunk_samples,
            "cross_compiler_exact": True,
            "pixels": pixels,
            "runs": run_evidence,
            "sample_count": sample_count,
            "threshold_identity": threshold_identity,
            "two_replays_exact": True,
        }
        stable_evidence_id = "sha256:" + hashlib.sha256(
            _canonical_bytes(evidence)
        ).hexdigest()
        return {
            "claim_ceiling": (
                "Windows-x86_64-host-streaming-runtime-only;"
                "no-media-or-product-admission"
            ),
            "evidence": evidence,
            "protocol": PROTOCOL,
            "stable_evidence_id": stable_evidence_id,
            "timing_observations": timing,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixels", type=int, required=True)
    parser.add_argument("--chunk-samples", type=int, required=True)
    parser.add_argument("--llvm-toolchain", type=Path, default=DEFAULT_LLVM)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_streaming_quantize(
        pixels=args.pixels,
        chunk_samples=args.chunk_samples,
        llvm_toolchain=args.llvm_toolchain,
    )
    payload = _canonical_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
