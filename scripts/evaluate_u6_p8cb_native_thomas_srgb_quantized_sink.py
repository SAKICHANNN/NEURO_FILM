#!/usr/bin/env python3
"""Run U6.P8CB one-final-quantization streaming evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_srgb_oetf_quantize_c_v1 import quantize_reference
from scripts.build_srgb_oetf_quantize_native_v1 import (
    build_llvm_mingw_dll,
    build_msvc_dll,
)
from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _exposure_fixture,
    _parent_payloads,
    _profiles,
)
from scripts.evaluate_u6_p8ca_native_thomas_gauged_sink import _gauge_payload
from src.eval.native_granularity_amplitude_conformance import (
    evaluate_conformance as evaluate_amplitude_conformance,
)
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rows_conformance import build_and_load
from src.eval.physical_native_gauge_conformance import (
    _load_gauge,
    build_msvc_native_gauge_dll,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_gauge_profile import native_gauge_profile_struct
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)
from src.film_physics.native_thomas_rows import (
    _pointer,
    load_native_thomas_rows_library,
    stream_native_exposure_thomas_rgb_gauged,
    stream_native_exposure_thomas_rgb_quantized,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8CB JSON must be an object")
    return payload


def _load_quantizer(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    apply = library.nf_srgb_oetf_quantize_apply_v1
    apply.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    apply.restype = ctypes.c_int
    identity = library.nf_srgb_oetf_quantize_thresholds_sha256_v1
    identity.argtypes = []
    identity.restype = ctypes.c_char_p
    return library


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8cb_native_thomas_srgb_quantized_sink_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("p8ca_arithmetic_and_gauge_unchanged") is not True
        or candidate.get("one_final_quantization_only") is not True
        or candidate.get("full_output_or_channel_plane_allowed") is not False
        or candidate.get("post_quantization_float_processing_allowed") is not False
        or candidate.get("image_encoding_allowed") is not False
        or candidate.get("model_or_profile_change_allowed") is not False
    ):
        raise RuntimeError("P8CB contract drift")
    parent = contract["parents"]["p8ca_evidence"]
    payload = _json(ROOT / parent["path"])
    if (
        sha256_file(ROOT / parent["path"]) != parent["sha256"]
        or payload.get("decision") != parent["required_decision"]
    ):
        raise RuntimeError("P8CB P8CA parent drift")
    quantizer = contract["quantizer"]
    if (
        sha256_file(ROOT / quantizer["source"]) != quantizer["source_sha256"]
        or sha256_file(ROOT / quantizer["header"])
        != quantizer["header_sha256"]
        or quantizer["bit_depths"] != [8, 16]
    ):
        raise RuntimeError("P8CB quantizer source drift")


class _HashSink:
    def __init__(self, dtype: np.dtype[Any]) -> None:
        self.dtype = np.dtype(dtype)
        self.digest = hashlib.sha256()
        self.minimum = np.iinfo(self.dtype).max
        self.maximum = 0
        self.rows: list[int] = []

    def __call__(self, row_start: int, values: np.ndarray) -> None:
        if (
            values.dtype != self.dtype
            or values.flags.writeable
            or not values.flags.c_contiguous
        ):
            raise RuntimeError("P8CB sink tile contract drift")
        self.digest.update(values.tobytes())
        self.minimum = min(self.minimum, int(np.min(values)))
        self.maximum = max(self.maximum, int(np.max(values)))
        self.rows.append(row_start)


class _ArraySink:
    def __init__(self) -> None:
        self.tiles: list[np.ndarray] = []

    def __call__(self, row_start: int, values: np.ndarray) -> None:
        del row_start
        self.tiles.append(values.copy())


def _worker(
    contract_path: Path,
    amplitude_dll: Path,
    rows_dll: Path,
    gauge_dll: Path,
    quantizer_dll: Path,
    result_path: Path,
) -> None:
    contract = _json(contract_path)
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(
        p4bw, prior_payload
    )
    shape = tuple(int(value) for value in contract["performance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    input_sha = hashlib.sha256(exposure.tobytes()).hexdigest()
    depth = int(contract["performance"]["primary_bit_depth"])
    sink = _HashSink(np.uint16 if depth == 16 else np.uint8)
    started = time.perf_counter()
    means, workspace_bytes, calls = stream_native_exposure_thomas_rgb_quantized(
        load_native_granularity_amplitude_library(amplitude_dll),
        load_native_thomas_rows_library(rows_dll),
        _load_gauge(gauge_dll),
        _load_quantizer(quantizer_dll),
        amplitude_profile,
        _profiles(_json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")),
        native_gauge_profile_struct(_gauge_payload()),
        exposure,
        sink,
        row_partition=int(contract["candidate"]["row_partition"]),
        bit_depth=depth,
    )
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("P8CB quantized sink mutated its input")
    result = {
        "schema": "neuro_film.u6_p8cb_native_thomas_quantized_worker.v1",
        "shape_chw": list(shape),
        "bit_depth": depth,
        "input_sha256": input_sha,
        "output_hwc_sha256": sink.digest.hexdigest(),
        "sample_range": [sink.minimum, sink.maximum],
        "raw_field_means": list(means),
        "sink_calls": calls,
        "sink_order_exact": sink.rows == sorted(sink.rows),
        "bounded_workspace_bytes": workspace_bytes,
        "wall_seconds": elapsed,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(canonical_bytes(result))


def _monitored(
    contract_path: Path,
    amplitude_dll: Path,
    rows_dll: Path,
    gauge_dll: Path,
    quantizer_dll: Path,
    result_path: Path,
) -> dict[str, Any]:
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--contract",
            str(contract_path),
            "--amplitude-dll",
            str(amplitude_dll),
            "--rows-dll",
            str(rows_dll),
            "--gauge-dll",
            str(gauge_dll),
            "--quantizer-dll",
            str(quantizer_dll),
            "--result",
            str(result_path),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    root_process = psutil.Process(process.pid)
    peak = 0
    observed: set[int] = set()
    started = time.perf_counter()
    while process.poll() is None:
        if time.perf_counter() - started > 35.0:
            for child in root_process.children(recursive=True):
                child.kill()
            root_process.kill()
            raise TimeoutError("P8CB worker exceeded timeout")
        try:
            total = 0
            for item in [root_process, *root_process.children(recursive=True)]:
                try:
                    observed.add(item.pid)
                    total += item.memory_info().rss
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
            peak = max(peak, total)
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.01)
    stdout, stderr = process.communicate(timeout=10)
    if process.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            "P8CB worker failed:\n"
            + stdout.decode(errors="replace")
            + stderr.decode(errors="replace")
        )
    return {
        "worker": _json(result_path),
        "peak_process_tree_rss_bytes": peak,
        "stderr_empty": not bool(stderr),
        "surviving_process_count": sum(1 for pid in observed if psutil.pid_exists(pid)),
    }


def evaluate(contract_path: Path, output_dir: Path, llvm: Path) -> dict[str, Any]:
    contract = _json(contract_path)
    _validate_contract(contract)
    amplitude = evaluate_amplitude_conformance(
        ROOT,
        _json(ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"),
        output_dir=output_dir / "amplitude",
        clang=llvm / "bin/clang.exe",
    )
    row_builds, row_libraries = build_and_load(
        ROOT, output_dir / "rows", llvm / "bin/clang.exe"
    )
    gauge_build = build_msvc_native_gauge_dll(
        root=ROOT, output_dir=output_dir / "gauge"
    )
    quantizer_builds = {
        "msvc": build_msvc_dll(output_dir / "quantizer/msvc/quantizer.dll"),
        "llvm_mingw": build_llvm_mingw_dll(
            llvm, output_dir / "quantizer/llvm/quantizer.dll"
        ),
    }
    quantizer_paths = {
        "msvc": output_dir / "quantizer/msvc/quantizer.dll",
        "llvm_mingw": output_dir / "quantizer/llvm/quantizer.dll",
    }
    quantizers = {
        name: _load_quantizer(path) for name, path in quantizer_paths.items()
    }
    for library in quantizers.values():
        identity = library.nf_srgb_oetf_quantize_thresholds_sha256_v1()
        if identity.decode("ascii") != contract["quantizer"]["threshold_identity"]:
            raise RuntimeError("P8CB threshold identity drift")
    gauge_library = _load_gauge(Path(gauge_build["dll_path"]))
    gauge_profile = native_gauge_profile_struct(_gauge_payload())
    amplitude_libraries = {
        name: load_native_granularity_amplitude_library(Path(row["dll_path"]))
        for name, row in amplitude["toolchains"].items()
    }
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(
        p4bw, prior_payload
    )
    shape = tuple(int(value) for value in contract["conformance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    profiles = _profiles(
        _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    display_sink = _ArraySink()
    expected_means, _, _ = stream_native_exposure_thomas_rgb_gauged(
        amplitude_libraries["msvc"], row_libraries["msvc"], gauge_library,
        amplitude_profile, profiles, gauge_profile, exposure, display_sink,
        row_partition=int(contract["candidate"]["row_partition"]),
    )
    display = np.concatenate(display_sink.tiles, axis=0)
    expected = {
        depth: np.ascontiguousarray(quantize_reference(display, depth))
        for depth in contract["quantizer"]["bit_depths"]
    }
    conformance: dict[str, Any] = {}
    exact = True
    for name, rows_library in row_libraries.items():
        by_depth: dict[str, Any] = {}
        for depth in contract["quantizer"]["bit_depths"]:
            sink = _HashSink(np.uint16 if depth == 16 else np.uint8)
            means, _, calls = stream_native_exposure_thomas_rgb_quantized(
                amplitude_libraries[name], rows_library, gauge_library,
                quantizers[name], amplitude_profile, profiles, gauge_profile,
                exposure, sink,
                row_partition=int(contract["candidate"]["row_partition"]),
                bit_depth=depth,
            )
            observed_hash = sink.digest.hexdigest()
            expected_hash = hashlib.sha256(expected[depth].tobytes()).hexdigest()
            by_depth[str(depth)] = {
                "output_hwc_sha256": observed_hash,
                "expected_hwc_sha256": expected_hash,
                "raw_field_means": list(means),
                "sink_calls": calls,
                "sink_order_exact": sink.rows == sorted(sink.rows),
            }
            exact = exact and observed_hash == expected_hash
            exact = exact and means == expected_means
        conformance[name] = by_depth

    calls = 0

    def fail_sink(row_start: int, values: np.ndarray) -> None:
        nonlocal calls
        del row_start, values
        calls += 1
        if calls == 2:
            raise RuntimeError("injected P8CB sink failure")

    try:
        stream_native_exposure_thomas_rgb_quantized(
            amplitude_libraries["msvc"], row_libraries["msvc"], gauge_library,
            quantizers["msvc"], amplitude_profile, profiles, gauge_profile,
            exposure, fail_sink,
            row_partition=int(contract["candidate"]["row_partition"]), bit_depth=16,
        )
        sink_failure = False
    except RuntimeError as error:
        sink_failure = str(error) == "injected P8CB sink failure" and calls == 2

    invalid_input = np.zeros(12, dtype=np.float32)
    invalid_input[0] = np.nan
    sentinel = np.full(12, 0xA5A5, dtype=np.uint16)
    before = sentinel.tobytes()
    status = quantizers["msvc"].nf_srgb_oetf_quantize_apply_v1(
        _pointer(invalid_input), invalid_input.size, 16,
        sentinel.ctypes.data, sentinel.size,
    )
    quantizer_atomic = status == 0 and sentinel.tobytes() == before

    runs = [
        _monitored(
            contract_path,
            Path(amplitude["toolchains"]["msvc"]["dll_path"]),
            Path(row_builds["msvc"]["dll_path"]),
            Path(gauge_build["dll_path"]),
            quantizer_paths["msvc"],
            output_dir / f"worker_{index}.json",
        )
        for index in (1, 2)
    ]
    walls = [float(row["worker"]["wall_seconds"]) for row in runs]
    peaks = [int(row["peak_process_tree_rss_bytes"]) for row in runs]
    hashes = [str(row["worker"]["output_hwc_sha256"]) for row in runs]
    limits = contract["performance"]
    gates = {
        "exact_full_array_quantizer_output": bool(exact),
        "sink_failure_propagation": sink_failure,
        "quantizer_failure_atomicity": quantizer_atomic,
        "fresh_stream_exact": len(set(hashes)) == 1,
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "maximum_rss": max(peaks) <= int(limits["maximum_process_tree_rss_bytes"]),
        "wall_repeat": max(walls) / min(walls)
        <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": max(peaks) / min(peaks)
        <= float(limits["maximum_repeat_rss_ratio"]),
        "worker_cleanup": all(
            row["stderr_empty"] and row["surviving_process_count"] == 0
            for row in runs
        ),
    }
    decision = (
        contract["decision_if_pass"]
        if all(gates.values())
        else contract["decision_if_fail"]
    )
    report = {
        "schema": "neuro_film.u6_p8cb_native_thomas_srgb_quantized_sink_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "row_builds": row_builds,
        "quantizer_builds": quantizer_builds,
        "threshold_identity": contract["quantizer"]["threshold_identity"],
        "conformance": conformance,
        "performance": {
            "runs": runs,
            "output_hwc_sha256": hashes[0],
            "maximum_wall_seconds": max(walls),
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "wall_repeat_ratio": max(walls) / min(walls),
            "rss_repeat_ratio": max(peaks) / min(peaks),
        },
        "gate_results": gates,
        "automatic_pass": all(gates.values()),
        "decision": decision,
        "claim_ceiling": contract["claim_ceiling"],
    }
    identity = {
        "experiment_id": report["experiment_id"],
        "contract_sha256": report["contract_sha256"],
        "threshold_identity": report["threshold_identity"],
        "conformance": conformance,
        "output_hwc_sha256": hashes[0],
        "gate_results": gates,
        "decision": decision,
    }
    report["stable_evidence_id"] = hashlib.sha256(
        canonical_bytes(identity)
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/u6_p8cb_native_thomas_srgb_quantized_sink_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/eval/u6_p8cb_native_thomas_srgb_quantized_sink_v1",
    )
    parser.add_argument(
        "--llvm", type=Path,
        default=ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--amplitude-dll", type=Path)
    parser.add_argument("--rows-dll", type=Path)
    parser.add_argument("--gauge-dll", type=Path)
    parser.add_argument("--quantizer-dll", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        values = (
            args.amplitude_dll, args.rows_dll, args.gauge_dll,
            args.quantizer_dll, args.result,
        )
        if any(value is None for value in values):
            parser.error("--worker requires four DLLs and --result")
        _worker(
            args.contract, args.amplitude_dll, args.rows_dll, args.gauge_dll,
            args.quantizer_dll, args.result,
        )
        return
    report = evaluate(args.contract, args.output_dir, args.llvm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "automatic_pass": report["automatic_pass"],
        "decision": report["decision"],
        "stable_evidence_id": report["stable_evidence_id"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
