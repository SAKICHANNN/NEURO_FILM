#!/usr/bin/env python3
"""Run U6.P8CF exact native Neumaier reduction evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import struct
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
from scripts.evaluate_u6_p8cb_native_thomas_srgb_quantized_sink import (
    _HashSink,
    _load_quantizer,
)
from src.eval.native_granularity_amplitude_conformance import (
    evaluate_conformance as evaluate_amplitude_conformance,
)
from src.eval.native_msvc import sha256_file
from src.eval.native_neumaier_conformance import build_and_load as build_reducers
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rows_conformance import build_and_load as build_rows
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
    load_native_neumaier_library,
    load_native_thomas_rows_library,
    stream_native_exposure_thomas_rgb_quantized_parallel,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8CF JSON must be an object")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8cf_native_neumaier_reducer_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("thomas_rows_v1_unchanged") is not True
        or candidate.get("ordered_float32_to_double_accumulation") is not True
        or candidate.get("parallel_colour_workers") != 3
        or candidate.get("model_profile_or_output_change_allowed") is not False
    ):
        raise RuntimeError("P8CF contract drift")
    parent = contract["parents"]["p8cd_evidence"]
    payload = _json(ROOT / parent["path"])
    if (
        sha256_file(ROOT / parent["path"]) != parent["sha256"]
        or payload.get("decision") != parent["required_decision"]
    ):
        raise RuntimeError("P8CF P8CD parent drift")


def _python_neumaier(values: np.ndarray) -> tuple[float, float]:
    total = 0.0
    compensation = 0.0
    for value in values:
        sample = float(value)
        updated = total + sample
        if abs(total) >= abs(sample):
            compensation += (total - updated) + sample
        else:
            compensation += (sample - updated) + total
        total = updated
    return total, compensation


def _direct_reducer_exact(library: ctypes.CDLL) -> bool:
    rng = np.random.default_rng(0x8CF)
    values = rng.standard_normal(100_003).astype(np.float32)
    expected = _python_neumaier(values)
    total = ctypes.c_double(0.0)
    compensation = ctypes.c_double(0.0)
    for start in range(0, values.size, 7919):
        tile = values[start : start + 7919]
        status = library.nf_neumaier_f32_accumulate_v1(
            _pointer(tile), tile.size, ctypes.byref(total), ctypes.byref(compensation)
        )
        if status != 1:
            return False
    return struct.pack("=dd", total.value, compensation.value) == struct.pack(
        "=dd", *expected
    )


def _invalid_atomic(library: ctypes.CDLL) -> bool:
    values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    total = ctypes.c_double(7.0)
    compensation = ctypes.c_double(-2.0)
    before = struct.pack("=dd", total.value, compensation.value)
    status = library.nf_neumaier_f32_accumulate_v1(
        _pointer(values), values.size, ctypes.byref(total), ctypes.byref(compensation)
    )
    return status == 0 and struct.pack("=dd", total.value, compensation.value) == before


def _worker(
    contract_path: Path,
    amplitude_dll: Path,
    rows_dll: Path,
    gauge_dll: Path,
    quantizer_dll: Path,
    reducer_dll: Path,
    result_path: Path,
) -> None:
    contract = _json(contract_path)
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    shape = tuple(int(value) for value in contract["performance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    input_sha = hashlib.sha256(exposure.tobytes()).hexdigest()
    depth = int(contract["performance"]["primary_bit_depth"])
    sink = _HashSink(np.uint16 if depth == 16 else np.uint8)
    started = time.perf_counter()
    means, workspace, calls = stream_native_exposure_thomas_rgb_quantized_parallel(
        load_native_granularity_amplitude_library(amplitude_dll),
        load_native_thomas_rows_library(rows_dll),
        _load_gauge(gauge_dll),
        _load_quantizer(quantizer_dll),
        compile_native_granularity_amplitude_profile(p4bw, prior_payload),
        _profiles(_json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")),
        native_gauge_profile_struct(_gauge_payload()),
        exposure,
        sink,
        row_partition=int(contract["candidate"]["row_partition"]),
        bit_depth=depth,
        reducer_library=load_native_neumaier_library(reducer_dll),
    )
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("P8CF mutated input exposure")
    result = {
        "schema": "neuro_film.u6_p8cf_native_neumaier_worker.v1",
        "shape_chw": list(shape),
        "bit_depth": depth,
        "input_sha256": input_sha,
        "output_hwc_sha256": sink.digest.hexdigest(),
        "raw_field_means": list(means),
        "sink_calls": calls,
        "sink_order_exact": sink.rows == sorted(sink.rows),
        "bounded_workspace_bytes": workspace,
        "wall_seconds": elapsed,
    }
    result_path.write_bytes(canonical_bytes(result))


def _monitored(
    contract_path: Path,
    amplitude_dll: Path,
    rows_dll: Path,
    gauge_dll: Path,
    quantizer_dll: Path,
    reducer_dll: Path,
    result_path: Path,
) -> dict[str, Any]:
    process = subprocess.Popen(
        [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--contract", str(contract_path),
            "--amplitude-dll", str(amplitude_dll),
            "--rows-dll", str(rows_dll),
            "--gauge-dll", str(gauge_dll),
            "--quantizer-dll", str(quantizer_dll),
            "--reducer-dll", str(reducer_dll),
            "--result", str(result_path),
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
        if time.perf_counter() - started > 30.0:
            for child in root_process.children(recursive=True):
                child.kill()
            root_process.kill()
            raise TimeoutError("P8CF worker exceeded timeout")
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
            "P8CF worker failed:\n" + stdout.decode(errors="replace") + stderr.decode(errors="replace")
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
    row_builds, row_libraries = build_rows(ROOT, output_dir / "rows", llvm / "bin/clang.exe")
    reducer_builds, reducers = build_reducers(ROOT, output_dir / "reducer", llvm / "bin/clang.exe")
    gauge_build = build_msvc_native_gauge_dll(root=ROOT, output_dir=output_dir / "gauge")
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
    amplitude_libraries = {
        name: load_native_granularity_amplitude_library(Path(row["dll_path"]))
        for name, row in amplitude["toolchains"].items()
    }
    quantizers = {name: _load_quantizer(path) for name, path in quantizer_paths.items()}
    gauge_library = _load_gauge(Path(gauge_build["dll_path"]))
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    profiles = _profiles(_json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json"))
    gauge_profile = native_gauge_profile_struct(_gauge_payload())
    shape = tuple(int(value) for value in contract["conformance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    conformance: dict[str, Any] = {}
    exact = True
    for name, row_library in row_libraries.items():
        by_depth: dict[str, Any] = {}
        direct = _direct_reducer_exact(reducers[name])
        atomic = _invalid_atomic(reducers[name])
        for depth in contract["conformance"]["bit_depths"]:
            python_sink = _HashSink(np.uint16 if depth == 16 else np.uint8)
            native_sink = _HashSink(np.uint16 if depth == 16 else np.uint8)
            python_means, _, python_calls = stream_native_exposure_thomas_rgb_quantized_parallel(
                amplitude_libraries[name], row_library, gauge_library, quantizers[name],
                amplitude_profile, profiles, gauge_profile, exposure, python_sink,
                row_partition=int(contract["candidate"]["row_partition"]), bit_depth=depth,
            )
            native_means, workspace, native_calls = stream_native_exposure_thomas_rgb_quantized_parallel(
                amplitude_libraries[name], row_library, gauge_library, quantizers[name],
                amplitude_profile, profiles, gauge_profile, exposure, native_sink,
                row_partition=int(contract["candidate"]["row_partition"]), bit_depth=depth,
                reducer_library=reducers[name],
            )
            means_exact = all(
                struct.pack("=d", left) == struct.pack("=d", right)
                for left, right in zip(python_means, native_means, strict=True)
            )
            row_exact = (
                python_sink.digest.hexdigest() == native_sink.digest.hexdigest()
                and means_exact
                and python_sink.rows == native_sink.rows
                and python_calls == native_calls
                and direct
                and atomic
            )
            exact = exact and row_exact
            by_depth[str(depth)] = {
                "python_sha256": python_sink.digest.hexdigest(),
                "native_sha256": native_sink.digest.hexdigest(),
                "raw_mean_bits_exact": means_exact,
                "direct_reducer_exact": direct,
                "invalid_output_unchanged": atomic,
                "exact": row_exact,
                "bounded_workspace_bytes": workspace,
            }
        conformance[name] = by_depth

    runs = [
        _monitored(
            contract_path,
            Path(amplitude["toolchains"]["msvc"]["dll_path"]),
            Path(row_builds["msvc"]["dll_path"]),
            Path(gauge_build["dll_path"]),
            quantizer_paths["msvc"],
            Path(reducer_builds["msvc"]["dll_path"]),
            output_dir / f"worker-{index}.json",
        )
        for index in (1, 2)
    ]
    walls = [float(row["worker"]["wall_seconds"]) for row in runs]
    peaks = [int(row["peak_process_tree_rss_bytes"]) for row in runs]
    hashes = [str(row["worker"]["output_hwc_sha256"]) for row in runs]
    reference = float(contract["parents"]["p8cd_evidence"]["reference_maximum_wall_seconds"])
    limits = contract["performance"]
    gates = {
        "exact_python_native_stream": exact,
        "fresh_stream_identity": len(set(hashes)) == 1,
        "exact_p8cd_stream_identity": set(hashes)
        == {"5a0f531fe9b28183dba4634b0c1d739f71c2677e2f9ba74dd16c33c2fa6530fd"},
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "material_wall_reduction": max(walls) / reference
        <= float(limits["maximum_wall_ratio_vs_p8cd"]),
        "maximum_rss": max(peaks) <= int(limits["maximum_process_tree_rss_bytes"]),
        "wall_repeat": max(walls) / min(walls) <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": max(peaks) / min(peaks) <= float(limits["maximum_repeat_rss_ratio"]),
        "worker_cleanup": all(
            row["stderr_empty"] and row["surviving_process_count"] == 0 for row in runs
        ),
    }
    decision = contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"]
    report = {
        "schema": "neuro_film.u6_p8cf_native_neumaier_reducer_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "reducer_builds": reducer_builds,
        "quantizer_builds": quantizer_builds,
        "conformance": conformance,
        "performance": {
            "runs": runs,
            "output_hwc_sha256": hashes[0],
            "maximum_wall_seconds": max(walls),
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "wall_ratio_vs_p8cd": max(walls) / reference,
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
        "reducer_builds": reducer_builds,
        "conformance": conformance,
        "output_hwc_sha256": hashes[0],
        "gate_results": gates,
        "decision": decision,
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/u6_p8cf_native_neumaier_reducer_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/eval/u6_p8cf_native_neumaier_reducer_v1",
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
    parser.add_argument("--reducer-dll", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        values = (
            args.amplitude_dll, args.rows_dll, args.gauge_dll,
            args.quantizer_dll, args.reducer_dll, args.result,
        )
        if any(value is None for value in values):
            parser.error("--worker requires five DLLs and --result")
        _worker(
            args.contract, args.amplitude_dll, args.rows_dll, args.gauge_dll,
            args.quantizer_dll, args.reducer_dll, args.result,
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
