#!/usr/bin/env python3
"""Run U6.P8CA exact non-retaining display-linear sink evidence."""

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

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _exposure_fixture,
    _parent_payloads,
    _profiles,
)
from src.eval.native_granularity_amplitude_conformance import (
    evaluate_conformance as evaluate_amplitude_conformance,
)
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_density_conformance import (
    evaluate_conformance as evaluate_density_conformance,
)
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rows_conformance import build_and_load
from src.eval.physical_native_gauge_conformance import (
    _load_gauge,
    build_msvc_native_gauge_dll,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_exposure_thomas_pipeline import (
    render_native_exposure_thomas_rgb,
)
from src.film_physics.native_gauge_profile import (
    native_gauge_payload_sha256,
    native_gauge_profile_struct,
)
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)
from src.film_physics.native_thomas_density import load_native_thomas_density_library
from src.film_physics.native_thomas_rows import (
    _pointer,
    load_native_thomas_rows_library,
    stream_native_exposure_thomas_rgb_gauged,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8CA JSON must be an object")
    return payload


def _gauge_payload() -> dict[str, Any]:
    decision = _json(
        ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_decision_v1.json"
    )
    report_path = (
        ROOT / "outputs/u6_p8b_artifact_only_cpu_consumer_v1/run_a/report.json"
    )
    if sha256_file(report_path) != decision["report_sha256"]:
        raise RuntimeError("P8CA persisted P8B artifact report drift")
    report = _json(report_path)
    if (
        report.get("artifact_canonical_sha256")
        != decision["artifact_canonical_sha256"]
    ):
        raise RuntimeError("P8CA persisted P8B artifact identity drift")
    payload = report["artifact"]["component_payloads"]["neutral-axis-gauge"]
    if not isinstance(payload, dict):
        raise TypeError("P8CA persisted neutral gauge payload drift")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8ca_native_thomas_gauged_sink_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("row_abi_and_p8bw_profiles_unchanged") is not True
        or candidate.get("full_output_or_channel_plane_allowed") is not False
        or candidate.get("model_or_profile_change_allowed") is not False
        or candidate.get("output_transfer_quantization_or_encoding_allowed")
        is not False
    ):
        raise RuntimeError("P8CA contract drift")
    p8bz = contract["parents"]["p8bz_evidence"]
    p8bz_payload = _json(ROOT / p8bz["path"])
    if (
        sha256_file(ROOT / p8bz["path"]) != p8bz["sha256"]
        or p8bz_payload.get("decision") != p8bz["required_decision"]
    ):
        raise RuntimeError("P8CA P8BZ parent drift")
    p8ag = contract["parents"]["p8ag_decision"]
    p8ag_payload = _json(ROOT / p8ag["path"])
    if (
        sha256_file(ROOT / p8ag["path"]) != p8ag["sha256"]
        or p8ag_payload.get("result", {}).get("status")
        != p8ag["required_result_status"]
    ):
        raise RuntimeError("P8CA P8AG parent drift")
    if native_gauge_payload_sha256(_gauge_payload()) != candidate[
        "neutral_gauge_payload_sha256"
    ]:
        raise RuntimeError("P8CA neutral gauge payload drift")


class _Sink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.rows: list[int] = []

    def __call__(self, row_start: int, values: np.ndarray) -> None:
        if values.flags.writeable or not values.flags.c_contiguous:
            raise RuntimeError("P8CA sink tile contract drift")
        self.digest.update(values.tobytes())
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.rows.append(row_start)


def _worker(
    contract_path: Path,
    amplitude_dll: Path,
    rows_dll: Path,
    gauge_dll: Path,
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
    sink = _Sink()
    started = time.perf_counter()
    means, workspace_bytes, calls = stream_native_exposure_thomas_rgb_gauged(
        load_native_granularity_amplitude_library(amplitude_dll),
        load_native_thomas_rows_library(rows_dll),
        _load_gauge(gauge_dll),
        amplitude_profile,
        _profiles(_json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")),
        native_gauge_profile_struct(_gauge_payload()),
        exposure,
        sink,
        row_partition=int(contract["candidate"]["row_partition"]),
    )
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("P8CA gauged sink mutated its input")
    result = {
        "schema": "neuro_film.u6_p8ca_native_thomas_gauged_worker.v1",
        "shape_chw": list(shape),
        "input_sha256": input_sha,
        "output_hwc_sha256": sink.digest.hexdigest(),
        "display_linear_range": [sink.minimum, sink.maximum],
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
            raise TimeoutError("P8CA worker exceeded timeout")
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
            "P8CA worker failed:\n"
            + stdout.decode(errors="replace")
            + stderr.decode(errors="replace")
        )
    return {
        "worker": _json(result_path),
        "peak_process_tree_rss_bytes": peak,
        "stderr_empty": not bool(stderr),
        "surviving_process_count": sum(1 for pid in observed if psutil.pid_exists(pid)),
    }


def evaluate(contract_path: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    contract = _json(contract_path)
    _validate_contract(contract)
    amplitude = evaluate_amplitude_conformance(
        ROOT,
        _json(ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"),
        output_dir=output_dir / "amplitude",
        clang=clang,
    )
    density = evaluate_density_conformance(
        ROOT,
        _json(ROOT / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"),
        output_dir=output_dir / "density",
        clang=clang,
    )
    row_builds, row_libraries = build_and_load(ROOT, output_dir / "rows", clang)
    gauge_build = build_msvc_native_gauge_dll(
        root=ROOT, output_dir=output_dir / "gauge"
    )
    gauge_library = _load_gauge(Path(gauge_build["dll_path"]))
    gauge_profile = native_gauge_profile_struct(_gauge_payload())
    amplitude_libraries = {
        name: load_native_granularity_amplitude_library(Path(row["dll_path"]))
        for name, row in amplitude["toolchains"].items()
    }
    density_library = load_native_thomas_density_library(
        Path(density["toolchains"]["msvc"]["dll_path"])
    )
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
    expected_chw, expected_means, _ = render_native_exposure_thomas_rgb(
        amplitude_libraries["msvc"],
        density_library,
        amplitude_profile,
        profiles,
        exposure,
    )
    expected_scan = np.ascontiguousarray(np.transpose(expected_chw, (1, 2, 0)))
    expected = np.empty_like(expected_scan)
    status = gauge_library.nf_neutral_gauge_f32_apply_v1(
        ctypes.byref(gauge_profile),
        _pointer(expected_scan),
        expected_scan.shape[0] * expected_scan.shape[1],
        _pointer(expected),
    )
    if status != 0:
        raise RuntimeError(f"P8CA full-array gauge oracle failed: {status}")
    expected_hash = hashlib.sha256(expected.tobytes()).hexdigest()
    conformance: dict[str, Any] = {}
    exact = True
    for name, rows_library in row_libraries.items():
        sink = _Sink()
        means, _, calls = stream_native_exposure_thomas_rgb_gauged(
            amplitude_libraries[name],
            rows_library,
            gauge_library,
            amplitude_profile,
            profiles,
            gauge_profile,
            exposure,
            sink,
            row_partition=int(contract["candidate"]["row_partition"]),
        )
        row = {
            "output_hwc_sha256": sink.digest.hexdigest(),
            "raw_field_means": list(means),
            "sink_calls": calls,
            "sink_order_exact": sink.rows == sorted(sink.rows),
        }
        conformance[name] = row
        exact = exact and row["output_hwc_sha256"] == expected_hash
        exact = exact and means == expected_means and row["sink_order_exact"]

    calls = 0

    def fail_sink(row_start: int, values: np.ndarray) -> None:
        nonlocal calls
        del row_start, values
        calls += 1
        if calls == 2:
            raise RuntimeError("injected P8CA sink failure")

    try:
        stream_native_exposure_thomas_rgb_gauged(
            amplitude_libraries["msvc"], row_libraries["msvc"], gauge_library,
            amplitude_profile, profiles, gauge_profile, exposure, fail_sink,
            row_partition=int(contract["candidate"]["row_partition"]),
        )
        sink_failure = False
    except RuntimeError as error:
        sink_failure = str(error) == "injected P8CA sink failure" and calls == 2

    invalid_gauge_output = np.full((2, 2, 3), np.float32(-13.0))
    invalid_gauge_input = np.zeros((2, 2, 3), dtype=np.float32)
    invalid_gauge_input[0, 0, 0] = np.nan
    before = invalid_gauge_output.tobytes()
    status = gauge_library.nf_neutral_gauge_f32_apply_v1(
        ctypes.byref(gauge_profile), _pointer(invalid_gauge_input), 4,
        _pointer(invalid_gauge_output),
    )
    gauge_atomic = status != 0 and invalid_gauge_output.tobytes() == before

    runs = [
        _monitored(
            contract_path,
            Path(amplitude["toolchains"]["msvc"]["dll_path"]),
            Path(row_builds["msvc"]["dll_path"]),
            Path(gauge_build["dll_path"]),
            output_dir / f"worker_{index}.json",
        )
        for index in (1, 2)
    ]
    walls = [float(row["worker"]["wall_seconds"]) for row in runs]
    peaks = [int(row["peak_process_tree_rss_bytes"]) for row in runs]
    hashes = [str(row["worker"]["output_hwc_sha256"]) for row in runs]
    limits = contract["performance"]
    gates = {
        "exact_full_array_gauge_output": bool(exact),
        "sink_failure_propagation": sink_failure,
        "gauge_failure_atomicity": gauge_atomic,
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
        "schema": "neuro_film.u6_p8ca_native_thomas_gauged_sink_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "row_builds": row_builds,
        "gauge_build": gauge_build,
        "neutral_gauge_payload_sha256": native_gauge_payload_sha256(_gauge_payload()),
        "expected_display_linear_hwc_sha256": expected_hash,
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
        "expected_display_linear_hwc_sha256": expected_hash,
        "neutral_gauge_payload_sha256": report["neutral_gauge_payload_sha256"],
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
        default=ROOT / "configs/u6_p8ca_native_thomas_gauged_sink_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/eval/u6_p8ca_native_thomas_gauged_sink_v1",
    )
    parser.add_argument(
        "--clang", type=Path,
        default=ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--amplitude-dll", type=Path)
    parser.add_argument("--rows-dll", type=Path)
    parser.add_argument("--gauge-dll", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if any(value is None for value in (args.amplitude_dll, args.rows_dll, args.gauge_dll, args.result)):
            parser.error("--worker requires amplitude, rows, gauge DLLs and --result")
        _worker(
            args.contract, args.amplitude_dll, args.rows_dll, args.gauge_dll,
            args.result,
        )
        return
    report = evaluate(args.contract, args.output_dir, args.clang)
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
