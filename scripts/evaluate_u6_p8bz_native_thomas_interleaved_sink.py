#!/usr/bin/env python3
"""Run U6.P8BZ exact interleaved row-sink evidence."""

from __future__ import annotations

import argparse
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
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_exposure_thomas_pipeline import (
    render_native_exposure_thomas_rgb,
)
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)
from src.film_physics.native_thomas_density import load_native_thomas_density_library
from src.film_physics.native_thomas_rows import (
    load_native_thomas_rows_library,
    stream_native_exposure_thomas_rgb_interleaved,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8BZ JSON must be an object")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    if (
        contract.get("schema")
        != "neuro_film.u6_p8bz_native_thomas_interleaved_sink_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or contract["candidate"].get("row_abi_and_profiles_unchanged") is not True
        or contract["candidate"].get("full_output_or_channel_plane_allowed")
        is not False
        or contract["candidate"].get("model_or_profile_change_allowed") is not False
    ):
        raise RuntimeError("P8BZ contract drift")
    for binding in contract["parents"].values():
        path = ROOT / binding["path"]
        payload = _json(path)
        if (
            sha256_file(path) != binding["sha256"]
            or payload.get("decision") != binding["required_decision"]
        ):
            raise RuntimeError("P8BZ parent drift")


class _Sink:
    def __init__(self) -> None:
        self.hwc = hashlib.sha256()
        self.channels = [hashlib.sha256() for _ in range(3)]
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.rows: list[int] = []

    def __call__(self, row_start: int, values: np.ndarray) -> None:
        self.hwc.update(values.tobytes())
        for channel in range(3):
            self.channels[channel].update(values[..., channel].tobytes())
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.rows.append(row_start)


def _worker(
    contract_path: Path, amplitude_dll: Path, rows_dll: Path, result_path: Path
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
    means, workspace_bytes, calls = stream_native_exposure_thomas_rgb_interleaved(
        load_native_granularity_amplitude_library(amplitude_dll),
        load_native_thomas_rows_library(rows_dll),
        amplitude_profile,
        _profiles(
            _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
        ),
        exposure,
        sink,
        row_partition=int(contract["candidate"]["row_partition"]),
    )
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("P8BZ interleaved sink mutated its input")
    result = {
        "schema": "neuro_film.u6_p8bz_native_thomas_interleaved_worker.v1",
        "shape_chw": list(shape),
        "input_sha256": input_sha,
        "output_hwc_sha256": sink.hwc.hexdigest(),
        "channel_sha256": [digest.hexdigest() for digest in sink.channels],
        "transmittance_range": [sink.minimum, sink.maximum],
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
        if time.perf_counter() - started > 30.0:
            for child in root_process.children(recursive=True):
                child.kill()
            root_process.kill()
            raise TimeoutError("P8BZ worker exceeded timeout")
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
            "P8BZ worker failed:\n"
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
    amplitude_contract = _json(
        ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"
    )
    density_contract = _json(
        ROOT / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"
    )
    amplitude = evaluate_amplitude_conformance(
        ROOT, amplitude_contract, output_dir=output_dir / "amplitude", clang=clang
    )
    density = evaluate_density_conformance(
        ROOT, density_contract, output_dir=output_dir / "density", clang=clang
    )
    row_builds, row_libraries = build_and_load(ROOT, output_dir / "rows", clang)
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
    expected, expected_means, _ = render_native_exposure_thomas_rgb(
        amplitude_libraries["msvc"], density_library, amplitude_profile, profiles, exposure
    )
    expected_hwc = np.transpose(expected, (1, 2, 0))
    expected_hash = hashlib.sha256(expected_hwc.tobytes()).hexdigest()
    conformance: dict[str, dict[str, Any]] = {}
    exact = True
    for name, rows_library in row_libraries.items():
        sink = _Sink()
        means, _, calls = stream_native_exposure_thomas_rgb_interleaved(
            amplitude_libraries[name],
            rows_library,
            amplitude_profile,
            profiles,
            exposure,
            sink,
            row_partition=int(contract["candidate"]["row_partition"]),
        )
        row = {
            "output_hwc_sha256": sink.hwc.hexdigest(),
            "raw_field_means": list(means),
            "sink_calls": calls,
            "sink_order_exact": sink.rows == sorted(sink.rows),
        }
        conformance[name] = row
        exact = exact and row["output_hwc_sha256"] == expected_hash
        exact = exact and means == expected_means and row["sink_order_exact"]

    failure_calls = 0

    def fail_sink(row_start: int, values: np.ndarray) -> None:
        nonlocal failure_calls
        del row_start, values
        failure_calls += 1
        if failure_calls == 2:
            raise RuntimeError("injected P8BZ sink failure")

    try:
        stream_native_exposure_thomas_rgb_interleaved(
            amplitude_libraries["msvc"],
            row_libraries["msvc"],
            amplitude_profile,
            profiles,
            exposure,
            fail_sink,
            row_partition=int(contract["candidate"]["row_partition"]),
        )
        sink_failure = False
    except RuntimeError as error:
        sink_failure = str(error) == "injected P8BZ sink failure" and failure_calls == 2

    runs = [
        _monitored(
            contract_path,
            Path(amplitude["toolchains"]["msvc"]["dll_path"]),
            Path(row_builds["msvc"]["dll_path"]),
            output_dir / f"worker_{index}.json",
        )
        for index in (1, 2)
    ]
    walls = [float(row["worker"]["wall_seconds"]) for row in runs]
    peaks = [int(row["peak_process_tree_rss_bytes"]) for row in runs]
    hashes = [str(row["worker"]["output_hwc_sha256"]) for row in runs]
    limits = contract["performance"]
    gates = {
        "exact_p8bw_pixels": bool(exact),
        "sink_failure_propagation": sink_failure,
        "fresh_stream_exact": len(set(hashes)) == 1,
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "maximum_rss": max(peaks) <= int(limits["maximum_process_tree_rss_bytes"]),
        "wall_repeat": max(walls) / min(walls)
        <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": max(peaks) / min(peaks)
        <= float(limits["maximum_repeat_rss_ratio"]),
        "worker_cleanup": all(
            row["stderr_empty"] and row["surviving_process_count"] == 0 for row in runs
        ),
    }
    decision = (
        contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"]
    )
    report = {
        "schema": "neuro_film.u6_p8bz_native_thomas_interleaved_sink_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "row_builds": row_builds,
        "expected_hwc_sha256": expected_hash,
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
        "source_sha256": row_builds["msvc"]["source_sha256"],
        "expected_hwc_sha256": expected_hash,
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
        default=ROOT / "configs/u6_p8bz_native_thomas_interleaved_sink_v1.json"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/eval/u6_p8bz_native_thomas_interleaved_sink_v1"
    )
    parser.add_argument(
        "--clang", type=Path,
        default=ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--amplitude-dll", type=Path)
    parser.add_argument("--rows-dll", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.amplitude_dll is None or args.rows_dll is None or args.result is None:
            parser.error("--worker requires both DLLs and --result")
        _worker(args.contract, args.amplitude_dll, args.rows_dll, args.result)
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
