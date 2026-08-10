#!/usr/bin/env python3
"""Run U6.P8BX exact row-stream conformance and 12MP resource evidence."""

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
from src.eval.native_thomas_rows_conformance import (
    build_and_load,
    failure_output_atomic,
    validate_contract,
)
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
    render_native_exposure_thomas_rgb_rows,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8BX JSON must be an object")
    return payload


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
    amplitude_library = load_native_granularity_amplitude_library(amplitude_dll)
    rows_library = load_native_thomas_rows_library(rows_dll)
    profiles = _profiles(
        _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    started = time.perf_counter()
    output, raw_means, workspace_bytes = render_native_exposure_thomas_rgb_rows(
        amplitude_library,
        rows_library,
        amplitude_profile,
        profiles,
        exposure,
        row_partition=int(contract["candidate"]["performance_row_partition"]),
    )
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("P8BX row stream mutated its input")
    result = {
        "schema": "neuro_film.u6_p8bx_native_thomas_row_worker.v1",
        "shape_chw": list(output.shape),
        "input_sha256": input_sha,
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "channel_sha256": [
            hashlib.sha256(output[index].tobytes()).hexdigest() for index in range(3)
        ],
        "transmittance_range": [float(np.min(output)), float(np.max(output))],
        "raw_field_means": list(raw_means),
        "input_bytes": exposure.nbytes,
        "output_bytes": output.nbytes,
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
    timeout: float,
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
        if time.perf_counter() - started > timeout:
            for child in root_process.children(recursive=True):
                child.kill()
            root_process.kill()
            raise TimeoutError("P8BX worker exceeded timeout")
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
            "P8BX worker failed:\n"
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
    validate_contract(ROOT, contract)
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
        amplitude_libraries["msvc"],
        density_library,
        amplitude_profile,
        profiles,
        exposure,
    )
    conformance_hashes: dict[str, dict[str, str]] = {}
    exact = True
    input_sha = hashlib.sha256(exposure.tobytes()).hexdigest()
    for name, rows_library in row_libraries.items():
        conformance_hashes[name] = {}
        exact = exact and failure_output_atomic(rows_library, ROOT)
        for partition in contract["candidate"]["row_partitions"]:
            actual, means, _ = render_native_exposure_thomas_rgb_rows(
                amplitude_libraries[name],
                rows_library,
                amplitude_profile,
                profiles,
                exposure,
                row_partition=int(partition),
            )
            conformance_hashes[name][str(partition)] = hashlib.sha256(
                actual.tobytes()
            ).hexdigest()
            exact = exact and np.array_equal(actual, expected) and means == expected_means
    exact = exact and hashlib.sha256(exposure.tobytes()).hexdigest() == input_sha

    amplitude_dll = Path(amplitude["toolchains"]["msvc"]["dll_path"])
    rows_dll = Path(row_builds["msvc"]["dll_path"])
    runs = [
        _monitored(
            contract_path,
            amplitude_dll,
            rows_dll,
            output_dir / f"worker_{index}.json",
            30.0,
        )
        for index in (1, 2)
    ]
    walls = [float(run["worker"]["wall_seconds"]) for run in runs]
    peaks = [int(run["peak_process_tree_rss_bytes"]) for run in runs]
    hashes = [str(run["worker"]["output_sha256"]) for run in runs]
    channel_hashes = [tuple(run["worker"]["channel_sha256"]) for run in runs]
    limits = contract["performance"]
    wall_ratio = max(walls) / min(walls)
    rss_ratio = max(peaks) / min(peaks)
    rss_reduction = int(limits["p8bw_reference_peak_rss_bytes"]) - max(peaks)
    gates = {
        "amplitude_conformance": bool(amplitude["automatic_pass"]),
        "density_conformance": bool(density["automatic_pass"]),
        "bit_exact_all_compilers_and_partitions": bool(exact),
        "fresh_output_exact": len(set(hashes)) == 1,
        "fresh_channel_exact": len(set(channel_hashes)) == 1,
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "maximum_rss": max(peaks) <= int(limits["maximum_process_tree_rss_bytes"]),
        "minimum_rss_reduction": rss_reduction
        >= int(limits["minimum_rss_reduction_bytes_vs_p8bw"]),
        "wall_repeat": wall_ratio <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": rss_ratio <= float(limits["maximum_repeat_rss_ratio"]),
        "strict_transmittance": all(
            float(run["worker"]["transmittance_range"][0]) > 0.0
            and float(run["worker"]["transmittance_range"][1]) <= 1.0
            for run in runs
        ),
        "worker_cleanup": all(
            run["stderr_empty"] and run["surviving_process_count"] == 0
            for run in runs
        ),
    }
    decision = (
        contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"]
    )
    report = {
        "schema": "neuro_film.u6_p8bx_native_thomas_row_stream_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "builds": row_builds,
        "conformance_output_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "conformance_hashes": conformance_hashes,
        "performance": {
            "runs": runs,
            "output_sha256": hashes[0],
            "channel_sha256": list(channel_hashes[0]),
            "maximum_wall_seconds": max(walls),
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "rss_reduction_bytes_vs_p8bw": rss_reduction,
            "wall_repeat_ratio": wall_ratio,
            "rss_repeat_ratio": rss_ratio,
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
        "header_sha256": row_builds["msvc"]["header_sha256"],
        "conformance_output_sha256": report["conformance_output_sha256"],
        "output_sha256": hashes[0],
        "channel_sha256": list(channel_hashes[0]),
        "gate_results": gates,
        "decision": decision,
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8bx_native_thomas_row_stream_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8bx_native_thomas_row_stream_v1",
    )
    parser.add_argument(
        "--clang",
        type=Path,
        default=ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe",
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
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
