"""Run U6.P8BW exact composition and 12MP resource evidence."""

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

from src.eval.native_granularity_amplitude_conformance import (
    evaluate_conformance as evaluate_amplitude_conformance,
)
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_density_conformance import (
    evaluate_conformance as evaluate_thomas_conformance,
)
from src.eval.native_thomas_field_conformance import (
    canonical_bytes,
    profile_from_contract,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_exposure_thomas_pipeline import (
    render_native_exposure_thomas_rgb,
)
from src.film_physics.native_granularity_amplitude import (
    apply_native_granularity_amplitude,
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)
from src.film_physics.native_thomas_density import (
    load_native_thomas_density_library,
    render_native_thomas_rgb_transmittance,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8BW JSON must be an object")
    return payload


def _profiles(contract: dict[str, Any]) -> tuple[Any, Any, Any]:
    template = profile_from_contract(
        _json(ROOT / "configs/u6_p8bs_native_thomas_field_v1.json")
    )
    rows = []
    for seed in contract["candidate"]["layer_realization_seeds"]:
        rows.append(
            type(template)(
                particle_sigma_pixels=template.particle_sigma_pixels,
                cluster_sigma_pixels=template.cluster_sigma_pixels,
                mean_offspring=template.mean_offspring,
                truncate=template.truncate,
                component_seeds=template.component_seeds,
                realization_seed=int(seed),
            )
        )
    return rows[0], rows[1], rows[2]


def _exposure_fixture(
    prior: ManufacturerCharacteristicPrior, shape: tuple[int, int]
) -> np.ndarray:
    height, width = shape
    exposure = np.empty((3, height, width), dtype=np.float32)
    x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) / np.float32(width)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        lower32 = np.float32(lower)
        if float(lower32) < lower:
            lower32 = np.nextafter(lower32, np.float32(np.inf))
        span = np.float32(upper - float(lower32))
        for row in range(height):
            y = np.float32((row + 0.5) / height)
            position = np.float32(0.05) + np.float32(0.85) * x + np.float32(0.10) * y
            exposure[channel, row] = lower32 + span * position
    return exposure


def _parent_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    amplitude_contract = _json(
        ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"
    )
    return (
        _json(ROOT / amplitude_contract["parents"]["p4bw_bundle"]["path"]),
        _json(ROOT / amplitude_contract["parents"]["p2q_bundle"]["path"]),
    )


def _worker(
    contract_path: Path, amplitude_dll: Path, thomas_dll: Path, result_path: Path
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
    thomas_library = load_native_thomas_density_library(thomas_dll)
    started = time.perf_counter()
    output, raw_means, reused_bytes = render_native_exposure_thomas_rgb(
        amplitude_library,
        thomas_library,
        amplitude_profile,
        _profiles(contract),
        exposure,
    )
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("P8BW pipeline mutated its input")
    result = {
        "schema": "neuro_film.u6_p8bw_native_exposure_thomas_worker.v1",
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
        "reused_intermediate_bytes": reused_bytes,
        "modeled_separated_intermediate_bytes": 2 * exposure.nbytes + 5 * exposure[0].nbytes,
        "modeled_live_array_reduction_bytes": 2 * exposure.nbytes + 5 * exposure[0].nbytes - reused_bytes,
        "wall_seconds": elapsed,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(canonical_bytes(result))


def _monitored(
    contract_path: Path,
    amplitude_dll: Path,
    thomas_dll: Path,
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
            "--thomas-dll",
            str(thomas_dll),
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
            raise TimeoutError("P8BW worker exceeded timeout")
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
            "P8BW worker failed:\n"
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
    for binding in contract["parents"].values():
        path = ROOT / binding["path"]
        payload = _json(path)
        if sha256_file(path) != binding["sha256"] or payload["decision"] != binding["required_decision"]:
            raise RuntimeError("P8BW parent evidence drift")
    amplitude_contract = _json(
        ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"
    )
    thomas_contract = _json(
        ROOT / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"
    )
    amplitude_conformance = evaluate_amplitude_conformance(
        ROOT, amplitude_contract, output_dir=output_dir / "amplitude", clang=clang
    )
    thomas_conformance = evaluate_thomas_conformance(
        ROOT, thomas_contract, output_dir=output_dir / "thomas", clang=clang
    )
    amplitude_dll = Path(amplitude_conformance["toolchains"]["msvc"]["dll_path"])
    thomas_dll = Path(thomas_conformance["toolchains"]["msvc"]["dll_path"])

    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    conformance_shape = tuple(int(value) for value in contract["conformance"]["shape"])
    exposure = _exposure_fixture(prior, (conformance_shape[1], conformance_shape[2]))
    amplitude_library = load_native_granularity_amplitude_library(amplitude_dll)
    thomas_library = load_native_thomas_density_library(thomas_dll)
    density, sigma = apply_native_granularity_amplitude(
        amplitude_library, amplitude_profile, exposure
    )
    expected, expected_means, separated_reused = render_native_thomas_rgb_transmittance(
        thomas_library, _profiles(contract), density, sigma
    )
    actual, actual_means, fused_reused = render_native_exposure_thomas_rgb(
        amplitude_library, thomas_library, amplitude_profile, _profiles(contract), exposure
    )
    modeled_reduction = density.nbytes + sigma.nbytes + separated_reused - fused_reused
    exact_separated = bool(np.array_equal(actual, expected) and actual_means == expected_means)

    runs = [
        _monitored(
            contract_path,
            amplitude_dll,
            thomas_dll,
            output_dir / f"worker_{index}.json",
            30.0,
        )
        for index in (1, 2)
    ]
    walls = [float(run["worker"]["wall_seconds"]) for run in runs]
    peaks = [int(run["peak_process_tree_rss_bytes"]) for run in runs]
    hashes = [str(run["worker"]["output_sha256"]) for run in runs]
    channel_hashes = [tuple(run["worker"]["channel_sha256"]) for run in runs]
    performance_reductions = [
        int(run["worker"]["modeled_live_array_reduction_bytes"]) for run in runs
    ]
    limits = contract["performance"]
    gates = contract["conformance"]
    wall_ratio = max(walls) / min(walls)
    rss_ratio = max(peaks) / min(peaks)
    gate_results = {
        "amplitude_conformance": bool(amplitude_conformance["automatic_pass"]),
        "thomas_conformance": bool(thomas_conformance["automatic_pass"]),
        "exact_separated_execution": exact_separated,
        "modeled_live_array_reduction": min(performance_reductions)
        >= int(gates["minimum_modeled_live_array_reduction_bytes_12mp"]),
        "fresh_output_exact": len(set(hashes)) == 1,
        "fresh_channel_exact": len(set(channel_hashes)) == 1,
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "maximum_rss": max(peaks) <= int(limits["maximum_process_tree_rss_bytes"]),
        "wall_repeat": wall_ratio <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": rss_ratio <= float(limits["maximum_repeat_rss_ratio"]),
        "strict_transmittance": all(
            float(run["worker"]["transmittance_range"][0]) > 0.0
            and float(run["worker"]["transmittance_range"][1]) <= 1.0
            for run in runs
        ),
        "worker_cleanup": all(
            run["stderr_empty"] and run["surviving_process_count"] == 0 for run in runs
        ),
    }
    decision = contract["decision_if_pass"] if all(gate_results.values()) else contract["decision_if_fail"]
    report = {
        "schema": "neuro_film.u6_p8bw_native_exposure_thomas_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "amplitude_conformance_id": amplitude_conformance["stable_evidence_id"],
        "thomas_conformance_id": thomas_conformance["stable_evidence_id"],
        "conformance_output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "modeled_conformance_reduction_bytes": modeled_reduction,
        "performance": {
            "runs": runs,
            "output_sha256": hashes[0],
            "channel_sha256": list(channel_hashes[0]),
            "maximum_wall_seconds": max(walls),
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "wall_repeat_ratio": wall_ratio,
            "rss_repeat_ratio": rss_ratio,
        },
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "decision": decision,
        "claim_ceiling": contract["claim_ceiling"],
    }
    identity = {
        "experiment_id": report["experiment_id"],
        "contract_sha256": report["contract_sha256"],
        "amplitude_conformance_id": report["amplitude_conformance_id"],
        "thomas_conformance_id": report["thomas_conformance_id"],
        "conformance_output_sha256": report["conformance_output_sha256"],
        "output_sha256": hashes[0],
        "channel_sha256": list(channel_hashes[0]),
        "gate_results": gate_results,
        "decision": decision,
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8bw_native_exposure_thomas_pipeline_v1",
    )
    parser.add_argument(
        "--clang",
        type=Path,
        default=ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--amplitude-dll", type=Path)
    parser.add_argument("--thomas-dll", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.amplitude_dll is None or args.thomas_dll is None or args.result is None:
            parser.error("--worker requires both DLLs and --result")
        _worker(args.contract, args.amplitude_dll, args.thomas_dll, args.result)
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
