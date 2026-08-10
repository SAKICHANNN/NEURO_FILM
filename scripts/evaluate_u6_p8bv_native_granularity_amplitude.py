"""Run U6.P8BV conformance and 12MP granularity-amplitude evidence."""

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

from src.eval.native_granularity_amplitude_conformance import evaluate_conformance
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_granularity_amplitude import (
    apply_native_granularity_amplitude,
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8BV JSON must be an object")
    return payload


def _bundles(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _json(ROOT / contract["parents"]["p4bw_bundle"]["path"]),
        _json(ROOT / contract["parents"]["p2q_bundle"]["path"]),
    )


def _exposure_fixture(
    prior: ManufacturerCharacteristicPrior, shape: tuple[int, int]
) -> np.ndarray:
    height, width = shape
    exposure = np.empty((3, height, width), dtype=np.float32)
    x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) / np.float32(width)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        span = np.float32(upper - lower)
        lower32 = np.float32(lower)
        for row in range(height):
            y = np.float32((row + 0.5) / height)
            position = np.float32(0.05) + np.float32(0.85) * x + np.float32(0.10) * y
            exposure[channel, row] = lower32 + span * position
    return exposure


def _worker(contract_path: Path, dll: Path, result_path: Path) -> None:
    contract = _json(contract_path)
    p4bw, prior_payload = _bundles(contract)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    profile = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    shape_chw = tuple(int(value) for value in contract["performance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape_chw[1], shape_chw[2]))
    exposure_sha = hashlib.sha256(exposure.tobytes()).hexdigest()
    library = load_native_granularity_amplitude_library(dll)
    started = time.perf_counter()
    density, sigma = apply_native_granularity_amplitude(library, profile, exposure)
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != exposure_sha:
        raise RuntimeError("P8BV native execution mutated its input")
    result = {
        "schema": "neuro_film.u6_p8bv_native_granularity_amplitude_worker.v1",
        "shape_chw": list(exposure.shape),
        "exposure_sha256": exposure_sha,
        "density_sha256": hashlib.sha256(density.tobytes()).hexdigest(),
        "point_sigma_sha256": hashlib.sha256(sigma.tobytes()).hexdigest(),
        "channel_density_sha256": [
            hashlib.sha256(density[index].tobytes()).hexdigest() for index in range(3)
        ],
        "channel_point_sigma_sha256": [
            hashlib.sha256(sigma[index].tobytes()).hexdigest() for index in range(3)
        ],
        "density_range": [float(np.min(density)), float(np.max(density))],
        "point_sigma_range": [float(np.min(sigma)), float(np.max(sigma))],
        "input_bytes": exposure.nbytes,
        "output_bytes": density.nbytes + sigma.nbytes,
        "wall_seconds": elapsed,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(canonical_bytes(result))


def _monitored(
    contract_path: Path, dll: Path, result_path: Path, timeout: float
) -> dict[str, Any]:
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--contract",
            str(contract_path),
            "--dll",
            str(dll),
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
            raise TimeoutError("P8BV worker exceeded timeout")
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
            "P8BV worker failed:\n"
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
    conformance = evaluate_conformance(
        ROOT, contract, output_dir=output_dir / "build", clang=clang
    )
    dll = Path(conformance["toolchains"]["msvc"]["dll_path"])
    runs = [
        _monitored(contract_path, dll, output_dir / f"worker_{index}.json", 30.0)
        for index in (1, 2)
    ]
    walls = [float(run["worker"]["wall_seconds"]) for run in runs]
    peaks = [int(run["peak_process_tree_rss_bytes"]) for run in runs]
    density_hashes = [str(run["worker"]["density_sha256"]) for run in runs]
    sigma_hashes = [str(run["worker"]["point_sigma_sha256"]) for run in runs]
    exposure_hashes = [str(run["worker"]["exposure_sha256"]) for run in runs]
    limits = contract["performance"]
    wall_ratio = max(walls) / min(walls)
    rss_ratio = max(peaks) / min(peaks)
    gate_results = {
        "conformance": bool(conformance["automatic_pass"]),
        "fresh_exposure_exact": len(set(exposure_hashes)) == 1,
        "fresh_density_exact": len(set(density_hashes)) == 1,
        "fresh_point_sigma_exact": len(set(sigma_hashes)) == 1,
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "maximum_rss": max(peaks) <= int(limits["maximum_process_tree_rss_bytes"]),
        "wall_repeat": wall_ratio <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": rss_ratio <= float(limits["maximum_repeat_rss_ratio"]),
        "finite_positive_outputs": all(
            float(run["worker"]["density_range"][0]) >= 0.0
            and float(run["worker"]["point_sigma_range"][0]) > 0.0
            for run in runs
        ),
        "worker_cleanup": all(
            run["stderr_empty"] and run["surviving_process_count"] == 0
            for run in runs
        ),
    }
    decision = (
        contract["decision_if_pass"]
        if all(gate_results.values())
        else contract["decision_if_fail"]
    )
    report = {
        "schema": "neuro_film.u6_p8bv_native_granularity_amplitude_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "conformance_id": conformance["stable_evidence_id"],
        "performance": {
            "runs": runs,
            "exposure_sha256": exposure_hashes[0],
            "density_sha256": density_hashes[0],
            "point_sigma_sha256": sigma_hashes[0],
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
        "conformance_id": report["conformance_id"],
        "exposure_sha256": exposure_hashes[0],
        "density_sha256": density_hashes[0],
        "point_sigma_sha256": sigma_hashes[0],
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
        default=ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8bv_native_granularity_amplitude_v1",
    )
    parser.add_argument(
        "--clang",
        type=Path,
        default=ROOT
        / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.dll is None or args.result is None:
            parser.error("--worker requires --dll and --result")
        _worker(args.contract, args.dll, args.result)
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
