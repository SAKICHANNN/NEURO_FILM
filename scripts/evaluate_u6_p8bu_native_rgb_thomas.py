"""Run U6.P8BU serial RGB conformance and 12MP resource evidence."""

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

from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_density_conformance import evaluate_conformance
from src.eval.native_thomas_field_conformance import (
    canonical_bytes,
    profile_from_contract,
)
from src.film_physics.native_thomas_density import (
    load_native_thomas_density_library,
    render_native_thomas_rgb_transmittance,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8BU JSON must be an object")
    return payload


def _profiles(contract: dict[str, Any]) -> tuple[Any, Any, Any]:
    p8bs = _json(ROOT / "configs/u6_p8bs_native_thomas_field_v1.json")
    template = profile_from_contract(p8bs)
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


def _rgb_fixture(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    base = np.empty((3, height, width), dtype=np.float32)
    sigma = np.empty_like(base)
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    for row in range(height):
        y = (row + 0.5) / height
        base_row = (
            0.45
            + 0.60 * x
            + 0.25 * y
            + 0.05 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        )
        sigma_row = 0.010 + 0.006 * x + 0.004 * y
        base[0, row] = base_row
        base[1, row] = base_row + 0.1
        base[2, row] = base_row + 0.2
        sigma[0, row] = sigma_row
        sigma[1, row] = 0.9 * sigma_row
        sigma[2, row] = 1.1 * sigma_row
    return base, sigma


def _worker(contract_path: Path, dll: Path, result_path: Path) -> None:
    contract = _json(contract_path)
    shape = tuple(int(value) for value in contract["performance"]["shape"])
    base, sigma = _rgb_fixture(shape)
    library = load_native_thomas_density_library(dll)
    started = time.perf_counter()
    output, raw_means, reused_bytes = render_native_thomas_rgb_transmittance(
        library, _profiles(contract), base, sigma
    )
    elapsed = time.perf_counter() - started
    result = {
        "schema": "neuro_film.u6_p8bu_native_rgb_thomas_worker.v1",
        "shape_chw": list(output.shape),
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "channel_sha256": [
            hashlib.sha256(output[index].tobytes()).hexdigest() for index in range(3)
        ],
        "minimum_transmittance": float(np.min(output)),
        "maximum_transmittance": float(np.max(output)),
        "raw_field_means": list(raw_means),
        "reused_workspace_and_scratch_bytes": reused_bytes,
        "input_bytes": base.nbytes + sigma.nbytes,
        "output_bytes": output.nbytes,
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
            raise TimeoutError("P8BU worker exceeded timeout")
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
            "P8BU worker failed:\n"
            + stdout.decode(errors="replace")
            + stderr.decode(errors="replace")
        )
    return {
        "worker": _json(result_path),
        "peak_process_tree_rss_bytes": peak,
        "stderr_empty": not bool(stderr),
        "surviving_process_count": sum(
            1 for pid in observed if psutil.pid_exists(pid)
        ),
    }


def evaluate(contract_path: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    contract = _json(contract_path)
    p8bt_contract_path = ROOT / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"
    p8bt_contract = _json(p8bt_contract_path)
    conformance = evaluate_conformance(
        ROOT, p8bt_contract, output_dir=output_dir / "build", clang=clang
    )
    dll = Path(conformance["toolchains"]["msvc"]["dll_path"])
    runs = [
        _monitored(contract_path, dll, output_dir / f"worker_{index}.json", 30.0)
        for index in (1, 2)
    ]
    walls = [float(run["worker"]["wall_seconds"]) for run in runs]
    peaks = [int(run["peak_process_tree_rss_bytes"]) for run in runs]
    hashes = [str(run["worker"]["output_sha256"]) for run in runs]
    channel_hashes = [tuple(run["worker"]["channel_sha256"]) for run in runs]
    limits = contract["performance"]
    wall_ratio = max(walls) / min(walls)
    rss_ratio = max(peaks) / min(peaks)
    gate_results = {
        "p8bt_conformance": bool(conformance["automatic_pass"]),
        "fresh_output_exact": len(set(hashes)) == 1,
        "channel_output_exact": len(set(channel_hashes)) == 1,
        "distinct_layers": len(set(channel_hashes[0])) == 3,
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "maximum_rss": max(peaks)
        <= int(limits["maximum_process_tree_rss_bytes"]),
        "wall_repeat": wall_ratio <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": rss_ratio <= float(limits["maximum_repeat_rss_ratio"]),
        "transmittance_domain": all(
            float(run["worker"]["minimum_transmittance"]) > 0.0
            and float(run["worker"]["maximum_transmittance"]) <= 1.0
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
    stable = {
        "schema": "neuro_film.u6_p8bu_native_rgb_thomas_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "p8bt_conformance_id": conformance["stable_evidence_id"],
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
        "experiment_id": stable["experiment_id"],
        "contract_sha256": stable["contract_sha256"],
        "p8bt_conformance_id": stable["p8bt_conformance_id"],
        "output_sha256": stable["performance"]["output_sha256"],
        "channel_sha256": stable["performance"]["channel_sha256"],
        "gate_results": gate_results,
        "decision": decision,
    }
    stable_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8bu_native_rgb_thomas_serial_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8bu_native_rgb_thomas_v1",
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
