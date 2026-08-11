#!/usr/bin/env python3
"""Run U6.P8CT 12MP profile-bound atomic native PNG evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_u6_p8ck_native_thomas_rgb16_png_program import _icc_payload
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_atomic_publication import _compile_profile, _json
from src.eval.native_thomas_export_profile import _configure_parallel
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rgb16_png_conformance import build_msvc, load_library
from src.film_physics.atomic_native_output import publish_native_thomas_rgb16_png
from src.preprocess.output_encode import srgb_icc_profile


class NativeThomasAtomicScaleError(RuntimeError):
    """Raised when the P8CT contract or execution drifts."""


def _validate_contract(contract_path: Path) -> dict[str, Any]:
    contract = _json(contract_path)
    fixture = contract.get("fixture", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8ct_thomas_atomic_publication_scale_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or fixture.get("height") != 3000
        or fixture.get("width") != 4000
        or fixture.get("row_partition") != 128
        or fixture.get("runs") != 2
        or fixture.get("maximum_output_bytes") != 80000000
    ):
        raise NativeThomasAtomicScaleError("P8CT contract drift")
    parent = contract["parent"]
    parent_path = ROOT / parent["path"]
    payload = _json(parent_path)
    if (
        sha256_file(parent_path) != parent["sha256"]
        or payload.get("decision") != parent["required_decision"]
    ):
        raise NativeThomasAtomicScaleError("P8CT parent drift")
    return contract


def _exposure_fixture(height: int, width: int) -> np.ndarray:
    """Build the exact P8CN code fixture with only one row-sized temporary."""
    exposure = np.empty((3, height, width), dtype=np.float32)
    columns = np.arange(width, dtype=np.uint64)
    for channel in range(3):
        for row in range(height):
            indices = columns + np.uint64(row * width)
            codes = ((indices * np.uint64(17) + np.uint64(channel * 13)) % 193).astype(
                np.int16
            )
            exposure[channel, row] = (codes - 96).astype(np.float32) / np.float32(64.0)
    return exposure


def _worker(
    contract_path: Path, dll_path: Path, destination: Path, result_path: Path
) -> None:
    contract = _validate_contract(contract_path)
    fixture = contract["fixture"]
    _prior, amplitude, fields, gauge = _compile_profile(ROOT, contract)
    exposure = _exposure_fixture(int(fixture["height"]), int(fixture["width"]))
    input_sha256 = hashlib.sha256(exposure.tobytes()).hexdigest()
    library = load_library(dll_path)
    _configure_parallel(library)
    started = time.perf_counter()
    published = publish_native_thomas_rgb16_png(
        library,
        amplitude,
        fields,
        gauge,
        exposure,
        row_partition=int(fixture["row_partition"]),
        destination=destination,
        maximum_output_bytes=int(fixture["maximum_output_bytes"]),
    )
    wall_seconds = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha256:
        raise NativeThomasAtomicScaleError("P8CT input mutated")
    result_path.write_bytes(
        canonical_bytes(
            {
                "schema": "neuro_film.u6_p8ct_thomas_atomic_publication_worker.v1",
                "input_sha256": input_sha256,
                "output_sha256": published["sha256"],
                "output_bytes": published["bytes"],
                "workspace_bytes": published["workspace_bytes"],
                "raw_field_means": published["raw_field_means"],
                "wall_seconds": wall_seconds,
            }
        )
    )


def _monitored(command: list[str], result_path: Path, timeout: float) -> dict[str, Any]:
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    monitored = psutil.Process(process.pid)
    observed: set[int] = set()
    peak = 0
    started = time.perf_counter()
    while process.poll() is None:
        if time.perf_counter() - started > timeout:
            for child in monitored.children(recursive=True):
                child.kill()
            monitored.kill()
            raise TimeoutError("P8CT worker exceeded timeout")
        try:
            total = 0
            for item in [monitored, *monitored.children(recursive=True)]:
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
        raise NativeThomasAtomicScaleError(
            "P8CT worker failed:\n"
            + stdout.decode(errors="replace")
            + stderr.decode(errors="replace")
        )
    return {
        "worker": _json(result_path),
        "peak_process_tree_rss_bytes": peak,
        "stderr_empty": not bool(stderr),
        "surviving_process_count": sum(1 for pid in observed if psutil.pid_exists(pid)),
    }


def evaluate(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = _validate_contract(contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    build = build_msvc(ROOT, output_dir / "build")
    runs: list[dict[str, Any]] = []
    paths: list[Path] = []
    for index in range(1, int(contract["fixture"]["runs"]) + 1):
        destination = (output_dir / f"run-{index}.png").resolve()
        result_path = output_dir / f"worker-{index}.json"
        destination.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        paths.append(destination)
        runs.append(
            _monitored(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--contract",
                    str(contract_path.resolve()),
                    "--dll",
                    str(Path(build["dll_path"]).resolve()),
                    "--destination",
                    str(destination),
                    "--result",
                    str(result_path.resolve()),
                ],
                result_path,
                timeout=max(45.0, float(contract["gates"]["maximum_wall_seconds"]) + 20.0),
            )
        )
    decoded_hashes: list[str] = []
    icc_hashes: list[str] = []
    for path in paths:
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.shape != (3000, 4000, 3) or decoded.dtype != np.uint16:
            raise NativeThomasAtomicScaleError("P8CT output decode drift")
        decoded_hashes.append(hashlib.sha256(decoded[..., ::-1].tobytes()).hexdigest())
        icc_hashes.append(hashlib.sha256(_icc_payload(path.read_bytes())).hexdigest())
    workers = [row["worker"] for row in runs]
    walls = [float(row["wall_seconds"]) for row in workers]
    peaks = [int(row["peak_process_tree_rss_bytes"]) for row in runs]
    gates_config = contract["gates"]
    stage_residue = list(output_dir.glob(".*.stage"))
    gates = {
        "exact_output_replay": len({row["output_sha256"] for row in workers}) == 1,
        "exact_input_replay": len({row["input_sha256"] for row in workers}) == 1,
        "exact_decoded_replay": len(set(decoded_hashes)) == 1,
        "icc_exact": len(set(icc_hashes)) == 1
        and icc_hashes[0] == hashlib.sha256(srgb_icc_profile()).hexdigest(),
        "maximum_wall": max(walls) <= float(gates_config["maximum_wall_seconds"]),
        "maximum_rss": max(peaks)
        <= int(gates_config["maximum_process_tree_rss_bytes"]),
        "wall_repeat": max(walls) / min(walls)
        <= float(gates_config["maximum_repeat_wall_ratio"]),
        "rss_repeat": max(peaks) / min(peaks)
        <= float(gates_config["maximum_repeat_rss_ratio"]),
        "no_stage_residue": not stage_residue,
        "worker_cleanup": all(
            row["stderr_empty"] and row["surviving_process_count"] == 0
            for row in runs
        ),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "profile_sha256": "923a985aa90e029332c723a33afb793afe05cae777b3c07061d54828b91e21b4",
        "input_sha256": workers[0]["input_sha256"],
        "output_sha256": workers[0]["output_sha256"],
        "output_bytes": workers[0]["output_bytes"],
        "decoded_rgb16_sha256": decoded_hashes[0],
        "icc_sha256": icc_hashes[0],
        "maximum_wall_seconds": max(walls),
        "maximum_peak_process_tree_rss_bytes": max(peaks),
        "wall_repeat_ratio": max(walls) / min(walls),
        "rss_repeat_ratio": max(peaks) / min(peaks),
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p8ct_thomas_atomic_publication_scale_report.v1",
        "experiment_id": contract["experiment_id"],
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest(),
        "stable": stable,
        "runs": runs,
        "build": {key: value for key, value in build.items() if key != "dll_path"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8ct_thomas_atomic_publication_scale_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8ct_thomas_atomic_publication_scale_v1",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.dll is None or args.destination is None or args.result is None:
            parser.error("--worker requires --dll, --destination and --result")
        _worker(args.contract, args.dll, args.destination, args.result)
        return
    report = evaluate(args.contract, args.output_dir)
    report_path = args.report or args.output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_bytes(report))
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["stable"]["decision"],
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
