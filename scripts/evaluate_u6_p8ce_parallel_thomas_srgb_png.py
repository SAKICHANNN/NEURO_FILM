#!/usr/bin/env python3
"""Run U6.P8CE exact parallel Thomas-to-PNG evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_srgb_oetf_quantize_native_v1 import build_msvc_dll
from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _exposure_fixture,
    _parent_payloads,
    _profiles,
)
from scripts.evaluate_u6_p8ca_native_thomas_gauged_sink import _gauge_payload
from scripts.evaluate_u6_p8cb_native_thomas_srgb_quantized_sink import _load_quantizer
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
    load_native_thomas_rows_library,
    stream_native_exposure_thomas_rgb_quantized_parallel,
)
from src.preprocess.output_encode import srgb_icc_profile_sha256
from src.preprocess.png_stream import StreamingSrgbPngWriter


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8CE JSON must be an object")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8ce_parallel_thomas_srgb_png_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("p8cc_encoder_unchanged") is not True
        or candidate.get("p8cd_scheduler_unchanged") is not True
        or candidate.get("full_output_allowed") is not False
        or candidate.get("model_profile_or_sample_change_allowed") is not False
    ):
        raise RuntimeError("P8CE contract drift")
    for parent in contract["parents"].values():
        payload = _json(ROOT / parent["path"])
        if (
            sha256_file(ROOT / parent["path"]) != parent["sha256"]
            or payload.get("decision") != parent["required_decision"]
        ):
            raise RuntimeError("P8CE parent drift")


def _worker(
    contract_path: Path,
    amplitude_dll: Path,
    rows_dll: Path,
    gauge_dll: Path,
    quantizer_dll: Path,
    png_path: Path,
    result_path: Path,
) -> None:
    contract = _json(contract_path)
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    shape = tuple(int(value) for value in contract["performance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    input_sha = hashlib.sha256(exposure.tobytes()).hexdigest()
    depth = int(contract["performance"]["primary_bit_depth"])
    profiles = _profiles(_json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json"))
    started = time.perf_counter()
    with StreamingSrgbPngWriter(
        png_path, width=shape[2], height=shape[1], bit_depth=depth
    ) as writer:
        means, workspace, calls = stream_native_exposure_thomas_rgb_quantized_parallel(
            load_native_granularity_amplitude_library(amplitude_dll),
            load_native_thomas_rows_library(rows_dll),
            _load_gauge(gauge_dll),
            _load_quantizer(quantizer_dll),
            amplitude_profile,
            profiles,
            native_gauge_profile_struct(_gauge_payload()),
            exposure,
            writer.write_rows,
            row_partition=int(contract["candidate"]["row_partition"]),
            bit_depth=depth,
        )
        digest = writer.finish()
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("P8CE mutated input exposure")
    result = {
        "schema": "neuro_film.u6_p8ce_parallel_png_worker.v1",
        "shape_chw": list(shape),
        "bit_depth": depth,
        "input_sha256": input_sha,
        "png_sha256": digest,
        "png_bytes": png_path.stat().st_size,
        "icc_sha256": srgb_icc_profile_sha256(),
        "raw_field_means": list(means),
        "sink_calls": calls,
        "bounded_compute_workspace_bytes": workspace,
        "wall_seconds": elapsed,
    }
    result_path.write_bytes(canonical_bytes(result))


def _monitored(
    contract_path: Path,
    amplitude_dll: Path,
    rows_dll: Path,
    gauge_dll: Path,
    quantizer_dll: Path,
    png_path: Path,
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
            "--png",
            str(png_path),
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
        if time.perf_counter() - started > 40.0:
            for child in root_process.children(recursive=True):
                child.kill()
            root_process.kill()
            raise TimeoutError("P8CE worker exceeded timeout")
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
            "P8CE worker failed:\n"
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
    row_builds, _ = build_and_load(ROOT, output_dir / "rows", llvm / "bin/clang.exe")
    gauge_build = build_msvc_native_gauge_dll(root=ROOT, output_dir=output_dir / "gauge")
    quantizer_path = output_dir / "quantizer/msvc/quantizer.dll"
    quantizer_build = build_msvc_dll(quantizer_path)
    runs = [
        _monitored(
            contract_path,
            Path(amplitude["toolchains"]["msvc"]["dll_path"]),
            Path(row_builds["msvc"]["dll_path"]),
            Path(gauge_build["dll_path"]),
            quantizer_path,
            output_dir / f"performance-{index}.png",
            output_dir / f"worker-{index}.json",
        )
        for index in (1, 2)
    ]
    walls = [float(row["worker"]["wall_seconds"]) for row in runs]
    peaks = [int(row["peak_process_tree_rss_bytes"]) for row in runs]
    hashes = [str(row["worker"]["png_sha256"]) for row in runs]
    p8cc = contract["parents"]["p8cc_evidence"]
    limits = contract["performance"]
    ratio = max(walls) / float(p8cc["reference_maximum_wall_seconds"])
    gates = {
        "exact_p8cc_png_identity": set(hashes) == {p8cc["reference_png_sha256"]},
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "material_wall_reduction": ratio <= float(limits["maximum_wall_ratio_vs_p8cc"]),
        "maximum_rss": max(peaks) <= int(limits["maximum_process_tree_rss_bytes"]),
        "wall_repeat": max(walls) / min(walls) <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": max(peaks) / min(peaks) <= float(limits["maximum_repeat_rss_ratio"]),
        "worker_cleanup": all(
            row["stderr_empty"] and row["surviving_process_count"] == 0 for row in runs
        ),
    }
    decision = contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"]
    report = {
        "schema": "neuro_film.u6_p8ce_parallel_thomas_srgb_png_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "row_builds": row_builds,
        "quantizer_build": quantizer_build,
        "performance": {
            "runs": runs,
            "png_sha256": hashes[0],
            "maximum_wall_seconds": max(walls),
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "wall_ratio_vs_p8cc": ratio,
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
        "png_sha256": hashes[0],
        "gate_results": gates,
        "decision": decision,
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/u6_p8ce_parallel_thomas_srgb_png_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/eval/u6_p8ce_parallel_thomas_srgb_png_v1",
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
    parser.add_argument("--png", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        values = (
            args.amplitude_dll, args.rows_dll, args.gauge_dll,
            args.quantizer_dll, args.png, args.result,
        )
        if any(value is None for value in values):
            parser.error("--worker requires four DLLs, --png and --result")
        _worker(
            args.contract, args.amplitude_dll, args.rows_dll, args.gauge_dll,
            args.quantizer_dll, args.png, args.result,
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
