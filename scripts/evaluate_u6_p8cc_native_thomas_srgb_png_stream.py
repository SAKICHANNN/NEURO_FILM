#!/usr/bin/env python3
"""Run U6.P8CC exact bounded-row sRGB PNG evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import cv2
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
from scripts.evaluate_u6_p8cb_native_thomas_srgb_quantized_sink import (
    _ArraySink,
    _load_quantizer,
)
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
    stream_native_exposure_thomas_rgb_gauged,
    stream_native_exposure_thomas_rgb_quantized,
)
from src.preprocess.output_encode import srgb_icc_profile
from src.preprocess.png_stream import StreamingSrgbPngWriter


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8CC JSON must be an object")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    encoder = contract.get("encoder", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8cc_native_thomas_srgb_png_stream_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("p8cb_arithmetic_and_samples_unchanged") is not True
        or candidate.get("one_final_quantization_only") is not True
        or candidate.get("full_output_allowed") is not False
        or candidate.get("post_quantization_float_processing_allowed") is not False
        or candidate.get("deterministic_png_stream") is not True
        or encoder.get("bit_depths") != [8, 16]
        or encoder.get("icc_profile_sha256")
        != hashlib.sha256(srgb_icc_profile()).hexdigest()
        or encoder.get("icc_profile_bytes") != len(srgb_icc_profile())
    ):
        raise RuntimeError("P8CC contract drift")
    parent = contract["parents"]["p8cb_evidence"]
    payload = _json(ROOT / parent["path"])
    if (
        sha256_file(ROOT / parent["path"]) != parent["sha256"]
        or payload.get("decision") != parent["required_decision"]
    ):
        raise RuntimeError("P8CC P8CB parent drift")


def _icc_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    offset = 8
    while offset < len(raw):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        kind = raw[offset + 4 : offset + 8]
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == b"iCCP":
            _, compressed = payload.split(b"\x00", 1)
            if not compressed or compressed[0] != 0:
                raise RuntimeError("P8CC iCCP compression drift")
            return zlib.decompress(compressed[1:])
        offset += 12 + length
    raise RuntimeError("P8CC PNG has no ICC")


def _encode_stream(
    path: Path,
    *,
    bit_depth: int,
    height: int,
    width: int,
    stream: Any,
) -> tuple[Any, int, int, str]:
    with StreamingSrgbPngWriter(
        path, width=width, height=height, bit_depth=bit_depth
    ) as writer:
        means, workspace, calls = stream(writer.write_rows)
        digest = writer.finish()
    return means, workspace, calls, digest


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
    amplitude_profile = compile_native_granularity_amplitude_profile(
        p4bw, prior_payload
    )
    shape = tuple(int(value) for value in contract["performance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    input_sha = hashlib.sha256(exposure.tobytes()).hexdigest()
    depth = int(contract["performance"]["primary_bit_depth"])
    profiles = _profiles(
        _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    started = time.perf_counter()

    def stream(sink: Any) -> tuple[Any, int, int]:
        return stream_native_exposure_thomas_rgb_quantized(
            load_native_granularity_amplitude_library(amplitude_dll),
            load_native_thomas_rows_library(rows_dll),
            _load_gauge(gauge_dll),
            _load_quantizer(quantizer_dll),
            amplitude_profile,
            profiles,
            native_gauge_profile_struct(_gauge_payload()),
            exposure,
            sink,
            row_partition=int(contract["candidate"]["row_partition"]),
            bit_depth=depth,
        )

    means, workspace, calls, digest = _encode_stream(
        png_path,
        bit_depth=depth,
        height=shape[1],
        width=shape[2],
        stream=stream,
    )
    elapsed = time.perf_counter() - started
    if hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha:
        raise RuntimeError("P8CC mutated input exposure")
    result = {
        "schema": "neuro_film.u6_p8cc_native_thomas_png_worker.v1",
        "shape_chw": list(shape),
        "bit_depth": depth,
        "input_sha256": input_sha,
        "png_sha256": digest,
        "png_bytes": png_path.stat().st_size,
        "icc_sha256": hashlib.sha256(_icc_payload(png_path)).hexdigest(),
        "raw_field_means": list(means),
        "sink_calls": calls,
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
        if time.perf_counter() - started > 45.0:
            for child in root_process.children(recursive=True):
                child.kill()
            root_process.kill()
            raise TimeoutError("P8CC worker exceeded timeout")
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
            "P8CC worker failed:\n"
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
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    shape = tuple(int(value) for value in contract["conformance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    profiles = _profiles(
        _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    gauge_library = _load_gauge(Path(gauge_build["dll_path"]))
    gauge_profile = native_gauge_profile_struct(_gauge_payload())
    display_sink = _ArraySink()
    expected_means, _, _ = stream_native_exposure_thomas_rgb_gauged(
        amplitude_libraries["msvc"],
        row_libraries["msvc"],
        gauge_library,
        amplitude_profile,
        profiles,
        gauge_profile,
        exposure,
        display_sink,
        row_partition=int(contract["candidate"]["row_partition"]),
    )
    display = np.concatenate(display_sink.tiles, axis=0)
    conformance: dict[str, Any] = {}
    png_hashes: list[str] = []
    exact = True
    for name, rows_library in row_libraries.items():
        by_depth: dict[str, Any] = {}
        for depth in contract["encoder"]["bit_depths"]:
            path = output_dir / f"conformance-{name}-{depth}.png"

            def stream(
                sink: Any,
                *,
                toolchain: str = name,
                row_library: Any = rows_library,
                selected_depth: int = depth,
            ) -> tuple[Any, int, int]:
                return stream_native_exposure_thomas_rgb_quantized(
                    amplitude_libraries[toolchain], row_library, gauge_library,
                    quantizers[toolchain], amplitude_profile, profiles, gauge_profile,
                    exposure, sink,
                    row_partition=int(contract["candidate"]["row_partition"]),
                    bit_depth=selected_depth,
                )

            means, _, calls, digest = _encode_stream(
                path,
                bit_depth=depth,
                height=shape[1],
                width=shape[2],
                stream=stream,
            )
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            expected = np.ascontiguousarray(quantize_reference(display, depth))
            sample_exact = decoded is not None and np.array_equal(decoded[..., ::-1], expected)
            icc_exact = _icc_payload(path) == srgb_icc_profile()
            exact = exact and sample_exact and icc_exact and means == expected_means
            png_hashes.append(digest)
            by_depth[str(depth)] = {
                "png_sha256": digest,
                "png_bytes": path.stat().st_size,
                "decoded_samples_exact": sample_exact,
                "icc_exact": icc_exact,
                "raw_field_means": list(means),
                "sink_calls": calls,
            }
        conformance[name] = by_depth

    incomplete = output_dir / "incomplete.png"
    try:
        with StreamingSrgbPngWriter(
            incomplete, width=7, height=3, bit_depth=16
        ) as writer:
            writer.write_rows(0, np.zeros((1, 7, 3), dtype=np.uint16))
            writer.finish()
        cleanup = False
    except ValueError:
        cleanup = not incomplete.exists() and not incomplete.with_suffix(
            ".png.stream.tmp"
        ).exists()

    runs = [
        _monitored(
            contract_path,
            Path(amplitude["toolchains"]["msvc"]["dll_path"]),
            Path(row_builds["msvc"]["dll_path"]),
            Path(gauge_build["dll_path"]),
            quantizer_paths["msvc"],
            output_dir / f"performance-{index}.png",
            output_dir / f"worker-{index}.json",
        )
        for index in (1, 2)
    ]
    walls = [float(row["worker"]["wall_seconds"]) for row in runs]
    peaks = [int(row["peak_process_tree_rss_bytes"]) for row in runs]
    hashes = [str(row["worker"]["png_sha256"]) for row in runs]
    limits = contract["performance"]
    compiler_depth_exact = all(
        conformance["msvc"][str(depth)]["png_sha256"]
        == conformance["llvm_mingw"][str(depth)]["png_sha256"]
        for depth in contract["encoder"]["bit_depths"]
    )
    gates = {
        "exact_decoded_samples_and_icc": bool(exact),
        "compiler_png_identity": compiler_depth_exact,
        "failure_cleanup": cleanup,
        "fresh_png_identity": len(set(hashes)) == 1,
        "maximum_wall": max(walls) <= float(limits["maximum_wall_seconds"]),
        "maximum_rss": max(peaks) <= int(limits["maximum_process_tree_rss_bytes"]),
        "wall_repeat": max(walls) / min(walls) <= float(limits["maximum_repeat_wall_ratio"]),
        "rss_repeat": max(peaks) / min(peaks) <= float(limits["maximum_repeat_rss_ratio"]),
        "worker_cleanup": all(
            row["stderr_empty"] and row["surviving_process_count"] == 0 for row in runs
        ),
    }
    decision = contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"]
    report = {
        "schema": "neuro_film.u6_p8cc_native_thomas_srgb_png_stream_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "row_builds": row_builds,
        "quantizer_builds": quantizer_builds,
        "conformance": conformance,
        "performance": {
            "runs": runs,
            "png_sha256": hashes[0],
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
        "conformance": conformance,
        "png_sha256": hashes[0],
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
        default=ROOT / "configs/u6_p8cc_native_thomas_srgb_png_stream_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8cc_native_thomas_srgb_png_stream_v1",
    )
    parser.add_argument(
        "--llvm",
        type=Path,
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
            args.amplitude_dll,
            args.rows_dll,
            args.gauge_dll,
            args.quantizer_dll,
            args.png,
            args.result,
        )
        if any(value is None for value in values):
            parser.error("--worker requires four DLLs, --png and --result")
        _worker(
            args.contract,
            args.amplitude_dll,
            args.rows_dll,
            args.gauge_dll,
            args.quantizer_dll,
            args.png,
            args.result,
        )
        return
    report = evaluate(args.contract, args.output_dir, args.llvm)
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
