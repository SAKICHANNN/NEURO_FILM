#!/usr/bin/env python3
"""Run U6.P8CK freestanding Thomas RGB16 PNG conformance."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, BinaryIO

import cv2
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
from scripts.evaluate_u6_p8ca_native_thomas_gauged_sink import _gauge_payload
from scripts.evaluate_u6_p8cj_native_thomas_rgb16_program import _run_program
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rgb16_png_conformance import (
    ByteSink,
    build_and_load,
    load_library,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_gauge_profile import native_gauge_profile_struct
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
)
from src.film_physics.native_thomas_field import NativeThomasFieldProfileV1
from src.preprocess.output_encode import srgb_icc_profile


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8CK JSON must be an object")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8ck_native_thomas_rgb16_png_program_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("language") != "freestanding-c11"
        or candidate.get("format") != "PNG"
        or candidate.get("bit_depth") != 16
        or candidate.get("compression") != "zlib-stored-blocks"
        or candidate.get("idat_payload_bytes") != 65536
        or candidate.get("icc_profile_sha256")
        != hashlib.sha256(srgb_icc_profile()).hexdigest()
        or candidate.get("icc_profile_bytes") != len(srgb_icc_profile())
        or candidate.get("one_final_quantization") is not True
        or candidate.get("full_output_allowed") is not False
        or candidate.get("output_transaction_owned_by_caller") is not True
    ):
        raise RuntimeError("P8CK contract drift")
    for parent in contract["parents"].values():
        path = ROOT / parent["path"]
        if sha256_file(path) != parent["sha256"]:
            raise RuntimeError("P8CK parent hash drift")
        if "required_decision" in parent and (
            _json(path).get("decision") != parent["required_decision"]
        ):
            raise RuntimeError("P8CK parent decision drift")


def _icc_payload(png: bytes) -> bytes:
    offset = 8
    while offset < len(png):
        length = int.from_bytes(png[offset : offset + 4], "big")
        kind = png[offset + 4 : offset + 8]
        payload = png[offset + 8 : offset + 8 + length]
        if kind == b"iCCP":
            _, compressed = payload.split(b"\x00", 1)
            if not compressed or compressed[0] != 0:
                raise RuntimeError("P8CK iCCP compression drift")
            return zlib.decompress(compressed[1:])
        offset += 12 + length
    raise RuntimeError("P8CK PNG has no ICC")


def _profiles_and_exposure(shape: tuple[int, int, int]) -> tuple[Any, Any, Any, Any]:
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    profiles = _profiles(
        _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    gauge = native_gauge_profile_struct(_gauge_payload())
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    return amplitude, profiles, gauge, exposure


def _run_png(
    library: ctypes.CDLL,
    amplitude: Any,
    profiles: tuple[Any, Any, Any],
    gauge: Any,
    exposure: np.ndarray,
    partition: int,
    *,
    target: BinaryIO | None = None,
    fail_on_call: int | None = None,
    workspace_delta: int = 0,
) -> tuple[int, bytes, tuple[float, float, float], int, int]:
    _, height, width = exposure.shape
    required = ctypes.c_size_t()
    status = library.nf_thomas_rgb16_png_f32_workspace_bytes_v1(
        width, partition, ctypes.byref(required)
    )
    if status != 0:
        raise RuntimeError("P8CK workspace request failed")
    workspace = np.empty(required.value, dtype=np.uint8)
    calls = 0
    captured = bytearray()

    @ByteSink
    def sink(_context: int, values: ctypes.POINTER(ctypes.c_uint8), count: int) -> int:
        nonlocal calls
        del _context
        calls += 1
        if fail_on_call is not None and calls == fail_on_call:
            return 0
        payload = ctypes.string_at(values, int(count))
        if target is None:
            captured.extend(payload)
        else:
            target.write(payload)
        return 1

    ProfileArray = NativeThomasFieldProfileV1 * 3
    abi_profiles = ProfileArray(*(profile.as_abi() for profile in profiles))
    means = (ctypes.c_double * 3)(-13.0, -13.0, -13.0)
    apply_status = library.nf_thomas_rgb16_png_f32_apply_v1(
        ctypes.byref(amplitude),
        abi_profiles,
        ctypes.byref(gauge),
        height,
        width,
        partition,
        exposure.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        exposure.size,
        ctypes.c_void_p(workspace.ctypes.data),
        max(0, workspace.nbytes + workspace_delta),
        sink,
        None,
        means,
    )
    return (
        int(apply_status),
        bytes(captured),
        tuple(float(value) for value in means),
        calls,
        workspace.nbytes,
    )


def _decode_rgb16(png: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16 or decoded.ndim != 3:
        raise RuntimeError("P8CK PNG decode failed")
    return np.ascontiguousarray(decoded[:, :, ::-1])


def _conformance(
    contract: dict[str, Any], output_dir: Path, clang: Path
) -> dict[str, Any]:
    builds, libraries = build_and_load(ROOT, output_dir / "build", clang)
    shape = tuple(int(value) for value in contract["conformance"]["shape_chw"])
    amplitude, profiles, gauge, exposure = _profiles_and_exposure(shape)
    rows: dict[str, Any] = {}
    pngs: dict[str, bytes] = {}
    exact = True
    expected_samples: np.ndarray | None = None
    expected_means: tuple[float, float, float] | None = None
    for name, library in libraries.items():
        rgb_status, samples, means, _, _ = _run_program(
            library, amplitude, profiles, gauge, exposure, 128
        )
        first = _run_png(library, amplitude, profiles, gauge, exposure, 128)
        second = _run_png(library, amplitude, profiles, gauge, exposure, 128)
        status, png, png_means, calls, workspace = first
        decoded = _decode_rgb16(png)
        icc = _icc_payload(png)
        row_exact = (
            rgb_status == 0
            and status == 0
            and first == second
            and means == png_means
            and np.array_equal(decoded, samples)
            and icc == srgb_icc_profile()
        )
        exact = exact and row_exact
        pngs[name] = png
        if expected_samples is None:
            expected_samples = samples
            expected_means = means
        else:
            exact = exact and np.array_equal(expected_samples, samples)
            exact = exact and expected_means == means
        invalid_exposure = exposure.copy()
        invalid_exposure[1, -1, -1] = np.nan
        invalid = _run_png(library, amplitude, profiles, gauge, invalid_exposure, 128)
        failed_sink = _run_png(
            library,
            amplitude,
            profiles,
            gauge,
            exposure,
            128,
            fail_on_call=2,
        )
        short = _run_png(
            library,
            amplitude,
            profiles,
            gauge,
            exposure,
            128,
            workspace_delta=-1,
        )
        failure_exact = (
            invalid[0] == 3
            and invalid[1] == b""
            and invalid[2] == (-13.0, -13.0, -13.0)
            and invalid[3] == 0
            and failed_sink[0] == 4
            and failed_sink[2] == (-13.0, -13.0, -13.0)
            and short[0] == 1
            and short[1] == b""
            and short[2] == (-13.0, -13.0, -13.0)
            and short[3] == 0
        )
        exact = exact and failure_exact
        rows[name] = {
            "png_sha256": hashlib.sha256(png).hexdigest(),
            "png_bytes": len(png),
            "decoded_samples_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            "icc_sha256": hashlib.sha256(icc).hexdigest(),
            "raw_mean_hex": [value.hex() for value in png_means],
            "byte_sink_calls": calls,
            "workspace_bytes": workspace,
            "repeat_exact": first == second,
            "sample_exact": np.array_equal(decoded, samples),
            "failure_exact": failure_exact,
        }
    compiler_exact = pngs["msvc"] == pngs["llvm_mingw"]
    exact = exact and compiler_exact
    return {
        "toolchains": builds,
        "rows": rows,
        "compiler_png_byte_exact": compiler_exact,
        "automatic_pass": exact,
    }


def _worker(
    contract_path: Path,
    dll_path: Path,
    png_path: Path,
    result_path: Path,
) -> None:
    contract = _json(contract_path)
    shape = tuple(int(value) for value in contract["performance"]["shape_chw"])
    amplitude, profiles, gauge, exposure = _profiles_and_exposure(shape)
    input_sha = hashlib.sha256(exposure.tobytes()).hexdigest()
    library = load_library(dll_path)
    started = time.perf_counter()
    with png_path.open("wb") as handle:
        status, captured, means, calls, workspace = _run_png(
            library,
            amplitude,
            profiles,
            gauge,
            exposure,
            int(contract["candidate"]["row_partition"]),
            target=handle,
        )
    elapsed = time.perf_counter() - started
    if (
        status != 0
        or captured
        or hashlib.sha256(exposure.tobytes()).hexdigest() != input_sha
    ):
        raise RuntimeError("P8CK worker execution failed")
    result = {
        "schema": "neuro_film.u6_p8ck_native_thomas_rgb16_png_worker.v1",
        "shape_chw": list(shape),
        "input_sha256": input_sha,
        "png_sha256": sha256_file(png_path),
        "png_bytes": png_path.stat().st_size,
        "raw_mean_hex": [value.hex() for value in means],
        "byte_sink_calls": calls,
        "workspace_bytes": workspace,
        "wall_seconds": elapsed,
    }
    result_path.write_bytes(canonical_bytes(result))


def _monitored(
    contract_path: Path,
    dll_path: Path,
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
            "--dll",
            str(dll_path),
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
            raise TimeoutError("P8CK worker exceeded timeout")
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
            "P8CK worker failed:\n"
            + stdout.decode(errors="replace")
            + stderr.decode(errors="replace")
        )
    return {
        "worker": _json(result_path),
        "peak_process_tree_rss_bytes": peak,
        "stderr_empty": not bool(stderr),
        "surviving_process_count": sum(1 for pid in observed if psutil.pid_exists(pid)),
    }


def _stable_payload(report: dict[str, Any]) -> dict[str, Any]:
    stable = copy.deepcopy(report)
    for build in stable["conformance"]["toolchains"].values():
        build.pop("dll_path", None)
    for row in stable["performance"]["runs"]:
        row["worker"].pop("wall_seconds", None)
        row.pop("peak_process_tree_rss_bytes", None)
    stable["performance"].pop("wall_seconds", None)
    stable["performance"].pop("peak_process_tree_rss_bytes", None)
    stable["performance"].pop("wall_repeat_ratio", None)
    stable["performance"].pop("rss_repeat_ratio", None)
    return stable


def evaluate(contract_path: Path, output_dir: Path, llvm: Path) -> dict[str, Any]:
    contract = _json(contract_path)
    _validate_contract(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    conformance = _conformance(
        contract, output_dir / "conformance", llvm / "bin/clang.exe"
    )
    dll = Path(conformance["toolchains"]["msvc"]["dll_path"])
    runs = [
        _monitored(
            contract_path,
            dll,
            output_dir / f"performance_run{index}.png",
            output_dir / f"performance_run{index}.json",
        )
        for index in (1, 2)
    ]
    performance = contract["performance"]
    walls = [float(row["worker"]["wall_seconds"]) for row in runs]
    peaks = [int(row["peak_process_tree_rss_bytes"]) for row in runs]
    identities = [row["worker"]["png_sha256"] for row in runs]
    performance_pass = (
        len(set(identities)) == 1
        and max(walls) <= float(performance["maximum_wall_seconds"])
        and max(peaks) <= int(performance["maximum_process_tree_rss_bytes"])
        and max(walls) / min(walls) <= float(performance["maximum_repeat_wall_ratio"])
        and max(peaks) / min(peaks) <= float(performance["maximum_repeat_rss_ratio"])
        and all(row["stderr_empty"] for row in runs)
        and all(row["surviving_process_count"] == 0 for row in runs)
    )
    passed = bool(conformance["automatic_pass"] and performance_pass)
    report = {
        "schema": "neuro_film.u6_p8ck_native_thomas_rgb16_png_program_report.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "conformance": conformance,
        "performance": {
            "runs": runs,
            "wall_seconds": walls,
            "peak_process_tree_rss_bytes": peaks,
            "wall_repeat_ratio": max(walls) / min(walls),
            "rss_repeat_ratio": max(peaks) / min(peaks),
            "png_sha256": identities[0],
            "png_identity_exact": len(set(identities)) == 1,
            "automatic_pass": performance_pass,
        },
        "automatic_pass": passed,
        "decision": (
            contract["decision_if_pass"] if passed else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable = _stable_payload(report)
    report["stable_evidence_id"] = hashlib.sha256(canonical_bytes(stable)).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8ck_native_thomas_rgb16_png_program_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/experiments/u6_p8ck_native_thomas_rgb16_png_program_v1",
    )
    parser.add_argument(
        "--llvm",
        type=Path,
        default=ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.dll is None or args.png is None or args.result is None:
            raise SystemExit("worker requires --dll, --png and --result")
        _worker(args.contract, args.dll, args.png, args.result)
        return
    report = evaluate(args.contract, args.output_dir, args.llvm)
    target = args.report or args.output_dir / "report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(report))
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={sha256_file(target)}")


if __name__ == "__main__":
    main()
