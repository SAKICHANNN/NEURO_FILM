#!/usr/bin/env python3
"""Run U6.P8CJ freestanding Thomas RGB16 program conformance."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
from src.eval.native_neumaier_conformance import build_and_load as build_reducers
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rgb16_conformance import Rgb16Sink, build_and_load
from src.eval.native_thomas_rows_conformance import build_and_load as build_rows
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
from src.film_physics.native_thomas_field import NativeThomasFieldProfileV1
from src.film_physics.native_thomas_rows import (
    _pointer,
    load_native_neumaier_library,
    load_native_thomas_rows_library,
    stream_native_exposure_thomas_rgb_quantized_parallel,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8CJ JSON must be an object")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8cj_native_thomas_rgb16_program_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("language") != "freestanding-c11"
        or candidate.get("output_layout") != "row-major-interleaved-rgb16"
        or candidate.get("row_partitions") != [1, 7, 31, 128]
        or candidate.get("one_final_quantization") is not True
        or candidate.get("full_output_allowed") is not False
        or candidate.get("model_profile_or_sample_change_allowed") is not False
        or contract.get("conformance", {}).get("shape_chw") != [3, 193, 257]
    ):
        raise RuntimeError("P8CJ contract drift")
    for parent in contract["parents"].values():
        payload = _json(ROOT / parent["path"])
        if (
            sha256_file(ROOT / parent["path"]) != parent["sha256"]
            or payload.get("decision") != parent["required_decision"]
        ):
            raise RuntimeError("P8CJ parent drift")


def _run_program(
    library: ctypes.CDLL,
    amplitude_profile: Any,
    profiles: tuple[Any, Any, Any],
    gauge_profile: Any,
    exposure: np.ndarray,
    partition: int,
    *,
    fail_on_call: int | None = None,
) -> tuple[int, np.ndarray, tuple[float, float, float], list[int], int]:
    _, height, width = exposure.shape
    required = ctypes.c_size_t()
    if (
        library.nf_thomas_rgb16_f32_workspace_bytes_v1(
            width, partition, ctypes.byref(required)
        )
        != 0
    ):
        raise RuntimeError("P8CJ workspace request failed")
    workspace = np.empty(required.value, dtype=np.uint8)
    output = np.full((height, width, 3), np.uint16(0xA5A5), dtype=np.uint16)
    order: list[int] = []

    @Rgb16Sink
    def sink(
        _context: int,
        row_start: int,
        row_count: int,
        values: ctypes.POINTER(ctypes.c_uint16),
        value_count: int,
    ) -> int:
        del _context
        order.append(int(row_start))
        if fail_on_call is not None and len(order) == fail_on_call:
            return 0
        expected = int(row_count) * width * 3
        if int(value_count) != expected:
            return 0
        view = np.ctypeslib.as_array(values, shape=(expected,))
        output[int(row_start) : int(row_start) + int(row_count)] = view.reshape(
            int(row_count), width, 3
        )
        return 1

    ProfileArray = NativeThomasFieldProfileV1 * 3
    abi_profiles = ProfileArray(*(profile.as_abi() for profile in profiles))
    means = (ctypes.c_double * 3)(-13.0, -13.0, -13.0)
    status = library.nf_thomas_rgb16_f32_apply_v1(
        ctypes.byref(amplitude_profile),
        abi_profiles,
        ctypes.byref(gauge_profile),
        height,
        width,
        partition,
        _pointer(exposure),
        exposure.size,
        ctypes.c_void_p(workspace.ctypes.data),
        workspace.nbytes,
        sink,
        None,
        means,
    )
    return (
        int(status),
        output,
        tuple(float(value) for value in means),
        order,
        int(workspace.nbytes),
    )


def evaluate(contract_path: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    contract = _json(contract_path)
    _validate_contract(contract)
    amplitude = evaluate_amplitude_conformance(
        ROOT,
        _json(ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"),
        output_dir=output_dir / "amplitude",
        clang=clang,
    )
    row_builds, _ = build_rows(ROOT, output_dir / "rows", clang)
    reducer_builds, _ = build_reducers(ROOT, output_dir / "reducer", clang)
    gauge_build = build_msvc_native_gauge_dll(
        root=ROOT, output_dir=output_dir / "gauge"
    )
    quantizer_path = output_dir / "quantizer/msvc/quantizer.dll"
    quantizer_build = build_msvc_dll(quantizer_path)
    builds, libraries = build_and_load(ROOT, output_dir / "rgb16", clang)
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(
        p4bw, prior_payload
    )
    profiles = _profiles(
        _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    gauge_profile = native_gauge_profile_struct(_gauge_payload())
    shape_chw = tuple(int(value) for value in contract["conformance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape_chw[1], shape_chw[2]))
    expected_tiles: list[np.ndarray] = []
    expected_means, _, _ = stream_native_exposure_thomas_rgb_quantized_parallel(
        load_native_granularity_amplitude_library(
            Path(amplitude["toolchains"]["msvc"]["dll_path"])
        ),
        load_native_thomas_rows_library(Path(row_builds["msvc"]["dll_path"])),
        _load_gauge(Path(gauge_build["dll_path"])),
        _load_quantizer(quantizer_path),
        amplitude_profile,
        profiles,
        gauge_profile,
        exposure,
        lambda _row_start, tile: expected_tiles.append(tile.copy()),
        row_partition=128,
        bit_depth=16,
        reducer_library=load_native_neumaier_library(
            Path(reducer_builds["msvc"]["dll_path"])
        ),
    )
    expected = np.concatenate(expected_tiles, axis=0)
    rows: dict[str, dict[str, Any]] = {}
    exact = bool(amplitude["automatic_pass"])
    maximum_workspace = 0
    for name, library in libraries.items():
        rows[name] = {}
        for partition in contract["candidate"]["row_partitions"]:
            status, actual, means, order, workspace = _run_program(
                library,
                amplitude_profile,
                profiles,
                gauge_profile,
                exposure,
                int(partition),
            )
            maximum_workspace = max(maximum_workspace, workspace)
            row_exact = (
                status == 0
                and means == expected_means
                and np.array_equal(actual, expected)
            )
            exact = exact and row_exact
            rows[name][str(partition)] = {
                "status": status,
                "raw_mean_hex": [value.hex() for value in means],
                "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "callback_order": order,
                "workspace_bytes": workspace,
                "exact": row_exact,
            }
        invalid_exposure = exposure.copy()
        invalid_exposure[2, -1, -1] = np.nan
        invalid = _run_program(
            library,
            amplitude_profile,
            profiles,
            gauge_profile,
            invalid_exposure,
            31,
        )
        failed = _run_program(
            library,
            amplitude_profile,
            profiles,
            gauge_profile,
            exposure,
            31,
            fail_on_call=2,
        )
        rows[name]["failures"] = {
            "invalid_status": invalid[0],
            "invalid_callback_count": len(invalid[3]),
            "invalid_means_unchanged": invalid[2] == (-13.0, -13.0, -13.0),
            "callback_status": failed[0],
            "callback_count": len(failed[3]),
            "callback_means_unchanged": failed[2] == (-13.0, -13.0, -13.0),
        }
        exact = exact and (
            invalid[0] == 3
            and len(invalid[3]) == 0
            and invalid[2] == (-13.0, -13.0, -13.0)
            and failed[0] == 4
            and len(failed[3]) == 2
            and failed[2] == (-13.0, -13.0, -13.0)
        )
    partitions = ("1", "7", "31", "128")
    output_hashes = {
        row[partition]["output_sha256"]
        for row in rows.values()
        for partition in partitions
    }
    mean_receipts = {
        tuple(row[partition]["raw_mean_hex"])
        for row in rows.values()
        for partition in partitions
    }
    expected_order = {
        partition: list(range(0, shape_chw[1], int(partition)))
        for partition in partitions
    }
    gates = {
        "amplitude_conformance": bool(amplitude["automatic_pass"]),
        "msvc_llvm_p8cb_samples_exact": exact and len(output_hashes) == 1,
        "msvc_llvm_raw_mean_bits_exact": len(mean_receipts) == 1,
        "partition_exact": all(
            row[partition]["exact"]
            for row in rows.values()
            for partition in partitions
        ),
        "strict_callback_order": all(
            row[partition]["callback_order"] == expected_order[partition]
            for row in rows.values()
            for partition in partitions
        ),
        "invalid_input_before_callback": all(
            row["failures"]["invalid_status"] == 3
            and row["failures"]["invalid_callback_count"] == 0
            and row["failures"]["invalid_means_unchanged"]
            for row in rows.values()
        ),
        "callback_failure_propagates": all(
            row["failures"]["callback_status"] == 4
            and row["failures"]["callback_count"] == 2
            and row["failures"]["callback_means_unchanged"]
            for row in rows.values()
        ),
        "workspace_bounded": maximum_workspace > 0,
    }
    decision = (
        contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"]
    )
    report = {
        "schema": "neuro_film.u6_p8cj_native_thomas_rgb16_program_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "builds": builds,
        "component_builds": {
            "amplitude": amplitude["toolchains"],
            "rows": row_builds,
            "reducer": reducer_builds,
            "gauge": gauge_build,
            "quantizer": quantizer_build,
        },
        "shape_chw": list(shape_chw),
        "conformance": rows,
        "maximum_workspace_bytes": maximum_workspace,
        "output_sha256": min(output_hashes),
        "raw_mean_hex": list(min(mean_receipts)),
        "gate_results": gates,
        "automatic_pass": all(gates.values()),
        "decision": decision,
        "claim_ceiling": contract["claim_ceiling"],
    }
    identity = {
        "experiment_id": report["experiment_id"],
        "contract_sha256": report["contract_sha256"],
        "source_sha256": builds["msvc"]["source_sha256"],
        "header_sha256": builds["msvc"]["header_sha256"],
        "output_sha256": report["output_sha256"],
        "raw_mean_hex": report["raw_mean_hex"],
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
        default=ROOT / "configs/u6_p8cj_native_thomas_rgb16_program_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8cj_native_thomas_rgb16_program_v1",
    )
    parser.add_argument(
        "--clang",
        type=Path,
        default=ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe",
    )
    args = parser.parse_args()
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
