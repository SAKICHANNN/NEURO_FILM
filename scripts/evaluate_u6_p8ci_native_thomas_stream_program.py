#!/usr/bin/env python3
"""Run U6.P8CI freestanding Thomas stream-program conformance."""

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

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _exposure_fixture,
    _parent_payloads,
    _profiles,
)
from src.eval.native_granularity_amplitude_conformance import (
    evaluate_conformance as evaluate_amplitude_conformance,
)
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_stream_conformance import StreamSink, build_and_load
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)
from src.film_physics.native_thomas_rows import (
    _neumaier_rows_native,
    _pointer,
    _workspace,
    load_native_neumaier_library,
    load_native_thomas_rows_library,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P8CI JSON must be an object")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8ci_native_thomas_stream_program_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("language") != "freestanding-c11"
        or candidate.get("field_density_and_reducer_sources_unchanged") is not True
        or candidate.get("row_partitions") != [1, 7, 31, 128]
        or candidate.get("model_profile_or_sample_change_allowed") is not False
        or contract.get("conformance", {}).get("shape_chw") != [3, 193, 257]
    ):
        raise RuntimeError("P8CI contract drift")
    for parent in contract["parents"].values():
        payload = _json(ROOT / parent["path"])
        if (
            sha256_file(ROOT / parent["path"]) != parent["sha256"]
            or payload.get("decision") != parent["required_decision"]
        ):
            raise RuntimeError("P8CI parent drift")


def _amplitude_rows(
    library: ctypes.CDLL,
    profile: Any,
    exposure: np.ndarray,
    channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.ascontiguousarray(exposure[channel])
    density = np.empty(values.size, dtype=np.float32)
    sigma = np.empty(values.size, dtype=np.float32)
    status = library.nf_granularity_amplitude_f32_apply_layer_v1(
        ctypes.byref(profile),
        channel,
        _pointer(values),
        values.size,
        _pointer(density),
        _pointer(sigma),
    )
    if status != 0:
        raise RuntimeError(f"P8CI amplitude failed: {status}")
    return density, sigma


def _direct(
    library: ctypes.CDLL,
    profile: Any,
    base_density: np.ndarray,
    sigma: np.ndarray,
    shape: tuple[int, int],
    partition: int,
) -> tuple[np.ndarray, float, list[int]]:
    height, width = shape
    rows_library = load_native_thomas_rows_library(Path(library._name))
    reducer_library = load_native_neumaier_library(Path(library._name))
    mean = _neumaier_rows_native(
        rows_library, reducer_library, profile, shape, partition
    )
    workspace = _workspace(rows_library, width, min(height, partition))
    tile = np.empty(min(height, partition) * width, dtype=np.float32)
    output = np.empty(height * width, dtype=np.float32)
    abi_profile = profile.as_abi()
    order: list[int] = []
    for row_start in range(0, height, partition):
        rows = min(partition, height - row_start)
        count = rows * width
        offset = row_start * width
        status = rows_library.nf_thomas_rows_f32_density_v1(
            ctypes.byref(abi_profile),
            height,
            width,
            row_start,
            rows,
            _pointer(base_density[offset : offset + count]),
            count,
            _pointer(sigma[offset : offset + count]),
            count,
            mean,
            _pointer(workspace),
            workspace.size,
            _pointer(tile),
            tile.size,
        )
        if status != 0:
            raise RuntimeError(f"P8CI direct row failed: {status}")
        output[offset : offset + count] = tile[:count]
        order.append(row_start)
    return output, mean, order


def _stream(
    library: ctypes.CDLL,
    profile: Any,
    base_density: np.ndarray,
    sigma: np.ndarray,
    shape: tuple[int, int],
    partition: int,
    *,
    fail_on_call: int | None = None,
) -> tuple[int, np.ndarray, float, list[int], int]:
    height, width = shape
    required = ctypes.c_size_t()
    if (
        library.nf_thomas_stream_f32_workspace_floats_v1(
            width, partition, ctypes.byref(required)
        )
        != 0
    ):
        raise RuntimeError("P8CI workspace request failed")
    workspace = np.empty(required.value, dtype=np.float32)
    output = np.full(height * width, np.float32(-17.0))
    order: list[int] = []

    @StreamSink
    def sink(
        _context: int,
        row_start: int,
        row_count: int,
        values: ctypes.POINTER(ctypes.c_float),
        value_count: int,
    ) -> int:
        del _context
        order.append(int(row_start))
        if fail_on_call is not None and len(order) == fail_on_call:
            return 0
        expected_count = int(row_count) * width
        if int(value_count) != expected_count:
            return 0
        view = np.ctypeslib.as_array(values, shape=(expected_count,))
        offset = int(row_start) * width
        output[offset : offset + expected_count] = view
        return 1

    mean = ctypes.c_double(-13.0)
    abi_profile = profile.as_abi()
    status = library.nf_thomas_stream_f32_density_v1(
        ctypes.byref(abi_profile),
        height,
        width,
        partition,
        _pointer(base_density),
        base_density.size,
        _pointer(sigma),
        sigma.size,
        _pointer(workspace),
        workspace.size,
        sink,
        None,
        ctypes.byref(mean),
    )
    return int(status), output, float(mean.value), order, int(workspace.nbytes)


def evaluate(contract_path: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    contract = _json(contract_path)
    _validate_contract(contract)
    amplitude = evaluate_amplitude_conformance(
        ROOT,
        _json(ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"),
        output_dir=output_dir / "amplitude",
        clang=clang,
    )
    builds, libraries = build_and_load(ROOT, output_dir / "stream", clang)
    amplitude_libraries = {
        name: load_native_granularity_amplitude_library(Path(report["dll_path"]))
        for name, report in amplitude["toolchains"].items()
    }
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(
        p4bw, prior_payload
    )
    shape_chw = tuple(int(value) for value in contract["conformance"]["shape_chw"])
    shape = (shape_chw[1], shape_chw[2])
    exposure = _exposure_fixture(prior, shape)
    profile = _profiles(
        _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )[0]
    rows: dict[str, dict[str, Any]] = {}
    exact = bool(amplitude["automatic_pass"])
    maximum_workspace = 0
    for name, library in libraries.items():
        density, sigma = _amplitude_rows(
            amplitude_libraries[name], amplitude_profile, exposure, 0
        )
        rows[name] = {}
        for partition in contract["candidate"]["row_partitions"]:
            direct, direct_mean, direct_order = _direct(
                library, profile, density, sigma, shape, int(partition)
            )
            status, actual, mean, order, workspace = _stream(
                library, profile, density, sigma, shape, int(partition)
            )
            maximum_workspace = max(maximum_workspace, workspace)
            row_exact = (
                status == 0
                and mean == direct_mean
                and order == direct_order
                and np.array_equal(actual, direct)
            )
            exact = exact and row_exact
            rows[name][str(partition)] = {
                "status": status,
                "raw_mean_hex": float(mean).hex(),
                "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "callback_order": order,
                "workspace_bytes": workspace,
                "exact": row_exact,
            }
        bad_sigma = sigma.copy()
        bad_sigma[-1] = np.nan
        invalid = _stream(library, profile, density, bad_sigma, shape, 31)
        failed = _stream(
            library, profile, density, sigma, shape, 31, fail_on_call=2
        )
        rows[name]["failures"] = {
            "invalid_status": invalid[0],
            "invalid_callback_count": len(invalid[3]),
            "invalid_mean_unchanged": invalid[2] == -13.0,
            "callback_status": failed[0],
            "callback_count": len(failed[3]),
            "callback_mean_unchanged": failed[2] == -13.0,
        }
        exact = exact and (
            invalid[0] == 3
            and len(invalid[3]) == 0
            and invalid[2] == -13.0
            and failed[0] == 4
            and len(failed[3]) == 2
            and failed[2] == -13.0
        )
    output_hashes = {
        row[partition]["output_sha256"]
        for row in rows.values()
        for partition in ("1", "7", "31", "128")
    }
    mean_bits = {
        row[partition]["raw_mean_hex"]
        for row in rows.values()
        for partition in ("1", "7", "31", "128")
    }
    gates = {
        "amplitude_conformance": bool(amplitude["automatic_pass"]),
        "msvc_llvm_partition_output_exact": exact and len(output_hashes) == 1,
        "msvc_llvm_partition_mean_exact": len(mean_bits) == 1,
        "strict_callback_order": all(
            row[partition]["callback_order"] == list(range(0, shape[0], int(partition)))
            for row in rows.values()
            for partition in ("1", "7", "31", "128")
        ),
        "invalid_input_before_callback": all(
            row["failures"]["invalid_status"] == 3
            and row["failures"]["invalid_callback_count"] == 0
            and row["failures"]["invalid_mean_unchanged"]
            for row in rows.values()
        ),
        "callback_failure_propagates": all(
            row["failures"]["callback_status"] == 4
            and row["failures"]["callback_count"] == 2
            and row["failures"]["callback_mean_unchanged"]
            for row in rows.values()
        ),
        "workspace_bounded": maximum_workspace > 0,
    }
    decision = (
        contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"]
    )
    report = {
        "schema": "neuro_film.u6_p8ci_native_thomas_stream_program_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "builds": builds,
        "shape_chw": list(shape_chw),
        "conformance": rows,
        "maximum_workspace_bytes": maximum_workspace,
        "output_sha256": min(output_hashes),
        "raw_mean_hex": min(mean_bits),
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
        default=ROOT / "configs/u6_p8ci_native_thomas_stream_program_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8ci_native_thomas_stream_program_v1",
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
