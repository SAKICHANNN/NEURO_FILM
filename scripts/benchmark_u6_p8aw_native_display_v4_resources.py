#!/usr/bin/env python3
"""Apply the frozen resource benchmark to exact AO6 display v4."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
from src.eval.physical_native_ao6_display_v4_conformance import (  # noqa: E402
    _load_display_v4,
    build_msvc_native_ao6_display_v4_dll,
)


SCHEMA = "neuro_film.u6_p8aw_native_display_v4_resources_contract.v1"
RESULT_SCHEMA = (
    "neuro_film.u6_p8aw_native_display_v4_resources_result.v1"
)


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or int(config["tile_rows"]) != 32
        or int(config["scenario"]["repeats"]) != 2
        or not config["execution"]["two_pass_source_context"]
    ):
        raise ValueError("unsupported P8AW contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AW"):
        raise ValueError("P8AW parent decision drift")
    frozen = p8aq._load_exact_json(
        ROOT / config["frozen_resource_contract"],
        config["frozen_resource_contract_sha256"],
    )
    if (
        frozen["scenario"] != config["scenario"]
        or frozen["tile_rows"] != config["tile_rows"]
        or frozen["safety"] != config["safety"]
        or frozen["gates"] != config["gates"]
        or frozen["profile_compiler_config"]
        != config["profile_compiler_config"]
        or frozen["profile_compiler_config_sha256"]
        != config["profile_compiler_config_sha256"]
        or frozen["expected_output_sha256"]
        != config["expected_output_sha256"]
    ):
        raise ValueError("P8AW frozen resource gates drift")
    p8aq._load_exact_json(
        ROOT / config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    frozen_components = frozen["component_dll_sha256"]
    for name in ("domains", "gaussian", "adjacency", "gauge", "context"):
        if config["component_dll_sha256"].get(name) != frozen_components.get(
            name
        ):
            raise ValueError(f"P8AW frozen {name} identity drift")
    if (
        config["component_dll_sha256"].get("display")
        != "a8a4aede80996d96f72ee805f4ae38d23caef3df786c0f4138a2ee5a37571ce9"
    ):
        raise ValueError("P8AW display v4 identity drift")
    return parent


def _apply_display_v4_rows(
    *,
    library: ctypes.CDLL,
    base_profile: Any,
    context: Any,
    residual_profile: Any,
    encoded: np.ndarray,
) -> np.ndarray:
    flat = np.ascontiguousarray(encoded.reshape(-1, 3))
    scratch = np.empty_like(flat)
    output = np.empty_like(flat)
    status = library.nf_ao6_display_f32_apply_v4(
        ctypes.byref(base_profile),
        ctypes.byref(context),
        ctypes.byref(residual_profile),
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        flat.shape[0],
        scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        raise RuntimeError(f"P8AW AO6 display v4 failed: {status}")
    return output.reshape(encoded.shape)


def _patch_runtime() -> None:
    p8aq.validate_contract = validate_contract
    p8aq.build_msvc_native_ao6_display_v2_dll = (
        build_msvc_native_ao6_display_v4_dll
    )
    p8aq._load_display_v2 = _load_display_v4
    p8aq._apply_display_v2_rows = _apply_display_v4_rows
    p8aq.__file__ = str(Path(__file__).resolve())


def _relabel_report(report: dict[str, Any]) -> dict[str, Any]:
    report["schema"] = RESULT_SCHEMA
    stable = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "worker_elapsed_seconds",
            "wall_seconds",
            "runs",
            "stable_evidence_id",
        }
    }
    report["stable_evidence_id"] = hashlib.sha256(
        p8aq._canonical_bytes(stable)
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8aw_native_display_v4_resources_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8aw_native_display_v4_resources_v1",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile-config")
    parser.add_argument("--domains-dll")
    parser.add_argument("--gaussian-dll")
    parser.add_argument("--adjacency-dll")
    parser.add_argument("--gauge-dll")
    parser.add_argument("--context-dll")
    parser.add_argument("--display-dll")
    parser.add_argument("--worker-output")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--tile-rows", type=int)
    arguments = parser.parse_args()
    _patch_runtime()
    if arguments.worker:
        p8aq._worker(
            profile_config=Path(arguments.profile_config),
            domains_dll=Path(arguments.domains_dll),
            gaussian_dll=Path(arguments.gaussian_dll),
            adjacency_dll=Path(arguments.adjacency_dll),
            gauge_dll=Path(arguments.gauge_dll),
            context_dll=Path(arguments.context_dll),
            display_dll=Path(arguments.display_dll),
            output=Path(arguments.worker_output),
            height=arguments.height,
            width=arguments.width,
            tile_rows=arguments.tile_rows,
        )
        return
    config = json.loads((ROOT / arguments.config).read_text())
    output_dir = ROOT / arguments.output_dir
    report = _relabel_report(p8aq.benchmark(config, output_dir))
    p8aq.write_report(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
