#!/usr/bin/env python3
"""Measure 12MP row-streamed native physical plus AO6 display execution."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p8d_cpu_consumer import _monitor  # noqa: E402
from src.eval.density_witness_frontier import (  # noqa: E402
    linear_srgb_to_encoded,
)
from src.eval.physical_native_ao6_context_conformance import (  # noqa: E402
    _load_context,
    build_msvc_native_ao6_context_dll,
)
from src.eval.physical_native_ao6_display_conformance import (  # noqa: E402
    _load_display,
    build_msvc_native_ao6_display_dll,
)
from src.eval.physical_native_composed_display_conformance import (  # noqa: E402
    _apply_gauge,
    _build_all,
)
from src.eval.physical_native_end_to_end_conformance import (  # noqa: E402
    _apply_display_rows,
)
from src.eval.physical_native_f32_conformance import (  # noqa: E402
    stream_tiled_f32_chain,
)
from src.film_physics.native_adjacency_profile import (  # noqa: E402
    compile_native_adjacency_profile_payload,
)
from src.film_physics.native_ao6_base_profile import (  # noqa: E402
    NativeAo6BaseContextF32V1,
    native_ao6_base_profile_struct,
)
from src.film_physics.native_ao6_context_profile import (  # noqa: E402
    NativeAo6ContextStateF32V1,
)
from src.film_physics.native_ao6_residual_profile import (  # noqa: E402
    native_ao6_residual_profile_struct,
)
from src.film_physics.native_gauge_profile import (  # noqa: E402
    native_gauge_profile_struct,
)
from src.film_physics.native_profile import (  # noqa: E402
    compile_native_domains_profile_payload,
)
from src.film_physics.native_spatial_profile import (  # noqa: E402
    compile_native_gaussian_profile_payload,
)
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
)


SCHEMA = "neuro_film.u6_p8an_native_end_to_end_resources_contract.v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"hash mismatch: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or int(config["tile_rows"]) != 32
        or int(config["scenario"]["repeats"]) != 2
        or not config["execution"]["two_pass_source_context"]
        or not config["execution"]["output_rows_consumed_by_hash_sink"]
    ):
        raise ValueError("unsupported P8AN contract")
    parent = _load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AN"):
        raise ValueError("P8AN parent decision drift")
    _load_exact_json(
        ROOT / config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    return parent


def _source_rows(
    *, y0: int, y1: int, height: int, width: int
) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    rows = np.empty((y1 - y0, width, 3), dtype=np.float32)
    rows[..., 0] = x[None, :]
    rows[..., 1] = y[y0:y1, None]
    rows[..., 2] = (
        np.float32(0.15)
        + np.float32(0.45) * x[None, :]
        + np.float32(0.35) * y[y0:y1, None]
    )
    np.clip(rows, 0.0, 1.0, out=rows)
    return rows


def _worker(
    *,
    profile_config: Path,
    domains_dll: Path,
    gaussian_dll: Path,
    adjacency_dll: Path,
    gauge_dll: Path,
    context_dll: Path,
    display_dll: Path,
    output: Path,
    height: int,
    width: int,
    tile_rows: int,
) -> None:
    config = json.loads(profile_config.read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=config
    )
    payloads = artifact["component_payloads"]
    domains_payload = compile_native_domains_profile_payload(artifact)
    spatial_payload = compile_native_gaussian_profile_payload(artifact)
    adjacency_payload = compile_native_adjacency_profile_payload(artifact)
    gauge_profile = native_gauge_profile_struct(
        payloads["neutral-axis-gauge"]
    )
    display_payload = payloads["ao6-source-context-display-look"]
    base_profile = native_ao6_base_profile_struct(display_payload)
    residual_profile = native_ao6_residual_profile_struct(display_payload)
    context_library = _load_context(context_dll)
    display_library = _load_display(display_dll)

    started = time.perf_counter()
    state = NativeAo6ContextStateF32V1()
    if context_library.nf_ao6_context_f32_init_v1(
        ctypes.byref(state)
    ) != 0:
        raise RuntimeError("P8AN context init failed")
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        linear = _source_rows(
            y0=y0, y1=y1, height=height, width=width
        )
        encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(linear.astype(np.float64)),
            dtype=np.float32,
        )
        status = context_library.nf_ao6_context_f32_update_v1(
            ctypes.byref(base_profile),
            ctypes.byref(state),
            encoded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            encoded.shape[0] * encoded.shape[1],
        )
        if status != 0:
            raise RuntimeError(f"P8AN context update failed: {status}")
    context = NativeAo6BaseContextF32V1()
    if context_library.nf_ao6_context_f32_finalize_v1(
        ctypes.byref(state), ctypes.byref(context)
    ) != 0:
        raise RuntimeError("P8AN context finalize failed")

    digest = hashlib.sha256()
    consumed_rows = 0

    def source_provider(y0: int, y1: int) -> np.ndarray:
        return _source_rows(
            y0=y0, y1=y1, height=height, width=width
        )

    def output_sink(y0: int, y1: int, rows: np.ndarray) -> None:
        nonlocal consumed_rows
        if y0 != consumed_rows or rows.shape != (y1 - y0, width, 3):
            raise RuntimeError("P8AN physical output row order drift")
        gauged = _apply_gauge(
            dll=gauge_dll, profile=gauge_profile, values=rows
        )
        gauged_encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(gauged.astype(np.float64)),
            dtype=np.float32,
        )
        final_rows = _apply_display_rows(
            library=display_library,
            base_profile=base_profile,
            context=context,
            residual_profile=residual_profile,
            encoded=gauged_encoded,
            tile_rows=y1 - y0,
            reverse=False,
        )
        digest.update(final_rows.tobytes())
        consumed_rows = y1

    stream_tiled_f32_chain(
        domains_dll=domains_dll,
        gaussian_dll=gaussian_dll,
        adjacency_dll=adjacency_dll,
        domains_payload=domains_payload,
        spatial_payload=spatial_payload,
        adjacency_payload=adjacency_payload,
        height=height,
        width=width,
        tile_rows=tile_rows,
        source_provider=source_provider,
        output_sink=output_sink,
    )
    elapsed = time.perf_counter() - started
    if consumed_rows != height:
        raise RuntimeError("P8AN did not consume every output row")
    result = {
        "height": height,
        "width": width,
        "pixels": height * width,
        "tile_rows": tile_rows,
        "elapsed_seconds": elapsed,
        "output_sha256": digest.hexdigest(),
        "output_dtype": "float32",
        "output_shape": [height, width, 3],
        "two_pass_source_context": True,
        "full_frame_source_retained": False,
        "full_frame_output_retained": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _build_components(
    config: dict[str, Any], output_dir: Path
) -> dict[str, dict[str, Any]]:
    physical = _build_all(root=ROOT, output_dir=output_dir / "physical")
    builds = {
        "domains": physical["domains"],
        "gaussian": physical["gaussian"],
        "adjacency": physical["adjacency"],
        "gauge": physical["gauge"],
        "context": build_msvc_native_ao6_context_dll(
            root=ROOT, output_dir=output_dir / "context"
        ),
        "display": build_msvc_native_ao6_display_dll(
            root=ROOT, output_dir=output_dir / "display"
        ),
    }
    for name, expected in config["component_dll_sha256"].items():
        if builds[name]["dll_sha256"] != expected:
            raise ValueError(f"P8AN {name} DLL identity drift")
    return builds


def benchmark(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_contract(config)
    available = int(psutil.virtual_memory().available)
    launch_floor = int(
        config["safety"]["minimum_available_memory_bytes_before_launch"]
    )
    if available < launch_floor:
        raise RuntimeError(
            f"available memory {available} is below launch floor {launch_floor}"
        )
    builds = _build_components(config, output_dir / "binaries")
    scenario = config["scenario"]
    runs = []
    script = Path(__file__).resolve()
    for repeat in range(int(scenario["repeats"])):
        worker_output = output_dir / "workers" / f"run_{repeat + 1}.json"
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--profile-config",
            str(ROOT / config["profile_compiler_config"]),
            "--domains-dll",
            builds["domains"]["dll_path"],
            "--gaussian-dll",
            builds["gaussian"]["dll_path"],
            "--adjacency-dll",
            builds["adjacency"]["dll_path"],
            "--gauge-dll",
            builds["gauge"]["dll_path"],
            "--context-dll",
            builds["context"]["dll_path"],
            "--display-dll",
            builds["display"]["dll_path"],
            "--worker-output",
            str(worker_output),
            "--height",
            str(scenario["height"]),
            "--width",
            str(scenario["width"]),
            "--tile-rows",
            str(config["tile_rows"]),
        ]
        run = _monitor(
            command,
            output=worker_output,
            timeout_seconds=int(scenario["timeout_seconds"]),
            maximum_rss=int(config["safety"]["worker_kill_rss_bytes"]),
            interval=float(config["monitor_interval_seconds"]),
        )
        run["repeat"] = repeat + 1
        runs.append(run)
        if not run["success"]:
            break
    successful = len(runs) == 2 and all(run["success"] for run in runs)
    output_hashes = [
        run["worker"]["output_sha256"] for run in runs if run["success"]
    ]
    exact = successful and len(set(output_hashes)) == 1
    peaks = [
        int(run["peak_process_tree_rss_bytes"])
        for run in runs
        if run["success"]
    ]
    elapsed = [
        float(run["worker"]["elapsed_seconds"])
        for run in runs
        if run["success"]
    ]
    median_peak = statistics.median(peaks) if peaks else None
    gates = config["gates"]
    memory_pass = bool(peaks) and max(peaks) <= int(
        gates["maximum_peak_process_tree_rss_bytes"]
    )
    elapsed_pass = bool(elapsed) and max(elapsed) <= float(
        gates["maximum_worker_elapsed_seconds"]
    )
    passed = successful and exact and memory_pass and elapsed_pass
    stable_core = {
        "schema": (
            "neuro_film.u6_p8an_native_end_to_end_resources_result.v1"
        ),
        "scenario": scenario["scenario_id"],
        "height": int(scenario["height"]),
        "width": int(scenario["width"]),
        "tile_rows": int(config["tile_rows"]),
        "component_dll_sha256": {
            name: build["dll_sha256"] for name, build in builds.items()
        },
        "repeat_count": len(runs),
        "all_runs_successful": successful,
        "output_repeat_exact": exact,
        "output_sha256": output_hashes[0] if exact else None,
        "peak_process_tree_rss_bytes": peaks,
        "median_peak_process_tree_rss_bytes": median_peak,
        "memory_gate_bytes": int(
            gates["maximum_peak_process_tree_rss_bytes"]
        ),
        "memory_pass": memory_pass,
        "elapsed_gate_seconds": float(
            gates["maximum_worker_elapsed_seconds"]
        ),
        "elapsed_pass": elapsed_pass,
        "pass": passed,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable_core,
        "worker_elapsed_seconds": elapsed,
        "wall_seconds": [float(run["wall_seconds"]) for run in runs],
        "runs": runs,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_core)
        ).hexdigest(),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8an_native_end_to_end_resources_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8an_native_end_to_end_resources_v1",
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
    if arguments.worker:
        _worker(
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
    report = benchmark(config, output_dir)
    write_report(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
