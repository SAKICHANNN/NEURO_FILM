#!/usr/bin/env python3
"""Select a bounded ordered tile-pipeline width without changing pixels."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
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

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
from src.film_physics.bounded_ordered_pipeline import (  # noqa: E402
    BoundedOrderedPipeline,
)
from src.eval.density_witness_frontier import (  # noqa: E402
    linear_srgb_to_encoded,
)
from src.eval.physical_native_ao6_fastpath_conformance import (  # noqa: E402
    _load_context_v2,
)
from src.eval.physical_native_gauge_conformance import (  # noqa: E402
    _load_gauge,
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


SCHEMA = "neuro_film.u6_p8ax_native_ordered_pipeline_grid_contract.v1"
RESULT_SCHEMA = (
    "neuro_film.u6_p8ax_native_ordered_pipeline_grid_result.v1"
)


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or config.get("pipeline_workers_candidates") != [1, 2, 4, 8]
        or config.get("execution_order") != [1, 2, 4, 8, 8, 4, 2, 1]
        or int(config["scenario"]["height"]) != 1024
        or int(config["scenario"]["width"]) != 2048
        or int(config["scenario"]["repeats_per_policy"]) != 2
        or int(config["tile_rows"]) != 32
        or int(config["max_in_flight_per_worker"]) != 1
    ):
        raise ValueError("unsupported P8AX contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AX"):
        raise ValueError("P8AX parent decision drift")
    p8aq._load_exact_json(
        ROOT / config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    p8aw_config = p8aq._load_exact_json(
        ROOT / config["p8aw_contract"],
        config["p8aw_contract_sha256"],
    )
    if (
        config["component_dll_sha256"]
        != p8aw_config["component_dll_sha256"]
        or config["profile_compiler_config"]
        != p8aw_config["profile_compiler_config"]
        or config["profile_compiler_config_sha256"]
        != p8aw_config["profile_compiler_config_sha256"]
    ):
        raise ValueError("P8AX frozen component identity drift")
    return parent


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _apply_gauge_loaded(
    *,
    library: ctypes.CDLL,
    profile: Any,
    values: np.ndarray,
) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float32)
    output = np.empty_like(source)
    status = library.nf_neutral_gauge_f32_apply_v1(
        ctypes.byref(profile),
        _pointer(source),
        source.shape[0] * source.shape[1],
        _pointer(output),
    )
    if status != 0:
        raise RuntimeError(f"P8AX native gauge failed: {status}")
    return output


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
    pipeline_workers: int,
    max_in_flight: int,
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
    context_library = _load_context_v2(context_dll)
    display_library = p8aw._load_display_v4(display_dll)
    gauge_library = _load_gauge(gauge_dll)

    started = time.perf_counter()
    state = NativeAo6ContextStateF32V1()
    if context_library.nf_ao6_context_f32_init_v1(
        ctypes.byref(state)
    ) != 0:
        raise RuntimeError("P8AX context init failed")
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        linear = p8aq._source_rows(
            y0=y0, y1=y1, height=height, width=width
        )
        encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(linear.astype(np.float64)),
            dtype=np.float32,
        )
        scratch_lab = np.empty_like(encoded)
        status = context_library.nf_ao6_context_f32_update_v2(
            ctypes.byref(base_profile),
            ctypes.byref(state),
            _pointer(encoded),
            encoded.shape[0] * encoded.shape[1],
            _pointer(scratch_lab),
        )
        if status != 0:
            raise RuntimeError(f"P8AX context update failed: {status}")
    context = NativeAo6BaseContextF32V1()
    if context_library.nf_ao6_context_f32_finalize_v1(
        ctypes.byref(state), ctypes.byref(context)
    ) != 0:
        raise RuntimeError("P8AX context finalize failed")

    digest = hashlib.sha256()
    consumed_rows = 0

    def transform(
        item: tuple[int, int, np.ndarray],
    ) -> tuple[int, int, np.ndarray]:
        y0, y1, rows = item
        gauged = _apply_gauge_loaded(
            library=gauge_library,
            profile=gauge_profile,
            values=rows,
        )
        encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(gauged.astype(np.float64)),
            dtype=np.float32,
        )
        final = p8aw._apply_display_v4_rows(
            library=display_library,
            base_profile=base_profile,
            context=context,
            residual_profile=residual_profile,
            encoded=encoded,
        )
        return y0, y1, final

    def consume(item: tuple[int, int, np.ndarray]) -> None:
        nonlocal consumed_rows
        y0, y1, rows = item
        if y0 != consumed_rows or rows.shape != (y1 - y0, width, 3):
            raise RuntimeError("P8AX ordered consumption drift")
        digest.update(rows.tobytes())
        consumed_rows = y1

    pipeline = BoundedOrderedPipeline[
        tuple[int, int, np.ndarray],
        tuple[int, int, np.ndarray],
    ](
        worker=transform,
        consumer=consume,
        max_workers=pipeline_workers,
        max_in_flight=max_in_flight,
    )

    def source_provider(y0: int, y1: int) -> np.ndarray:
        return p8aq._source_rows(
            y0=y0, y1=y1, height=height, width=width
        )

    def output_sink(y0: int, y1: int, rows: np.ndarray) -> None:
        owned = np.array(rows, dtype=np.float32, order="C", copy=True)
        pipeline.submit((y0, y1, owned))

    try:
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
        stats = pipeline.finish()
    except BaseException:
        pipeline.abort()
        raise
    elapsed = time.perf_counter() - started
    if consumed_rows != height:
        raise RuntimeError("P8AX did not consume every output row")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "height": height,
                "width": width,
                "pixels": height * width,
                "tile_rows": tile_rows,
                "elapsed_seconds": elapsed,
                "output_sha256": digest.hexdigest(),
                "output_dtype": "float32",
                "output_shape": [height, width, 3],
                "pipeline_workers": pipeline_workers,
                "max_in_flight": max_in_flight,
                "maximum_observed_in_flight": stats.maximum_in_flight,
                "submitted_tiles": stats.submitted,
                "consumed_tiles": stats.consumed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _worker_command(
    *,
    script: Path,
    builds: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output: Path,
    workers: int,
    serial: bool,
) -> list[str]:
    scenario = config["scenario"]
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
        str(output),
        "--height",
        str(scenario["height"]),
        "--width",
        str(scenario["width"]),
        "--tile-rows",
        str(config["tile_rows"]),
    ]
    if not serial:
        command.extend(
            [
                "--pipeline-workers",
                str(workers),
                "--max-in-flight",
                str(
                    workers * int(config["max_in_flight_per_worker"])
                ),
            ]
        )
    return command


def benchmark(
    config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    validate_contract(config)
    if psutil.virtual_memory().available < int(
        config["safety"]["minimum_available_memory_bytes_before_launch"]
    ):
        raise RuntimeError("P8AX available memory is below launch floor")
    p8aw._patch_runtime()
    builds = p8aq._build_components(config, output_dir / "binaries")
    timeout = int(config["scenario"]["timeout_seconds"])
    maximum_rss = int(config["safety"]["worker_kill_rss_bytes"])
    interval = float(config["monitor_interval_seconds"])

    baseline_runs = []
    for repeat in range(2):
        output = output_dir / "workers" / f"serial_r{repeat + 1}.json"
        run = p8aq._monitor(
            _worker_command(
                script=Path(p8aw.__file__).resolve(),
                builds=builds,
                config=config,
                output=output,
                workers=1,
                serial=True,
            ),
            output=output,
            timeout_seconds=timeout,
            maximum_rss=maximum_rss,
            interval=interval,
        )
        run["repeat"] = repeat + 1
        baseline_runs.append(run)
        if not run["success"]:
            break
    if len(baseline_runs) != 2 or not all(
        run["success"] for run in baseline_runs
    ):
        raise RuntimeError("P8AX serial baseline failed")
    baseline_hashes = {
        run["worker"]["output_sha256"] for run in baseline_runs
    }
    if len(baseline_hashes) != 1:
        raise RuntimeError("P8AX serial baseline is not repeat-exact")
    expected_hash = next(iter(baseline_hashes))

    policy_runs: dict[int, list[dict[str, Any]]] = {
        workers: [] for workers in config["pipeline_workers_candidates"]
    }
    seen: dict[int, int] = {
        workers: 0 for workers in config["pipeline_workers_candidates"]
    }
    script = Path(__file__).resolve()
    for workers in config["execution_order"]:
        seen[workers] += 1
        output = (
            output_dir
            / "workers"
            / f"pipeline_w{workers}_r{seen[workers]}.json"
        )
        run = p8aq._monitor(
            _worker_command(
                script=script,
                builds=builds,
                config=config,
                output=output,
                workers=workers,
                serial=False,
            ),
            output=output,
            timeout_seconds=timeout,
            maximum_rss=maximum_rss,
            interval=interval,
        )
        run["repeat"] = seen[workers]
        policy_runs[workers].append(run)

    serial_elapsed = [
        float(run["worker"]["elapsed_seconds"])
        for run in baseline_runs
    ]
    serial_median = statistics.median(serial_elapsed)
    rows = []
    for workers in config["pipeline_workers_candidates"]:
        runs = policy_runs[workers]
        successful = len(runs) == 2 and all(
            run["success"] for run in runs
        )
        hashes = [
            run["worker"]["output_sha256"]
            for run in runs
            if run["success"]
        ]
        exact = (
            successful
            and len(set(hashes)) == 1
            and hashes[0] == expected_hash
        )
        elapsed = [
            float(run["worker"]["elapsed_seconds"])
            for run in runs
            if run["success"]
        ]
        median = statistics.median(elapsed) if elapsed else None
        rows.append(
            {
                "pipeline_workers": workers,
                "max_in_flight": workers
                * int(config["max_in_flight_per_worker"]),
                "successful": successful,
                "serial_exact": exact,
                "worker_elapsed_seconds": elapsed,
                "median_worker_elapsed_seconds": median,
                "median_elapsed_ratio_vs_serial": (
                    median / serial_median if median is not None else None
                ),
                "maximum_observed_in_flight": [
                    run["worker"]["maximum_observed_in_flight"]
                    for run in runs
                    if run["success"]
                ],
                "peak_process_tree_rss_bytes": [
                    int(run["peak_process_tree_rss_bytes"])
                    for run in runs
                    if run["success"]
                ],
                "runs": runs,
            }
        )
    eligible = [
        row
        for row in rows
        if row["successful"] and row["serial_exact"]
    ]
    if not eligible:
        raise RuntimeError("P8AX has no exact successful policy")
    fastest = min(
        eligible,
        key=lambda row: (
            row["median_worker_elapsed_seconds"],
            row["pipeline_workers"],
        ),
    )
    promotion_ratio = float(
        config["selection"]["maximum_median_ratio_vs_serial"]
    )
    selected = (
        fastest
        if fastest["median_elapsed_ratio_vs_serial"] <= promotion_ratio
        else next(row for row in rows if row["pipeline_workers"] == 1)
    )
    stable = {
        "schema": RESULT_SCHEMA,
        "scenario": config["scenario"]["scenario_id"],
        "height": int(config["scenario"]["height"]),
        "width": int(config["scenario"]["width"]),
        "tile_rows": int(config["tile_rows"]),
        "candidate_workers": config["pipeline_workers_candidates"],
        "all_policies_serial_exact": all(
            row["serial_exact"] for row in rows
        ),
        "output_sha256": expected_hash,
        "selected_pipeline_workers": selected["pipeline_workers"],
        "selected_max_in_flight": selected["max_in_flight"],
        "selection_gate_pass": (
            fastest["median_elapsed_ratio_vs_serial"] <= promotion_ratio
        ),
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable,
        "serial_worker_elapsed_seconds": serial_elapsed,
        "serial_median_worker_elapsed_seconds": serial_median,
        "rows": rows,
        "stable_evidence_id": hashlib.sha256(
            p8aq._canonical_bytes(stable)
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8ax_native_ordered_pipeline_grid_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8ax_native_ordered_pipeline_grid_v1",
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
    parser.add_argument("--pipeline-workers", type=int, default=1)
    parser.add_argument("--max-in-flight", type=int, default=1)
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
            pipeline_workers=arguments.pipeline_workers,
            max_in_flight=arguments.max_in_flight,
        )
        return
    config = json.loads((ROOT / arguments.config).read_text())
    output_dir = ROOT / arguments.output_dir
    report = benchmark(config, output_dir)
    p8aq.write_report(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
