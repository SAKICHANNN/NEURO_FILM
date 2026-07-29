#!/usr/bin/env python3
"""Benchmark the exact fresh-process native Standard strength transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2
import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
from src.film_physics.native_standard_factory import (  # noqa: E402
    create_opt_in_native_standard_runtime,
)
from src.film_physics.native_standard_strength_file import (  # noqa: E402
    render_native_standard_strength_file_to_png16,
)
from src.film_physics.native_standard_strength_output import (  # noqa: E402
    verify_native_standard_strength_png16,
)
from src.film_physics.native_standard_strength_staging import (  # noqa: E402
    verify_native_standard_strength_staging,
)
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
)


SCHEMA = (
    "neuro_film.u6_p8bm_native_standard_strength_transaction_contract.v1"
)
RESULT_SCHEMA = (
    "neuro_film.u6_p8bm_native_standard_strength_transaction_result.v1"
)
CONFIG_PATH = (
    ROOT
    / "configs/u6_p8bm_native_standard_strength_transaction_v1.json"
)
_COMPONENTS = (
    "domains",
    "gaussian",
    "adjacency",
    "gauge",
    "context",
    "display",
)


def validate_contract(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config["strength"] != 0.8
        or config["strength_domain"] != "encoded-display-srgb"
        or config["repeats"] != 2
        or config["production_default_changed"]
    ):
        raise ValueError("unsupported P8BM contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    base = p8aq._load_exact_json(
        ROOT / config["working_image_contract"],
        config["working_image_contract_sha256"],
    )
    fixture = ROOT / config["raw_fixture"]
    legacy = ROOT / config["legacy_quantized_sweep_output"]
    if (
        parent["result"]["status"] != "pass_limited"
        or not str(parent["next_leaf"]).startswith("U6.P8BM")
        or parent["result"]["autonomous_visual_review"][
            "preferred_global_development_strength"
        ]
        != config["strength"]
        or hashlib.sha256(fixture.read_bytes()).hexdigest()
        != config["raw_fixture_sha256"]
        or hashlib.sha256(legacy.read_bytes()).hexdigest()
        != config["legacy_quantized_sweep_output_sha256"]
    ):
        raise ValueError("P8BM parent or RAW identity drift")
    return parent, base


def _worker(
    *,
    config_path: Path,
    library_paths: dict[str, Path],
    worker_output: Path,
    transaction_dir: Path,
) -> None:
    config = json.loads(config_path.read_text())
    _, base = validate_contract(config)
    package = json.loads((ROOT / base["package"]).read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(
            (ROOT / base["profile_compiler_config"]).read_text()
        ),
    )
    runtime, factory_receipt = create_opt_in_native_standard_runtime(
        package=package,
        artifact=artifact,
        library_paths=library_paths,
    )
    transaction_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "raw_output_path": transaction_dir / "render.f32",
        "raw_report_path": transaction_dir / "render.raw.json",
        "png_output_path": transaction_dir / "render.png",
        "png_report_path": transaction_dir / "render.png.json",
    }
    started = time.perf_counter()
    result = render_native_standard_strength_file_to_png16(
        runtime,
        input_path=ROOT / config["raw_fixture"],
        strength=config["strength"],
        **paths,
    )
    png_first = verify_native_standard_strength_png16(
        report_path=paths["png_report_path"],
        expected_report_sha256=result["png_report_sha256"],
        expected_delivery_id=result["png_delivery_id"],
    )
    png_second = verify_native_standard_strength_png16(
        report_path=paths["png_report_path"],
        expected_report_sha256=result["png_report_sha256"],
        expected_delivery_id=result["png_delivery_id"],
    )
    raw_first = verify_native_standard_strength_staging(
        report_path=paths["raw_report_path"],
        expected_report_sha256=result["raw_report_sha256"],
        expected_run_id=result["raw_run_id"],
    )
    raw_second = verify_native_standard_strength_staging(
        report_path=paths["raw_report_path"],
        expected_report_sha256=result["raw_report_sha256"],
        expected_run_id=result["raw_run_id"],
    )
    elapsed = time.perf_counter() - started
    code = cv2.imread(
        str(paths["png_output_path"]), cv2.IMREAD_UNCHANGED
    )
    if code is None or code.dtype.name != "uint16" or code.ndim != 3:
        raise RuntimeError("P8BM PNG decode drift")
    boundary = float(
        ((code == 0) | (code == 65535)).any(axis=2).mean()
    )
    legacy = cv2.imread(
        str(ROOT / config["legacy_quantized_sweep_output"]),
        cv2.IMREAD_UNCHANGED,
    )
    if legacy is None or legacy.shape != code.shape:
        raise RuntimeError("P8BM legacy sweep decode drift")
    code_delta = np.abs(
        code.astype(np.int32) - legacy.astype(np.int32)
    )
    payload = {
        "elapsed_seconds": elapsed,
        "input_file_sha256": result["input_file_sha256"],
        "factory_receipt_sha256": factory_receipt["receipt_sha256"],
        "strength": result["strength"],
        "raw_run_id": result["raw_run_id"],
        "raw_report_sha256": result["raw_report_sha256"],
        "raw_verification_repeat_exact": raw_first == raw_second,
        "png_delivery_id": result["png_delivery_id"],
        "png_report_sha256": result["png_report_sha256"],
        "png_output_sha256": result["png_output_sha256"],
        "png_verification_id": png_first["verification_id"],
        "png_verification_repeat_exact": png_first == png_second,
        "png_bytes": paths["png_output_path"].stat().st_size,
        "output_code_boundary_fraction": boundary,
        "legacy_sweep_max_abs_code_delta": int(code_delta.max()),
        "legacy_sweep_changed_code_fraction": float(
            np.mean(code_delta != 0)
        ),
    }
    for path in (
        paths["png_report_path"],
        paths["png_output_path"],
        paths["raw_report_path"],
        paths["raw_output_path"],
    ):
        path.unlink()
    transaction_dir.rmdir()
    payload["cleanup_pass"] = not transaction_dir.exists()
    worker_output.parent.mkdir(parents=True, exist_ok=True)
    worker_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def benchmark(
    config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    _, base = validate_contract(config)
    if int(psutil.virtual_memory().available) < int(
        config["safety"]["minimum_available_memory_bytes_before_launch"]
    ):
        raise RuntimeError("P8BM available memory below launch floor")
    p8aw._patch_runtime()
    builds = p8aq._build_components(base, output_dir / "binaries")
    library_paths = {
        name: Path(row["dll_path"]) for name, row in builds.items()
    }
    runs: list[dict[str, Any]] = []
    for repeat in range(config["repeats"]):
        worker_output = (
            output_dir / "workers" / f"run_{repeat + 1}.json"
        )
        transaction_dir = (
            output_dir / "workers" / f"transaction_{repeat + 1}"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(CONFIG_PATH),
            "--worker-output",
            str(worker_output),
            "--transaction-dir",
            str(transaction_dir),
        ]
        for name, path in library_paths.items():
            command.extend([f"--{name}-dll", str(path)])
        run = p8aq._monitor(
            command,
            output=worker_output,
            timeout_seconds=config["timeout_seconds"],
            maximum_rss=config["safety"]["worker_kill_rss_bytes"],
            interval=config["monitor_interval_seconds"],
        )
        run["repeat"] = repeat + 1
        runs.append(run)
        if not run["success"]:
            break

    successful = len(runs) == 2 and all(row["success"] for row in runs)
    workers = [row["worker"] for row in runs if row["success"]]
    exact = (
        successful
        and len({row["png_output_sha256"] for row in workers}) == 1
        and len({row["input_file_sha256"] for row in workers}) == 1
        and all(row["raw_verification_repeat_exact"] for row in workers)
        and all(row["png_verification_repeat_exact"] for row in workers)
    )
    cleanup = successful and all(row["cleanup_pass"] for row in workers)
    peaks = [
        int(row["peak_process_tree_rss_bytes"])
        for row in runs
        if row["success"]
    ]
    elapsed = [float(row["elapsed_seconds"]) for row in workers]
    boundary = [
        float(row["output_code_boundary_fraction"])
        for row in workers
    ]
    legacy_delta = [
        int(row["legacy_sweep_max_abs_code_delta"])
        for row in workers
    ]
    gates = config["gates"]
    memory_pass = bool(peaks) and max(peaks) <= gates[
        "maximum_peak_process_tree_rss_bytes"
    ]
    elapsed_pass = bool(elapsed) and max(elapsed) <= gates[
        "maximum_worker_elapsed_seconds"
    ]
    boundary_pass = bool(boundary) and max(boundary) <= gates[
        "maximum_output_code_boundary_fraction"
    ]
    legacy_compatibility_pass = (
        bool(legacy_delta)
        and max(legacy_delta)
        <= gates["maximum_legacy_sweep_abs_code_delta"]
    )
    stable = {
        "schema": RESULT_SCHEMA,
        "all_runs_successful": successful,
        "strength": config["strength"],
        "strength_domain": config["strength_domain"],
        "raw_to_png_repeat_exact": exact,
        "png_output_sha256": (
            workers[0]["png_output_sha256"] if exact else None
        ),
        "cleanup_pass": cleanup,
        "peak_process_tree_rss_bytes": peaks,
        "median_peak_process_tree_rss_bytes": (
            statistics.median(peaks) if peaks else None
        ),
        "memory_gate_bytes": gates[
            "maximum_peak_process_tree_rss_bytes"
        ],
        "memory_pass": memory_pass,
        "worker_elapsed_seconds": elapsed,
        "elapsed_gate_seconds": gates[
            "maximum_worker_elapsed_seconds"
        ],
        "elapsed_pass": elapsed_pass,
        "output_code_boundary_fraction": boundary,
        "boundary_gate": gates[
            "maximum_output_code_boundary_fraction"
        ],
        "boundary_pass": boundary_pass,
        "legacy_sweep_max_abs_code_delta": legacy_delta,
        "legacy_sweep_changed_code_fraction": [
            float(row["legacy_sweep_changed_code_fraction"])
            for row in workers
        ],
        "legacy_compatibility_pass": legacy_compatibility_pass,
        "pass": (
            successful
            and exact
            and cleanup
            and memory_pass
            and elapsed_pass
            and boundary_pass
            and legacy_compatibility_pass
        ),
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable,
        "wall_seconds": [float(row["wall_seconds"]) for row in runs],
        "runs": runs,
        "stable_evidence_id": hashlib.sha256(
            p8aq._canonical_bytes(stable)
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH.relative_to(ROOT)),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8bm_native_standard_strength_transaction_v1",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output")
    parser.add_argument("--transaction-dir")
    for name in _COMPONENTS:
        parser.add_argument(f"--{name}-dll")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if args.worker:
        _worker(
            config_path=config_path,
            library_paths={
                name: Path(getattr(args, f"{name}_dll"))
                for name in _COMPONENTS
            },
            worker_output=Path(args.worker_output),
            transaction_dir=Path(args.transaction_dir),
        )
        return
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    report = benchmark(
        json.loads(config_path.read_text()), output_dir
    )
    p8aq.write_report(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
