#!/usr/bin/env python3
"""Attribute RSS by phase through the actual current P8M renderer entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p8d_cpu_consumer import (  # noqa: E402
    _analytic_scene,
    _monitor,
)
from scripts.benchmark_u6_p8l_phase_rss import _PhaseSampler  # noqa: E402
from src.eval.global_frontier import sha256_file  # noqa: E402
import src.film_physics.profile_consumer as profile_consumer  # noqa: E402
from src.film_physics.profile_compiler import _canonical_bytes  # noqa: E402
from src.preprocess.types import SourceProfile, WorkingImage  # noqa: E402


SCHEMA = "neuro_film.u6_p8n_current_phase_rss_contract.v1"


def _load_exact_json(path: Path, expected: str) -> dict[str, Any]:
    if sha256_file(path) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    measurement = config["measurement"]
    scenario = config["scenario"]
    if (
        config.get("schema") != SCHEMA
        or int(scenario["repeats"]) != 2
        or int(scenario["height"]) * int(scenario["width"]) != 12_000_000
        or int(scenario["tile_rows"]) <= 0
        or float(measurement["sample_interval_seconds"]) <= 0.0
        or not measurement["production_entry_required"]
        or not measurement[
            "timing_and_rss_excluded_from_stable_evidence_id"
        ]
        or measurement["post_result_retuning_allowed"]
        or len(measurement["expected_output_sha256"]) != 64
        or len(set(measurement["phase_order"]))
        != len(measurement["phase_order"])
    ):
        raise ValueError("unsupported U6.P8N contract")
    decision = _load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    profile = _load_exact_json(
        ROOT / config["profile_contract"],
        config["profile_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8N")
        or not decision["candidate_retained"]
        or decision["production_default_changed"]
    ):
        raise ValueError("U6.P8N parent decision drift")
    return profile


def _run_worker(config: dict[str, Any], output: Path) -> None:
    profile_config = _load_exact_json(
        ROOT / config["profile_contract"],
        config["profile_contract_sha256"],
    )
    scenario = config["scenario"]
    phases = [str(value) for value in config["measurement"]["phase_order"]]
    sampler = _PhaseSampler(
        phases,
        float(config["measurement"]["sample_interval_seconds"]),
    )
    sampler.start()

    original_scene = profile_consumer.scene_exposure_from_working_image
    original_hash = profile_consumer._array_sha256
    original_encode = (
        profile_consumer._linear_srgb_to_encoded_row_staged
    )
    original_reconstruct = (
        profile_consumer.reconstruct_standalone_runtime
    )
    original_decode = profile_consumer.encoded_srgb_to_linear
    original_physical = profile_consumer._render_physical
    original_gauge = profile_consumer.apply_gauge_to_intermediate
    original_display_builder = (
        profile_consumer.build_source_context_display_look_row_streamed
    )
    hash_calls = 0

    def scene_wrapper(working: WorkingImage) -> Any:
        sampler.enter("ingress-owned-scene")
        return original_scene(working)

    def hash_wrapper(value: np.ndarray) -> str:
        nonlocal hash_calls
        hash_calls += 1
        sampler.enter("input-hash" if hash_calls == 1 else "receipt-hash")
        return original_hash(value)

    def encode_wrapper(
        value: np.ndarray, *, tile_rows: int
    ) -> np.ndarray:
        sampler.enter("encoded-source")
        return original_encode(value, tile_rows=tile_rows)

    def reconstruct_wrapper(artifact: dict[str, Any]) -> Any:
        sampler.enter("runtime-reconstruct")
        return original_reconstruct(artifact)

    def decode_wrapper(value: np.ndarray) -> np.ndarray:
        sampler.enter("roundtrip-linear")
        return original_decode(value)

    def physical_wrapper(value: np.ndarray, runtime: Any) -> np.ndarray:
        sampler.enter("physical-spatial")
        return original_physical(value, runtime)

    def gauge_wrapper(value: np.ndarray, gauge: Any) -> np.ndarray:
        sampler.enter("gauge-and-encode")
        return original_gauge(value, gauge)

    def display_builder_wrapper(
        payload: dict[str, Any],
        source: np.ndarray,
        *,
        tile_rows: int,
        reuse_input_buffer: bool = False,
    ) -> Callable[[np.ndarray], np.ndarray]:
        sampler.enter("source-context-build")
        apply = original_display_builder(
            payload,
            source,
            tile_rows=tile_rows,
            reuse_input_buffer=reuse_input_buffer,
        )

        def apply_wrapper(value: np.ndarray) -> np.ndarray:
            sampler.enter("display-base-residual")
            return apply(value)

        return apply_wrapper

    profile_consumer.scene_exposure_from_working_image = scene_wrapper
    profile_consumer._array_sha256 = hash_wrapper
    profile_consumer._linear_srgb_to_encoded_row_staged = encode_wrapper
    profile_consumer.reconstruct_standalone_runtime = reconstruct_wrapper
    profile_consumer.encoded_srgb_to_linear = decode_wrapper
    profile_consumer._render_physical = physical_wrapper
    profile_consumer.apply_gauge_to_intermediate = gauge_wrapper
    profile_consumer.build_source_context_display_look_row_streamed = (
        display_builder_wrapper
    )

    started = time.perf_counter()
    try:
        sampler.enter("setup")
        artifact = profile_consumer.compile_standalone_profile_artifact(
            root=ROOT, config=profile_config
        )
        working = WorkingImage(
            pixels=_analytic_scene(
                int(scenario["height"]), int(scenario["width"])
            ),
            working_space="linear_srgb_d65",
            transfer_state="scene_linear",
            source_transfer_state="scene_linear",
            source_profile=SourceProfile(
                "raw_metadata", "U6.P8N analytic scene"
            ),
            hdr_metadata={},
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=32,
            source_path=Path("u6-p8n-analytic.scene-linear"),
        )
        rendered, receipt = (
            profile_consumer.render_working_image_fully_row_streamed(
                artifact,
                working,
                tile_rows=int(scenario["tile_rows"]),
            )
        )
        elapsed = time.perf_counter() - started
        output_sha256 = original_hash(rendered)
        if output_sha256 != receipt["output"]["array_sha256"]:
            raise RuntimeError("P8N receipt/output identity mismatch")
    finally:
        phase_peaks, phase_samples = sampler.finish()

    result = {
        "scenario_id": scenario["scenario_id"],
        "height": int(scenario["height"]),
        "width": int(scenario["width"]),
        "pixels": int(scenario["height"]) * int(scenario["width"]),
        "tile_rows": int(scenario["tile_rows"]),
        "elapsed_seconds": elapsed,
        "output_sha256": output_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "output_dtype": rendered.dtype.name,
        "output_shape": list(rendered.shape),
        "phase_peak_rss_bytes": phase_peaks,
        "phase_sample_counts": phase_samples,
        "dominant_phase": max(phase_peaks, key=phase_peaks.__getitem__),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_contract(config)
    available = int(psutil.virtual_memory().available)
    if available < int(
        config["safety"]["minimum_available_memory_bytes_before_launch"]
    ):
        raise RuntimeError("available memory is below P8N launch floor")
    scenario = config["scenario"]
    runs: list[dict[str, Any]] = []
    for repeat in range(int(scenario["repeats"])):
        worker_output = (
            output_dir / "workers" / f"current-phase-rss-r{repeat + 1}.json"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-output",
            str(worker_output),
            "--config",
            str(ROOT / "configs/u6_p8n_current_phase_rss_v1.json"),
        ]
        observed = _monitor(
            command,
            output=worker_output,
            timeout_seconds=int(scenario["timeout_seconds"]),
            maximum_rss=int(
                config["safety"]["maximum_process_tree_rss_bytes"]
            ),
            interval=float(
                config["measurement"]["sample_interval_seconds"]
            ),
        )
        runs.append({"repeat": repeat + 1, **observed})
        if not observed["success"]:
            break
    expected = config["measurement"]["expected_output_sha256"]
    success = len(runs) == 2 and all(row["success"] for row in runs)
    exact = success and all(
        row["worker"]["output_sha256"] == expected for row in runs
    )
    dominant = {
        row["worker"]["dominant_phase"]
        for row in runs
        if row["success"]
    }
    stable = {
        "schema": (
            "neuro_film.u6_p8n_current_phase_rss_stable_evidence.v1"
        ),
        "phase_order": config["measurement"]["phase_order"],
        "all_success": success,
        "all_output_identity_exact": exact,
        "dominant_phase_stable": len(dominant) == 1,
        "dominant_phase": next(iter(dominant)) if len(dominant) == 1 else None,
    }
    return {
        "schema": "neuro_film.u6_p8n_current_phase_rss_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "total_memory_bytes": int(psutil.virtual_memory().total),
            "available_memory_bytes_before_launch": available,
        },
        "runs": runs,
        "stable_evidence": stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "decision": (
            "attribution-complete"
            if success and exact and len(dominant) == 1
            else "resource-identity-or-attribution-failure"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p8n_current_phase_rss_v1.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.worker:
        if args.worker_output is None:
            parser.error("--worker-output is required")
        _run_worker(config, args.worker_output)
        return 0
    if args.output is None:
        parser.error("--output is required")
    report = evaluate(config, args.output.parent)
    raw = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={hashlib.sha256(raw).hexdigest()}")
    for run in report["runs"]:
        print(
            f"r{run['repeat']}: success={run['success']} "
            f"wall={run['wall_seconds']:.3f}s "
            f"tree_peak={run['peak_process_tree_rss_bytes'] / 2**30:.3f}GiB "
            f"phase={run['worker'].get('dominant_phase')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
