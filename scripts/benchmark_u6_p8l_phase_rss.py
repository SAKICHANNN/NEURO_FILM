#!/usr/bin/env python3
"""Attribute phase-local RSS on the exact restored P8H renderer."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p8d_cpu_consumer import (  # noqa: E402
    _analytic_scene,
    _monitor,
)
from src.eval.density_witness_frontier import (  # noqa: E402
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import sha256_file  # noqa: E402
from src.eval.physical_neutral_gauged_chain import (  # noqa: E402
    apply_gauge_to_intermediate,
)
from src.eval.physical_virtual_scan_sampling import (  # noqa: E402
    _render_physical,
    compile_virtual_scan_profile,
)
from src.film_physics.display_look import (  # noqa: E402
    build_source_context_display_look_row_streamed,
)
from src.film_physics.profile_compiler import _canonical_bytes  # noqa: E402
from src.film_physics.profile_consumer import (  # noqa: E402
    _array_sha256,
    compile_standalone_profile_artifact,
    reconstruct_standalone_runtime,
    validate_standalone_profile_artifact,
)
from src.film_physics.contracts import (  # noqa: E402
    scene_exposure_from_working_image,
)
from src.film_physics.spatial_response import (  # noqa: E402
    required_spatial_response_halo,
)
from src.preprocess.types import SourceProfile, WorkingImage  # noqa: E402


SCHEMA = "neuro_film.u6_p8l_phase_rss_attribution_contract.v1"


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
        or not measurement[
            "timing_and_rss_excluded_from_stable_evidence_id"
        ]
        or measurement["post_result_retuning_allowed"]
        or len(measurement["expected_output_sha256"]) != 64
        or len(set(measurement["phase_order"]))
        != len(measurement["phase_order"])
    ):
        raise ValueError("unsupported U6.P8L contract")
    decision = _load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    profile = _load_exact_json(
        ROOT / config["profile_contract"],
        config["profile_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8L")
        or decision["candidate_retained"]
        or decision["production_default_changed"]
    ):
        raise ValueError("U6.P8L parent decision drift")
    return profile


class _PhaseSampler:
    def __init__(self, phases: list[str], interval: float) -> None:
        self._process = psutil.Process()
        self._interval = interval
        self._lock = threading.Lock()
        self._current = phases[0]
        self._peaks = {phase: 0 for phase in phases}
        self._samples = {phase: 0 for phase in phases}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="u6-p8l-rss-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def enter(self, phase: str) -> None:
        with self._lock:
            if phase not in self._peaks:
                raise ValueError(f"unknown phase: {phase}")
            self._current = phase
        self.sample_once()

    def sample_once(self) -> None:
        rss = int(self._process.memory_info().rss)
        with self._lock:
            phase = self._current
            self._peaks[phase] = max(self._peaks[phase], rss)
            self._samples[phase] += 1

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample_once()

    def finish(self) -> tuple[dict[str, int], dict[str, int]]:
        self.sample_once()
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("phase RSS sampler did not stop")
        return dict(self._peaks), dict(self._samples)


def _run_worker(
    config: dict[str, Any],
    output: Path,
) -> None:
    profile_config = _load_exact_json(
        ROOT / config["profile_contract"],
        config["profile_contract_sha256"],
    )
    scenario = config["scenario"]
    height = int(scenario["height"])
    width = int(scenario["width"])
    tile_rows = int(scenario["tile_rows"])
    phases = [str(value) for value in config["measurement"]["phase_order"]]
    sampler = _PhaseSampler(
        phases,
        float(config["measurement"]["sample_interval_seconds"]),
    )
    sampler.start()
    started = time.perf_counter()
    try:
        sampler.enter("setup")
        artifact = compile_standalone_profile_artifact(
            root=ROOT, config=profile_config
        )
        bundle = validate_standalone_profile_artifact(artifact)
        working = WorkingImage(
            pixels=_analytic_scene(height, width),
            working_space="linear_srgb_d65",
            transfer_state="scene_linear",
            source_transfer_state="scene_linear",
            source_profile=SourceProfile(
                "raw_metadata", "U6.P8L analytic scene"
            ),
            hdr_metadata={},
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=32,
            source_path=Path("u6-p8l-analytic.scene-linear"),
        )

        sampler.enter("ingress-owned-scene")
        scene = scene_exposure_from_working_image(working)
        if np.any(scene.values > 1.0):
            raise ValueError("P8L scene left frozen [0,1] ingress")
        input_array_sha256 = _array_sha256(scene.values)
        input_shape = list(scene.values.shape)

        sampler.enter("encoded-source")
        encoded = linear_srgb_to_encoded(
            scene.values.astype(np.float64)
        )
        del scene

        sampler.enter("runtime-reconstruct")
        runtime, gauge = reconstruct_standalone_runtime(artifact)
        compiled = replace(
            runtime,
            profile=compile_virtual_scan_profile(
                runtime.profile,
                sampling_dpi=int(artifact["reference_sampling_dpi"]),
            ),
        )
        halo = required_spatial_response_halo(compiled.profile)

        sampler.enter("roundtrip-linear")
        linear = encoded_srgb_to_linear(encoded)
        ranges = [
            (y0, min(encoded.shape[0], y0 + tile_rows))
            for y0 in range(0, encoded.shape[0], tile_rows)
        ]
        gauged_encoded = np.empty_like(encoded, dtype=np.float64)
        seams: list[int] = []
        for y0, y1 in ranges:
            source_y0 = max(0, y0 - halo)
            source_y1 = min(encoded.shape[0], y1 + halo)
            sampler.enter("physical-spatial")
            physical = _render_physical(
                linear[source_y0:source_y1], compiled
            )
            core = physical[y0 - source_y0 : y1 - source_y0]
            sampler.enter("gauge-and-encode")
            gauged = apply_gauge_to_intermediate(core, gauge)
            gauged_encoded[y0:y1] = linear_srgb_to_encoded(gauged)
            if 0 < y0 < encoded.shape[0]:
                seams.append(y0)
        del linear

        sampler.enter("source-context-build")
        display = build_source_context_display_look_row_streamed(
            artifact["component_payloads"][
                "ao6-source-context-display-look"
            ],
            encoded,
            tile_rows=tile_rows,
            reuse_input_buffer=True,
        )
        del encoded

        sampler.enter("display-base-residual")
        output_pixels = display(gauged_encoded)

        sampler.enter("receipt-hash")
        output_sha256 = _array_sha256(output_pixels)
        receipt_core = {
            "profile_id": bundle.profile_id,
            "bundle_sha256": bundle.bundle_sha256,
            "input_array_sha256": input_array_sha256,
            "input_shape": input_shape,
            "output_array_sha256": output_sha256,
            "seam_rows": sorted(seams),
        }
        elapsed = time.perf_counter() - started
    finally:
        phase_peaks, phase_samples = sampler.finish()

    result = {
        "scenario_id": scenario["scenario_id"],
        "height": height,
        "width": width,
        "pixels": height * width,
        "tile_rows": tile_rows,
        "elapsed_seconds": elapsed,
        "output_sha256": output_sha256,
        "receipt_core_sha256": hashlib.sha256(
            _canonical_bytes(receipt_core)
        ).hexdigest(),
        "output_dtype": output_pixels.dtype.name,
        "output_shape": list(output_pixels.shape),
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
        raise RuntimeError("available memory is below P8L launch floor")
    scenario = config["scenario"]
    runs: list[dict[str, Any]] = []
    for repeat in range(int(scenario["repeats"])):
        worker_output = (
            output_dir / "workers" / f"phase-rss-r{repeat + 1}.json"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-output",
            str(worker_output),
            "--config",
            str(ROOT / "configs/u6_p8l_phase_rss_attribution_v1.json"),
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
            "neuro_film.u6_p8l_phase_rss_stable_evidence.v1"
        ),
        "phase_order": config["measurement"]["phase_order"],
        "all_success": success,
        "all_output_identity_exact": exact,
        "dominant_phase_stable": len(dominant) == 1,
        "dominant_phase": next(iter(dominant)) if len(dominant) == 1 else None,
    }
    return {
        "schema": "neuro_film.u6_p8l_phase_rss_attribution_report.v1",
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
        default=ROOT / "configs/u6_p8l_phase_rss_attribution_v1.json",
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
