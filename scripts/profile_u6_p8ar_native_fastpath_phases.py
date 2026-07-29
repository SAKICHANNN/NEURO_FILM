#!/usr/bin/env python3
"""Attribute 12MP runtime after exact native AO6 v2 fast paths."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p8an_native_end_to_end_resources import (  # noqa: E402
    _canonical_bytes,
    _load_exact_json,
    _source_rows,
)
from scripts.benchmark_u6_p8aq_native_fastpath_resources import (  # noqa: E402
    _apply_display_v2_rows,
    _build_components,
)
from src.eval.density_witness_frontier import (  # noqa: E402
    linear_srgb_to_encoded,
)
from src.eval.physical_native_ao6_fastpath_conformance import (  # noqa: E402
    _load_context_v2,
    _load_display_v2,
)
from src.eval.physical_native_composed_display_conformance import (  # noqa: E402
    _apply_gauge,
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


SCHEMA = "neuro_film.u6_p8ar_native_fastpath_phases_contract.v1"


def validate_contract(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != SCHEMA
        or int(config["tile_rows"]) != 32
        or int(config["scenario"]["repeats"]) != 2
    ):
        raise ValueError("unsupported P8AR contract")
    parent = _load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AR"):
        raise ValueError("P8AR parent decision drift")
    _load_exact_json(
        ROOT / config["resource_contract"],
        config["resource_contract_sha256"],
    )
    _load_exact_json(
        ROOT / config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )


def _profile_once(
    *,
    config: dict[str, Any],
    builds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    profile_config = json.loads(
        (ROOT / config["profile_compiler_config"]).read_text()
    )
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=profile_config
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
    context_library = _load_context_v2(
        Path(builds["context"]["dll_path"])
    )
    display_library = _load_display_v2(
        Path(builds["display"]["dll_path"])
    )
    height = int(config["scenario"]["height"])
    width = int(config["scenario"]["width"])
    tile_rows = int(config["tile_rows"])
    phases = {
        "context_source_and_oetf_seconds": 0.0,
        "context_native_update_v2_seconds": 0.0,
        "physical_source_provider_seconds": 0.0,
        "physical_native_core_seconds": 0.0,
        "gauge_native_seconds": 0.0,
        "display_input_oetf_seconds": 0.0,
        "ao6_native_display_v2_seconds": 0.0,
        "hash_sink_seconds": 0.0,
    }
    total_started = time.perf_counter()
    state = NativeAo6ContextStateF32V1()
    if context_library.nf_ao6_context_f32_init_v1(
        ctypes.byref(state)
    ) != 0:
        raise RuntimeError("P8AR context init failed")
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        phase_started = time.perf_counter()
        linear = _source_rows(
            y0=y0, y1=y1, height=height, width=width
        )
        encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(linear.astype(np.float64)),
            dtype=np.float32,
        )
        phases["context_source_and_oetf_seconds"] += (
            time.perf_counter() - phase_started
        )
        scratch_lab = np.empty_like(encoded)
        phase_started = time.perf_counter()
        status = context_library.nf_ao6_context_f32_update_v2(
            ctypes.byref(base_profile),
            ctypes.byref(state),
            encoded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            encoded.shape[0] * encoded.shape[1],
            scratch_lab.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        phases["context_native_update_v2_seconds"] += (
            time.perf_counter() - phase_started
        )
        if status != 0:
            raise RuntimeError(f"P8AR context update failed: {status}")
    context = NativeAo6BaseContextF32V1()
    if context_library.nf_ao6_context_f32_finalize_v1(
        ctypes.byref(state), ctypes.byref(context)
    ) != 0:
        raise RuntimeError("P8AR context finalize failed")

    digest = hashlib.sha256()
    consumed_rows = 0
    sink_seconds = 0.0

    def source_provider(y0: int, y1: int) -> np.ndarray:
        started = time.perf_counter()
        rows = _source_rows(
            y0=y0, y1=y1, height=height, width=width
        )
        phases["physical_source_provider_seconds"] += (
            time.perf_counter() - started
        )
        return rows

    def output_sink(y0: int, y1: int, rows: np.ndarray) -> None:
        nonlocal consumed_rows, sink_seconds
        sink_started = time.perf_counter()
        phase_started = time.perf_counter()
        gauged = _apply_gauge(
            dll=Path(builds["gauge"]["dll_path"]),
            profile=gauge_profile,
            values=rows,
        )
        phases["gauge_native_seconds"] += (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        gauged_encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(gauged.astype(np.float64)),
            dtype=np.float32,
        )
        phases["display_input_oetf_seconds"] += (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        final_rows = _apply_display_v2_rows(
            library=display_library,
            base_profile=base_profile,
            context=context,
            residual_profile=residual_profile,
            encoded=gauged_encoded,
        )
        phases["ao6_native_display_v2_seconds"] += (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        digest.update(final_rows.tobytes())
        phases["hash_sink_seconds"] += time.perf_counter() - phase_started
        consumed_rows = y1
        sink_seconds += time.perf_counter() - sink_started

    physical_started = time.perf_counter()
    stream_tiled_f32_chain(
        domains_dll=Path(builds["domains"]["dll_path"]),
        gaussian_dll=Path(builds["gaussian"]["dll_path"]),
        adjacency_dll=Path(builds["adjacency"]["dll_path"]),
        domains_payload=domains_payload,
        spatial_payload=spatial_payload,
        adjacency_payload=adjacency_payload,
        height=height,
        width=width,
        tile_rows=tile_rows,
        source_provider=source_provider,
        output_sink=output_sink,
    )
    physical_total = time.perf_counter() - physical_started
    phases["physical_native_core_seconds"] = max(
        0.0,
        physical_total
        - sink_seconds
        - phases["physical_source_provider_seconds"],
    )
    total_seconds = time.perf_counter() - total_started
    if consumed_rows != height:
        raise RuntimeError("P8AR did not consume every output row")
    phase_sum = sum(phases.values())
    shares = {
        name: value / phase_sum for name, value in phases.items()
    }
    dominant = max(phases, key=phases.__getitem__)
    return {
        "total_seconds": total_seconds,
        "attributed_seconds": phase_sum,
        "unattributed_seconds": total_seconds - phase_sum,
        "phases": phases,
        "phase_shares": shares,
        "dominant_phase": dominant,
        "dominant_share": shares[dominant],
        "output_sha256": digest.hexdigest(),
    }


def profile(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_contract(config)
    builds = _build_components(config, output_dir / "binaries")
    runs = [
        _profile_once(config=config, builds=builds)
        for _ in range(int(config["scenario"]["repeats"]))
    ]
    output_exact = (
        len({run["output_sha256"] for run in runs}) == 1
        and runs[0]["output_sha256"] == config["expected_output_sha256"]
    )
    dominant_exact = len({run["dominant_phase"] for run in runs}) == 1
    minimum_dominant_share = min(run["dominant_share"] for run in runs)
    gate_share = float(config["gates"]["minimum_dominant_phase_share"])
    passed = (
        output_exact and dominant_exact and minimum_dominant_share >= gate_share
    )
    phase_medians = {
        name: statistics.median(run["phases"][name] for run in runs)
        for name in runs[0]["phases"]
    }
    stable = {
        "schema": "neuro_film.u6_p8ar_native_fastpath_phases_result.v1",
        "scenario": config["scenario"]["scenario_id"],
        "height": int(config["scenario"]["height"]),
        "width": int(config["scenario"]["width"]),
        "tile_rows": int(config["tile_rows"]),
        "repeat_count": len(runs),
        "output_repeat_and_v1_exact": output_exact,
        "output_sha256": runs[0]["output_sha256"] if output_exact else None,
        "dominant_phase_repeat_exact": dominant_exact,
        "dominant_phase": (
            runs[0]["dominant_phase"] if dominant_exact else None
        ),
        "minimum_dominant_phase_share": minimum_dominant_share,
        "dominant_phase_share_gate": gate_share,
        "phase_median_seconds": phase_medians,
        "pass": passed,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable,
        "runs": runs,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8ar_native_fastpath_phases_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8ar_native_fastpath_phases_v1",
    )
    arguments = parser.parse_args()
    config = json.loads((ROOT / arguments.config).read_text())
    output_dir = ROOT / arguments.output_dir
    report = profile(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
