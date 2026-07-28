"""U6.P8E stage-level attribution for the canonical CPU profile."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import psutil

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import sha256_file
from src.eval.physical_neutral_gauged_chain import (
    apply_gauge_to_intermediate,
)
from src.eval.physical_neutral_gauged_invariance import (
    render_challenger_row_tiled,
)
from src.eval.physical_virtual_scan_sampling import (
    _render_physical,
    compile_virtual_scan_profile,
)
from src.film_physics import required_spatial_response_halo
from src.film_physics.display_look import (
    build_source_context_display_look_stages,
)
from src.film_physics.profile_compiler import _canonical_bytes
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
    reconstruct_standalone_runtime,
)


SCHEMA = "neuro_film.u6_p8e_cpu_stage_attribution_contract.v1"


def _load_exact_json(path: Path, expected: str) -> dict[str, Any]:
    if sha256_file(path) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    expected_stages = [
        "artifact-reconstruct",
        "scene-linear-to-encoded",
        "encoded-to-linear",
        "physical-spatial",
        "neutral-gauge",
        "physical-intermediate-oetf",
        "source-context-build",
        "safe-lab-base",
        "t15-c35-residual",
    ]
    if (
        config.get("schema") != SCHEMA
        or config["stages"] != expected_stages
        or int(config["height"]) != 1024
        or int(config["width"]) != 1024
        or int(config["repeats"]) != 2
        or int(config["tile_rows"]) <= 0
        or not config["execution"]["exact_reference_output_required"]
        or not config["execution"]["rss_is_stage_end_snapshot_not_peak"]
        or not config["execution"][
            "timing_excluded_from_stable_evidence_id"
        ]
        or config["execution"]["post_result_retuning_allowed"]
    ):
        raise ValueError("unsupported U6.P8E contract")
    decision = _load_exact_json(
        root / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    profile = _load_exact_json(
        root / config["profile_contract"],
        config["profile_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8E")
        or decision["performance_target_pass"]
        or decision["production_default_changed"]
    ):
        raise ValueError("U6.P8E parent decision drift")
    return profile


def _analytic_scene(height: int, width: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    result = np.empty((height, width, 3), dtype=np.float32)
    result[..., 0] = x[None, :]
    result[..., 1] = y[:, None]
    result[..., 2] = 0.15 + 0.45 * x[None, :] + 0.35 * y[:, None]
    np.clip(result, 0.0, 1.0, out=result)
    return result


def _timed(
    stages: dict[str, dict[str, float]],
    name: str,
    function: Any,
) -> Any:
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    stages[name] = {
        "elapsed_seconds": elapsed,
        "rss_bytes_after_stage": int(
            psutil.Process().memory_info().rss
        ),
    }
    return result


def _profile_once(
    artifact: dict[str, Any],
    scene: np.ndarray,
    *,
    tile_rows: int,
) -> dict[str, Any]:
    stages: dict[str, dict[str, float]] = {}
    runtime, gauge = _timed(
        stages,
        "artifact-reconstruct",
        lambda: reconstruct_standalone_runtime(artifact),
    )
    encoded = _timed(
        stages,
        "scene-linear-to-encoded",
        lambda: linear_srgb_to_encoded(scene.astype(np.float64)),
    )
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=4000
        ),
    )
    halo = required_spatial_response_halo(compiled.profile)
    linear = _timed(
        stages,
        "encoded-to-linear",
        lambda: encoded_srgb_to_linear(encoded),
    )
    gauged_encoded = np.empty_like(encoded, dtype=np.float64)
    physical_seconds = 0.0
    gauge_seconds = 0.0
    oetf_seconds = 0.0
    for y0 in range(0, encoded.shape[0], tile_rows):
        y1 = min(encoded.shape[0], y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(encoded.shape[0], y1 + halo)
        started = time.perf_counter()
        physical = _render_physical(
            linear[source_y0:source_y1], compiled
        )
        physical_seconds += time.perf_counter() - started
        core = physical[y0 - source_y0 : y1 - source_y0]
        started = time.perf_counter()
        gauged = apply_gauge_to_intermediate(core, gauge)
        gauge_seconds += time.perf_counter() - started
        started = time.perf_counter()
        gauged_encoded[y0:y1] = linear_srgb_to_encoded(gauged)
        oetf_seconds += time.perf_counter() - started
    rss = int(psutil.Process().memory_info().rss)
    stages["physical-spatial"] = {
        "elapsed_seconds": physical_seconds,
        "rss_bytes_after_stage": rss,
    }
    stages["neutral-gauge"] = {
        "elapsed_seconds": gauge_seconds,
        "rss_bytes_after_stage": rss,
    }
    stages["physical-intermediate-oetf"] = {
        "elapsed_seconds": oetf_seconds,
        "rss_bytes_after_stage": rss,
    }
    apply_base, apply_residual = _timed(
        stages,
        "source-context-build",
        lambda: build_source_context_display_look_stages(
            artifact["component_payloads"][
                "ao6-source-context-display-look"
            ],
            encoded,
        ),
    )
    base = _timed(
        stages, "safe-lab-base", lambda: apply_base(gauged_encoded)
    )
    output = _timed(
        stages, "t15-c35-residual", lambda: apply_residual(base)
    )
    reference, _ = render_challenger_row_tiled(
        encoded,
        runtime,
        gauge,
        sampling_dpi=4000,
        tile_rows=tile_rows,
    )
    exact = np.array_equal(output, reference)
    return {
        "stages": stages,
        "reference_output_exact": exact,
        "output_sha256": hashlib.sha256(
            np.ascontiguousarray(output).tobytes()
        ).hexdigest(),
        "total_attributed_seconds": float(
            sum(row["elapsed_seconds"] for row in stages.values())
        ),
    }


def evaluate_stage_attribution(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    profile = validate_contract(root, config)
    artifact = compile_standalone_profile_artifact(
        root=root, config=profile
    )
    scene = _analytic_scene(
        int(config["height"]), int(config["width"])
    )
    runs = [
        _profile_once(
            artifact, scene, tile_rows=int(config["tile_rows"])
        )
        for _ in range(int(config["repeats"]))
    ]
    exact = all(row["reference_output_exact"] for row in runs)
    hashes = {row["output_sha256"] for row in runs}
    stable_core = {
        "schema": "neuro_film.u6_p8e_cpu_stage_stable_evidence.v1",
        "height": int(config["height"]),
        "width": int(config["width"]),
        "tile_rows": int(config["tile_rows"]),
        "reference_output_exact": exact,
        "repeat_output_exact": len(hashes) == 1,
        "output_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
    }
    return {
        "schema": "neuro_film.u6_p8e_cpu_stage_attribution_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "runs": runs,
        "stable_evidence": stable_core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_core)
        ).hexdigest(),
        "decision": (
            "attribution-complete"
            if exact and len(hashes) == 1
            else "identity-failure"
        ),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_stage_attribution",
    "validate_contract",
    "write_report",
]
