"""U6.P7G1 stagewise attribution of cross-resolution noncommutation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import sha256_file
from src.eval.physical_neutral_gauged_chain import (
    apply_gauge_to_intermediate,
    validate_contract as validate_p7f_contract,
)
from src.eval.physical_neutral_gauged_invariance import (
    _analytic_chart,
    _bounded_real_source,
    _resize,
    _resolution_metrics,
)
from src.eval.physical_virtual_scan_sampling import (
    _render_physical,
    compile_virtual_scan_profile,
)


SCHEMA = "neuro_film.u6_p7g1_cross_resolution_attribution_contract.v1"


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: dict[str, Any]) -> tuple[Any, Any]:
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("post_result_retuning_allowed")
        or config["stages"]
        != [
            "source_encoded_control",
            "physical_ungauged_encoded",
            "neutral_gauged_encoded",
            "ao6_colour_only_control",
            "full_challenger",
        ]
    ):
        raise ValueError("unsupported U6.P7G1 contract")
    decision = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    parent = _load_exact_json(
        root,
        config["parent_contract"],
        config["parent_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P7G1")
        or decision["tile_result"]["decision"] != "pass"
        or decision["resolution_result"]["decision"] != "fail"
        or decision["production_default_changed"]
        or int(parent["sampling_dpi"])
        != int(config["reference_sampling_dpi"])
    ):
        raise ValueError("U6.P7G1 parent evidence drift")
    return validate_p7f_contract(root, parent)


def render_stages(
    source: np.ndarray,
    runtime: Any,
    gauge: Any,
    *,
    sampling_dpi: int,
) -> dict[str, np.ndarray]:
    encoded = np.asarray(source, dtype=np.float64)
    if (
        encoded.ndim != 3
        or encoded.shape[-1] != 3
        or not np.all(np.isfinite(encoded))
        or np.any(encoded < 0.0)
        or np.any(encoded > 1.0)
    ):
        raise ValueError("stage input must be finite encoded HxWx3")
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=sampling_dpi
        ),
    )
    physical = _render_physical(
        encoded_srgb_to_linear(encoded),
        compiled,
    )
    physical_encoded = linear_srgb_to_encoded(physical)
    gauged_encoded = linear_srgb_to_encoded(
        apply_gauge_to_intermediate(physical, gauge)
    )
    apply_colour = runtime.build_source_context_colour(encoded)
    stages = {
        "source_encoded_control": encoded,
        "physical_ungauged_encoded": physical_encoded,
        "neutral_gauged_encoded": gauged_encoded,
        "ao6_colour_only_control": apply_colour(encoded),
        "full_challenger": apply_colour(gauged_encoded),
    }
    for name, values in stages.items():
        if (
            values.shape != encoded.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise RuntimeError(f"{name} left encoded RGB")
    return stages


def _passes(metrics: dict[str, float], gates: dict[str, Any]) -> bool:
    return bool(
        metrics["median_delta_e76"]
        <= float(gates["maximum_median_delta_e76"])
        and metrics["p95_delta_e76"]
        <= float(gates["maximum_p95_delta_e76"])
        and metrics["maximum_delta_e76"]
        <= float(gates["maximum_delta_e76"])
        and float(gates["minimum_edge_energy_ratio"])
        <= metrics["edge_energy_ratio"]
        <= float(gates["maximum_edge_energy_ratio"])
    )


def evaluate_attribution(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    runtime, gauge = validate_contract(root, config)
    dpi = int(config["reference_sampling_dpi"])
    high_shape = tuple(int(value) for value in config["synthetic_shape"])
    sources = {"synthetic": _analytic_chart(high_shape)}
    for sample_id in config["sample_ids"]:
        sources[sample_id] = _bounded_real_source(
            root,
            runtime,
            sample_id,
            max_long_edge=int(config["real_max_long_edge"]),
        )
    rows: list[dict[str, Any]] = []
    gates = config["comparison_gates"]
    for sample_id, high_source in sources.items():
        high_stages = render_stages(
            high_source, runtime, gauge, sampling_dpi=dpi
        )
        for scale in config["scale_factors"]:
            factor = float(scale)
            low_shape = (
                int(round(high_source.shape[0] * factor)),
                int(round(high_source.shape[1] * factor)),
            )
            low_source = _resize(high_source, low_shape)
            low_stages = render_stages(
                low_source,
                runtime,
                gauge,
                sampling_dpi=int(round(dpi * factor)),
            )
            for stage in config["stages"]:
                metrics = _resolution_metrics(
                    low_stages[stage],
                    _resize(high_stages[stage], low_shape),
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "scale_factor": factor,
                        "sampling_dpi": int(round(dpi * factor)),
                        "stage": stage,
                        **metrics,
                        "passes_p7g_comparison_gates": _passes(
                            metrics, gates
                        ),
                    }
                )
    summaries: dict[str, Any] = {}
    for stage in config["stages"]:
        subset = [row for row in rows if row["stage"] == stage]
        summaries[stage] = {
            "failed_cases": int(
                sum(
                    not row["passes_p7g_comparison_gates"]
                    for row in subset
                )
            ),
            "maximum_median_delta_e76": float(
                max(row["median_delta_e76"] for row in subset)
            ),
            "maximum_p95_delta_e76": float(
                max(row["p95_delta_e76"] for row in subset)
            ),
            "maximum_delta_e76": float(
                max(row["maximum_delta_e76"] for row in subset)
            ),
            "minimum_edge_energy_ratio": float(
                min(row["edge_energy_ratio"] for row in subset)
            ),
            "maximum_edge_energy_ratio": float(
                max(row["edge_energy_ratio"] for row in subset)
            ),
        }
    stage_pass = {
        stage: summaries[stage]["failed_cases"] == 0
        for stage in config["stages"]
    }
    if not stage_pass["source_encoded_control"]:
        attribution = "invalid_resampling_control"
    elif not stage_pass["physical_ungauged_encoded"]:
        attribution = "physical_or_nonlinear_resampling_first"
    elif not stage_pass["neutral_gauged_encoded"]:
        attribution = "neutral_gauge_first"
    elif (
        not stage_pass["ao6_colour_only_control"]
        or not stage_pass["full_challenger"]
    ):
        attribution = "ao6_or_source_context_first"
    else:
        attribution = "p7g_aspect_ratio_stress_only"
    core = {
        "schema": (
            "neuro_film.u6_p7g1_cross_resolution_attribution_report.v1"
        ),
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "rows": rows,
        "summaries": summaries,
        "stage_pass": stage_pass,
        "attribution": attribution,
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_attribution",
    "render_stages",
    "validate_contract",
    "write_report",
]
