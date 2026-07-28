"""U6.P7B source-context-fixed colour and physical-path ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.global_frontier import sha256_file
from src.eval.physical_joint_ablation import (
    evaluate_ablation,
    load_contracts,
    render_arms_with_source_context,
)
from src.eval.physical_joint_mechanism import _refresh_blind
from src.film_physics import required_spatial_response_halo


SCHEMA = "neuro_film.u6_p7b_source_context_joint_ablation_contract.v1"


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != SCHEMA or config.get("post_result_retuning_allowed"):
        raise ValueError("unsupported U6.P7B contract")
    parent = _load_exact_json(
        root, config["parent_audit"], config["parent_audit_sha256"]
    )
    decision = _load_exact_json(
        root, config["parent_decision"], config["parent_decision_sha256"]
    )
    runtime_parent = _load_exact_json(
        root, config["runtime_parent"], config["runtime_parent_sha256"]
    )
    if (
        parent.get("node") != "U6.P7A4"
        or decision.get("next_leaf")
        != "U6.P7B source-derived fixed AO6 context ablation"
        or config["single_change"].get("colour_only_equivalence_required")
        != "byte_exact"
        or not config["single_change"].get(
            "factorized_tone_chroma_residual_unchanged"
        )
        or not config["single_change"].get("physical_pipeline_unchanged")
        or not config["single_change"].get("ao6_strengths_unchanged")
    ):
        raise ValueError("U6.P7B parent or single-change boundary drift")
    return runtime_parent


def evaluate_mechanism_smoke(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    runtime_parent = validate_contract(root, config)
    _, runtime = load_contracts(root, runtime_parent)
    controls = config["mechanism_smoke"]
    shape = tuple(int(value) for value in controls["shape"])
    flat_rows: list[dict[str, Any]] = []
    repeat_exact = True
    colour_equivalent = True
    for rgb in controls["constant_rgb"]:
        source = np.broadcast_to(
            np.asarray(rgb, dtype=np.float64), (*shape, 3)
        ).copy()
        first = render_arms_with_source_context(source, runtime)
        second = render_arms_with_source_context(source, runtime)
        repeat_exact = repeat_exact and all(
            np.array_equal(first[name], second[name]) for name in first
        )
        colour_equivalent = colour_equivalent and np.array_equal(
            first["colour_only"], runtime.apply_colour(source)
        )
        flat_rows.append(
            {
                "input_rgb": rgb,
                "combined_minus_cheap_max_abs": float(
                    np.max(np.abs(first["combined"] - first["cheap"]))
                ),
                "maximum_spatial_range": float(
                    max(
                        np.max(np.ptp(values, axis=(0, 1)))
                        for values in first.values()
                    )
                ),
            }
        )

    background = np.asarray(
        controls["impulse_background_rgb"], dtype=np.float64
    )
    source = np.broadcast_to(background, (*shape, 3)).copy()
    center = (shape[0] // 2, shape[1] // 2)
    source[center] = np.asarray(controls["impulse_rgb"], dtype=np.float64)
    first = render_arms_with_source_context(source, runtime)
    second = render_arms_with_source_context(source, runtime)
    repeat_exact = repeat_exact and all(
        np.array_equal(first[name], second[name]) for name in first
    )
    colour_equivalent = colour_equivalent and np.array_equal(
        first["colour_only"], runtime.apply_colour(source)
    )
    difference = np.max(
        np.abs(first["combined"] - first["cheap"]), axis=-1
    )
    halo = required_spatial_response_halo(runtime.profile)
    yy, xx = np.indices(shape)
    outside = (np.abs(yy - center[0]) > halo) | (
        np.abs(xx - center[1]) > halo
    )
    outside_max = float(np.max(difference[outside]))
    flat_tolerance = float(
        controls["maximum_flat_combined_minus_cheap_abs"]
    )
    decisions = {
        "colour_only_equivalence": colour_equivalent,
        "flat_invariance": max(
            row["maximum_spatial_range"] for row in flat_rows
        )
        <= flat_tolerance,
        "flat_full_vs_cheap": max(
            row["combined_minus_cheap_max_abs"] for row in flat_rows
        )
        <= flat_tolerance,
        "finite_support": outside_max
        <= float(controls["maximum_outside_halo_combined_minus_cheap_abs"]),
        "repeat_exact": repeat_exact,
    }
    return {
        "flat_rows": flat_rows,
        "impulse": {
            "shape": list(shape),
            "center_yx": list(center),
            "analytic_halo_pixels": halo,
            "outside_halo_max_abs": outside_max,
            "inside_halo_max_abs": float(np.max(difference[~outside])),
        },
        "decisions": decisions,
        "automatic_pass": all(decisions.values()),
    }


def evaluate_source_context_ablation(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    runtime_parent = validate_contract(root, config)
    smoke = evaluate_mechanism_smoke(root=root, config=config)
    if not smoke["automatic_pass"]:
        photo: dict[str, Any] | None = None
        decisions = dict(smoke["decisions"])
        branch_key = "mechanism_smoke_fail"
    else:
        photo = evaluate_ablation(
            root=root,
            correction=runtime_parent,
            output_dir=output_dir,
            render_fn=render_arms_with_source_context,
            evaluate_artifact_gate=False,
            save_visual_on_automatic_pass_only=True,
        )
        if photo["automatic_pass"]:
            _, runtime = load_contracts(root, runtime_parent)
            parent = _load_exact_json(
                root,
                config["parent_audit"],
                config["parent_audit_sha256"],
            )
            refreshed = _refresh_blind(
                root=root,
                config={
                    "node": config["node"],
                    "visual_refresh": parent["visual_refresh"],
                },
                runtime=runtime,
                inherited_run_dir=output_dir,
                output_dir=output_dir,
            )
            photo["visual_evidence"] = {
                "diagnostic_sha256": photo["visual_evidence"][
                    "diagnostic_sha256"
                ],
                **refreshed,
            }
        decisions = {**smoke["decisions"], **photo["decisions"]}
        branch_key = "complete_pass" if all(decisions.values()) else "automatic_fail"
    core = {
        "schema": "neuro_film.u6_p7b_source_context_joint_ablation_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "mechanism_smoke": smoke,
        "photographic_ablation": photo,
        "decisions": decisions,
        "automatic_pass": all(decisions.values()),
        "branch": config["branch_rule"][branch_key],
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
    "evaluate_mechanism_smoke",
    "evaluate_source_context_ablation",
    "validate_contract",
    "write_report",
]
