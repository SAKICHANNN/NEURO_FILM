"""U6.P7F neutral-gauged full-print and spatial-chain evaluation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import (
    new_hard_clipping_fraction,
    sha256_file,
)
from src.eval.physical_joint_ablation import load_contracts
from src.eval.physical_nonspatial_attribution import _metric_sample
from src.eval.physical_spatial_stage_attribution import _gradient_energy
from src.eval.physical_virtual_scan_sampling import (
    _median_delta_e76,
    _render_physical,
    compile_virtual_scan_profile,
)
from src.real_film.gold_matrix_transplant import style_and_basic_residual
from src.roll2film.sensitometry_gauge import NeutralAxisGaugeOperator


SCHEMA = "neuro_film.u6_p7f_neutral_gauged_physical_chain_contract.v1"


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[Any, NeutralAxisGaugeOperator]:
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("post_result_retuning_allowed")
        or config["selection"].get("production_promotion_allowed")
    ):
        raise ValueError("unsupported U6.P7F contract")
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    gauge_config = _load_exact_json(
        root,
        config["gauge_config"],
        config["gauge_config_sha256"],
    )
    gauge_decision = _load_exact_json(
        root,
        config["gauge_decision"],
        config["gauge_decision_sha256"],
    )
    runtime_parent = _load_exact_json(
        root,
        config["runtime_parent"],
        config["runtime_parent_sha256"],
    )
    if (
        not parent["next_leaf"].startswith("U6.P7F")
        or parent["production_default_changed"]
        or gauge_decision["decision"] != "neutral_axis_gauge_numerical_pass"
        or int(gauge_config["gauge_knots"]) != int(config["gauge_knots"])
        or config["candidate_preference_order"]
        != ["gauged_spatial_4000", "gauged_full_print"]
    ):
        raise ValueError("U6.P7F parent or gauge drift")
    _, runtime = load_contracts(root, runtime_parent)
    gauge = NeutralAxisGaugeOperator.from_base(
        runtime.print_operator, int(config["gauge_knots"])
    )
    return runtime, gauge


def apply_gauge_to_intermediate(
    values: np.ndarray, gauge: NeutralAxisGaugeOperator
) -> np.ndarray:
    intermediate = np.asarray(values, dtype=np.float64)
    if (
        intermediate.ndim != 3
        or intermediate.shape[-1] != 3
        or not np.all(np.isfinite(intermediate))
        or np.any(intermediate < 0.0)
        or np.any(intermediate > 1.0)
    ):
        raise ValueError("gauge intermediate must be finite HxWx3 in [0, 1]")
    output = np.stack(
        [
            gauge.inverse_neutral_splines[channel].apply(
                intermediate[..., channel]
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -1e-12)
        or np.any(output > 1.0 + 1e-12)
    ):
        raise RuntimeError("gauged intermediate escaped [0, 1]")
    return np.clip(output, 0.0, 1.0)


def _render_arms(
    source: np.ndarray,
    runtime: Any,
    gauge: NeutralAxisGaugeOperator,
    *,
    sampling_dpi: int,
) -> dict[str, np.ndarray]:
    encoded = np.asarray(source, dtype=np.float64)
    linear = encoded_srgb_to_linear(encoded)
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=sampling_dpi
        ),
    )
    full_print = runtime.print_operator.apply(linear)
    full_spatial = _render_physical(linear, compiled)
    gauged_print = apply_gauge_to_intermediate(full_print, gauge)
    gauged_spatial = apply_gauge_to_intermediate(full_spatial, gauge)
    apply_colour = runtime.build_source_context_colour(encoded)
    outputs = {
        "colour_only": apply_colour(encoded),
        "gauged_full_print": apply_colour(
            linear_srgb_to_encoded(gauged_print)
        ),
        "gauged_spatial_4000": apply_colour(
            linear_srgb_to_encoded(gauged_spatial)
        ),
        "full_print_control": apply_colour(
            linear_srgb_to_encoded(full_print)
        ),
        "full_spatial_4000_control": apply_colour(
            linear_srgb_to_encoded(full_spatial)
        ),
    }
    for arm_id, values in outputs.items():
        if (
            values.shape != encoded.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise RuntimeError(f"{arm_id} left encoded RGB domain")
    return outputs


def _neutral_axis(
    runtime: Any, gauge: NeutralAxisGaugeOperator, levels: int
) -> dict[str, float]:
    axis = np.linspace(0.0, 1.0, levels, dtype=np.float64)
    linear = np.repeat(axis[:, None], 3, axis=1)
    ungauged = runtime.print_operator.apply(linear)
    gauged = np.stack(
        [
            gauge.inverse_neutral_splines[channel].apply(
                ungauged[..., channel]
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    encoded = linear_srgb_to_encoded(gauged.reshape(-1, 1, 3))
    lab = rgb2lab(encoded).reshape(-1, 3)
    return {
        "maximum_lab_chroma": float(
            np.max(np.linalg.norm(lab[:, 1:3], axis=1))
        ),
        "maximum_encoded_channel_spread": float(
            np.max(np.ptp(encoded.reshape(-1, 3), axis=1))
        ),
        "minimum_lstar_step": float(np.min(np.diff(lab[:, 0]))),
    }


def _save_rgb(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        path, format="PNG", compress_level=6
    )
    return sha256_file(path)


def evaluate_gauged_chain(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    runtime, gauge = validate_contract(root, config)
    synthetic = np.random.default_rng(20260729).random((65, 67, 3))
    first_synthetic = _render_arms(
        synthetic,
        runtime,
        gauge,
        sampling_dpi=int(config["sampling_dpi"]),
    )
    second_synthetic = _render_arms(
        synthetic,
        runtime,
        gauge,
        sampling_dpi=int(config["sampling_dpi"]),
    )
    repeat = all(
        np.array_equal(first_synthetic[name], second_synthetic[name])
        for name in first_synthetic
    )
    neutral = _neutral_axis(runtime, gauge, 16385)
    rows: list[dict[str, Any]] = []
    for sample_id in runtime.eligible_ids:
        source_row = runtime.source_rows[sample_id]
        source_path = root / source_row["decoded_path"]
        if sha256_file(source_path) != source_row["decoded_sha256"]:
            raise ValueError(f"source hash drift: {sample_id}")
        with Image.open(source_path) as image:
            source = (
                np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    dtype=np.float64,
                )
                / 255.0
            )
        outputs = _render_arms(
            source,
            runtime,
            gauge,
            sampling_dpi=int(config["sampling_dpi"]),
        )
        source_sample = _metric_sample(source)
        nonspatial_energy = _gradient_energy(
            encoded_srgb_to_linear(outputs["gauged_full_print"])
        )
        edge_mask = nonspatial_energy >= max(
            float(np.percentile(nonspatial_energy, 75.0)), 1e-12
        )
        baseline_edge = max(
            float(np.mean(nonspatial_energy[edge_mask])), 1e-12
        )
        for arm_id in config["arms"][1:]:
            values = outputs[arm_id]
            sample = _metric_sample(values)
            style, non_basic = style_and_basic_residual(
                source_sample, sample
            )
            spatial_edge_ratio = None
            if arm_id == "gauged_spatial_4000":
                spatial_edge_ratio = float(
                    np.mean(
                        _gradient_energy(
                            encoded_srgb_to_linear(values)
                        )[edge_mask]
                    )
                ) / baseline_edge
            rows.append(
                {
                    "sample_id": sample_id,
                    "make": source_row["make"],
                    "arm_id": arm_id,
                    "source_sha256": source_row["decoded_sha256"],
                    "output_sha256": _save_rgb(
                        output_dir
                        / "renders"
                        / arm_id
                        / f"{sample_id}.png",
                        values,
                    ),
                    "metrics": {
                        "style_delta_e76": style,
                        "non_basic_delta_e76": non_basic,
                        "to_colour_delta_e76": _median_delta_e76(
                            values, outputs["colour_only"]
                        ),
                        "to_gauged_nonspatial_delta_e76": (
                            _median_delta_e76(
                                values, outputs["gauged_full_print"]
                            )
                        ),
                        "new_hard_boundary_fraction": (
                            new_hard_clipping_fraction(
                                source_sample,
                                sample,
                                0.5 / 255.0,
                            )
                        ),
                        "edge_energy_ratio_to_gauged_nonspatial": (
                            spatial_edge_ratio
                        ),
                    },
                }
            )
    if len(rows) != 4 * 16:
        raise ValueError("U6.P7F population drift")

    gates = config["automatic_gates"]
    summaries: dict[str, Any] = {}
    decisions: dict[str, dict[str, bool]] = {}
    for arm_id in config["arms"][1:]:
        subset = [row for row in rows if row["arm_id"] == arm_id]
        styles = [row["metrics"]["style_delta_e76"] for row in subset]
        non_basic = [row["metrics"]["non_basic_delta_e76"] for row in subset]
        to_colour = [row["metrics"]["to_colour_delta_e76"] for row in subset]
        boundaries = [
            row["metrics"]["new_hard_boundary_fraction"] for row in subset
        ]
        summary = {
            "maximum_style_delta_e76": float(max(styles)),
            "median_non_basic_delta_e76": float(np.median(non_basic)),
            "median_to_colour_delta_e76": float(np.median(to_colour)),
            "images_to_colour_delta_e76_at_least_0_5": int(
                sum(value >= 0.5 for value in to_colour)
            ),
            "maximum_new_hard_boundary_fraction": float(max(boundaries)),
        }
        if arm_id == "gauged_spatial_4000":
            spatial_delta = [
                row["metrics"]["to_gauged_nonspatial_delta_e76"]
                for row in subset
            ]
            edge = [
                row["metrics"]["edge_energy_ratio_to_gauged_nonspatial"]
                for row in subset
            ]
            summary.update(
                {
                    "median_to_gauged_nonspatial_delta_e76": float(
                        np.median(spatial_delta)
                    ),
                    "median_edge_energy_ratio_to_gauged_nonspatial": float(
                        np.median(edge)
                    ),
                    "worst_edge_energy_ratio_to_gauged_nonspatial": float(
                        min(edge)
                    ),
                }
            )
        summaries[arm_id] = summary
        checks = {
            "repeat_exact": repeat,
            "new_boundaries": max(boundaries)
            <= float(gates["maximum_new_hard_boundary_fraction"]),
            "style_envelope": max(styles)
            <= float(gates["maximum_per_image_style_delta_e76"]),
            "non_basic": float(np.median(non_basic))
            >= float(gates["minimum_median_non_basic_delta_e76"]),
            "material": float(np.median(to_colour))
            >= float(gates["minimum_to_colour_median_delta_e76"]),
            "population": sum(value >= 0.5 for value in to_colour)
            >= int(
                gates[
                    "minimum_images_to_colour_delta_e76_at_least_0_5"
                ]
            ),
        }
        if arm_id in {
            "gauged_full_print",
            "gauged_spatial_4000",
        }:
            checks["neutral_chroma"] = neutral["maximum_lab_chroma"] <= float(
                gates["maximum_neutral_axis_lab_chroma"]
            )
        if arm_id == "gauged_spatial_4000":
            checks.update(
                {
                    "spatial_material": summary[
                        "median_to_gauged_nonspatial_delta_e76"
                    ]
                    >= float(
                        gates[
                            "minimum_spatial_to_nonspatial_median_delta_e76"
                        ]
                    ),
                    "median_edge_retention": summary[
                        "median_edge_energy_ratio_to_gauged_nonspatial"
                    ]
                    >= float(
                        gates["minimum_spatial_median_edge_energy_ratio"]
                    ),
                    "worst_edge_retention": summary[
                        "worst_edge_energy_ratio_to_gauged_nonspatial"
                    ]
                    >= float(
                        gates["minimum_spatial_worst_edge_energy_ratio"]
                    ),
                }
            )
        decisions[arm_id] = checks
    eligible = [
        arm_id
        for arm_id in config["candidate_preference_order"]
        if all(decisions[arm_id].values())
    ]
    selected = eligible[0] if eligible else None
    core = {
        "schema": "neuro_film.u6_p7f_neutral_gauged_physical_chain_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "neutral_axis": neutral,
        "rows": rows,
        "summaries": summaries,
        "decisions": decisions,
        "automatic_eligible_arm_ids": eligible,
        "selected_candidate_arm_id": selected,
        "branch": config["branch_rule"][
            "automatic_pass"
            if selected is not None
            else "no_candidate_automatic_pass"
        ],
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
    "_neutral_axis",
    "_render_arms",
    "apply_gauge_to_intermediate",
    "evaluate_gauged_chain",
    "validate_contract",
    "write_report",
]
