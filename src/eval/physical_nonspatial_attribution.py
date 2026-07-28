"""U6.P7E non-spatial sensitometry/print interpretation attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

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
from src.eval.physical_joint_ablation import (
    _canonicalize_endpoint_roundoff,
    load_contracts,
)
from src.eval.physical_virtual_scan_sampling import _median_delta_e76
from src.real_film.gold_matrix_transplant import style_and_basic_residual
from src.roll2film.sensitometry_print import DensityToPrintInterpretation


SCHEMA = (
    "neuro_film.u6_p7e_nonspatial_interpretation_attribution_contract.v1"
)


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: dict[str, Any]) -> Any:
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("post_result_retuning_allowed")
        or config["selection"].get("production_promotion_allowed")
    ):
        raise ValueError("unsupported U6.P7E contract")
    decision = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    runtime_parent = _load_exact_json(
        root,
        config["runtime_parent"],
        config["runtime_parent_sha256"],
    )
    if (
        decision["visual_result"]["decision"] != "preference_fail"
        or decision["production_default_changed"]
        or not decision["next_leaf"].startswith("U6.P7E")
        or config["candidate_arm_ids"]
        != [
            "normalized_density",
            "channelwise_paper",
            "dye_matrix_only",
            "print_matrix_only",
        ]
        or [row["arm_id"] for row in config["arms"]]
        != [
            "normalized_density",
            "channelwise_paper",
            "dye_matrix_only",
            "print_matrix_only",
            "full_print_control",
        ]
    ):
        raise ValueError("U6.P7E parent or arm identity drift")
    _, runtime = load_contracts(root, runtime_parent)
    return runtime


def _interpretations(runtime: Any) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    source = runtime.print_operator.interpretation
    identity = np.eye(3, dtype=np.float64)

    def build(
        dye: np.ndarray, print_matrix: np.ndarray
    ) -> DensityToPrintInterpretation:
        return DensityToPrintInterpretation(
            dye,
            print_matrix,
            source.paper_midpoints,
            source.paper_slopes,
            source.paper_maximum_densities,
            source.black_reference_density,
            source.white_reference_density,
        )

    channelwise = build(identity, identity)
    dye_only = build(source.dye_absorption_matrix, identity)
    print_only = build(identity, source.print_matrix)

    def normalized_density(density: np.ndarray) -> np.ndarray:
        output = (
            density - source.black_reference_density
        ) / (
            source.white_reference_density - source.black_reference_density
        )
        return _canonicalize_endpoint_roundoff(output)

    return {
        "normalized_density": normalized_density,
        "channelwise_paper": channelwise.apply,
        "dye_matrix_only": dye_only.apply,
        "print_matrix_only": print_only.apply,
        "full_print_control": source.apply,
    }


def _render_arms(source: np.ndarray, runtime: Any) -> dict[str, np.ndarray]:
    encoded = np.asarray(source, dtype=np.float64)
    linear = encoded_srgb_to_linear(encoded)
    density = runtime.print_operator.sensitometry.apply(linear)
    apply_colour = runtime.build_source_context_colour(encoded)
    colour = apply_colour(encoded)
    outputs = {"colour_only": colour}
    for arm_id, interpret in _interpretations(runtime).items():
        physical = linear_srgb_to_encoded(
            _canonicalize_endpoint_roundoff(interpret(density))
        )
        outputs[arm_id] = apply_colour(physical)
    for arm_id, values in outputs.items():
        if (
            values.shape != encoded.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise RuntimeError(f"{arm_id} left encoded RGB domain")
    return outputs


def _neutral_axis(runtime: Any, levels: int) -> dict[str, Any]:
    axis = np.linspace(0.0, 1.0, levels, dtype=np.float64)
    linear = np.repeat(axis[:, None], 3, axis=1)
    density = runtime.print_operator.sensitometry.apply(linear)
    result: dict[str, Any] = {}
    for arm_id, interpret in _interpretations(runtime).items():
        interpreted = _canonicalize_endpoint_roundoff(
            interpret(density)
        ).reshape(-1, 1, 3)
        encoded = linear_srgb_to_encoded(interpreted)
        lab = rgb2lab(encoded).reshape(-1, 3)
        chroma = np.linalg.norm(lab[:, 1:3], axis=1)
        result[arm_id] = {
            "maximum_lab_chroma": float(np.max(chroma)),
            "median_lab_a": float(np.median(lab[:, 1])),
            "median_lab_b": float(np.median(lab[:, 2])),
            "minimum_lstar_step": float(np.min(np.diff(lab[:, 0]))),
        }
    return result


def _save_rgb(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        path, format="PNG", compress_level=6
    )
    return sha256_file(path)


def evaluate_attribution(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    runtime = validate_contract(root, config)
    neutral = _neutral_axis(
        runtime, int(config["neutral_axis"]["linear_levels"])
    )
    synthetic = np.random.default_rng(20260729).random((65, 67, 3))
    synthetic_first = _render_arms(synthetic, runtime)
    synthetic_second = _render_arms(synthetic, runtime)
    repeat = all(
        np.array_equal(synthetic_first[name], synthetic_second[name])
        for name in synthetic_first
    )
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
        first = _render_arms(source, runtime)
        source_sample = source[::8, ::8]
        colour_sample = first["colour_only"][::8, ::8]
        for arm in config["arms"]:
            arm_id = arm["arm_id"]
            values = first[arm_id]
            sample = values[::8, ::8]
            style, non_basic = style_and_basic_residual(
                source_sample, sample
            )
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
                            values, first["colour_only"]
                        ),
                        "to_full_print_delta_e76": _median_delta_e76(
                            values, first["full_print_control"]
                        ),
                        "new_hard_boundary_fraction": (
                            new_hard_clipping_fraction(
                                source_sample,
                                sample,
                                0.5 / 255.0,
                            )
                        ),
                    },
                }
            )
    if len(rows) != len(config["arms"]) * 16:
        raise ValueError("U6.P7E population drift")

    gates = config["automatic_gates"]
    summaries: dict[str, Any] = {}
    decisions: dict[str, dict[str, bool]] = {}
    for arm in config["arms"]:
        arm_id = arm["arm_id"]
        subset = [row for row in rows if row["arm_id"] == arm_id]
        styles = [row["metrics"]["style_delta_e76"] for row in subset]
        non_basic = [row["metrics"]["non_basic_delta_e76"] for row in subset]
        to_colour = [row["metrics"]["to_colour_delta_e76"] for row in subset]
        to_full = [row["metrics"]["to_full_print_delta_e76"] for row in subset]
        boundaries = [
            row["metrics"]["new_hard_boundary_fraction"] for row in subset
        ]
        summaries[arm_id] = {
            "maximum_style_delta_e76": float(max(styles)),
            "median_non_basic_delta_e76": float(np.median(non_basic)),
            "median_to_colour_delta_e76": float(np.median(to_colour)),
            "images_to_colour_delta_e76_at_least_0_5": int(
                sum(value >= 0.5 for value in to_colour)
            ),
            "median_to_full_print_delta_e76": float(np.median(to_full)),
            "maximum_new_hard_boundary_fraction": float(max(boundaries)),
            **neutral[arm_id],
        }
        decisions[arm_id] = {
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
            "neutral_chroma": neutral[arm_id]["maximum_lab_chroma"]
            <= float(
                config["neutral_axis"]["maximum_encoded_lab_chroma"]
            ),
            "neutral_monotone": neutral[arm_id]["minimum_lstar_step"]
            >= float(config["neutral_axis"]["minimum_lstar_step"]),
        }
    eligible = [
        arm_id
        for arm_id in config["candidate_arm_ids"]
        if all(decisions[arm_id].values())
    ]
    selected = (
        min(
            eligible,
            key=lambda arm_id: (
                summaries[arm_id]["maximum_lab_chroma"],
                -summaries[arm_id]["median_to_colour_delta_e76"],
            ),
        )
        if eligible
        else None
    )
    core = {
        "schema": (
            "neuro_film.u6_p7e_nonspatial_interpretation_attribution_report.v1"
        ),
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
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
    "evaluate_attribution",
    "validate_contract",
    "write_report",
]
