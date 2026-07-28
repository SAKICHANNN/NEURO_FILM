"""U6.P7D explicit virtual-scan sampling compiler and audit."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
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
from src.eval.physical_spatial_stage_attribution import (
    _gradient_energy,
    _sample,
)
from src.real_film.gold_matrix_transplant import style_and_basic_residual
from src.film_physics import (
    SpatialResponseProfile,
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
)


SCHEMA = "neuro_film.u6_p7d_virtual_scan_sampling_audit_contract.v1"


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def compile_virtual_scan_profile(
    profile: SpatialResponseProfile, *, sampling_dpi: int
) -> SpatialResponseProfile:
    """Compile physical micrometre sigmas to an explicit virtual scan pitch."""

    if (
        isinstance(sampling_dpi, bool)
        or not isinstance(sampling_dpi, int)
        or sampling_dpi <= 0
    ):
        raise ValueError("sampling_dpi must be a positive integer")
    return replace(profile, pixel_pitch_um=25_400.0 / float(sampling_dpi))


def validate_contract(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("post_result_retuning_allowed")
        or config["selection"].get("production_promotion_allowed")
    ):
        raise ValueError("unsupported U6.P7D contract")
    _load_exact_json(
        root,
        config["parent_attribution"],
        config["parent_attribution_sha256"],
    )
    decision = _load_exact_json(
        root, config["parent_decision"], config["parent_decision_sha256"]
    )
    runtime_parent = _load_exact_json(
        root, config["runtime_parent"], config["runtime_parent_sha256"]
    )
    if (
        decision.get("next_leaf")
        != "U6.P7D explicit virtual-scan sampling compiler and fixed-DPI sensitivity audit"
        or decision.get("production_default_changed")
    ):
        raise ValueError("U6.P7D parent decision drift")
    seen: set[int] = set()
    for row in [*config["sampling_candidates"], config["negative_control"]]:
        dpi = int(row["sampling_dpi"])
        pitch = float(row["pixel_pitch_um"])
        if dpi in seen or abs(pitch - 25_400.0 / dpi) > 1e-12:
            raise ValueError("virtual scan candidate identity drift")
        seen.add(dpi)
    return runtime_parent


def _render_physical(
    linear: np.ndarray, runtime: Any
) -> np.ndarray:
    exposure = apply_forward_scatter(linear, runtime.profile)
    density = runtime.print_operator.sensitometry.apply(exposure)
    density = runtime.apply_adjacency(density, runtime.profile)
    density = apply_dye_diffusion(density, runtime.profile)
    interpreted = _canonicalize_endpoint_roundoff(
        runtime.print_operator.interpretation.apply(density)
    )
    return _canonicalize_endpoint_roundoff(
        apply_scanner_mtf(interpreted, runtime.profile)
    )


def _render_candidate(
    source: np.ndarray,
    runtime: Any,
    *,
    sampling_dpi: int,
) -> dict[str, np.ndarray]:
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=sampling_dpi
        ),
    )
    apply_colour = runtime.build_source_context_colour(source)
    colour = apply_colour(source)
    if not np.array_equal(colour, runtime.apply_colour(source)):
        raise RuntimeError("virtual-scan audit changed colour-only pixels")
    linear = encoded_srgb_to_linear(source)
    cheap_linear = _canonicalize_endpoint_roundoff(
        runtime.print_operator.apply(linear)
    )
    cheap = apply_colour(linear_srgb_to_encoded(cheap_linear))
    physical_linear = _render_physical(linear, compiled)
    combined = apply_colour(linear_srgb_to_encoded(physical_linear))
    wrong_linear = _render_physical(
        encoded_srgb_to_linear(colour), compiled
    )
    wrong = linear_srgb_to_encoded(wrong_linear)
    outputs = {
        "colour_only": colour,
        "cheap": cheap,
        "combined": combined,
        "wrong_order": wrong,
    }
    for name, values in outputs.items():
        if (
            values.shape != source.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise RuntimeError(f"{name} left encoded RGB domain")
    return outputs


def _median_delta_e76(left: np.ndarray, right: np.ndarray) -> float:
    first = rgb2lab(_sample(left).reshape(-1, 1, 3))
    second = rgb2lab(_sample(right).reshape(-1, 1, 3))
    return float(np.median(np.linalg.norm(first - second, axis=-1)))


def _save_rgb(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        path, format="PNG", compress_level=6
    )
    return sha256_file(path)


def _save_diagnostic(
    rows: dict[str, dict[str, np.ndarray]],
    selected: str,
    output_dir: Path,
) -> str:
    columns = ("source", "colour_only", "cheap", "combined")
    tile_size = (320, 210)
    header = 24
    canvas = Image.new(
        "RGB",
        (
            len(columns) * tile_size[0],
            len(rows) * (tile_size[1] + header),
        ),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (sample_id, values) in enumerate(rows.items()):
        y = row_index * (tile_size[1] + header)
        for column_index, name in enumerate(columns):
            pixels = np.rint(values[name] * 255.0).astype(np.uint8)
            tile = ImageOps.fit(
                Image.fromarray(pixels, mode="RGB"),
                tile_size,
                method=Image.Resampling.LANCZOS,
            )
            canvas.paste(
                tile, (column_index * tile_size[0], y + header)
            )
            draw.text(
                (column_index * tile_size[0] + 4, y + 4),
                (
                    sample_id
                    if column_index == 0
                    else f"{name} {selected}"
                ),
                fill=(235, 235, 235),
            )
    path = output_dir / "selected_diagnostic.png"
    canvas.save(path, format="PNG", compress_level=6)
    return sha256_file(path)


def _synthetic_repeat(
    runtime: Any, candidates: list[dict[str, Any]]
) -> dict[str, bool]:
    source = np.random.default_rng(20260728).random((65, 67, 3))
    result: dict[str, bool] = {}
    for row in candidates:
        first = _render_candidate(
            source, runtime, sampling_dpi=int(row["sampling_dpi"])
        )
        second = _render_candidate(
            source, runtime, sampling_dpi=int(row["sampling_dpi"])
        )
        result[row["candidate_id"]] = all(
            np.array_equal(first[name], second[name]) for name in first
        )
    return result


def evaluate_sampling_audit(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    runtime_parent = validate_contract(root, config)
    _, runtime = load_contracts(root, runtime_parent)
    candidates = [
        *config["sampling_candidates"],
        config["negative_control"],
    ]
    repeat = _synthetic_repeat(runtime, candidates)
    rows: list[dict[str, Any]] = []
    visual_ids = {
        "canon_eos_kiss_f",
        "nikon_d2x",
        "sony_nex_3n",
        "fujifilm_finepix_s5000",
        "olympus_sp550uz",
        "panasonic_dmc_gf2",
        "pentax_k_r",
        "leica_d_lux_6",
        "kodak_dcs_pro_14n",
    }
    visual_bank: dict[str, dict[str, dict[str, np.ndarray]]] = {
        row["candidate_id"]: {} for row in candidates
    }
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
        source_energy = _gradient_energy(encoded_srgb_to_linear(source))
        edge_mask = source_energy >= max(
            float(np.percentile(source_energy, 75.0)), 1e-12
        )
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            outputs = _render_candidate(
                source,
                runtime,
                sampling_dpi=int(candidate["sampling_dpi"]),
            )
            source_sample = _sample(source)
            combined_sample = _sample(outputs["combined"])
            style, non_basic = style_and_basic_residual(
                source_sample, combined_sample
            )
            cheap_energy = max(
                float(
                    np.mean(
                        _gradient_energy(
                            encoded_srgb_to_linear(outputs["cheap"])
                        )[edge_mask]
                    )
                ),
                1e-12,
            )
            combined_energy = float(
                np.mean(
                    _gradient_energy(
                        encoded_srgb_to_linear(outputs["combined"])
                    )[edge_mask]
                )
            )
            output_hash = _save_rgb(
                output_dir
                / "renders"
                / candidate_id
                / f"{sample_id}.png",
                outputs["combined"],
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "sampling_dpi": int(candidate["sampling_dpi"]),
                    "pixel_pitch_um": float(candidate["pixel_pitch_um"]),
                    "sample_id": sample_id,
                    "make": source_row["make"],
                    "source_sha256": source_row["decoded_sha256"],
                    "output_sha256": output_hash,
                    "metrics": {
                        "combined_style_delta_e76": style,
                        "combined_non_basic_delta_e76": non_basic,
                        "combined_to_colour_delta_e76": _median_delta_e76(
                            outputs["combined"], outputs["colour_only"]
                        ),
                        "full_to_cheap_delta_e76": _median_delta_e76(
                            outputs["combined"], outputs["cheap"]
                        ),
                        "combined_to_wrong_order_delta_e76": _median_delta_e76(
                            outputs["combined"], outputs["wrong_order"]
                        ),
                        "post_ao6_edge_energy_ratio_to_cheap": (
                            combined_energy / cheap_energy
                        ),
                        "new_hard_boundary_fraction": (
                            new_hard_clipping_fraction(
                                source_sample,
                                combined_sample,
                                0.5 / 255.0,
                            )
                        ),
                        "synthetic_repeat_identity": repeat[candidate_id],
                    },
                }
            )
            if sample_id in visual_ids:
                visual_bank[candidate_id][sample_id] = {
                    "source": source,
                    **outputs,
                }
    if len(rows) != len(candidates) * 16:
        raise ValueError("U6.P7D population drift")

    gates = config["automatic_gates"]
    summaries: dict[str, Any] = {}
    decisions: dict[str, dict[str, bool]] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        subset = [row for row in rows if row["candidate_id"] == candidate_id]
        styles = [
            row["metrics"]["combined_style_delta_e76"] for row in subset
        ]
        non_basic = [
            row["metrics"]["combined_non_basic_delta_e76"] for row in subset
        ]
        combined_delta = [
            row["metrics"]["combined_to_colour_delta_e76"] for row in subset
        ]
        full_cheap = [
            row["metrics"]["full_to_cheap_delta_e76"] for row in subset
        ]
        wrong = [
            row["metrics"]["combined_to_wrong_order_delta_e76"]
            for row in subset
        ]
        edge = [
            row["metrics"]["post_ao6_edge_energy_ratio_to_cheap"]
            for row in subset
        ]
        boundaries = [
            row["metrics"]["new_hard_boundary_fraction"] for row in subset
        ]
        summaries[candidate_id] = {
            "sampling_dpi": int(candidate["sampling_dpi"]),
            "pixel_pitch_um": float(candidate["pixel_pitch_um"]),
            "maximum_style_delta_e76": float(max(styles)),
            "median_non_basic_delta_e76": float(np.median(non_basic)),
            "median_combined_to_colour_delta_e76": float(
                np.median(combined_delta)
            ),
            "images_combined_to_colour_at_least_0_5": int(
                sum(value >= 0.5 for value in combined_delta)
            ),
            "median_full_to_cheap_delta_e76": float(
                np.median(full_cheap)
            ),
            "median_combined_to_wrong_order_delta_e76": float(
                np.median(wrong)
            ),
            "median_post_ao6_edge_energy_ratio_to_cheap": float(
                np.median(edge)
            ),
            "worst_post_ao6_edge_energy_ratio_to_cheap": float(min(edge)),
            "maximum_new_hard_boundary_fraction": float(max(boundaries)),
        }
        decisions[candidate_id] = {
            "repeat_exact": repeat[candidate_id],
            "new_boundaries": max(boundaries)
            <= float(gates["maximum_new_hard_boundary_fraction"]),
            "style_envelope": max(styles)
            <= float(gates["maximum_combined_per_image_style_delta_e76"]),
            "non_basic": float(np.median(non_basic))
            >= float(gates["minimum_combined_median_non_basic_delta_e76"]),
            "physics_material": float(np.median(combined_delta))
            >= float(gates["minimum_combined_to_colour_median_delta_e76"]),
            "physics_population": sum(value >= 0.5 for value in combined_delta)
            >= int(
                gates[
                    "minimum_images_with_combined_to_colour_delta_e76_at_least_0_5"
                ]
            ),
            "full_not_cheap": float(np.median(full_cheap))
            >= float(gates["minimum_full_to_cheap_median_delta_e76"]),
            "order_identifiable": float(np.median(wrong))
            >= float(
                gates["minimum_combined_to_wrong_order_median_delta_e76"]
            ),
            "median_edge_retention": float(np.median(edge))
            >= float(
                gates[
                    "minimum_median_post_ao6_edge_energy_ratio_to_cheap"
                ]
            ),
            "worst_edge_retention": min(edge)
            >= float(
                gates[
                    "minimum_worst_image_post_ao6_edge_energy_ratio_to_cheap"
                ]
            ),
        }
    eligible = [
        row["candidate_id"]
        for row in config["sampling_candidates"]
        if all(decisions[row["candidate_id"]].values())
    ]
    selected = (
        max(
            eligible,
            key=lambda candidate_id: summaries[candidate_id][
                "sampling_dpi"
            ],
        )
        if eligible
        else None
    )
    negative_id = config["negative_control"]["candidate_id"]
    negative_hashes = {
        row["sample_id"]: row["output_sha256"]
        for row in rows
        if row["candidate_id"] == negative_id
    }
    p7c_report = json.loads(
        (
            root
            / "outputs"
            / "u6_p7c_spatial_stage_attribution_v1"
            / "run_a"
            / "report.json"
        ).read_text(encoding="utf-8")
    )
    expected_negative = {
        row["sample_id"]: row["output_sha256"]["scanner_mtf"]
        for row in p7c_report["rows"]
    }
    negative_reproduced = negative_hashes == expected_negative
    visual: dict[str, Any] | str
    if selected is None:
        visual = "forbidden_by_automatic_gate"
    else:
        visual = {
            "selected_candidate_id": selected,
            "diagnostic_sha256": _save_diagnostic(
                visual_bank[selected], selected, output_dir
            ),
            "blind_preference_performed": False,
        }
    core = {
        "schema": "neuro_film.u6_p7d_virtual_scan_sampling_audit_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "rows": rows,
        "summaries": summaries,
        "decisions": decisions,
        "negative_control_reproduced_p7c": negative_reproduced,
        "automatic_eligible_candidate_ids": eligible,
        "selected_candidate_id": selected,
        "visual_evidence": visual,
        "branch": config["branch_rule"][
            "no_candidate_automatic_pass"
            if selected is None
            else "complete_pass"
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
    "compile_virtual_scan_profile",
    "evaluate_sampling_audit",
    "validate_contract",
    "write_report",
]
