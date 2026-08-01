"""U6.P2W fixed photographic value and safety audit."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import new_hard_clipping_fraction, sha256_file
from src.eval.physical_neutral_gauged_chain import (
    apply_gauge_to_intermediate,
    validate_contract as validate_p7f,
)
from src.eval.physical_silver_photographic import _isolated
from src.eval.physical_spatial_stage_attribution import _gradient_energy, _sample
from src.eval.physical_virtual_scan_sampling import (
    _canonicalize_endpoint_roundoff,
    _median_delta_e76,
    compile_virtual_scan_profile,
)
from src.film_physics import (
    LangmuirDonorProfile,
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
)
from src.real_film.gold_matrix_transplant import style_and_basic_residual


SCHEMA = "neuro_film.u6_p2w_langmuir_interimage_photographic_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2W contract")
    return value


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"U6.P2W parent hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _donor_profile(config: dict[str, Any], *, linear: bool) -> LangmuirDonorProfile:
    fixed = config["fixed_operator"]
    knee = (float("inf"),) * 3 if linear else tuple(fixed["langmuir_knee_fraction"])
    return LangmuirDonorProfile(
        (1.0, 1.0, 1.0),
        knee,
        tuple(fixed["match_fraction"]),
    )


def _develop_density(
    sensitometry: Any,
    linear_exposure: np.ndarray,
    coupling: np.ndarray,
    donor: LangmuirDonorProfile,
) -> np.ndarray:
    values = np.asarray(linear_exposure, dtype=np.float64)
    log_exposure = sensitometry.log_exposure(values)
    base_density = np.stack(
        [
            sensitometry.curves[channel].apply(log_exposure[..., channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    minimum = np.asarray(
        [curve.spline.y_knots[0] for curve in sensitometry.curves],
        dtype=np.float64,
    )
    maximum = np.asarray(
        [curve.spline.y_knots[-1] for curve in sensitometry.curves],
        dtype=np.float64,
    )
    fraction = (base_density - minimum) / (maximum - minimum)
    if np.any(fraction < 0.0) or np.any(fraction > 1.0):
        raise RuntimeError("sensitometry density left frozen donor normalization")
    inhibition = donor.apply(fraction) @ coupling.T
    floor = float(sensitometry.encoder.minimum_log_exposure)
    available = log_exposure - floor
    ratio = np.full_like(inhibition, np.inf)
    np.divide(available, inhibition, out=ratio, where=inhibition > 0.0)
    scale = np.minimum(1.0, np.min(ratio, axis=-1))
    if np.any(scale < 0.0) or not np.all(np.isfinite(scale)):
        raise RuntimeError("invalid analytical donor safety scale")
    corrected_exposure = log_exposure - scale[..., None] * inhibition
    roundoff = (corrected_exposure < floor) & (corrected_exposure >= floor - 1e-12)
    corrected_exposure = np.where(roundoff, floor, corrected_exposure)
    if np.any(corrected_exposure < floor):
        raise RuntimeError("analytical donor safety left the exposure domain")
    density = np.stack(
        [
            sensitometry.curves[channel].apply(corrected_exposure[..., channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    if not np.all(np.isfinite(density)):
        raise RuntimeError("interimage development produced non-finite density")
    return density


def _render_pair(
    source: np.ndarray,
    runtime: Any,
    gauge: Any,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(source, dtype=np.float64)
    linear = encoded_srgb_to_linear(encoded)
    fixed = config["fixed_operator"]
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=int(fixed["sampling_dpi"])
        ),
    )
    exposure = apply_forward_scatter(linear, compiled.profile)
    coupling = np.asarray(fixed["coupling"], dtype=np.float64)
    linear_density = _develop_density(
        runtime.print_operator.sensitometry,
        exposure,
        coupling,
        _donor_profile(config, linear=True),
    )
    langmuir_density = _develop_density(
        runtime.print_operator.sensitometry,
        exposure,
        coupling,
        _donor_profile(config, linear=False),
    )

    def finish(density: np.ndarray) -> np.ndarray:
        developed = runtime.apply_adjacency(density, compiled.profile)
        developed = apply_dye_diffusion(developed, compiled.profile)
        scan = _canonicalize_endpoint_roundoff(
            runtime.print_operator.interpretation.apply(developed)
        )
        scan = _canonicalize_endpoint_roundoff(
            apply_scanner_mtf(scan, compiled.profile)
        )
        scan = apply_gauge_to_intermediate(scan, gauge)
        apply_colour = runtime.build_source_context_colour(encoded)
        output = apply_colour(linear_srgb_to_encoded(scan))
        if (
            output.shape != encoded.shape
            or not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError("U6.P2W output escaped encoded RGB")
        return output

    return finish(linear_density), finish(langmuir_density)


def _save(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", compress_level=6)
    return sha256_file(path)


def _copy_exact(source: Path, destination: Path, expected: str) -> str:
    if sha256_file(source) != expected:
        raise ValueError(f"frozen incumbent hash drift: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != expected:
        raise RuntimeError("frozen incumbent copy changed bytes")
    return expected


def evaluate_langmuir_photographic(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parents = config["parents"]
    p2v = _load_exact(
        root, parents["p2v_decision_path"], parents["p2v_decision_sha256"]
    )
    p7f = _load_exact(
        root, parents["p7f_contract_path"], parents["p7f_contract_sha256"]
    )
    p7f_decision = _load_exact(
        root, parents["p7f_decision_path"], parents["p7f_decision_sha256"]
    )
    p7f_report = _load_exact(
        root, parents["p7f_report_path"], parents["p7f_report_sha256"]
    )
    if (
        p2v["decision"] != "retain-clean-room-generic-primitive"
        or p7f_decision["visual_result"]["decision"] != "complete_pass"
        or p7f_decision["production_default_changed"]
        or p7f_report["selected_candidate_arm_id"] != "gauged_spatial_4000"
    ):
        raise ValueError("U6.P2W parent decision mismatch")
    runtime, gauge = validate_p7f(root, p7f)
    population = config["population"]
    if (
        list(runtime.eligible_ids) != population["expected_ids"]
        or len(runtime.eligible_ids) != population["expected_rows"]
        or len({runtime.source_rows[row]["make"] for row in runtime.eligible_ids})
        != population["expected_makes"]
    ):
        raise ValueError("U6.P2W population drift")
    metric = config["metric_sampling"]
    incumbent_rows = {
        row["sample_id"]: row
        for row in p7f_report["rows"]
        if row["arm_id"] == "gauged_spatial_4000"
    }
    if set(incumbent_rows) != set(runtime.eligible_ids):
        raise ValueError("frozen P7F incumbent population drift")
    incumbent_root = (root / parents["p7f_report_path"]).parent
    rows = []
    for sample_id in runtime.eligible_ids:
        source_row = runtime.source_rows[sample_id]
        source_path = root / source_row["decoded_path"]
        if sha256_file(source_path) != source_row["decoded_sha256"]:
            raise ValueError(f"source hash drift: {sample_id}")
        with Image.open(source_path) as image:
            source = (
                np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float64
                )
                / 255.0
            )
        linear, langmuir = _render_pair(source, runtime, gauge, config)
        source_sample = _sample(source, int(metric["maximum_pixels_per_image"]))
        linear_sample = _sample(linear, int(metric["maximum_pixels_per_image"]))
        candidate_sample = _sample(langmuir, int(metric["maximum_pixels_per_image"]))
        effect = _median_delta_e76(langmuir, linear)
        style, non_basic = style_and_basic_residual(source_sample, candidate_sample)
        linear_lab = rgb2lab(linear_sample.reshape(-1, 1, 3)).reshape(-1, 3)
        candidate_lab = rgb2lab(candidate_sample.reshape(-1, 1, 3)).reshape(-1, 3)
        linear_chroma = np.linalg.norm(linear_lab[:, 1:], axis=1)
        candidate_chroma = np.linalg.norm(candidate_lab[:, 1:], axis=1)
        luma_weights = np.array([0.2126, 0.7152, 0.0722])
        gradient_linear = _gradient_energy(encoded_srgb_to_linear(linear))
        gradient_candidate = _gradient_energy(encoded_srgb_to_linear(langmuir))
        difference = langmuir - linear
        rows.append(
            {
                "sample_id": sample_id,
                "make": source_row["make"],
                "source_sha256": source_row["decoded_sha256"],
                "linear_sha256": _save(
                    output_dir / "linear_donor" / f"{sample_id}.png", linear
                ),
                "langmuir_sha256": _save(
                    output_dir / "langmuir_donor" / f"{sample_id}.png", langmuir
                ),
                "incumbent_sha256": _copy_exact(
                    incumbent_root
                    / "renders"
                    / "gauged_spatial_4000"
                    / f"{sample_id}.png",
                    output_dir / "frozen_p7f_incumbent" / f"{sample_id}.png",
                    incumbent_rows[sample_id]["output_sha256"],
                ),
                "metrics": {
                    "effect_median_delta_e76": effect,
                    "style_delta_e76_vs_input": style,
                    "non_basic_delta_e76_vs_input": non_basic,
                    "new_hard_boundary_fraction_vs_linear": new_hard_clipping_fraction(
                        linear_sample,
                        candidate_sample,
                        float(metric["hard_boundary_epsilon"]),
                    ),
                    "median_luma_absolute_shift_vs_linear": float(
                        np.median(
                            np.abs(
                                candidate_sample @ luma_weights
                                - linear_sample @ luma_weights
                            )
                        )
                    ),
                    "lab_chroma_ratio_vs_linear": float(
                        np.median(candidate_chroma)
                        / max(float(np.median(linear_chroma)), 1e-12)
                    ),
                    "gradient_energy_ratio_vs_linear": float(
                        np.mean(gradient_candidate)
                        / max(float(np.mean(gradient_linear)), 1e-12)
                    ),
                    "isolated_excursion_count": _isolated(
                        difference,
                        threshold=float(metric["isolated_excursion_threshold"]),
                        radius=int(metric["isolated_support_radius_pixels"]),
                        support=int(metric["minimum_isolated_support_count"]),
                    ),
                },
            }
        )
    values = [row["metrics"] for row in rows]
    summary = {
        "source_count": len(rows),
        "all_outputs_finite_and_bounded": True,
        "population_median_effect_delta_e76": float(
            np.median([row["effect_median_delta_e76"] for row in values])
        ),
        "maximum_per_image_median_effect_delta_e76": max(
            row["effect_median_delta_e76"] for row in values
        ),
        "images_with_effect_delta_e76_at_least_0_1": sum(
            row["effect_median_delta_e76"] >= 0.1 for row in values
        ),
        "population_median_style_delta_e76_vs_input": float(
            np.median([row["style_delta_e76_vs_input"] for row in values])
        ),
        "population_median_non_basic_delta_e76_vs_input": float(
            np.median([row["non_basic_delta_e76_vs_input"] for row in values])
        ),
        "worst_new_hard_boundary_fraction_vs_linear": max(
            row["new_hard_boundary_fraction_vs_linear"] for row in values
        ),
        "population_median_luma_absolute_shift_vs_linear": float(
            np.median([row["median_luma_absolute_shift_vs_linear"] for row in values])
        ),
        "population_median_lab_chroma_ratio_vs_linear": float(
            np.median([row["lab_chroma_ratio_vs_linear"] for row in values])
        ),
        "worst_gradient_energy_ratio_vs_linear": min(
            row["gradient_energy_ratio_vs_linear"] for row in values
        ),
        "maximum_gradient_energy_ratio_vs_linear": max(
            row["gradient_energy_ratio_vs_linear"] for row in values
        ),
        "maximum_isolated_excursion_count": max(
            row["isolated_excursion_count"] for row in values
        ),
    }
    gates = config["automatic_gates"]
    checks = {
        "source_count": summary["source_count"] == gates["source_count_exact"],
        "finite_and_bounded": summary["all_outputs_finite_and_bounded"]
        is gates["all_outputs_finite_and_bounded"],
        "effect_floor": summary["population_median_effect_delta_e76"]
        >= gates["minimum_population_median_effect_delta_e76"],
        "effect_ceiling": summary["maximum_per_image_median_effect_delta_e76"]
        <= gates["maximum_per_image_median_effect_delta_e76"],
        "effect_population": summary["images_with_effect_delta_e76_at_least_0_1"]
        >= gates["minimum_images_with_effect_delta_e76_at_least_0_1"],
        "style": summary["population_median_style_delta_e76_vs_input"]
        >= gates["minimum_population_median_style_delta_e76_vs_input"],
        "non_basic": summary["population_median_non_basic_delta_e76_vs_input"]
        >= gates["minimum_population_median_non_basic_delta_e76_vs_input"],
        "boundary": summary["worst_new_hard_boundary_fraction_vs_linear"]
        <= gates["maximum_worst_new_hard_boundary_fraction_vs_linear"],
        "luma": summary["population_median_luma_absolute_shift_vs_linear"]
        <= gates["maximum_population_median_luma_absolute_shift_vs_linear"],
        "chroma_min": summary["population_median_lab_chroma_ratio_vs_linear"]
        >= gates["minimum_population_median_lab_chroma_ratio_vs_linear"],
        "chroma_max": summary["population_median_lab_chroma_ratio_vs_linear"]
        <= gates["maximum_population_median_lab_chroma_ratio_vs_linear"],
        "gradient_min": summary["worst_gradient_energy_ratio_vs_linear"]
        >= gates["minimum_worst_gradient_energy_ratio_vs_linear"],
        "gradient_max": summary["maximum_gradient_energy_ratio_vs_linear"]
        <= gates["maximum_worst_gradient_energy_ratio_vs_linear"],
        "isolated": summary["maximum_isolated_excursion_count"]
        <= gates["maximum_isolated_excursion_count"],
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": "neuro_film.u6_p2w_langmuir_interimage_photographic_report.v1",
        "node": "U6.P2W",
        "summary": summary,
        "gate_checks": checks,
        "automatic_pass": automatic_pass,
        "visual_review_allowed": automatic_pass,
        "decision": "automatic-pass" if automatic_pass else "close-photographic-use",
        "rows": rows,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    return {**core, "stable_evidence_id": hashlib.sha256(stable).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "_develop_density",
    "_render_pair",
    "evaluate_langmuir_photographic",
    "load_contract",
    "write_report",
]
