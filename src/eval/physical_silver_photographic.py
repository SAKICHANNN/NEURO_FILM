"""U6.P2O photographic audit of the fixed retained-silver density branch."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import uniform_filter
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
from src.eval.physical_nonspatial_attribution import _metric_sample
from src.eval.physical_virtual_scan_sampling import (
    _canonicalize_endpoint_roundoff,
    compile_virtual_scan_profile,
)
from src.film_physics import (
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
)
from src.film_physics.silver_retention import (
    SilverRetentionProfile,
    apply_silver_retention,
)
from src.real_film.gold_matrix_transplant import style_and_basic_residual


SCHEMAS = {
    "neuro_film.u6_p2o_silver_retention_photographic_contract.v1",
    "neuro_film.u6_p2o1_silver_retention_photographic_contract.v1",
}


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in SCHEMAS:
        raise ValueError("unsupported U6.P2O/P2O1 contract")
    return payload


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"parent hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _silver_profile(p2n: dict[str, Any]) -> SilverRetentionProfile:
    row = p2n["profile"]
    return SilverRetentionProfile(
        float(row["retention_fraction"]),
        tuple(row["dye_to_silver_weights"]),
        float(row["maximum_input_dye_density"]),
        float(row["maximum_output_total_density"]),
    )


def apply_bounded_silver_density(
    density: np.ndarray,
    profile: SilverRetentionProfile,
    white_reference_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(density)
    white = np.asarray(white_reference_density, dtype=values.dtype)
    if white.shape != (3,) or np.any(values > white):
        raise ValueError("dye density is outside print-reference headroom")
    requested = apply_silver_retention(values, profile).silver_density
    headroom = np.min(white - values, axis=-1)
    scale = np.ones_like(requested)
    positive = requested > 0.0
    scale[positive] = np.minimum(
        values.dtype.type(1.0),
        np.maximum(values.dtype.type(0.0), headroom[positive])
        / requested[positive],
    )
    applied = requested * scale
    output = values + applied[..., None]
    if np.any(output > white) or np.any(output < values):
        raise RuntimeError("bounded silver density escaped analytical headroom")
    return output, scale


def _isolated(
    difference: np.ndarray, *, threshold: float, radius: int, support: int
) -> int:
    mask = np.max(np.abs(difference), axis=-1) > threshold
    width = 2 * radius + 1
    count = uniform_filter(
        mask.astype(np.float64), size=width, mode="constant", cval=0.0
    ) * (width * width)
    return int(np.count_nonzero(mask & (count < support)))


def _save(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        path, format="PNG", compress_level=6
    )
    return sha256_file(path)


def _render(
    source: np.ndarray,
    runtime: Any,
    gauge: Any,
    silver: SilverRetentionProfile,
    *,
    sampling_dpi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded = np.asarray(source, dtype=np.float64)
    linear = encoded_srgb_to_linear(encoded)
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=sampling_dpi
        ),
    )
    exposure = apply_forward_scatter(linear, compiled.profile)
    density = runtime.print_operator.sensitometry.apply(exposure)
    density = runtime.apply_adjacency(density, compiled.profile)
    density = apply_dye_diffusion(density, compiled.profile)
    normal = _canonicalize_endpoint_roundoff(
        runtime.print_operator.interpretation.apply(density)
    )
    normal = _canonicalize_endpoint_roundoff(
        apply_scanner_mtf(normal, compiled.profile)
    )
    bounded_density, scale = apply_bounded_silver_density(
        density,
        silver,
        runtime.print_operator.interpretation.white_reference_density,
    )
    retained = _canonicalize_endpoint_roundoff(
        runtime.print_operator.interpretation.apply(bounded_density)
    )
    retained = _canonicalize_endpoint_roundoff(
        apply_scanner_mtf(retained, compiled.profile)
    )
    normal = apply_gauge_to_intermediate(normal, gauge)
    retained = apply_gauge_to_intermediate(retained, gauge)
    apply_colour = runtime.build_source_context_colour(encoded)
    baseline = apply_colour(linear_srgb_to_encoded(normal))
    candidate = apply_colour(linear_srgb_to_encoded(retained))
    for output in (baseline, candidate):
        if (
            output.shape != encoded.shape
            or not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError("P2O output escaped encoded RGB domain")
    return baseline, candidate, scale


def evaluate_silver_photographic(
    *, root: Path, contract: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    p2n = _load_exact(
        root, parents["p2n_contract_path"], parents["p2n_contract_sha256"]
    )
    decision = _load_exact(
        root, parents["p2n_decision_path"], parents["p2n_decision_sha256"]
    )
    p7f = _load_exact(
        root, parents["p7f_contract_path"], parents["p7f_contract_sha256"]
    )
    if not decision["automatic_pass"]:
        raise ValueError("P2N parent did not pass")
    runtime, gauge = validate_p7f(root, p7f)
    if (
        len(runtime.eligible_ids) != int(contract["population"]["expected_rows"])
        or len(
            {
                runtime.source_rows[sample_id]["make"]
                for sample_id in runtime.eligible_ids
            }
        )
        != int(contract["population"]["expected_makes"])
    ):
        raise ValueError("P2O population drift")
    if "expected_ids" in contract["population"] and list(runtime.eligible_ids) != list(
        contract["population"]["expected_ids"]
    ):
        raise ValueError("P2O eligible ID drift")
    profile = _silver_profile(p2n)
    gates = contract["automatic_gates"]
    rows = []
    for sample_id in runtime.eligible_ids:
        row = runtime.source_rows[sample_id]
        path = root / row["decoded_path"]
        if sha256_file(path) != row["decoded_sha256"]:
            raise ValueError(f"source hash drift: {sample_id}")
        with Image.open(path) as image:
            source = (
                np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    dtype=np.float64,
                )
                / 255.0
            )
        baseline, candidate, scale = _render(
            source,
            runtime,
            gauge,
            profile,
            sampling_dpi=int(contract["execution"]["sampling_dpi"]),
        )
        baseline_sample = _metric_sample(baseline)
        candidate_sample = _metric_sample(candidate)
        delta, non_basic = style_and_basic_residual(
            baseline_sample, candidate_sample
        )
        baseline_lab = rgb2lab(baseline_sample.reshape(-1, 1, 3)).reshape(-1, 3)
        candidate_lab = rgb2lab(candidate_sample.reshape(-1, 1, 3)).reshape(-1, 3)
        baseline_chroma = np.linalg.norm(baseline_lab[:, 1:], axis=1)
        candidate_chroma = np.linalg.norm(candidate_lab[:, 1:], axis=1)
        chroma_ratio = float(
            np.median(candidate_chroma)
            / max(float(np.median(baseline_chroma)), 1e-12)
        )
        luma_shift = float(
            np.median(
                np.abs(
                    candidate_sample @ np.array([0.2126, 0.7152, 0.0722])
                    - baseline_sample @ np.array([0.2126, 0.7152, 0.0722])
                )
            )
        )
        difference = candidate - baseline
        difference_view = np.clip(0.5 + 4.0 * difference, 0.0, 1.0)
        rows.append(
            {
                "sample_id": sample_id,
                "make": row["make"],
                "source_sha256": row["decoded_sha256"],
                "baseline_sha256": _save(
                    output_dir / "baseline" / f"{sample_id}.png", baseline
                ),
                "candidate_sha256": _save(
                    output_dir / "silver-retention" / f"{sample_id}.png",
                    candidate,
                ),
                "difference_sha256": _save(
                    output_dir / "difference-x4" / f"{sample_id}.png",
                    difference_view,
                ),
                "metrics": {
                    "median_delta_e76": delta,
                    "non_basic_delta_e76": non_basic,
                    "new_hard_boundary_fraction": new_hard_clipping_fraction(
                        baseline_sample, candidate_sample, 0.5 / 255.0
                    ),
                    "median_applied_retention_scale": float(np.median(scale)),
                    "median_luma_absolute_shift": luma_shift,
                    "lab_chroma_ratio": chroma_ratio,
                    "isolated_excursion_count": _isolated(
                        difference,
                        threshold=float(gates["isolated_excursion_threshold"]),
                        radius=int(gates["isolated_support_radius_pixels"]),
                        support=int(gates["minimum_isolated_support_count"]),
                    ),
                },
            }
        )
    values = [row["metrics"] for row in rows]
    summary = {
        "maximum_new_hard_boundary_fraction": max(
            row["new_hard_boundary_fraction"] for row in values
        ),
        "maximum_per_image_median_delta_e76": max(
            row["median_delta_e76"] for row in values
        ),
        "population_median_delta_e76": float(
            np.median([row["median_delta_e76"] for row in values])
        ),
        "population_median_non_basic_delta_e76": float(
            np.median([row["non_basic_delta_e76"] for row in values])
        ),
        "images_with_delta_e76_at_least_1": sum(
            row["median_delta_e76"] >= 1.0 for row in values
        ),
        "population_median_applied_retention_scale": float(
            np.median([row["median_applied_retention_scale"] for row in values])
        ),
        "population_median_luma_absolute_shift": float(
            np.median([row["median_luma_absolute_shift"] for row in values])
        ),
        "population_median_lab_chroma_ratio": float(
            np.median([row["lab_chroma_ratio"] for row in values])
        ),
        "maximum_isolated_excursion_count": max(
            row["isolated_excursion_count"] for row in values
        ),
    }
    decisions = {
        "boundary": summary["maximum_new_hard_boundary_fraction"]
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "maximum_style": summary["maximum_per_image_median_delta_e76"]
        <= float(gates["maximum_per_image_median_delta_e76"]),
        "style_floor": summary["population_median_delta_e76"]
        >= float(gates["minimum_population_median_delta_e76"]),
        "non_basic": summary["population_median_non_basic_delta_e76"]
        >= float(gates["minimum_population_median_non_basic_delta_e76"]),
        "population_support": summary["images_with_delta_e76_at_least_1"]
        >= int(gates["minimum_images_with_delta_e76_at_least_1"]),
        "retention_support": summary["population_median_applied_retention_scale"]
        >= float(gates["minimum_median_applied_retention_scale"]),
        "luma": summary["population_median_luma_absolute_shift"]
        <= float(gates["maximum_median_luma_absolute_shift"]),
        "chroma_min": summary["population_median_lab_chroma_ratio"]
        >= float(gates["minimum_population_median_lab_chroma_ratio"]),
        "chroma_max": summary["population_median_lab_chroma_ratio"]
        <= float(gates["maximum_population_median_lab_chroma_ratio"]),
        "isolated": summary["maximum_isolated_excursion_count"]
        <= int(gates["maximum_isolated_excursion_count"]),
    }
    automatic_pass = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p2o_silver_retention_photographic_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "summary": summary,
        "decisions": decisions,
        "rows": rows,
        "automatic_pass": automatic_pass,
        "branch": contract["branch_rule"][
            "automatic_pass" if automatic_pass else "automatic_fail"
        ],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
