"""U6.P5Q photographic severe/OOD stress for the P5P minimax envelope."""

from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import uniform_filter

from src.eval.physical_interimage_photographic_stress import _crop_bounds
from src.eval.physical_measured_mtf_domain_placement import _apply_arms
from src.eval.physical_spatial_photographic_stress import (
    _flat_region_p99,
    _isolated_excursions,
    _load_source,
    _preview,
)
from src.eval.physical_spatial_response import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    apply_bounded_development_adjacency,
    apply_scanner_mtf,
    apply_spatial_response_pipeline,
)
from src.film_physics.measured_mtf_budget import MeasuredTotalFilmSpatialBudget
from src.film_physics.measured_mtf_envelope import (
    PLACEMENT_ARMS,
    build_minimax_transmittance_envelope,
)

SCHEMA = "neuro_film.u6_p5q_measured_mtf_photographic_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p5q_measured_mtf_photographic_stress_report.v1"


class MeasuredMtfPhotographicError(RuntimeError):
    """Raised when P5Q exact parents, inputs or contract drift."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MeasuredMtfPhotographicError("P5Q paths must be repository-relative")
    return path


def _load_exact(root: Path, value: str, expected: str) -> dict[str, Any]:
    relative = _relative(value)
    path = root / relative
    if _hash(path) != expected:
        raise MeasuredMtfPhotographicError(f"P5Q parent hash mismatch: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("input", {}).get("expected_rows") != 18
        or payload.get("input", {}).get("expected_camera_makes") != 9
        or payload.get("pipeline", {}).get("row_partitions") != [257, 509]
        or payload.get("automatic_gates", {}).get(
            "minimum_strong_edge_gradient_ratio"
        )
        != 0.5
        or payload.get("automatic_gates", {}).get(
            "maximum_population_p95_uncertainty_half_range"
        )
        != 0.01
    ):
        raise MeasuredMtfPhotographicError("P5Q frozen contract drift")
    return payload


def _luma_gradient(values: np.ndarray) -> np.ndarray:
    luma = values @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    dx = np.zeros_like(luma)
    dy = np.zeros_like(luma)
    dx[:, 1:] = np.diff(luma, axis=1)
    dy[1:, :] = np.diff(luma, axis=0)
    return np.hypot(dx, dy)


def _strong_edge_ratio(baseline: np.ndarray, candidate: np.ndarray) -> float:
    baseline_gradient = _luma_gradient(baseline)
    candidate_gradient = _luma_gradient(candidate)
    threshold = float(np.quantile(baseline_gradient, 0.9))
    mask = baseline_gradient >= max(threshold, np.finfo(np.float64).eps)
    denominator = float(np.sum(baseline_gradient[mask]))
    if denominator <= 0.0:
        raise MeasuredMtfPhotographicError("P5Q strong-edge support is empty")
    return float(np.sum(candidate_gradient[mask]) / denominator)


def _texture_energy_ratio(baseline: np.ndarray, candidate: np.ndarray) -> float:
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    base_luma = baseline @ weights
    candidate_luma = candidate @ weights
    base_high = base_luma - uniform_filter(base_luma, size=3, mode="nearest")
    candidate_high = candidate_luma - uniform_filter(
        candidate_luma, size=3, mode="nearest"
    )
    denominator = float(np.mean(np.abs(base_high)))
    if denominator <= 0.0:
        raise MeasuredMtfPhotographicError("P5Q texture support is empty")
    return float(np.mean(np.abs(candidate_high)) / denominator)


def _render_candidate(
    source: np.ndarray,
    operator: Any,
    budget: MeasuredTotalFilmSpatialBudget,
    profile: Any,
    *,
    tile_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arms = _apply_arms(
        source,
        sensitometry=operator,
        budget=budget,
        tile_rows=tile_rows,
    )
    envelope = build_minimax_transmittance_envelope(
        {name: arms[name]["transmittance"] for name in PLACEMENT_ARMS}
    )
    return apply_scanner_mtf(envelope.transmittance, profile), envelope.uncertainty_half_range


def _save_overview(rows: list[dict[str, Any]], fixed_ids: list[str], path: Path) -> str:
    by_id = {row["id"]: row for row in rows}
    tile_width, tile_height, header = 240, 160, 22
    canvas = Image.new(
        "RGB", (5 * tile_width, len(fixed_ids) * (tile_height + header)), (24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        views = (
            row["source"],
            row["sensitometry"],
            row["generic"],
            row["candidate"],
            np.clip(0.5 + 8.0 * row["difference"], 0.0, 1.0),
        )
        for column, values in enumerate(views):
            image = _preview(values, (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return _hash(path)


def _save_worst_patches(
    rows: list[dict[str, Any]], fixed_ids: list[str], size: int, path: Path
) -> str:
    by_id = {row["id"]: row for row in rows}
    header = 22
    canvas = Image.new(
        "RGB", (5 * size, len(fixed_ids) * (size + header)), (24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = row_index * (size + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        magnitude = np.max(np.abs(row["difference"]), axis=-1)
        centre = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        ys, xs = _crop_bounds(magnitude.shape, centre, size)
        views = (
            row["source"][ys, xs],
            row["sensitometry"][ys, xs],
            row["generic"][ys, xs],
            row["candidate"][ys, xs],
            np.clip(0.5 + 8.0 * row["difference"][ys, xs], 0.0, 1.0),
        )
        for column, values in enumerate(views):
            image = _preview(values, (size, size))
            if image.size != (size, size):
                image = ImageOps.pad(
                    image,
                    (size, size),
                    method=Image.Resampling.NEAREST,
                    color=(24, 24, 24),
                )
            canvas.paste(image, (column * size, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return _hash(path)


def evaluate_photographic_stress(
    *, root: Path, contract: dict[str, Any], overview_path: Path, patch_path: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    p5p_decision = _load_exact(
        root, parents["p5p_decision"], parents["p5p_decision_sha256"]
    )
    if p5p_decision.get("decision") != "retain_minimax_envelope_open_photographic_challenger":
        raise MeasuredMtfPhotographicError("P5Q requires exact P5P advancement")
    p5p_contract = _load_exact(
        root,
        "configs/u6_p5p_measured_mtf_minimax_envelope_v1.json",
        p5p_decision["contract_sha256"],
    )
    p5d = _load_exact(root, parents["p5d_contract"], parents["p5d_contract_sha256"])
    p5c_decision = _load_exact(
        root, parents["p5c_decision"], parents["p5c_decision_sha256"]
    )
    p5c = _load_exact(
        root,
        "configs/u6_p5c_bounded_adjacency_v1.json",
        p5c_decision["contract_sha256"],
    )
    p5b = _load_exact(
        root,
        "configs/u6_p5b_spatial_halo_audit_v1.json",
        p5c["parents"]["p5b_contract_sha256"],
    )
    p5a = _load_exact(
        root,
        "configs/u6_p5a_spatial_response_primitives_v1.json",
        p5b["parents"]["p5a_contract_sha256"],
    )
    sens_config = _load_exact(
        root,
        p5p_contract["parents"]["sensitometry_contract"],
        p5p_contract["parents"]["sensitometry_contract_sha256"],
    )
    if _hash(root / "configs/u2_2a_sensitometry_primitive_v1.json") != p5a["parents"][
        "sensitometry_config_sha256"
    ]:
        raise MeasuredMtfPhotographicError("P5Q sensitometry lineage disagrees")
    lod = _load_exact(
        root,
        p5p_contract["parents"]["p5l_bundle"],
        p5p_contract["parents"]["p5l_bundle_sha256"],
    )
    p5d_input = p5d["input"]
    manifest = _load_exact(
        root, p5d_input["manifest"], p5d_input["manifest_sha256"]
    )
    _load_exact(
        root,
        p5d_input["preflight_report"],
        p5d_input["preflight_report_sha256"],
    )
    if len(manifest) != 18 or len({row["make"] for row in manifest}) != 9:
        raise MeasuredMtfPhotographicError("P5Q cohort support drift")
    for row in manifest:
        if (
            row["allowed_use"] != p5d_input["required_allowed_use"]
            or row["rights_scope"] != p5d_input["required_rights_scope"]
            or row["decoded_color_state"] != p5d_input["decoded_color_state"]
        ):
            raise MeasuredMtfPhotographicError("P5Q cohort rights or rail drift")

    operator = build_operator(sens_config)
    budget = MeasuredTotalFilmSpatialBudget.from_lod_bundle(lod)
    profile = _profile(p5a)
    p5c_candidate = p5c["candidate"]
    p5c_apply = partial(
        apply_bounded_development_adjacency,
        maximum_absolute_transmittance_delta=float(
            p5c_candidate["maximum_absolute_transmittance_delta"]
        ),
        maximum_absolute_density_delta=float(
            p5c_candidate["maximum_absolute_density_delta"]
        ),
    )
    gates = contract["automatic_gates"]
    rows = []
    visuals = []
    for source_row in manifest:
        source, input_sha = _load_source(source_row, root)
        density = operator.apply(source)
        sensitometry = apply_scanner_mtf(np.power(10.0, -density), profile)
        generic = apply_spatial_response_pipeline(
            source,
            profile,
            sensitometry_apply=operator.apply,
            adjacency_apply=p5c_apply,
        )
        candidate, uncertainty = _render_candidate(source, operator, budget, profile)
        partitions = {}
        for tile_rows in contract["pipeline"]["row_partitions"]:
            tiled, tiled_uncertainty = _render_candidate(
                source,
                operator,
                budget,
                profile,
                tile_rows=int(tile_rows),
            )
            partitions[str(tile_rows)] = bool(
                np.array_equal(candidate, tiled)
                and np.array_equal(uncertainty, tiled_uncertainty)
            )
        difference = candidate - sensitometry
        generic_difference = candidate - generic
        absolute = np.abs(difference)
        new_boundary = (
            ((candidate <= 0.0) | (candidate >= 1.0))
            & ~((sensitometry <= 0.0) | (sensitometry >= 1.0))
        )
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "input_sha256": input_sha,
            "shape": list(source.shape),
            "finite_domain": bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate > 0.0)
                and np.all(candidate <= 1.0)
            ),
            "partition_exact": partitions,
            "candidate_vs_sensitometry_max_abs": float(np.max(absolute)),
            "candidate_vs_sensitometry_p95_abs": float(np.quantile(absolute, 0.95)),
            "candidate_vs_generic_p95_abs": float(
                np.quantile(np.abs(generic_difference), 0.95)
            ),
            "flat_region_p99_abs": _flat_region_p99(source, difference),
            "maximum_difference_channel_spread": float(
                np.max(np.max(difference, axis=-1) - np.min(difference, axis=-1))
            ),
            "isolated_excursion_count": _isolated_excursions(
                difference,
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            ),
            "new_hard_boundary_fraction": float(np.mean(new_boundary)),
            "uncertainty_half_range_max": float(np.max(uncertainty)),
            "uncertainty_half_range_p95": float(np.quantile(uncertainty, 0.95)),
            "strong_edge_gradient_ratio": _strong_edge_ratio(sensitometry, candidate),
            "texture_energy_ratio": _texture_energy_ratio(sensitometry, candidate),
        }
        rows.append(row)
        if source_row["id"] in contract["visual_protocol"]["fixed_ids"]:
            visuals.append(
                {
                    "id": source_row["id"],
                    "source": source,
                    "sensitometry": sensitometry,
                    "generic": generic,
                    "candidate": candidate,
                    "difference": difference,
                }
            )
    fixed_ids = contract["visual_protocol"]["fixed_ids"]
    if sorted(row["id"] for row in visuals) != sorted(fixed_ids):
        raise MeasuredMtfPhotographicError("P5Q fixed visual IDs are incomplete")
    overview_sha = _save_overview(visuals, fixed_ids, overview_path)
    patch_sha = _save_worst_patches(
        visuals,
        fixed_ids,
        int(contract["visual_protocol"]["worst_patch_size"]),
        patch_path,
    )
    population_candidate_p95 = float(
        np.quantile([row["candidate_vs_sensitometry_p95_abs"] for row in rows], 0.95)
    )
    population_generic_p95 = float(
        np.quantile([row["candidate_vs_generic_p95_abs"] for row in rows], 0.95)
    )
    population_uncertainty_p95 = float(
        np.quantile([row["uncertainty_half_range_p95"] for row in rows], 0.95)
    )
    decisions = {
        "input_hashes": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, manifest, strict=True)
        ),
        "finite_domain": all(row["finite_domain"] for row in rows),
        "partition_exact": all(all(row["partition_exact"].values()) for row in rows),
        "output_bound": max(row["candidate_vs_sensitometry_max_abs"] for row in rows)
        <= float(gates["maximum_candidate_vs_sensitometry_abs"]),
        "nontrivial_population": population_candidate_p95
        >= float(gates["minimum_population_p95_candidate_vs_sensitometry_abs"]),
        "distinct_from_generic": population_generic_p95
        >= float(gates["minimum_population_p95_candidate_vs_generic_abs"]),
        "flat_regions": max(row["flat_region_p99_abs"] for row in rows)
        <= float(gates["maximum_flat_region_p99_abs"]),
        "channel_spread": max(row["maximum_difference_channel_spread"] for row in rows)
        <= float(gates["maximum_difference_channel_spread"]),
        "isolated_excursions": sum(row["isolated_excursion_count"] for row in rows)
        <= int(gates["maximum_isolated_excursion_count"]),
        "new_boundaries": max(row["new_hard_boundary_fraction"] for row in rows)
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "uncertainty_max": max(row["uncertainty_half_range_max"] for row in rows)
        <= float(gates["maximum_uncertainty_half_range"]),
        "uncertainty_population": population_uncertainty_p95
        <= float(gates["maximum_population_p95_uncertainty_half_range"]),
        "strong_edge_retention": min(row["strong_edge_gradient_ratio"] for row in rows)
        >= float(gates["minimum_strong_edge_gradient_ratio"])
        and max(row["strong_edge_gradient_ratio"] for row in rows)
        <= float(gates["maximum_strong_edge_gradient_ratio"]),
        "texture_retention": min(row["texture_energy_ratio"] for row in rows)
        >= float(gates["minimum_texture_energy_ratio"])
        and max(row["texture_energy_ratio"] for row in rows)
        <= float(gates["maximum_texture_energy_ratio"]),
    }
    automatic_pass = all(decisions.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "claim_ceiling": contract["claim_ceiling"],
        "parent_spatial_budget_id": budget.budget_id,
        "rows": rows,
        "population_p95_candidate_vs_sensitometry_abs": population_candidate_p95,
        "population_p95_candidate_vs_generic_abs": population_generic_p95,
        "population_p95_uncertainty_half_range": population_uncertainty_p95,
        "overview_sha256": overview_sha,
        "worst_patches_sha256": patch_sha,
        "decisions": decisions,
        "automatic_pass": automatic_pass,
        "decision": "retain_photo_safe_open_value_comparison"
        if automatic_pass
        else "close_measured_mtf_envelope_photographic_use",
    }
    return {
        **core,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "ascii"
            )
        ).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "SCHEMA",
    "MeasuredMtfPhotographicError",
    "_render_candidate",
    "_strong_edge_ratio",
    "_texture_energy_ratio",
    "evaluate_photographic_stress",
    "load_contract",
    "write_report",
]
