"""U6.P3R photographic comparison of analytical and Gaussian halation topology."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.fresh_native_standard_confirmation import (
    load_confirmation_working_image,
    sha256_file,
)
from src.eval.physical_backing_return_combined_ablation import _interpret
from src.eval.physical_backing_return_photographic_stress import _isolated_excursions
from src.eval.response_bounded_fresh_confirmation import (
    _diagnostic_map,
    _save_json,
    _save_rgb,
    _save_sheet,
)
from src.eval.scene_linear_backing_return import _resize_scene_linear
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.base_return_geometry import (
    BaseReturnGeometryProfile,
    apply_positive_base_return_geometry,
    base_return_geometry_kernel,
    equal_second_moment_gaussian,
)
from src.film_physics.response_bounded_exposure import (
    apply_response_bounded_exposure_residual,
)

SCHEMA = "neuro_film.u6_p3r_base_return_topology_photographic_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3r_base_return_topology_photographic_report.v1"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_exact(root: Path, path: str, expected_sha256: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"P3R evidence hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_contract(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    arms = config.get("fixed_arms", {})
    gates = config.get("automatic_gates", {})
    if (
        config.get("schema") != SCHEMA
        or config.get("experiment_id") != "U6.P3R"
        or config.get("training_allowed")
        or config.get("production_integration_allowed")
        or arms.get("diagnostic_return_fraction_scale") != 4.0
        or arms.get("maximum_transmittance_delta") != 0.015
        or not arms.get("shared_rgb_response_scale")
        or arms.get("per_image_parameter_selection_allowed")
        or arms.get("profile_fitting_allowed")
        or arms.get("hard_clipping_allowed")
        or arms.get("physical_energy_claim_allowed")
        or gates.get("minimum_population_p95_effect") != 0.006
        or gates.get("minimum_population_p95_topology_difference") != 0.001
        or not gates.get("repeat_report_and_render_hashes_exact")
    ):
        raise ValueError("invalid frozen U6.P3R contract")
    return config


def validate_contract(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    parent = config["parent"]
    decision = _load_exact(root, parent["decision_path"], parent["decision_sha256"])
    geometry = _load_exact(root, parent["geometry_contract_path"], parent["geometry_contract_sha256"])
    sensitometry = _load_exact(root, parent["sensitometry_contract_path"], parent["sensitometry_contract_sha256"])
    if (
        decision.get("decision") != parent["required_decision"]
        or decision["formal_evidence"].get("stable_evidence_id") != parent["stable_evidence_id"]
        or geometry.get("experiment_id") != "U6.P3Q1"
    ):
        raise ValueError("P3R parent decision drift")

    population = config["population"]
    preflight = _load_exact(root, population["preflight_decision_path"], population["preflight_decision_sha256"])
    manifest = _load_exact(root, population["manifest_path"], population["manifest_sha256"])
    review = _load_exact(root, population["visual_source_review_path"], population["visual_source_review_sha256"])
    p3o_manifest = _load_exact(root, population["p3o_manifest_path"], population["p3o_manifest_sha256"])
    if (
        preflight["result"].get("decision") != population["required_preflight_decision"]
        or not preflight["result"].get("automatic_pass")
        or not preflight["result"].get("visual_pass")
        or review.get("confirmed_severe_source_artifact_count") != 0
        or population.get("selection_used_image_appearance")
    ):
        raise ValueError("P3R source preflight drift")
    eligible = set(review["eligible_ids"])
    selected = [row for row in manifest if row["id"] in eligible]
    p3o_hashes = {row["raw_sha256"] for row in p3o_manifest}
    overlap = {row["raw_sha256"] for row in selected} & p3o_hashes
    if (
        len(selected) != int(population["expected_source_count"])
        or len({row["make"] for row in selected}) != int(population["expected_camera_make_count"])
        or (population["require_zero_exact_raw_hash_overlap_with_p3o"] and overlap)
    ):
        raise ValueError("P3R population support drift")
    return selected, geometry, sensitometry


def _blind_rounds(
    rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    candidate_key: str,
    control_key: str,
    prefix: str,
    visual_root: Path,
) -> list[dict[str, Any]]:
    ids = [row["id"] for row in rows]
    rounds = []
    for round_index, seed in enumerate(seeds, start=1):
        rng = random.Random(int(seed))
        mapping: dict[str, dict[str, str]] = {}
        keys_by_id: dict[str, tuple[str, str]] = {}
        for sample_id in ids:
            candidate_is_a = bool(rng.getrandbits(1))
            mapping[sample_id] = {
                "A": candidate_key if candidate_is_a else control_key,
                "B": control_key if candidate_is_a else candidate_key,
            }
            keys_by_id[sample_id] = (mapping[sample_id]["A"], mapping[sample_id]["B"])
        mapping_sha = _save_json(
            {
                "schema": "neuro_film.u6_p3r_blind_mapping.v1",
                "protocol": prefix,
                "round": round_index,
                "seed": int(seed),
                "mapping": mapping,
            },
            visual_root / f"{prefix}_round_{round_index}_mapping.json",
        )
        sheet_sha = _save_sheet(
            rows,
            keys_by_id=keys_by_id,
            path=visual_root / f"{prefix}_round_{round_index}.png",
        )
        rounds.append(
            {
                "round": round_index,
                "seed": int(seed),
                "mapping_sha256": mapping_sha,
                "sheet_sha256": sheet_sha,
            }
        )
    return rounds


def _save_direct_sheet(rows: list[dict[str, Any]], path: Path) -> str:
    keys = ("no_spatial", "gaussian", "analytical")
    tile_width, tile_height, header = 384, 256, 24
    canvas = Image.new(
        "RGB",
        (len(keys) * tile_width, len(rows) * (tile_height + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), row["id"], fill=(235, 235, 235))
        for column, key in enumerate(keys):
            values = np.rint(np.clip(row["visual"][key], 0.0, 1.0) * 255.0).astype(np.uint8)
            preview = ImageOps.contain(
                Image.fromarray(values, mode="RGB"),
                (tile_width, tile_height),
                method=Image.Resampling.LANCZOS,
            )
            x = column * tile_width + (tile_width - preview.width) // 2
            canvas.paste(preview, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return sha256_file(path)


def evaluate_photographic_topology(
    *, root: Path, config: dict[str, Any], visual_root: Path
) -> dict[str, Any]:
    manifest, geometry, sensitometry = validate_contract(root, config)
    profile_row = dict(geometry["generic_hypothesis_profile"])
    base_fractions = np.asarray(profile_row.pop("return_fraction_rgb"), dtype=np.float64)
    profile = BaseReturnGeometryProfile(**profile_row)
    analytical_kernel = base_return_geometry_kernel(profile)
    gaussian_kernel = equal_second_moment_gaussian(analytical_kernel, profile.pixel_pitch_um)
    fractions = tuple(
        float(value * float(config["fixed_arms"]["diagnostic_return_fraction_scale"]))
        for value in base_fractions
    )
    operator = build_operator(sensitometry)
    cap = float(config["fixed_arms"]["maximum_transmittance_delta"])
    gates = config["automatic_gates"]

    rows: list[dict[str, Any]] = []
    visual_assets: dict[str, dict[str, str]] = {}
    for source_row in manifest:
        raw_path = root / source_row["raw_path"]
        if sha256_file(raw_path) != source_row["raw_sha256"]:
            raise ValueError("P3R RAW identity drift")
        working = load_confirmation_working_image(raw_path)
        source = _resize_scene_linear(
            working.pixels, int(config["population"]["maximum_long_edge"])
        ).astype(np.float64)
        no_spatial = _interpret(source, operator)
        arm_outputs: dict[str, np.ndarray] = {}
        arm_scales: dict[str, float] = {}
        for arm, kernel in (("gaussian", gaussian_kernel), ("analytical", analytical_kernel)):
            unbounded = apply_positive_base_return_geometry(source, kernel, fractions)
            bounded = apply_response_bounded_exposure_residual(
                source,
                unbounded.output,
                operator,
                maximum_transmittance_delta=cap,
            )
            arm_outputs[arm] = _interpret(bounded.exposure, operator)
            arm_scales[arm] = float(np.median(bounded.shared_scale))
        effect = np.abs(arm_outputs["analytical"] - no_spatial)
        gaussian_effect = np.abs(arm_outputs["gaussian"] - no_spatial)
        topology = np.abs(arm_outputs["analytical"] - arm_outputs["gaussian"])
        pixel_effect = np.max(effect, axis=-1)
        new_boundary = []
        isolated = 0
        for values, delta in ((arm_outputs["analytical"], effect), (arm_outputs["gaussian"], gaussian_effect)):
            new_boundary.append(
                float(
                    np.mean(
                        ((values <= 0.0) | (values >= 1.0))
                        & ~((no_spatial <= 0.0) | (no_spatial >= 1.0))
                    )
                )
            )
            isolated += _isolated_excursions(
                delta,
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            )
        visual = {
            "no_spatial": _diagnostic_map(no_spatial),
            "gaussian": _diagnostic_map(arm_outputs["gaussian"]),
            "analytical": _diagnostic_map(arm_outputs["analytical"]),
        }
        visual_assets[source_row["id"]] = {
            key: _save_rgb(values, visual_root / source_row["id"] / f"{key}.png")
            for key, values in visual.items()
        }
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "model": source_row["model"],
                "raw_sha256": source_row["raw_sha256"],
                "shape": list(source.shape),
                "analytical_maximum_effect": float(np.max(effect)),
                "analytical_p95_effect": float(np.quantile(effect, 0.95)),
                "analytical_fraction_pixels_over_0p005": float(np.mean(pixel_effect > 0.005)),
                "topology_p95_difference": float(np.quantile(topology, 0.95)),
                "topology_maximum_difference": float(np.max(topology)),
                "maximum_new_hard_boundary_fraction": max(new_boundary),
                "isolated_excursion_count": isolated,
                "shared_scale_median": arm_scales,
                "visual": visual,
            }
        )

    metrics = {
        "source_count": len(rows),
        "camera_make_count": len({row["make"] for row in rows}),
        "p3o_raw_hash_overlap": 0,
        "analytical_maximum_effect": max(row["analytical_maximum_effect"] for row in rows),
        "analytical_population_p95_effect": float(np.quantile([row["analytical_p95_effect"] for row in rows], 0.95)),
        "analytical_median_fraction_pixels_over_0p005": float(np.median([row["analytical_fraction_pixels_over_0p005"] for row in rows])),
        "population_p95_topology_difference": float(np.quantile([row["topology_p95_difference"] for row in rows], 0.95)),
        "maximum_new_hard_boundary_fraction": max(row["maximum_new_hard_boundary_fraction"] for row in rows),
        "total_isolated_excursion_count": sum(row["isolated_excursion_count"] for row in rows),
    }
    checks = {
        "source_count": metrics["source_count"] == int(gates["source_count_exact"]),
        "camera_make_count": metrics["camera_make_count"] == int(gates["camera_make_count_exact"]),
        "p3o_disjoint": metrics["p3o_raw_hash_overlap"] <= int(gates["maximum_p3o_raw_hash_overlap"]),
        "response_bound": metrics["analytical_maximum_effect"] <= float(gates["maximum_response_delta"]),
        "visible_effect": metrics["analytical_population_p95_effect"] >= float(gates["minimum_population_p95_effect"]),
        "visible_coverage": metrics["analytical_median_fraction_pixels_over_0p005"] >= float(gates["minimum_median_fraction_pixels_over_0p005"]),
        "topology_distinguishable": metrics["population_p95_topology_difference"] >= float(gates["minimum_population_p95_topology_difference"]),
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"] <= float(gates["maximum_new_hard_boundary_fraction"]),
        "isolated_excursions": metrics["total_isolated_excursion_count"] <= int(gates["maximum_isolated_excursion_count"]),
    }
    automatic_pass = all(checks.values())
    report_rows = [{key: value for key, value in row.items() if key != "visual"} for row in rows]
    direct_sheet_sha = _save_direct_sheet(rows, visual_root / "direct_severe_sheet.png")
    topology_rounds = _blind_rounds(
        rows,
        seeds=[int(value) for value in config["visual_protocol"]["topology_blind_round_seeds"]],
        candidate_key="analytical",
        control_key="gaussian",
        prefix="topology_blind",
        visual_root=visual_root,
    )
    value_rounds = _blind_rounds(
        rows,
        seeds=[int(value) for value in config["visual_protocol"]["value_blind_round_seeds"]],
        candidate_key="analytical",
        control_key="no_spatial",
        prefix="value_blind",
        visual_root=visual_root,
    )
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "claim_ceiling": config["claim_ceiling"],
        "rows": report_rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "visual_assets": visual_assets,
        "direct_severe_sheet_sha256": direct_sheet_sha,
        "topology_blind_rounds": topology_rounds,
        "value_blind_rounds": value_rounds,
        "branch": config["branches"]["pass" if automatic_pass else "automatic_fail"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical_bytes(core)).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


__all__ = ["REPORT_SCHEMA", "SCHEMA", "evaluate_photographic_topology", "load_contract", "validate_contract", "write_report"]
