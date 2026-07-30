"""U6.P3O fresh confirmation for the response-bounded exposure residual."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.eval.fresh_native_standard_confirmation import (
    load_confirmation_working_image,
    sha256_file,
)
from src.eval.physical_backing_return_combined_ablation import (
    _array,
    _compile_profiles,
    _interpret,
)
from src.eval.physical_backing_return_photographic_stress import (
    _isolated_excursions,
)
from src.eval.response_bounded_backing_return import (
    load_contract as load_p3n_contract,
)
from src.eval.scene_linear_backing_return import (
    _resize_scene_linear,
    validate_contract as validate_p3l_contract,
)
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    apply_compiled_scatter,
    apply_fft_backing_return,
    apply_response_bounded_exposure_residual,
)


SCHEMA = "neuro_film.u6_p3o_response_bounded_fresh_confirmation_contract.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P3O contract")
    return payload


def _load_exact(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"evidence hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _diagnostic_map(transmittance: np.ndarray) -> np.ndarray:
    return linear_srgb_to_encoded(
        np.clip(1.0 - transmittance, 0.0, 1.0)
    )


def _save_rgb(values: np.ndarray, path: Path) -> str:
    encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="RGB").save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return sha256_file(path)


def _preview(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    return ImageOps.contain(
        Image.fromarray(encoded, mode="RGB"),
        size,
        method=Image.Resampling.LANCZOS,
    )


def _save_sheet(
    rows: list[dict[str, Any]],
    *,
    keys_by_id: dict[str, tuple[str, str]],
    path: Path,
) -> str:
    by_id = {row["id"]: row for row in rows}
    ids = list(keys_by_id)
    tile_width, tile_height, header = 384, 256, 24
    canvas = Image.new(
        "RGB",
        (2 * tile_width, len(ids) * (tile_height + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(ids):
        row = by_id[sample_id]
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        for column, key in enumerate(keys_by_id[sample_id]):
            image = _preview(row["visual"][key], (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return sha256_file(path)


def _save_json(payload: Any, path: Path) -> str:
    raw = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _validate_contract(
    root: Path,
    contract: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("training_allowed")
        or contract.get("production_integration_allowed")
        or contract["population"]["selection_used_image_appearance"]
        or contract["candidate"]["profile_fitting_allowed"]
        or contract["candidate"]["per_image_parameter_selection_allowed"]
        or contract["candidate"]["hard_clipping_allowed"]
        or not contract["candidate"]["shared_rgb_residual_scale"]
    ):
        raise ValueError("invalid U6.P3O contract")

    parents = contract["parents"]
    p3n_contract = _load_exact(
        root,
        parents["p3n_contract_path"],
        parents["p3n_contract_sha256"],
    )
    if p3n_contract != load_p3n_contract(root / parents["p3n_contract_path"]):
        raise ValueError("P3N contract parser drift")
    p3n_decision = _load_exact(
        root,
        parents["p3n_decision_path"],
        parents["p3n_decision_sha256"],
    )
    if (
        p3n_decision.get("decision")
        != "retain_development_challenger_require_fresh_value_confirmation"
        or not p3n_decision["automatic_gate"]["passed"]
        or p3n_decision["production_integration_allowed"]
    ):
        raise ValueError("P3N parent decision drift")
    if (
        float(contract["candidate"]["maximum_transmittance_delta"])
        != float(p3n_contract["candidate"]["maximum_transmittance_delta"])
    ):
        raise ValueError("P3N response cap drift")

    population = contract["population"]
    preflight = _load_exact(
        root,
        population["preflight_decision_path"],
        population["preflight_decision_sha256"],
    )
    manifest = _load_exact(
        root,
        population["manifest_path"],
        population["manifest_sha256"],
    )
    source_review = _load_exact(
        root,
        population["visual_source_review_path"],
        population["visual_source_review_sha256"],
    )
    development = _load_exact(
        root,
        population["development_manifest_path"],
        population["development_manifest_sha256"],
    )
    if (
        preflight.get("decision") != population["required_preflight_decision"]
        or not preflight["automatic_evidence"]["automatic_pass"]
        or not preflight["visual_and_selection_evidence"]["visual_gate_pass"]
        or preflight["operator_applied"]
        or preflight["training_allowed"]
        or preflight["operator_fitting_allowed"]
    ):
        raise ValueError("fresh source preflight drift")
    eligible_ids = list(
        preflight["visual_and_selection_evidence"]["eligible_ids"]
    )
    excluded_ids = list(
        preflight["visual_and_selection_evidence"]["excluded_ids"]
    )
    if (
        len(eligible_ids) != int(population["eligible_source_count"])
        or set(excluded_ids) != set(population["excluded_ids"])
        or set(source_review["excluded_ids"]) != set(excluded_ids)
    ):
        raise ValueError("fresh source eligibility drift")
    by_id = {row["id"]: row for row in manifest}
    if len(by_id) != len(manifest) or not set(eligible_ids).issubset(by_id):
        raise ValueError("fresh source manifest drift")
    selected = [by_id[sample_id] for sample_id in eligible_ids]
    if len({row["make"] for row in selected}) != int(
        population["camera_make_count"]
    ):
        raise ValueError("fresh source make support drift")

    development_hashes = {row["raw_sha256"] for row in development}
    overlap = {
        row["raw_sha256"] for row in selected
    }.intersection(development_hashes)
    if (
        population["require_zero_exact_raw_hash_overlap_with_development"]
        and overlap
    ):
        raise ValueError("fresh source RAW overlaps development population")

    p3l_contract = _load_exact(
        root,
        p3n_contract["parents"]["p3l_contract_path"],
        p3n_contract["parents"]["p3l_contract_sha256"],
    )
    _, _, p1, p3d, sensitometry = validate_p3l_contract(
        root,
        p3l_contract,
    )
    return p3n_contract, selected, p1, p3d, sensitometry, {
        "overlap_count": len(overlap),
        "eligible_ids": eligible_ids,
    }


def evaluate_response_bounded_fresh_confirmation(
    *,
    root: Path,
    contract: dict[str, Any],
    visual_root: Path,
) -> dict[str, Any]:
    (
        _p3n_contract,
        manifest,
        p1,
        p3d,
        sensitometry,
        population_evidence,
    ) = _validate_contract(root, contract)
    legacy, forward, backing = _compile_profiles(p1, p3d)
    operator = build_operator(sensitometry)
    cap = float(contract["candidate"]["maximum_transmittance_delta"])
    gates = contract["automatic_gates"]
    fixed_ids = list(contract["visual_protocol"]["fixed_ids"])
    if not set(fixed_ids).issubset({row["id"] for row in manifest}):
        raise ValueError("fixed visual IDs are outside the eligible population")

    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    visual_assets: dict[str, dict[str, str]] = {}
    for source_row in manifest:
        raw_path = root / source_row["raw_path"]
        if sha256_file(raw_path) != source_row["raw_sha256"]:
            raise ValueError("fresh RAW identity drift")
        working = load_confirmation_working_image(raw_path)
        source = _resize_scene_linear(
            working.pixels,
            int(contract["population"]["maximum_long_edge"]),
        ).astype(np.float64)
        source_array = _array(
            source.astype(np.float32),
            legacy.pixel_pitch_um,
        )
        forward_exposure = apply_compiled_scatter(source_array, forward)
        unbounded_exposure = apply_fft_backing_return(
            forward_exposure,
            backing,
        ).values.astype(np.float64)
        bounded = apply_response_bounded_exposure_residual(
            source,
            unbounded_exposure,
            operator,
            maximum_transmittance_delta=cap,
        )
        no_spatial = _interpret(source, operator)
        bounded_output = _interpret(bounded.exposure, operator)
        bounded_delta = np.abs(bounded_output - no_spatial)
        pixel_delta = np.max(bounded_delta, axis=-1)
        new_boundary = (
            ((bounded_output <= 0.0) | (bounded_output >= 1.0))
            & ~((no_spatial <= 0.0) | (no_spatial >= 1.0))
        )
        shared_direction_error = float(
            np.max(
                np.abs(
                    (bounded.exposure - source)
                    - bounded.shared_scale
                    * (unbounded_exposure - source)
                )
            )
        )
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "model": source_row["model"],
                "raw_sha256": source_row["raw_sha256"],
                "scene_linear_sha256": hashlib.sha256(
                    source.tobytes()
                ).hexdigest(),
                "shape": list(source.shape),
                "bounded_max_abs": float(np.max(bounded_delta)),
                "bounded_p95_abs": float(
                    np.quantile(bounded_delta, 0.95)
                ),
                "bounded_fraction_pixels_over_0p005": float(
                    np.mean(pixel_delta > 0.005)
                ),
                "bounded_isolated_excursion_count": _isolated_excursions(
                    bounded_delta,
                    threshold=float(gates["isolated_excursion_threshold"]),
                    radius=int(gates["isolated_support_radius_pixels"]),
                    minimum_support=int(gates["minimum_isolated_support_count"]),
                ),
                "new_hard_boundary_fraction": float(np.mean(new_boundary)),
                "shared_scale_minimum": float(np.min(bounded.shared_scale)),
                "shared_scale_median": float(np.median(bounded.shared_scale)),
                "shared_direction_error": shared_direction_error,
            }
        )
        if source_row["id"] in fixed_ids:
            visual = {
                "no_spatial": _diagnostic_map(no_spatial),
                "bounded": _diagnostic_map(bounded_output),
            }
            visual_rows.append({"id": source_row["id"], "visual": visual})
            visual_assets[source_row["id"]] = {
                key: _save_rgb(
                    values,
                    visual_root / source_row["id"] / f"{key}.png",
                )
                for key, values in visual.items()
            }

    if {row["id"] for row in visual_rows} != set(fixed_ids):
        raise ValueError("fresh visual support is incomplete")
    ordered_visual_rows = sorted(
        visual_rows,
        key=lambda row: fixed_ids.index(row["id"]),
    )
    direct_keys = {sample_id: ("no_spatial", "bounded") for sample_id in fixed_ids}
    direct_sheet_sha256 = _save_sheet(
        ordered_visual_rows,
        keys_by_id=direct_keys,
        path=visual_root / "direct_severe_sheet.png",
    )

    blind_rounds = []
    for round_index, seed in enumerate(
        contract["visual_protocol"]["blind_value_round_seeds"],
        start=1,
    ):
        rng = random.Random(int(seed))
        mapping: dict[str, dict[str, str]] = {}
        keys_by_id: dict[str, tuple[str, str]] = {}
        for sample_id in fixed_ids:
            candidate_is_a = bool(rng.getrandbits(1))
            mapping[sample_id] = {
                "A": "bounded" if candidate_is_a else "no_spatial",
                "B": "no_spatial" if candidate_is_a else "bounded",
            }
            keys_by_id[sample_id] = (
                mapping[sample_id]["A"],
                mapping[sample_id]["B"],
            )
        mapping_path = visual_root / f"blind_round_{round_index}_mapping.json"
        mapping_sha256 = _save_json(
            {
                "schema": "neuro_film.u6_p3o_blind_mapping.v1",
                "round": round_index,
                "seed": int(seed),
                "mapping": mapping,
            },
            mapping_path,
        )
        sheet_sha256 = _save_sheet(
            ordered_visual_rows,
            keys_by_id=keys_by_id,
            path=visual_root / f"blind_round_{round_index}.png",
        )
        blind_rounds.append(
            {
                "round": round_index,
                "seed": int(seed),
                "mapping_sha256": mapping_sha256,
                "sheet_sha256": sheet_sha256,
            }
        )

    metrics = {
        "source_count": len(rows),
        "camera_make_count": len({row["make"] for row in rows}),
        "exact_raw_hash_overlap_with_development": population_evidence[
            "overlap_count"
        ],
        "bounded_maximum_abs": max(row["bounded_max_abs"] for row in rows),
        "bounded_population_p95_abs": float(
            np.quantile([row["bounded_p95_abs"] for row in rows], 0.95)
        ),
        "bounded_median_fraction_pixels_over_0p005": float(
            np.median(
                [row["bounded_fraction_pixels_over_0p005"] for row in rows]
            )
        ),
        "bounded_median_shared_scale": float(
            np.median([row["shared_scale_median"] for row in rows])
        ),
        "bounded_total_isolated_excursion_count": sum(
            row["bounded_isolated_excursion_count"] for row in rows
        ),
        "maximum_new_hard_boundary_fraction": max(
            row["new_hard_boundary_fraction"] for row in rows
        ),
        "maximum_shared_direction_error": max(
            row["shared_direction_error"] for row in rows
        ),
    }
    checks = {
        "source_count": metrics["source_count"]
        == int(gates["source_count_exact"]),
        "camera_make_count": metrics["camera_make_count"]
        == int(gates["camera_make_count_exact"]),
        "population_disjoint": metrics[
            "exact_raw_hash_overlap_with_development"
        ]
        <= int(gates["maximum_exact_raw_hash_overlap_with_development"]),
        "bounded_maximum": metrics["bounded_maximum_abs"]
        <= float(gates["bounded_maximum_abs_maximum"]),
        "bounded_nontrivial_p95": metrics["bounded_population_p95_abs"]
        >= float(gates["bounded_population_p95_abs_minimum"]),
        "bounded_nontrivial_coverage": metrics[
            "bounded_median_fraction_pixels_over_0p005"
        ]
        >= float(
            gates["bounded_median_fraction_pixels_over_0p005_minimum"]
        ),
        "bounded_scale": metrics["bounded_median_shared_scale"]
        >= float(gates["bounded_median_shared_scale_minimum"]),
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"]
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "isolated_excursions": metrics[
            "bounded_total_isolated_excursion_count"
        ]
        <= int(gates["maximum_isolated_candidate_excursion_count"]),
        "shared_direction": metrics["maximum_shared_direction_error"]
        <= float(gates["maximum_shared_direction_error"]),
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": "neuro_film.u6_p3o_response_bounded_fresh_confirmation_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "population": {
            "eligible_ids": population_evidence["eligible_ids"],
            "development_overlap_count": population_evidence["overlap_count"],
        },
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "visual_assets": visual_assets,
        "direct_severe_sheet_sha256": direct_sheet_sha256,
        "blind_rounds": blind_rounds,
        "branch": contract["branch_rule"][
            "value_pass" if automatic_pass else "automatic_fail"
        ],
    }
    return {
        **core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(core)
        ).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "SCHEMA",
    "evaluate_response_bounded_fresh_confirmation",
    "load_contract",
    "write_report",
]
