"""U6.P3N response-bounded backing-return development evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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


SCHEMA = "neuro_film.u6_p3n_response_bounded_backing_return_contract.v1"


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
        raise ValueError("unsupported U6.P3N contract")
    return payload


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
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
        path, format="PNG", optimize=False, compress_level=9
    )
    return sha256_file(path)


def _preview(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(encoded, mode="RGB")
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _save_sheet(
    rows: list[dict[str, Any]],
    fixed_ids: list[str],
    path: Path,
) -> str:
    by_id = {row["id"]: row for row in rows}
    tile_width, tile_height, header = 320, 214, 24
    canvas = Image.new(
        "RGB",
        (3 * tile_width, len(fixed_ids) * (tile_height + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        for column, key in enumerate(("no_spatial", "unbounded", "bounded")):
            image = _preview(row["visual"][key], (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return sha256_file(path)


def evaluate_response_bounded_backing_return(
    *,
    root: Path,
    contract: dict[str, Any],
    visual_root: Path,
) -> dict[str, Any]:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("training_allowed")
        or contract.get("production_integration_allowed")
        or contract["candidate"]["profile_fitting_allowed"]
        or contract["candidate"]["per_image_parameter_selection_allowed"]
        or contract["candidate"]["hard_clipping_allowed"]
        or contract["input"]["physical_scale_claimed"]
    ):
        raise ValueError("invalid U6.P3N contract")

    parents = contract["parents"]
    p3l = _load_exact(
        root,
        parents["p3l_contract_path"],
        parents["p3l_contract_sha256"],
    )
    p3l_decision = _load_exact(
        root,
        parents["p3l_decision_path"],
        parents["p3l_decision_sha256"],
    )
    parent_report = _load_exact(
        root,
        parents["p3l_report_path"],
        parents["p3l_report_sha256"],
    )
    if (
        p3l_decision["decision"]
        != "close_fixed_scene_linear_backing_return_topology"
        or p3l_decision["automatic_gate"]["passed"]
        or parent_report["automatic_pass"]
    ):
        raise ValueError("P3L negative parent drift")

    manifest, p8bp, p1, p3d, sensitometry = validate_p3l_contract(root, p3l)
    if (
        len(manifest) != int(contract["input"]["source_count"])
        or len({row["make"] for row in manifest})
        != int(contract["input"]["camera_make_count"])
        or int(p3l["input"]["maximum_long_edge"])
        != int(contract["input"]["maximum_long_edge"])
    ):
        raise ValueError("P3N source support drift")

    legacy, forward, backing = _compile_profiles(p1, p3d)
    operator = build_operator(sensitometry)
    candidates = {row["id"]: row for row in p8bp["candidates"]}
    cap = float(contract["candidate"]["maximum_transmittance_delta"])
    gates = contract["automatic_gates"]
    fixed_ids = list(contract["visual_protocol"]["fixed_ids"])
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    visual_assets: dict[str, dict[str, str]] = {}

    for source_row in manifest:
        candidate_record = candidates[source_row["id"]]
        raw_path = root / candidate_record["path"]
        if sha256_file(raw_path) != candidate_record["sha256"]:
            raise ValueError("P3N RAW identity drift")
        working = load_confirmation_working_image(raw_path)
        source = _resize_scene_linear(
            working.pixels,
            int(contract["input"]["maximum_long_edge"]),
        ).astype(np.float64)
        source_array = _array(
            source.astype(np.float32), legacy.pixel_pitch_um
        )
        forward_exposure = apply_compiled_scatter(source_array, forward)
        unbounded_exposure = apply_fft_backing_return(
            forward_exposure, backing
        ).values.astype(np.float64)
        bounded = apply_response_bounded_exposure_residual(
            source,
            unbounded_exposure,
            operator,
            maximum_transmittance_delta=cap,
        )

        no_spatial = _interpret(source, operator)
        unbounded_output = _interpret(unbounded_exposure, operator)
        bounded_output = _interpret(bounded.exposure, operator)
        unbounded_delta = np.abs(unbounded_output - no_spatial)
        bounded_delta = np.abs(bounded_output - no_spatial)
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
        pixel_delta = np.max(bounded_delta, axis=-1)
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "raw_sha256": candidate_record["sha256"],
            "scene_linear_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "shape": list(source.shape),
            "unbounded_max_abs": float(np.max(unbounded_delta)),
            "unbounded_p95_abs": float(np.quantile(unbounded_delta, 0.95)),
            "unbounded_isolated_excursion_count": _isolated_excursions(
                unbounded_delta,
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            ),
            "bounded_max_abs": float(np.max(bounded_delta)),
            "bounded_p95_abs": float(np.quantile(bounded_delta, 0.95)),
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
            "shared_scale_p10": float(
                np.quantile(bounded.shared_scale, 0.1)
            ),
            "shared_scale_median": float(np.median(bounded.shared_scale)),
            "shared_direction_error": shared_direction_error,
        }
        rows.append(row)
        if source_row["id"] in fixed_ids:
            visual = {
                "no_spatial": _diagnostic_map(no_spatial),
                "unbounded": _diagnostic_map(unbounded_output),
                "bounded": _diagnostic_map(bounded_output),
            }
            visual_rows.append({"id": source_row["id"], "visual": visual})
            sample_assets = {}
            for key, values in visual.items():
                relative = Path(source_row["id"]) / f"{key}.png"
                sample_assets[key] = _save_rgb(values, visual_root / relative)
            visual_assets[source_row["id"]] = sample_assets

    if {row["id"] for row in visual_rows} != set(fixed_ids):
        raise ValueError("P3N fixed visual support is incomplete")
    sheet_sha256 = _save_sheet(
        visual_rows, fixed_ids, visual_root / "contact_sheet.png"
    )

    metrics = {
        "source_count": len(rows),
        "camera_make_count": len({row["make"] for row in rows}),
        "unbounded_maximum_abs": max(row["unbounded_max_abs"] for row in rows),
        "unbounded_total_isolated_excursion_count": sum(
            row["unbounded_isolated_excursion_count"] for row in rows
        ),
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
    parent_metrics = parent_report["metrics"]
    parent_reproduced = (
        metrics["unbounded_maximum_abs"]
        == float(parent_metrics["maximum_candidate_vs_no_spatial_abs"])
        and metrics["unbounded_total_isolated_excursion_count"]
        == int(parent_metrics["total_isolated_candidate_excursion_count"])
    )
    checks = {
        "source_support": metrics["source_count"]
        == int(contract["input"]["source_count"])
        and metrics["camera_make_count"]
        == int(contract["input"]["camera_make_count"]),
        "parent_failure_reproduced": parent_reproduced,
        "unbounded_maximum_negative": metrics["unbounded_maximum_abs"]
        >= float(gates["parent_unbounded_maximum_abs_minimum"]),
        "unbounded_isolated_negative": metrics[
            "unbounded_total_isolated_excursion_count"
        ]
        >= int(gates["parent_unbounded_isolated_excursion_minimum"]),
        "bounded_maximum": metrics["bounded_maximum_abs"]
        <= float(gates["bounded_maximum_abs_maximum"]),
        "bounded_nontrivial_p95": metrics["bounded_population_p95_abs"]
        >= float(gates["bounded_population_p95_abs_minimum"]),
        "bounded_nontrivial_coverage": metrics[
            "bounded_median_fraction_pixels_over_0p005"
        ]
        >= float(
            gates[
                "bounded_median_fraction_pixels_over_0p005_minimum"
            ]
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
    passed = all(checks.values())
    core = {
        "schema": "neuro_film.u6_p3n_response_bounded_backing_return_report.v1",
        "node": contract["node"],
        "epistemic_scope": contract["epistemic_scope"],
        "claim_ceiling": contract["claim_ceiling"],
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "visual_assets": visual_assets,
        "contact_sheet_sha256": sheet_sha256,
        "branch": contract["branch_rule"][
            "development_pass" if passed else "automatic_fail"
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
    "evaluate_response_bounded_backing_return",
    "load_contract",
    "write_report",
]
