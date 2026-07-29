"""Fixed-chain U6.P3G ablation for backing-return topology."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.eval.physical_backing_return_photographic_stress import (
    _isolated_excursions,
    _load_source,
    _sha256,
)
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_compiled_backing_return,
    apply_compiled_scatter,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
    compile_scatter_profile,
    density_to_scan_transmittance,
)
from src.film_physics.reference_scatter import profile_from_contract


SCHEMA = "neuro_film.u6_p3g_backing_return_combined_ablation_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P3G contract")
    return payload


def _array(values: np.ndarray, pitch: float) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(pitch),
    )


def _compile_profiles(
    p1_contract: dict[str, Any], p3d_contract: dict[str, Any]
) -> tuple[Any, Any, Any]:
    legacy_reference = profile_from_contract(p1_contract)
    forward_contract = json.loads(json.dumps(p1_contract))
    forward_contract["components"] = [
        component
        for component in forward_contract["components"]
        if component["component_id"] == "forward-scatter"
    ]
    if len(forward_contract["components"]) != 1:
        raise ValueError("P1 forward-scatter component identity drift")
    forward_reference = profile_from_contract(forward_contract)
    backing_reference = backing_return_profile_from_contract(p3d_contract)
    if not (
        legacy_reference.pixel_pitch_um
        == forward_reference.pixel_pitch_um
        == backing_reference.pixel_pitch_um
    ):
        raise ValueError("combined-chain physical scale mismatch")
    return (
        compile_scatter_profile(legacy_reference),
        compile_scatter_profile(forward_reference),
        compile_backing_return_profile(backing_reference),
    )


def _interpret(exposure: np.ndarray, operator: Any) -> np.ndarray:
    density = operator.apply(exposure.astype(np.float64))
    return density_to_scan_transmittance(density)


def _variants(
    source: np.ndarray,
    *,
    legacy: Any,
    forward: Any,
    backing: Any,
    operator: Any,
) -> dict[str, np.ndarray]:
    source_array = _array(source, legacy.pixel_pitch_um)
    legacy_exposure = apply_compiled_scatter(source_array, legacy).values
    forward_exposure = apply_compiled_scatter(source_array, forward)
    split_exposure = apply_compiled_backing_return(
        forward_exposure, backing
    ).values
    legacy_array = _array(legacy_exposure, legacy.pixel_pitch_um)
    double_exposure = apply_compiled_backing_return(
        legacy_array, backing
    ).values
    return {
        "no_spatial": _interpret(source, operator),
        "legacy_full": _interpret(legacy_exposure, operator),
        "split_candidate": _interpret(split_exposure, operator),
        "forbidden_double": _interpret(double_exposure, operator),
    }


def _diagnostic_map(transmittance: np.ndarray) -> np.ndarray:
    return linear_srgb_to_encoded(
        np.clip(1.0 - transmittance, 0.0, 1.0)
    )


def _preview(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(
        np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    )
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _save_contact_sheet(
    visual_rows: list[dict[str, Any]], fixed_ids: list[str], path: Path
) -> str:
    by_id = {row["id"]: row for row in visual_rows}
    tile_width, tile_height, header = 260, 174, 24
    canvas = Image.new(
        "RGB",
        (4 * tile_width, len(fixed_ids) * (tile_height + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        variants = row["variants"]
        views = (
            variants["no_spatial"],
            variants["legacy_full"],
            variants["split_candidate"],
            variants["forbidden_double"],
        )
        for column, values in enumerate(views):
            image = _preview(_diagnostic_map(values), (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return _sha256(path)


def evaluate_combined_ablation(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    p1_contract: dict[str, Any],
    p3d_contract: dict[str, Any],
    sensitometry: dict[str, Any],
    *,
    root: Path,
    contact_sheet_path: Path,
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("invalid U6.P3G contract")
    expected = contract["input"]
    if _sha256(root / expected["manifest"]) != expected["manifest_sha256"]:
        raise ValueError("photographic manifest hash drift")
    if (
        _sha256(root / expected["preflight_report"])
        != expected["preflight_report_sha256"]
    ):
        raise ValueError("photographic preflight hash drift")
    if len(manifest) != int(expected["expected_rows"]):
        raise ValueError("photographic manifest row count drift")
    if len({row["make"] for row in manifest}) != int(
        expected["expected_camera_makes"]
    ):
        raise ValueError("photographic camera-make count drift")
    for row in manifest:
        if row["allowed_use"] != expected["required_allowed_use"]:
            raise ValueError("photographic allowed-use drift")
        if row["rights_scope"] != expected["required_rights_scope"]:
            raise ValueError("photographic rights-scope drift")
        if row["decoded_color_state"] != expected["decoded_color_state"]:
            raise ValueError("photographic color-state drift")

    legacy, forward, backing = _compile_profiles(p1_contract, p3d_contract)
    parents = contract["parents"]
    if legacy.parent_profile_sha256 != parents["p1_profile_sha256"]:
        raise ValueError("P1 profile identity drift")
    if backing.parent_profile_sha256 != parents["p3d_profile_sha256"]:
        raise ValueError("P3D profile identity drift")
    operator = build_operator(sensitometry)
    gates = contract["automatic_gates"]
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for source_row in manifest:
        source, input_sha = _load_source(source_row, root)
        variants = _variants(
            source,
            legacy=legacy,
            forward=forward,
            backing=backing,
            operator=operator,
        )
        candidate = variants["split_candidate"]
        no_spatial = variants["no_spatial"]
        legacy_full = variants["legacy_full"]
        forbidden_double = variants["forbidden_double"]
        candidate_delta = np.abs(candidate - no_spatial)
        new_boundary = (
            ((candidate <= 0.0) | (candidate >= 1.0))
            & ~((no_spatial <= 0.0) | (no_spatial >= 1.0))
        )
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "input_sha256": input_sha,
            "shape": list(source.shape),
            "finite_domain": all(
                np.all(np.isfinite(value))
                and np.all(value > 0.0)
                and np.all(value <= 1.0)
                for value in variants.values()
            ),
            "candidate_vs_no_max_abs": float(np.max(candidate_delta)),
            "candidate_vs_no_p95_abs": float(
                np.quantile(candidate_delta, 0.95)
            ),
            "candidate_vs_legacy_p95_abs": float(
                np.quantile(np.abs(candidate - legacy_full), 0.95)
            ),
            "candidate_vs_forbidden_double_p95_abs": float(
                np.quantile(np.abs(candidate - forbidden_double), 0.95)
            ),
            "new_hard_boundary_fraction": float(np.mean(new_boundary)),
            "isolated_candidate_excursion_count": _isolated_excursions(
                candidate_delta,
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            ),
        }
        rows.append(row)
        if source_row["id"] in contract["visual_protocol"]["fixed_ids"]:
            visual_rows.append(
                {"id": source_row["id"], "variants": variants}
            )
    fixed_ids = contract["visual_protocol"]["fixed_ids"]
    if {row["id"] for row in visual_rows} != set(fixed_ids):
        raise ValueError("fixed visual IDs are incomplete")
    contact_sha = _save_contact_sheet(visual_rows, fixed_ids, contact_sheet_path)

    population_p95_no = float(
        np.quantile([row["candidate_vs_no_p95_abs"] for row in rows], 0.95)
    )
    population_p95_legacy = float(
        np.quantile(
            [row["candidate_vs_legacy_p95_abs"] for row in rows], 0.95
        )
    )
    population_p95_double = float(
        np.quantile(
            [row["candidate_vs_forbidden_double_p95_abs"] for row in rows],
            0.95,
        )
    )
    decisions = {
        "input_hashes": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, manifest, strict=True)
        ),
        "finite_domain": all(row["finite_domain"] for row in rows),
        "candidate_bound": max(row["candidate_vs_no_max_abs"] for row in rows)
        <= float(gates["maximum_candidate_vs_no_spatial_abs"]),
        "candidate_nontrivial": population_p95_no
        >= float(gates["minimum_population_p95_candidate_vs_no_spatial_abs"]),
        "legacy_distinct": population_p95_legacy
        >= float(gates["minimum_population_p95_candidate_vs_legacy_abs"]),
        "double_counting_distinct": population_p95_double
        >= float(
            gates[
                "minimum_population_p95_candidate_vs_forbidden_double_abs"
            ]
        ),
        "new_boundaries": max(
            row["new_hard_boundary_fraction"] for row in rows
        )
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "isolated_excursions": sum(
            row["isolated_candidate_excursion_count"] for row in rows
        )
        <= int(gates["maximum_isolated_candidate_excursion_count"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p3g_backing_return_combined_ablation_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "rows": rows,
        "metrics": {
            "maximum_candidate_vs_no_spatial_abs": max(
                row["candidate_vs_no_max_abs"] for row in rows
            ),
            "population_p95_candidate_vs_no_spatial_abs": population_p95_no,
            "population_p95_candidate_vs_legacy_abs": population_p95_legacy,
            "population_p95_candidate_vs_forbidden_double_abs": (
                population_p95_double
            ),
            "total_isolated_candidate_excursion_count": sum(
                row["isolated_candidate_excursion_count"] for row in rows
            ),
            "maximum_new_hard_boundary_fraction": max(
                row["new_hard_boundary_fraction"] for row in rows
            ),
        },
        "contact_sheet_sha256": contact_sha,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
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
