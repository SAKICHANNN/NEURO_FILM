"""Automatic AO6 value gate for the fixed P4IL sigmoid/scanner candidate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from skimage import data

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.characteristic_scanner_spatial_smoke import _structured_transmittance
from src.eval.layer_gamma_photographic_development import _high_frequency_chroma_p999
from src.eval.physical_spatial_photographic_stress import (
    _flat_region_p99,
    _isolated_excursions,
)
from src.eval.sigmoid_characteristic_spatial_d0 import _runtime
from src.film_physics.display_look import build_source_context_display_look_stages
from src.film_physics.profile_consumer import compile_standalone_profile_artifact
from src.film_physics.sigmoid_characteristic import render_sigmoid_scanner_positive
from src.film_physics.spatial_response import SpatialResponseProfile, apply_scanner_mtf

SCHEMA = "neuro-film.u6-p4im-sigmoid-scanner-ao6-value-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4im-sigmoid-scanner-ao6-value-d1-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IM contract")
    return value


def _boundary(candidate: np.ndarray, reference: np.ndarray | None = None) -> dict[str, float]:
    code = np.rint(np.asarray(candidate) * 65535.0).astype(np.uint16)
    mask = np.any((code == 0) | (code == 65535), axis=-1)
    result = {"output_code_boundary_fraction": float(np.mean(mask))}
    if reference is not None:
        ref = np.rint(np.asarray(reference) * 65535.0).astype(np.uint16)
        ref_mask = np.any((ref == 0) | (ref == 65535), axis=-1)
        result["new_boundary_fraction"] = float(np.mean(mask & ~ref_mask))
    return result


def _thumbnail(array: np.ndarray, size: tuple[int, int] = (320, 220)) -> Image.Image:
    encoded = np.asarray(array, dtype=np.float64)
    image = Image.fromarray(np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#181818")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _write_contact(rows: list[dict[str, Any]], images: list[list[np.ndarray]], path: Path) -> str:
    cell_w, cell_h, label_h = 320, 220, 28
    sheet = Image.new("RGB", (cell_w * 5, (cell_h + label_h) * len(rows)), "#101010")
    draw = ImageDraw.Draw(sheet)
    labels = ("SOURCE", "CURRENT AO6", "MATCHED AO6", "PHYSICAL", "COMBINED")
    for y, (row, row_images) in enumerate(zip(rows, images, strict=True)):
        top = y * (cell_h + label_h)
        for x, (label, image) in enumerate(zip(labels, row_images, strict=True)):
            sheet.paste(_thumbnail(image), (x * cell_w, top + label_h))
            draw.text((x * cell_w + 6, top + 7), f"{row['id']} | {label}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=False)
    return _sha(path)


def evaluate(
    contract: Mapping[str, Any],
    root: Path,
    *,
    contact_path: Path | None = None,
    structure_direction: tuple[float, float, float] | None = None,
    structure_builder: Callable[
        [np.ndarray, np.ndarray, int, float], np.ndarray
    ]
    | None = None,
    scan_structure_builder: Callable[
        [np.ndarray, np.ndarray, int, float], np.ndarray
    ]
    | None = None,
    post_base_structure_builder: Callable[
        [np.ndarray, np.ndarray, int, float], np.ndarray
    ]
    | None = None,
    stage_observer: Callable[[str, Mapping[str, np.ndarray]], None] | None = None,
) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    if _sha(parent_path) != contract["parent"]["sha256"]:
        raise ValueError("P4IM parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("P4IM parent decision drift")
    ao6_path = root / str(contract["ao6"]["config_path"])
    if _sha(ao6_path) != contract["ao6"]["config_sha256"]:
        raise ValueError("P4IM AO6 config drift")
    artifact = compile_standalone_profile_artifact(
        root=root, config=json.loads(ao6_path.read_text(encoding="utf-8"))
    )
    if artifact["bundle_sha256"] != contract["ao6"]["required_bundle_sha256"]:
        raise ValueError("P4IM AO6 bundle drift")
    display_payload = artifact["component_payloads"]["ao6-source-context-display-look"]
    p4il_contract = json.loads((root / "configs/u6_p4il_sigmoid_scanner_bundled_natural_d1_v1.json").read_text(encoding="utf-8"))
    curves, _, compiler = _runtime(root, json.loads((root / "configs/u6_p4ij_sigmoid_characteristic_spatial_d0_v1.json").read_text(encoding="utf-8")))
    mechanism = p4il_contract["mechanism"]
    spatial = SpatialResponseProfile(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), tuple(mechanism["scanner_mtf_sigma_pixels_rgb"]), float(mechanism["gaussian_truncate"]))
    rows: list[dict[str, Any]] = []
    contact_images: list[list[np.ndarray]] = []
    for index, source_id in enumerate(p4il_contract["source"]["ids"]):
        raw = np.ascontiguousarray(getattr(data, source_id)())
        original = np.ascontiguousarray(raw.astype(np.float64) / 255.0, dtype=np.float32)
        linear = np.ascontiguousarray(encoded_srgb_to_linear(original.astype(np.float64)), dtype=np.float32)
        density = np.stack([curve.apply_normalized(linear[..., c]) for c, curve in enumerate(curves)], axis=-1)
        base_t = np.ascontiguousarray(np.power(10.0, -density), dtype=np.float32)
        baseline, _ = render_sigmoid_scanner_positive(linear, base_t, curves=curves, compiler=compiler)
        if sum(
            builder is not None
            for builder in (
                structure_builder,
                scan_structure_builder,
                post_base_structure_builder,
            )
        ) > 1:
            raise ValueError("P4IM structure builders are mutually exclusive")
        if structure_builder is not None:
            structured = structure_builder(
                base_t,
                linear,
                index,
                float(mechanism["structure_amplitude"]),
            )
            if (
                structured.shape != base_t.shape
                or structured.dtype != np.float32
                or not np.all(np.isfinite(structured))
                or np.any(structured <= 0.0)
                or np.any(structured > 1.0)
            ):
                raise ValueError("invalid P4IM structured transmittance")
        elif structure_direction is None:
            structured = _structured_transmittance(
                base_t, index, float(mechanism["structure_amplitude"])
            )
        else:
            height, width = base_t.shape[:2]
            y, x = np.mgrid[0:height, 0:width].astype(np.float64)
            field = np.sin((x + 11 * index) / 23.0) * np.cos(
                (y - 7 * index) / 19.0
            )
            direction = np.asarray(structure_direction, dtype=np.float64)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)):
                raise ValueError("invalid P4IM structure direction")
            residual = (
                float(mechanism["structure_amplitude"])
                * field[..., None]
                * direction
            )
            structured = np.ascontiguousarray(
                base_t.astype(np.float64) * np.power(10.0, -residual),
                dtype=np.float32,
            )
        baseline = apply_scanner_mtf(baseline, spatial).astype(np.float32)
        if scan_structure_builder is None:
            physical, _ = render_sigmoid_scanner_positive(
                linear, structured, curves=curves, compiler=compiler
            )
            physical = apply_scanner_mtf(physical, spatial).astype(np.float32)
        else:
            physical = scan_structure_builder(
                baseline,
                linear,
                index,
                float(mechanism["structure_amplitude"]),
            )
            if (
                physical.shape != baseline.shape
                or physical.dtype != np.float32
                or not np.all(np.isfinite(physical))
                or np.any(physical < 0.0)
                or np.any(physical > 1.0)
            ):
                raise ValueError("invalid P4IM scan-linear structure")
        baseline_encoded = np.ascontiguousarray(linear_srgb_to_encoded(baseline.astype(np.float64)), dtype=np.float32)
        physical_encoded = np.ascontiguousarray(linear_srgb_to_encoded(physical.astype(np.float64)), dtype=np.float32)
        apply_base, apply_residual = build_source_context_display_look_stages(display_payload, original)

        def ao6(
            value: np.ndarray,
            *,
            base: Callable[[np.ndarray], np.ndarray] = apply_base,
            residual: Callable[[np.ndarray], np.ndarray] = apply_residual,
        ) -> np.ndarray:
            return np.ascontiguousarray(residual(base(value)), dtype=np.float32)

        current_base = apply_base(original)
        matched_base = apply_base(baseline_encoded)
        if post_base_structure_builder is None:
            physical_base = apply_base(physical_encoded)
            physical_arm = physical_encoded
        else:
            physical_base = post_base_structure_builder(
                matched_base,
                linear,
                index,
                float(mechanism["structure_amplitude"]),
            )
            if (
                physical_base.shape != matched_base.shape
                or physical_base.dtype != np.float32
                or not np.all(np.isfinite(physical_base))
                or np.any(physical_base < 0.0)
                or np.any(physical_base > 1.0)
            ):
                raise ValueError("invalid P4IM post-base structure")
            physical_arm = physical_base
        current = np.ascontiguousarray(apply_residual(current_base), dtype=np.float32)
        matched = np.ascontiguousarray(apply_residual(matched_base), dtype=np.float32)
        combined = np.ascontiguousarray(apply_residual(physical_base), dtype=np.float32)
        replay = (
            ao6(physical_encoded)
            if post_base_structure_builder is None
            else np.ascontiguousarray(
                apply_residual(physical_base), dtype=np.float32
            )
        )
        if stage_observer is not None:
            stage_observer(
                source_id,
                {
                    "baseline_encoded": baseline_encoded,
                    "physical_encoded": physical_encoded,
                    "matched_base": matched_base,
                    "physical_base": physical_base,
                    "matched_output": matched,
                    "physical_output": combined,
                },
            )
        arms = (original, current, matched, physical_arm, combined)
        if any(not np.all(np.isfinite(arm)) or np.any(arm < 0.0) or np.any(arm > 1.0) for arm in arms):
            raise ValueError(f"P4IM arm left bounded display RGB: {source_id}")
        delta = combined.astype(np.float64) - matched.astype(np.float64)
        product_delta = combined.astype(np.float64) - current.astype(np.float64)
        boundary = _boundary(combined, matched)
        rows.append({
            "id": source_id,
            "source_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "arm_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in zip(contract["arms"], arms, strict=True)},
            "p95_combined_vs_matched_abs": float(np.percentile(np.abs(delta), 95)),
            "p99_combined_vs_matched_abs": float(np.percentile(np.abs(delta), 99)),
            "flat_region_p99_combined_vs_matched_abs": _flat_region_p99(original, delta),
            "high_frequency_chroma_p999": _high_frequency_chroma_p999(delta),
            "isolated_excursions": _isolated_excursions(delta, threshold=0.05, radius=2, minimum_support=3),
            "new_boundary_fraction_vs_matched": boundary["new_boundary_fraction"],
            "maximum_arm_output_code_boundary_fraction": max(_boundary(arm)["output_code_boundary_fraction"] for arm in arms[1:]),
            "p95_combined_vs_current_abs": float(np.percentile(np.abs(product_delta), 95)),
            "repeat_error": float(np.max(np.abs(combined - replay))),
        })
        contact_images.append(list(arms))
    metrics = {
        "row_count": len(rows),
        "population_p95_combined_vs_matched_abs": float(np.percentile([r["p95_combined_vs_matched_abs"] for r in rows], 95)),
        "population_p99_combined_vs_matched_abs": float(np.percentile([r["p99_combined_vs_matched_abs"] for r in rows], 99)),
        "maximum_flat_region_p99_combined_vs_matched_abs": max(r["flat_region_p99_combined_vs_matched_abs"] for r in rows),
        "maximum_high_frequency_chroma_p999": max(r["high_frequency_chroma_p999"] for r in rows),
        "total_isolated_excursions": sum(r["isolated_excursions"] for r in rows),
        "maximum_new_boundary_fraction_vs_matched": max(r["new_boundary_fraction_vs_matched"] for r in rows),
        "maximum_arm_output_code_boundary_fraction": max(r["maximum_arm_output_code_boundary_fraction"] for r in rows),
        "population_p95_combined_vs_current_abs": float(np.percentile([r["p95_combined_vs_current_abs"] for r in rows], 95)),
        "maximum_repeat_error": max(r["repeat_error"] for r in rows),
    }
    g = contract["gates"]
    checks = {
        "material_increment": metrics["population_p95_combined_vs_matched_abs"] >= g["minimum_population_p95_combined_vs_matched_abs"],
        "tail": metrics["population_p99_combined_vs_matched_abs"] <= g["maximum_population_p99_combined_vs_matched_abs"],
        "flat": metrics["maximum_flat_region_p99_combined_vs_matched_abs"] <= g["maximum_flat_region_p99_combined_vs_matched_abs"],
        "chroma": metrics["maximum_high_frequency_chroma_p999"] <= g["maximum_high_frequency_chroma_p999"],
        "isolated": metrics["total_isolated_excursions"] <= g["maximum_isolated_excursions"],
        "new_boundary": metrics["maximum_new_boundary_fraction_vs_matched"] <= g["maximum_new_boundary_fraction_vs_matched"],
        "output_boundary": metrics["maximum_arm_output_code_boundary_fraction"] <= g["maximum_output_code_boundary_fraction"],
        "product_separation": metrics["population_p95_combined_vs_current_abs"] >= g["minimum_population_p95_combined_vs_current_abs"],
        "repeat": metrics["maximum_repeat_error"] <= g["maximum_repeat_error"],
    }
    passed = all(checks.values())
    core: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "ao6_bundle_sha256": artifact["bundle_sha256"],
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "blind_review_allowed": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    if contact_path is not None:
        core["contact_sheet_sha256"] = _write_contact(rows, contact_images, contact_path)
    stable = {key: value for key, value in core.items() if key != "contact_sheet_sha256"}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
