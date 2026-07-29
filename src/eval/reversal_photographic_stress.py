"""Photographic severe-artifact stress for generic reversal development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.eval.generic_reversal_development import _file_sha256
from src.eval.negative_route_photographic_stress import (
    _array_sha256,
    _isolated_noise,
    _load_source,
    _preview,
)
from src.eval.physical_scanner_profile import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    GenericReversalDevelopment,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    develop_reversal_layer_exposure,
    prepare_interpretation_medium,
    reversal_development_contract,
    reversal_development_identity,
    scan_interpretation_medium,
)


SCHEMA = "neuro_film.u6_p2g_reversal_photographic_stress.v1"


def _contact_sheet(
    visual_rows: list[dict[str, Any]],
    fixed_ids: list[str],
    path: Path,
) -> str:
    by_id = {row["id"]: row for row in visual_rows}
    tile_width, tile_height, header = 420, 276, 24
    canvas = Image.new(
        "RGB",
        (2 * tile_width, len(fixed_ids) * (tile_height + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = index * (tile_height + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        for column, image in enumerate(row["previews"]):
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return _file_sha256(path)


def evaluate(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    *,
    root: Path,
    contact_sheet_path: Path,
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2G contract")
    parents = contract["parents"]
    if (
        _file_sha256(root / parents["p2f_contract_path"])
        != parents["p2f_contract_sha256"]
    ):
        raise ValueError("P2F contract hash drift")
    if (
        _file_sha256(root / parents["p2f_decision_path"])
        != parents["p2f_decision_sha256"]
    ):
        raise ValueError("P2F decision hash drift")
    p2f = json.loads(
        (root / parents["p2f_contract_path"]).read_text(encoding="utf-8")
    )
    template_path = root / p2f["parents"]["negative_sensitometry_path"]
    if (
        _file_sha256(template_path)
        != p2f["parents"]["negative_sensitometry_config_sha256"]
    ):
        raise ValueError("reversal template hash drift")
    operator = GenericReversalDevelopment(
        build_operator(json.loads(template_path.read_text(encoding="utf-8"))),
        float(p2f["candidate"]["maximum_relative_layer_exposure"]),
    )
    if (
        reversal_development_identity(operator)
        != parents["reversal_operator_sha256"]
    ):
        raise ValueError("reversal operator identity drift")

    expected = contract["input"]
    manifest_path = root / expected["manifest"]
    preflight_path = root / expected["preflight_report"]
    if _file_sha256(manifest_path) != expected["manifest_sha256"]:
        raise ValueError("photographic manifest hash drift")
    if _file_sha256(preflight_path) != expected["preflight_report_sha256"]:
        raise ValueError("photographic preflight hash drift")
    if len(manifest) != int(expected["expected_rows"]):
        raise ValueError("photographic manifest row count drift")
    if len({row["make"] for row in manifest}) != int(
        expected["expected_camera_makes"]
    ):
        raise ValueError("photographic camera-make count drift")
    for row in manifest:
        if (
            row["allowed_use"] != expected["required_allowed_use"]
            or row["rights_scope"] != expected["required_rights_scope"]
        ):
            raise ValueError("photographic rights or allowed-use drift")

    scanner_contract = json.loads(
        (root / parents["scanner_contract_path"]).read_text(encoding="utf-8")
    )
    scanner = _profile(
        scanner_contract["profiles"][parents["scanner_profile"]]
    )
    stages = tuple(contract["pipeline"]["scanner_stages"])
    no_noise_stages = tuple(
        stage
        for stage in stages
        if stage != contract["pipeline"]["noise_ablation_stage"]
    )
    gates = contract["automatic_gates"]
    scale = float(contract["pipeline"]["pseudo_exposure_scale"])
    pitch = float(contract["pipeline"]["pixel_pitch_um"])
    route_contract = reversal_development_contract(operator)
    rows: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for source_row in manifest:
        encoded, linear, input_sha = _load_source(source_row, root)
        exposure = PhysicalDomainArray(
            (linear * np.float32(scale)).astype(np.float32),
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
            ("red", "green", "blue"),
        )
        developed = develop_reversal_layer_exposure(
            exposure, operator, route_contract
        )
        medium = prepare_interpretation_medium(developed)
        first = scan_interpretation_medium(
            medium, scanner, pixel_pitch_um=pitch, stages=stages
        ).scan_linear.values
        second = scan_interpretation_medium(
            medium, scanner, pixel_pitch_um=pitch, stages=stages
        ).scan_linear.values
        repeat_exact = first.tobytes() == second.tobytes()
        del second
        no_noise = scan_interpretation_medium(
            medium, scanner, pixel_pitch_um=pitch, stages=no_noise_stages
        ).scan_linear.values
        isolated_noise_count = _isolated_noise(
            first - no_noise,
            threshold=float(gates["isolated_noise_threshold"]),
            radius=int(gates["isolated_noise_radius_pixels"]),
            minimum_support=int(gates["isolated_noise_minimum_support"]),
        )
        del no_noise
        boundary = (
            ((first <= 0.0) | (first >= 1.0))
            & ~((linear <= 0.0) | (linear >= 1.0))
        )
        luma = (
            np.float32(0.2126) * first[..., 0]
            + np.float32(0.7152) * first[..., 1]
            + np.float32(0.0722) * first[..., 2]
        )
        p05, p95 = np.percentile(luma, [5.0, 95.0])
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "input_sha256": input_sha,
                "shape": list(linear.shape),
                "output_sha256": _array_sha256(first),
                "repeat_exact": repeat_exact,
                "finite_bounded": bool(
                    np.all(np.isfinite(first))
                    and np.all(first >= 0.0)
                    and np.all(first <= 1.0)
                ),
                "mean_abs_change": float(np.mean(np.abs(first - linear))),
                "new_boundary_fraction": float(np.mean(boundary)),
                "isolated_noise_count": isolated_noise_count,
                "near_black_fraction": float(
                    np.mean(luma <= float(gates["near_black_threshold"]))
                ),
                "near_white_fraction": float(
                    np.mean(luma >= float(gates["near_white_threshold"]))
                ),
                "luma_p95_minus_p05": float(p95 - p05),
            }
        )
        if source_row["id"] in contract["visual_protocol"]["fixed_ids"]:
            visuals.append(
                {
                    "id": source_row["id"],
                    "previews": (
                        _preview(encoded, (420, 276)),
                        _preview(
                            linear_srgb_to_encoded(first), (420, 276)
                        ),
                    ),
                }
            )
    fixed_ids = contract["visual_protocol"]["fixed_ids"]
    if {row["id"] for row in visuals} != set(fixed_ids):
        raise ValueError("fixed visual IDs are incomplete")
    contact_sha = _contact_sheet(visuals, fixed_ids, contact_sheet_path)
    population = {
        "median_mean_abs_change": float(
            np.median([row["mean_abs_change"] for row in rows])
        ),
        "maximum_new_boundary_fraction": max(
            row["new_boundary_fraction"] for row in rows
        ),
        "total_isolated_noise_count": sum(
            row["isolated_noise_count"] for row in rows
        ),
        "maximum_near_black_fraction": max(
            row["near_black_fraction"] for row in rows
        ),
        "maximum_near_white_fraction": max(
            row["near_white_fraction"] for row in rows
        ),
        "median_near_black_fraction": float(
            np.median([row["near_black_fraction"] for row in rows])
        ),
        "median_near_white_fraction": float(
            np.median([row["near_white_fraction"] for row in rows])
        ),
        "minimum_luma_p95_minus_p05": min(
            row["luma_p95_minus_p05"] for row in rows
        ),
    }
    decisions = {
        "input_hashes": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, manifest, strict=True)
        ),
        "finite_bounded": all(row["finite_bounded"] for row in rows),
        "repeat": all(row["repeat_exact"] for row in rows),
        "material": population["median_mean_abs_change"]
        >= float(gates["minimum_population_median_mean_abs_change"]),
        "boundary": population["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
        "isolated_noise": population["total_isolated_noise_count"]
        <= int(gates["maximum_isolated_noise_count"]),
        "per_image_near_black": population["maximum_near_black_fraction"]
        <= float(gates["maximum_per_image_near_black_fraction"]),
        "per_image_near_white": population["maximum_near_white_fraction"]
        <= float(gates["maximum_per_image_near_white_fraction"]),
        "population_near_black": population["median_near_black_fraction"]
        <= float(gates["maximum_population_median_near_black_fraction"]),
        "population_near_white": population["median_near_white_fraction"]
        <= float(gates["maximum_population_median_near_white_fraction"]),
        "photographic_luma_range": population["minimum_luma_p95_minus_p05"]
        >= float(gates["minimum_per_image_luma_p95_minus_p05"]),
    }
    passed = all(decisions.values())
    report = {
        "schema": "neuro_film.u6_p2g_reversal_photographic_stress_report.v1",
        "node": contract["node"],
        "operator_sha256": reversal_development_identity(operator),
        "rows": rows,
        "population": population,
        "contact_sheet_sha256": contact_sha,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rules"][
            "automatic_pass" if passed else "automatic_fail"
        ],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return report


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
