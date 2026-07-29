"""Photographic severe-artifact stress for generic negative interpretations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import uniform_filter

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.physical_scanner_profile import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationRoute,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    compile_print_interpretation,
    develop_layer_exposure,
    prepare_interpretation_medium,
    scan_interpretation_medium,
)


SCHEMA = "neuro_film.u6_p2d_negative_route_photographic_stress.v1"
REFERENCE_GAUGE_SCHEMA = (
    "neuro_film.u6_p2e_reference_gauge_negative_stress.v1"
)
_REPORT_SCHEMAS = {
    SCHEMA: (
        "neuro_film.u6_p2d_negative_route_photographic_stress_report.v1"
    ),
    REFERENCE_GAUGE_SCHEMA: (
        "neuro_film.u6_p2e_reference_gauge_negative_stress_report.v1"
    ),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(values)).cast("B")
    ).hexdigest()


def _load_source(
    row: dict[str, Any], root: Path
) -> tuple[np.ndarray, np.ndarray, str]:
    path = root / row["decoded_path"]
    digest = _file_sha256(path)
    if digest != row["decoded_sha256"]:
        raise ValueError(f"decoded input hash drift for {row['id']}")
    with Image.open(path) as image:
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    linear = encoded_srgb_to_linear(encoded).astype(np.float32)
    return encoded, linear, digest


def _isolated_noise(
    difference: np.ndarray,
    *,
    threshold: float,
    radius: int,
    minimum_support: int,
) -> int:
    excursion = np.max(np.abs(difference), axis=-1) > threshold
    width = 2 * radius + 1
    support = uniform_filter(
        excursion.astype(np.float32),
        size=width,
        mode="constant",
        cval=0.0,
    ) * float(width * width)
    return int(np.count_nonzero(excursion & (support < minimum_support)))


def _preview(encoded: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(
        np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    )
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _contact_sheet(
    visual_rows: list[dict[str, Any]],
    fixed_ids: list[str],
    path: Path,
) -> str:
    by_id = {row["id"]: row for row in visual_rows}
    tile_width, tile_height, header = 320, 210, 24
    canvas = Image.new(
        "RGB",
        (3 * tile_width, len(fixed_ids) * (tile_height + header)),
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
    contract_schema = contract.get("schema")
    if contract_schema not in _REPORT_SCHEMAS:
        raise ValueError("unsupported negative-route photographic contract")
    reference_gauge = contract_schema == REFERENCE_GAUGE_SCHEMA
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
        if row["allowed_use"] != expected["required_allowed_use"]:
            raise ValueError("photographic allowed-use drift")
        if row["rights_scope"] != expected["required_rights_scope"]:
            raise ValueError("photographic rights-scope drift")

    u2 = json.loads(
        (root / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )
    sensitometry = build_operator(u2)
    witness = json.loads(
        (
            root / "configs/u5_r2e0_density_domain_operator_v1.json"
        ).read_text(encoding="utf-8")
    )["witnesses"][contract["pipeline"]["print_witness"]]
    print_operator = compile_print_interpretation(
        sensitometry,
        dye_absorption_matrix=np.asarray(witness["dye_absorption_matrix"]),
        print_matrix=np.asarray(witness["print_matrix"]),
        paper_midpoints=np.asarray(witness["paper_midpoints"]),
        paper_slopes=np.asarray(witness["paper_slopes"]),
        paper_maximum_densities=np.asarray(
            witness["paper_maximum_densities"]
        ),
        maximum_relative_layer_exposure=float(
            contract["pipeline"].get(
                "print_maximum_relative_layer_exposure",
                contract["pipeline"]["pseudo_exposure_scale"],
            )
        ),
    )
    scanner_contract = json.loads(
        (
            root / contract["parents"]["scanner_contract_path"]
        ).read_text(encoding="utf-8")
    )
    scanner = _profile(
        scanner_contract["profiles"][
            contract["parents"]["scanner_profile"]
        ]
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
    rows = []
    visuals = []
    for source_row in manifest:
        encoded, linear, input_sha = _load_source(source_row, root)
        exposure = PhysicalDomainArray(
            (linear * np.float32(scale)).astype(np.float32),
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
            ("red", "green", "blue"),
        )
        route_outputs: dict[str, np.ndarray] = {}
        route_rows = {}
        for name, route, print_arg in (
            (
                "negative",
                InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
                None,
            ),
            (
                "print",
                InterpretationRoute.COLOR_NEGATIVE_PRINT,
                print_operator,
            ),
        ):
            route_contract = DevelopmentInterpretationContract.for_operator(
                sensitometry,
                emulsion_family=EmulsionFamily.COLOR_NEGATIVE,
                interpretation_route=route,
            )
            developed = develop_layer_exposure(
                exposure, sensitometry, route_contract
            )
            medium = prepare_interpretation_medium(
                developed, print_interpretation=print_arg
            )
            first = scan_interpretation_medium(
                medium,
                scanner,
                pixel_pitch_um=pitch,
                stages=stages,
            ).scan_linear.values
            second = scan_interpretation_medium(
                medium,
                scanner,
                pixel_pitch_um=pitch,
                stages=stages,
            ).scan_linear.values
            repeat_exact = first.tobytes() == second.tobytes()
            del second
            no_noise = scan_interpretation_medium(
                medium,
                scanner,
                pixel_pitch_um=pitch,
                stages=no_noise_stages,
            ).scan_linear.values
            isolated_noise_count = _isolated_noise(
                first - no_noise,
                threshold=float(gates["isolated_noise_threshold"]),
                radius=int(gates["isolated_noise_radius_pixels"]),
                minimum_support=int(
                    gates["isolated_noise_minimum_support"]
                ),
            )
            del no_noise
            boundary = (
                ((first <= 0.0) | (first >= 1.0))
                & ~((linear <= 0.0) | (linear >= 1.0))
            )
            route_outputs[name] = first
            route_row = {
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
            }
            if reference_gauge:
                luma = (
                    np.float32(0.2126) * first[..., 0]
                    + np.float32(0.7152) * first[..., 1]
                    + np.float32(0.0722) * first[..., 2]
                )
                route_row["near_white_fraction"] = float(
                    np.mean(
                        luma
                        >= float(gates["near_white_threshold"])
                    )
                )
                p05, p95 = np.percentile(luma, [5.0, 95.0])
                route_row["luma_p95_minus_p05"] = float(p95 - p05)
            route_rows[name] = route_row
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "input_sha256": input_sha,
                "shape": list(linear.shape),
                "routes": route_rows,
            }
        )
        if source_row["id"] in contract["visual_protocol"]["fixed_ids"]:
            visuals.append(
                {
                    "id": source_row["id"],
                    "previews": (
                        _preview(encoded, (320, 210)),
                        _preview(
                            linear_srgb_to_encoded(
                                route_outputs["negative"]
                            ),
                            (320, 210),
                        ),
                        _preview(
                            linear_srgb_to_encoded(route_outputs["print"]),
                            (320, 210),
                        ),
                    ),
                }
            )
    fixed_ids = contract["visual_protocol"]["fixed_ids"]
    if {row["id"] for row in visuals} != set(fixed_ids):
        raise ValueError("fixed visual IDs are incomplete")
    contact_sha = _contact_sheet(visuals, fixed_ids, contact_sheet_path)
    route_names = ("negative", "print")
    decisions = {
        "input_hashes": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, manifest, strict=True)
        ),
        "finite_bounded": all(
            row["routes"][route]["finite_bounded"]
            for row in rows
            for route in route_names
        ),
        "repeat": all(
            row["routes"][route]["repeat_exact"]
            for row in rows
            for route in route_names
        ),
        "material": all(
            float(
                np.median(
                    [
                        row["routes"][route]["mean_abs_change"]
                        for row in rows
                    ]
                )
            )
            >= float(gates["minimum_population_median_mean_abs_change"])
            for route in route_names
        ),
        "boundary": max(
            row["routes"][route]["new_boundary_fraction"]
            for row in rows
            for route in route_names
        )
        <= float(gates["maximum_new_boundary_fraction"]),
        "isolated_noise": sum(
            row["routes"][route]["isolated_noise_count"]
            for row in rows
            for route in route_names
        )
        <= int(gates["maximum_isolated_noise_count"]),
    }
    if reference_gauge:
        decisions["per_image_near_white"] = max(
            row["routes"][route]["near_white_fraction"]
            for row in rows
            for route in route_names
        ) <= float(gates["maximum_per_image_near_white_fraction"])
        decisions["population_near_white"] = all(
            float(
                np.median(
                    [
                        row["routes"][route]["near_white_fraction"]
                        for row in rows
                    ]
                )
            )
            <= float(
                gates["maximum_population_median_near_white_fraction"]
            )
            for route in route_names
        )
        decisions["photographic_luma_range"] = min(
            row["routes"][route]["luma_p95_minus_p05"]
            for row in rows
            for route in route_names
        ) >= float(gates["minimum_per_image_luma_p95_minus_p05"])
    passed = all(decisions.values())
    population = {
        route: {
            "median_mean_abs_change": float(
                np.median(
                    [
                        row["routes"][route]["mean_abs_change"]
                        for row in rows
                    ]
                )
            ),
            "maximum_new_boundary_fraction": max(
                row["routes"][route]["new_boundary_fraction"]
                for row in rows
            ),
            "total_isolated_noise_count": sum(
                row["routes"][route]["isolated_noise_count"]
                for row in rows
            ),
        }
        for route in route_names
    }
    if reference_gauge:
        for route in route_names:
            population[route].update(
                {
                    "maximum_near_white_fraction": max(
                        row["routes"][route]["near_white_fraction"]
                        for row in rows
                    ),
                    "median_near_white_fraction": float(
                        np.median(
                            [
                                row["routes"][route][
                                    "near_white_fraction"
                                ]
                                for row in rows
                            ]
                        )
                    ),
                    "minimum_luma_p95_minus_p05": min(
                        row["routes"][route]["luma_p95_minus_p05"]
                        for row in rows
                    ),
                }
            )
    core = {
        "schema": _REPORT_SCHEMAS[contract_schema],
        "node": contract["node"],
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
    core["stable_evidence_id"] = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return core


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
