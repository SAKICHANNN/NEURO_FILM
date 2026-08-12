"""Photographic ablation for the retained P4GQ physical residual."""

from __future__ import annotations

import _ctypes
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.native_cloud_residual_display import _apply_residual
from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure_source_derived,
    render_physical_partition,
)
from src.eval.physical_spatial_photographic_stress import (
    _flat_region_p99,
    _isolated_excursions,
)
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.bounded_common_density_residual import (
    apply_bounded_common_density_residual,
)
from src.film_physics.bounded_linear_residual import apply_bounded_linear_residual
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.density_lod_residual import apply_density_lod_residual
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile as scatter_profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime

SCHEMA = "neuro-film.u6-p4gr-neutral-base-photographic-ablation-contract.v1"
SHARED_DENSITY_SCHEMA = (
    "neuro-film.u6-p4gs-shared-density-photographic-development-contract.v1"
)
DENSITY_LOD_SCHEMA = (
    "neuro-film.u6-p4gt-density-lod-photographic-development-contract.v1"
)
COMPOUND_POISSON_SCHEMA = (
    "neuro-film.u6-p4gu-compound-poisson-density-photographic-development-contract.v1"
)
NPS_COMPILED_SCHEMA = (
    "neuro-film.u6-p4gv-nps-compiled-density-photographic-development-contract.v1"
)
P4FB = Path("configs/u6_p4fb_native_cloud_spatial_partition_v1.json")
CAPACITY = Path("configs/u6_p4di_sensitometry_cloud_capacity_v2.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in {
        SCHEMA,
        SHARED_DENSITY_SCHEMA,
        DENSITY_LOD_SCHEMA,
        COMPOUND_POISSON_SCHEMA,
        NPS_COMPILED_SCHEMA,
    }:
        raise ValueError("unsupported photographic physical-residual contract")
    return payload


def _load_source(row: dict[str, Any], root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = root / row["decoded_path"]
    if sha256_file(path) != row["decoded_sha256"]:
        raise ValueError(f"P4GR decoded input hash drift for {row['id']}")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"P4GR decoded input mode drift for {row['id']}")
        encoded = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    linear = np.ascontiguousarray(
        encoded_srgb_to_linear(encoded.astype(np.float64)), dtype=np.float32
    )
    return np.ascontiguousarray(encoded), linear


def _new_boundary_fraction(reference: np.ndarray, candidate: np.ndarray) -> float:
    epsilon = 1.0 / 65535.0
    reference_boundary = (reference <= epsilon) | (reference >= 1.0 - epsilon)
    candidate_boundary = (candidate <= epsilon) | (candidate >= 1.0 - epsilon)
    return float(np.mean(candidate_boundary & ~reference_boundary))


def _preview(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(
        np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB"
    )
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _contact_sheet(rows: list[dict[str, Any]], path: Path) -> str:
    tile_width, tile_height, header = 256, 180, 22
    canvas = Image.new(
        "RGB", (5 * tile_width, len(rows) * (tile_height + header)), (24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), f"{row['id']} / {row['make']}", fill=(235, 235, 235))
        values = (
            row["source"],
            row["ao6"],
            row["physical"],
            row["combined"],
            np.clip(0.5 + 8.0 * (row["combined"] - row["ao6"]), 0.0, 1.0),
        )
        for column, panel in enumerate(values):
            image = _preview(panel, (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return sha256_file(path)


def evaluate(
    contract: dict[str, Any],
    *,
    root: Path,
    contact_sheet_path: Path,
) -> dict[str, Any]:
    parent_contract = contract.get(
        "parent", contract.get("parents", {}).get("photographic")
    )
    if not isinstance(parent_contract, dict):
        raise TypeError("P4GR parent contract missing")
    parent_path = root / parent_contract["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        sha256_file(parent_path) != parent_contract["sha256"]
        or parent["decision"] != parent_contract["required_decision"]
    ):
        raise ValueError("P4GR parent drift")
    source_contract = contract["source"]
    manifest_path = root / source_contract["manifest"]
    if sha256_file(manifest_path) != source_contract["manifest_sha256"]:
        raise ValueError("P4GR source manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest) != source_contract["expected_rows"]:
        raise ValueError("P4GR source row count drift")
    if (
        len({row["make"] for row in manifest})
        != source_contract["expected_camera_makes"]
    ):
        raise ValueError("P4GR camera make count drift")
    for row in manifest:
        if (
            row["allowed_use"] != source_contract["required_allowed_use"]
            or row["rights_scope"] != source_contract["required_rights_scope"]
            or row["decoded_color_state"] != source_contract["required_color_state"]
        ):
            raise ValueError(f"P4GR source policy drift for {row['id']}")

    capacity = evaluate_capacity(root, root / CAPACITY)
    cloud_profile = CrossLayerCloudReferenceProfile.from_payload(
        capacity["compiled_profile"]
    )
    physical_base = json.loads((root / P4FB).read_text(encoding="utf-8"))
    gates = contract["automatic_gates"]
    metric_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    component_sources = [
        root / "src/film_physics/bounded_linear_residual.py",
        root / "native/film_physics/nf_cloud_post_spatial_f32_v1.c",
    ]
    if contract["candidate"].get("residual_projection") in {
        "density-lod-multiplicative",
        "compound-poisson-density-multiplicative",
        "nps-compiled-compound-poisson-density-multiplicative",
    }:
        component_sources.append(root / "src/film_physics/density_lod_residual.py")
    component_sha = hashlib.sha256(
        b"".join(path.read_bytes() for path in component_sources)
    ).hexdigest()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        temporary = Path(directory)
        standard = _runtime(temporary)
        dll = _build(
            root,
            temporary / "native",
            None,
            bridge_source="nf_sensitometry_cloud_bridge_f32_v2.c",
        )
        library = _configure_source_derived(dll)
        try:
            for source_row in manifest:
                source_encoded, source = _load_source(source_row, root)
                height, width, _ = source.shape
                physical_contract = json.loads(json.dumps(physical_base))
                physical_contract["fixture"]["full_height"] = height
                physical_contract["fixture"]["width"] = width
                source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
                physical_outputs: list[np.ndarray] = []
                all_diagnostics: list[dict[str, float]] = []
                density_lod = contract["candidate"].get("residual_projection") in {
                    "density-lod-multiplicative",
                    "compound-poisson-density-multiplicative",
                    "nps-compiled-compound-poisson-density-multiplicative",
                }
                compound_poisson = contract["candidate"].get("residual_projection") in {
                    "compound-poisson-density-multiplicative",
                    "nps-compiled-compound-poisson-density-multiplicative",
                }
                channel_density_gain = None
                if (
                    contract["candidate"].get("residual_projection")
                    == "nps-compiled-compound-poisson-density-multiplicative"
                ):
                    nps_parent = contract["parents"]["nps_compiler"]
                    nps_path = root / nps_parent["path"]
                    nps_evidence = json.loads(nps_path.read_text(encoding="utf-8"))
                    if (
                        sha256_file(nps_path) != nps_parent["sha256"]
                        or nps_evidence["decision"] != nps_parent["required_decision"]
                        or nps_evidence["compiled_channel_gain"]
                        != nps_parent["compiled_channel_gain"]
                    ):
                        raise ValueError("P4GV NPS compiler drift")
                    channel_density_gain = tuple(nps_parent["compiled_channel_gain"])
                finite_tail_density = None
                if compound_poisson:
                    rates = np.asarray(
                        cloud_profile.count_profile.marginal_rates_cmy,
                        dtype=np.float64,
                    )
                    marks = np.asarray(
                        cloud_profile.count_profile.mark_optical_density_cmy,
                        dtype=np.float64,
                    )
                    if channel_density_gain is not None:
                        marks = marks * np.asarray(channel_density_gain)
                    correlation = cloud_profile.count_profile.analytic_correlation()
                    covariance = correlation * np.sqrt(rates[:, None] * rates[None, :])
                    common_sigma = float(np.sqrt(marks @ covariance @ marks / 9.0))
                    aperture = np.asarray(
                        contract["candidate"]["pixel_aperture_kernel"],
                        dtype=np.float64,
                    )
                    aperture_sigma_scale = float(np.sum(aperture * aperture))
                    finite_tail_density = (
                        float(contract["candidate"]["tail_sigma"])
                        * common_sigma
                        * aperture_sigma_scale
                    )
                for tile_rows in contract["candidate"]["tile_rows"]:
                    diagnostics: list[dict[str, float]] = []

                    def provider(
                        forward_rows: np.ndarray,
                        y0: int,
                        count: int,
                        *,
                        _physical_contract: dict[str, Any] = physical_contract,
                        _source: np.ndarray = source,
                        _diagnostics: list[dict[str, float]] = diagnostics,
                        _density_lod: bool = density_lod,
                        _finite_tail_density: float | None = finite_tail_density,
                        _channel_density_gain: tuple[float, float, float]
                        | None = channel_density_gain,
                        _height: int = height,
                    ) -> np.ndarray:
                        render_y0 = max(0, y0 - 2) if _density_lod else y0
                        render_y1 = (
                            min(_height, y0 + count + 2) if _density_lod else y0 + count
                        )
                        render_count = render_y1 - render_y0
                        render_forward_rows = lambda start, requested: forward_rows(
                            start, requested
                        )
                        baseline: list[np.ndarray] = []
                        cloud_scan = render_physical_partition(
                            library,
                            _physical_contract,
                            render_forward_rows,
                            render_y0,
                            render_count,
                            source_derived_expected=True,
                            enforce_density_envelope=True,
                            exact_sensitometry_endpoints=True,
                            cloud_profile=cloud_profile,
                            physical_baseline_outputs=baseline,
                        )
                        if _density_lod:
                            full_result, row_diagnostics = apply_density_lod_residual(
                                _source[render_y0:render_y1],
                                cloud_scan,
                                baseline[0],
                                finite_tail_density=_finite_tail_density,
                                channel_density_gain=_channel_density_gain,
                            )
                            crop_start = y0 - render_y0
                            result = np.ascontiguousarray(
                                full_result[crop_start : crop_start + count]
                            )
                        elif (
                            contract["candidate"].get("residual_projection")
                            == "shared-density-multiplicative"
                        ):
                            result, row_diagnostics = (
                                apply_bounded_common_density_residual(
                                    _source[y0 : y0 + count], cloud_scan, baseline[0]
                                )
                            )
                        else:
                            result, row_diagnostics = apply_bounded_linear_residual(
                                _source[y0 : y0 + count], cloud_scan, baseline[0]
                            )
                        _diagnostics.append(row_diagnostics)
                        return result

                    cloud = WindowedNativeCloudScanRuntime(
                        gaussian_library=dll,
                        forward_scatter_profile=scatter_profile(),
                        physical_rows=provider,
                        physical_component_sha256=component_sha,
                        tile_rows=int(tile_rows),
                    )
                    pieces: list[np.ndarray] = []
                    cloud.render_to_sink(
                        source,
                        output_sink=lambda _a, _b, value, _pieces=pieces: (
                            _pieces.append(value.copy())
                        ),
                    )
                    physical_outputs.append(np.concatenate(pieces))
                    all_diagnostics.extend(diagnostics)

                physical = physical_outputs[0]
                ao6_linear = _apply_residual(standard, source)
                combined_linear = _apply_residual(standard, physical)
                physical_encoded = np.ascontiguousarray(
                    linear_srgb_to_encoded(physical.astype(np.float64)), np.float32
                )
                ao6_encoded = np.ascontiguousarray(
                    linear_srgb_to_encoded(ao6_linear.astype(np.float64)), np.float32
                )
                combined_encoded = np.ascontiguousarray(
                    linear_srgb_to_encoded(combined_linear.astype(np.float64)),
                    np.float32,
                )
                difference = combined_encoded - ao6_encoded
                absolute = np.abs(difference)
                row = {
                    "id": source_row["id"],
                    "make": source_row["make"],
                    "decoded_sha256": source_row["decoded_sha256"],
                    "linear_input_sha256": source_sha,
                    "shape": list(source.shape),
                    "physical_output_sha256": hashlib.sha256(
                        memoryview(physical).cast("B")
                    ).hexdigest(),
                    "combined_output_sha256": hashlib.sha256(
                        memoryview(combined_encoded).cast("B")
                    ).hexdigest(),
                    "finite": bool(
                        np.all(np.isfinite(physical))
                        and np.all(np.isfinite(combined_encoded))
                    ),
                    "partition_exact": bool(
                        np.array_equal(physical_outputs[0], physical_outputs[1])
                    ),
                    "physical_residual_rms": float(
                        np.sqrt(np.mean((physical.astype(np.float64) - source) ** 2))
                    ),
                    "maximum_limited_fraction": max(
                        item["limited_fraction"] for item in all_diagnostics
                    ),
                    "hard_clipping_used": any(
                        item["hard_clipping_used"] != 0.0 for item in all_diagnostics
                    ),
                    "combined_vs_ao6_p95_abs": float(np.quantile(absolute, 0.95)),
                    "combined_vs_ao6_p99_abs": float(np.quantile(absolute, 0.99)),
                    "flat_region_p99_abs": _flat_region_p99(source, difference),
                    "isolated_excursion_count": _isolated_excursions(
                        difference,
                        threshold=float(gates["isolated_excursion_threshold"]),
                        radius=int(gates["isolated_support_radius_pixels"]),
                        minimum_support=int(gates["minimum_isolated_support_count"]),
                    ),
                    "new_boundary_fraction_vs_ao6": _new_boundary_fraction(
                        ao6_encoded, combined_encoded
                    ),
                }
                if density_lod:
                    positive = np.all(source > 1.0e-6, axis=-1)
                    if np.any(positive):
                        source_ratios = source[positive, :2] / source[positive, 1:]
                        output_ratios = physical[positive, :2] / physical[positive, 1:]
                        row["maximum_rgb_ratio_error"] = float(
                            np.max(np.abs(output_ratios - source_ratios))
                        )
                    else:
                        row["maximum_rgb_ratio_error"] = 0.0
                metric_rows.append(row)
                visual_rows.append(
                    {
                        "id": row["id"],
                        "make": row["make"],
                        "source": source_encoded,
                        "ao6": ao6_encoded,
                        "physical": physical_encoded,
                        "combined": combined_encoded,
                    }
                )
        finally:
            _ctypes.FreeLibrary(library._handle)

    population_p95 = float(
        np.quantile([row["combined_vs_ao6_p95_abs"] for row in metric_rows], 0.95)
    )
    population_p99 = float(
        np.quantile([row["combined_vs_ao6_p99_abs"] for row in metric_rows], 0.99)
    )
    checks = {
        "physical_residual": min(row["physical_residual_rms"] for row in metric_rows)
        >= float(gates["minimum_per_row_physical_residual_rms"]),
        "limited_fraction": max(row["maximum_limited_fraction"] for row in metric_rows)
        <= float(gates["maximum_per_row_limited_fraction"]),
        "nontrivial_population": population_p95
        >= float(gates["minimum_population_p95_combined_vs_ao6_abs"]),
        "population_tail": population_p99
        <= float(gates["maximum_population_p99_combined_vs_ao6_abs"]),
        "flat_regions": max(row["flat_region_p99_abs"] for row in metric_rows)
        <= float(gates["maximum_flat_region_p99_combined_vs_ao6_abs"]),
        "isolated_excursions": sum(
            row["isolated_excursion_count"] for row in metric_rows
        )
        <= int(gates["maximum_isolated_excursion_count"]),
        "new_boundaries": max(
            row["new_boundary_fraction_vs_ao6"] for row in metric_rows
        )
        <= float(gates["maximum_new_boundary_fraction_vs_ao6"]),
        "partition_repeat_exact": all(row["partition_exact"] for row in metric_rows),
        "finite": all(row["finite"] for row in metric_rows),
        "no_hard_clipping": not any(row["hard_clipping_used"] for row in metric_rows),
    }
    if "maximum_rgb_ratio_error" in gates:
        checks["rgb_ratio_preservation"] = max(
            row["maximum_rgb_ratio_error"] for row in metric_rows
        ) <= float(gates["maximum_rgb_ratio_error"])
    sheet_sha = _contact_sheet(visual_rows, contact_sheet_path)
    automatic_pass = all(checks.values())
    stable = {
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "manifest_sha256": source_contract["manifest_sha256"],
        "compiled_profile_identity": cloud_profile.identity(),
        "rows": metric_rows,
        "population_p95_combined_vs_ao6_abs": population_p95,
        "population_p99_combined_vs_ao6_abs": population_p99,
        "contact_sheet_sha256": sheet_sha,
        "gates": checks,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass"]
        if automatic_pass
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("contract", "result"),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


__all__ = [
    "SCHEMA",
    "SHARED_DENSITY_SCHEMA",
    "evaluate",
    "load_contract",
    "sha256_file",
]
