"""U6.P2BB fresh photographic value ablation for positive B&W structure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import uniform_filter

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.nonuniform_positive_density_wedge import load_reference_profiles
from src.film_physics.bw_density_scanner_chain import (
    build_bound_bw_negative_direct_scan,
    build_typed_neutral_density_scanner_chain,
    validate_bound_bw_negative_direct_scan,
)
from src.film_physics.positive_density_field import (
    render_nonuniform_positive_density_region,
)
from src.film_physics.spatial_response import SpatialResponseProfile
from src.film_physics.thomas_dc_projection import build_thomas_dc_receipt
from src.preprocess.output_encode import save_srgb16_png


class BWStructurePhotographicValueError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2bb_bw_structure_photographic_value_contract.v1"
    ):
        raise BWStructurePhotographicValueError("unsupported P2BB contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any]) -> Any:
    path = root / binding["path"]
    if not path.is_file() or _sha(path) != binding["sha256"]:
        raise BWStructurePhotographicValueError("P2BB parent hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "required_automatic_pass" in binding and (
        payload.get("automatic_pass") is not binding["required_automatic_pass"]
    ):
        raise BWStructurePhotographicValueError("P2BB parent state mismatch")
    return payload


def _load_manifest(root: Path, binding: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _load_bound(root, binding)
    if not isinstance(rows, list) or len(rows) != int(binding["expected_rows"]):
        raise BWStructurePhotographicValueError("P2BB source row count mismatch")
    ids = [str(row["id"]) for row in rows]
    makes = [str(row["make"]) for row in rows]
    if len(ids) != len(set(ids)) or len(set(makes)) != int(binding["expected_makes"]):
        raise BWStructurePhotographicValueError("P2BB source identity drift")
    for row in rows:
        if (
            row.get("allowed_use") != binding["required_allowed_use"]
            or row.get("rights_scope") != binding["required_rights_scope"]
            or row.get("decoded_color_state") != binding["required_color_state"]
        ):
            raise BWStructurePhotographicValueError("P2BB source policy drift")
    return rows


def _mean_density(encoded: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    linear = encoded_srgb_to_linear(encoded)
    weights = np.asarray(mapping["linear_luma_weights"], dtype=np.float64)
    if weights.shape != (3,) or not np.isclose(float(np.sum(weights)), 1.0):
        raise BWStructurePhotographicValueError("P2BB luma weights drift")
    luma = linear @ weights
    low = float(mapping["minimum_density"])
    high = float(mapping["maximum_density"])
    if not 0.0 < low < high or mapping["per_image_normalization_allowed"]:
        raise BWStructurePhotographicValueError("P2BB density mapping drift")
    return np.ascontiguousarray(low + (high - low) * luma, dtype=np.float64)


def _scanner_profile(contract: dict[str, Any]) -> SpatialResponseProfile:
    scanner = contract["scanner"]
    pitch = float(scanner["pixel_pitch_micrometres"])
    sigma = tuple(pitch * float(value) for value in scanner["scanner_mtf_sigma_pixels_rgb"])
    if (
        scanner["spectral_matrix"] != "identity"
        or float(scanner["flare_fraction"]) != 0.0
        or float(scanner["noise_variance"]) != 0.0
        or scanner["scanner_calibration_claimed"]
    ):
        raise BWStructurePhotographicValueError("P2BB scanner policy drift")
    return SpatialResponseProfile(
        pixel_pitch_um=pitch,
        forward_scatter_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_gain_rgb=(0.0, 0.0, 0.0),
        dye_diffusion_sigma_um_rgb=(0.0, 0.0, 0.0),
        scanner_mtf_sigma_um_rgb=sigma,
        gaussian_truncate=float(scanner["gaussian_truncate"]),
    )


def _new_boundary_fraction(reference: np.ndarray, candidate: np.ndarray) -> float:
    epsilon = 1.0 / 65535.0
    reference_boundary = np.any(
        (reference <= epsilon) | (reference >= 1.0 - epsilon), axis=-1
    )
    candidate_boundary = np.any(
        (candidate <= epsilon) | (candidate >= 1.0 - epsilon), axis=-1
    )
    return float(np.mean(candidate_boundary & ~reference_boundary))


def _flat_region_p99(reference: np.ndarray, difference: np.ndarray) -> float:
    luma = reference[..., 0]
    gradient = np.zeros_like(luma)
    gradient[:, 1:] += np.abs(np.diff(luma, axis=1))
    gradient[1:, :] += np.abs(np.diff(luma, axis=0))
    flat = gradient <= float(np.quantile(gradient, 0.25))
    values = np.max(np.abs(difference), axis=-1)[flat]
    return float(np.quantile(values, 0.99)) if values.size else 0.0


def _isolated_excursions(
    difference: np.ndarray, *, threshold: float, radius: int, minimum_support: int
) -> int:
    excursion = np.max(np.abs(difference), axis=-1) > threshold
    width = 2 * radius + 1
    support = uniform_filter(
        excursion.astype(np.float64), size=width, mode="constant", cval=0.0
    ) * float(width * width)
    return int(np.count_nonzero(excursion & (support < minimum_support)))


def _save_png_exact(encoded: np.ndarray, path: Path) -> tuple[str, str]:
    if np.any(encoded < 0.0) or np.any(encoded > 1.0) or not np.all(np.isfinite(encoded)):
        raise BWStructurePhotographicValueError("P2BB encoder input escaped bounds")
    expected = np.rint(encoded * 65535.0).astype(np.uint16)
    save_srgb16_png(encoded, path)
    reopened = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        reopened is None
        or reopened.dtype != np.uint16
        or reopened.shape != expected.shape
        or not np.array_equal(reopened[..., ::-1], expected)
    ):
        raise BWStructurePhotographicValueError("P2BB RGB16 PNG readback mismatch")
    return _sha(path), _array_sha(expected, "<u2")


def _preview(encoded: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.rint(encoded * 255.0).astype(np.uint8), mode="RGB")
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _contact(rows: list[dict[str, Any]], output: Path) -> str:
    tile_w, tile_h, header = 320, 220, 24
    canvas = Image.new("RGB", (3 * tile_w, len(rows) * (tile_h + header)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        y = index * (tile_h + header)
        draw.text((5, y + 5), str(row["id"]), fill=(235, 235, 235))
        for column, values in enumerate((row["source"], row["baseline"], row["structure"])):
            image = _preview(values, (tile_w, tile_h))
            x = column * tile_w + (tile_w - image.width) // 2
            canvas.paste(image, (x, y + header))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "PNG", compress_level=9, optimize=False)
    return _sha(output)


def evaluate(*, root: Path, contract: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        _load_bound(root, binding)
    rows = _load_manifest(root, contract["parents"]["source_manifest"])
    wedge_contract = _load_bound(root, contract["parents"]["nonuniform_wedge_contract"])
    amplitude_profile, parameter_profile = load_reference_profiles(
        root=root, contract=wedge_contract
    )
    physical_gain_contract = _load_bound(root, contract["parents"]["physical_gain_contract"])
    spatial = physical_gain_contract["spatial_mechanism"]
    structure = contract["structure"]
    if (
        structure["spatial_profile_id"] != spatial["spatial_profile_id"]
        or structure["parameter_refit_allowed"]
        or structure["realized_normalization_allowed"]
        or structure["hard_clipping_allowed"]
    ):
        raise BWStructurePhotographicValueError("P2BB structure policy drift")
    seeds = structure["realization_seeds"]
    if len(seeds) != len(rows) or len(set(seeds)) != len(seeds):
        raise BWStructurePhotographicValueError("P2BB realization seed drift")
    scanner_profile = _scanner_profile(contract)
    metric_vectors: list[np.ndarray] = []
    visual_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for index, (row, seed) in enumerate(zip(rows, seeds, strict=True)):
        source_path = root / row["decoded_path"]
        if _sha(source_path) != row["decoded_sha256"]:
            raise BWStructurePhotographicValueError("P2BB decoded source hash drift")
        with Image.open(source_path) as opened:
            source = np.asarray(opened.convert("RGB"), dtype=np.float64) / 255.0
        mean_density = _mean_density(source, contract["mean_density_mapping"])
        receipt = build_thomas_dc_receipt(
            mean_density.shape,
            profile_id=spatial["spatial_profile_id"],
            particle_sigma_pixels=float(spatial["particle_sigma_samples"]),
            cluster_sigma_pixels=float(spatial["cluster_sigma_samples"]),
            mean_offspring=float(spatial["mean_offspring"]),
            component_seeds=tuple(spatial["component_seeds"]),
            realization_seed=int(seed),
            truncate=float(spatial["truncate_sigma"]),
            canonical_row_block_height=int(structure["canonical_row_block_height"]),
        )
        structured_density = render_nonuniform_positive_density_region(
            receipt,
            mean_density=mean_density,
            origin_yx=(0, 0),
            shape=mean_density.shape,
            amplitude_profile=amplitude_profile,
            parameter_profile=parameter_profile,
        )
        partitioned = []
        block = int(structure["verification_row_partition_height"])
        for y0 in range(0, mean_density.shape[0], block):
            height = min(block, mean_density.shape[0] - y0)
            partitioned.append(
                render_nonuniform_positive_density_region(
                    receipt,
                    mean_density=mean_density,
                    origin_yx=(y0, 0),
                    shape=(height, mean_density.shape[1]),
                    amplitude_profile=amplitude_profile,
                    parameter_profile=parameter_profile,
                )
            )
        partition_exact = np.array_equal(structured_density, np.concatenate(partitioned))
        baseline_chain = build_typed_neutral_density_scanner_chain(
            mean_density, scanner_profile
        )
        structure_chain = build_typed_neutral_density_scanner_chain(
            structured_density, scanner_profile
        )
        baseline_result = build_bound_bw_negative_direct_scan(baseline_chain.scan_linear)
        structure_result = build_bound_bw_negative_direct_scan(structure_chain.scan_linear)
        validate_bound_bw_negative_direct_scan(baseline_chain.scan_linear, baseline_result)
        validate_bound_bw_negative_direct_scan(structure_chain.scan_linear, structure_result)
        baseline = linear_srgb_to_encoded(baseline_result.display_linear.values)
        candidate = linear_srgb_to_encoded(structure_result.display_linear.values)
        difference = candidate - baseline
        absolute = np.max(np.abs(difference), axis=-1)
        metric_vectors.append(np.asarray(absolute.reshape(-1), dtype=np.float32))
        baseline_path = output_dir / "renders" / contract["arms"][0] / f"{row['id']}.png"
        structure_path = output_dir / "renders" / contract["arms"][1] / f"{row['id']}.png"
        baseline_png, baseline_samples = _save_png_exact(baseline, baseline_path)
        structure_png, structure_samples = _save_png_exact(candidate, structure_path)
        isolated = _isolated_excursions(
            difference,
            threshold=float(contract["automatic_gates"]["isolated_excursion_threshold"]),
            radius=int(contract["automatic_gates"]["isolated_support_radius_pixels"]),
            minimum_support=int(contract["automatic_gates"]["minimum_isolated_support_count"]),
        )
        channel_spread = float(np.max(np.ptp(candidate, axis=-1)))
        output_boundary = max(
            float(np.mean(np.any((baseline <= 1 / 65535) | (baseline >= 1 - 1 / 65535), axis=-1))),
            float(np.mean(np.any((candidate <= 1 / 65535) | (candidate >= 1 - 1 / 65535), axis=-1))),
        )
        result_rows.append(
            {
                "id": row["id"],
                "make": row["make"],
                "decoded_sha256": row["decoded_sha256"],
                "shape": list(source.shape),
                "realization_seed": int(seed),
                "receipt_id": receipt.receipt_id,
                "mean_density_sha256": _array_sha(mean_density),
                "structured_density_sha256": _array_sha(structured_density),
                "minimum_structured_density": float(np.min(structured_density)),
                "partition_exact": bool(partition_exact),
                "baseline_scan_receipt": baseline_result.receipt.__dict__,
                "structure_scan_receipt": structure_result.receipt.__dict__,
                "encoded_rms_difference": float(np.sqrt(np.mean(np.square(difference)))),
                "encoded_p99_absolute_difference": float(np.quantile(absolute, 0.99)),
                "flat_region_p99_absolute_difference": _flat_region_p99(baseline, difference),
                "new_boundary_fraction": _new_boundary_fraction(baseline, candidate),
                "output_code_boundary_fraction": output_boundary,
                "isolated_excursion_count": isolated,
                "maximum_channel_spread": channel_spread,
                "outputs": {
                    contract["arms"][0]: {
                        "relative_path": baseline_path.relative_to(output_dir).as_posix(),
                        "png_sha256": baseline_png,
                        "uint16_samples_sha256": baseline_samples,
                        "encoded_array_sha256": _array_sha(baseline),
                    },
                    contract["arms"][1]: {
                        "relative_path": structure_path.relative_to(output_dir).as_posix(),
                        "png_sha256": structure_png,
                        "uint16_samples_sha256": structure_samples,
                        "encoded_array_sha256": _array_sha(candidate),
                    },
                },
            }
        )
        visual_rows.append({"id": row["id"], "source": source, "baseline": baseline, "structure": candidate})
    population = np.concatenate(metric_vectors)
    metrics = {
        "minimum_per_row_encoded_rms_difference": min(row["encoded_rms_difference"] for row in result_rows),
        "population_p99_encoded_absolute_difference": float(np.quantile(population, 0.99)),
        "maximum_flat_region_p99_encoded_absolute_difference": max(row["flat_region_p99_absolute_difference"] for row in result_rows),
        "maximum_new_boundary_fraction_vs_baseline": max(row["new_boundary_fraction"] for row in result_rows),
        "maximum_output_code_boundary_fraction": max(row["output_code_boundary_fraction"] for row in result_rows),
        "total_isolated_excursion_count": sum(row["isolated_excursion_count"] for row in result_rows),
        "maximum_channel_spread": max(row["maximum_channel_spread"] for row in result_rows),
        "minimum_developed_density": min(row["minimum_structured_density"] for row in result_rows),
        "all_partition_exact": all(row["partition_exact"] for row in result_rows),
        "all_inputs_and_outputs_finite": True,
        "rgb16_png_exact_readback": True,
    }
    limits = contract["automatic_gates"]
    gates = {
        "visible_separation": metrics["minimum_per_row_encoded_rms_difference"] >= limits["minimum_per_row_encoded_rms_difference"],
        "p99_tail": metrics["population_p99_encoded_absolute_difference"] <= limits["maximum_population_p99_encoded_absolute_difference"],
        "flat_tail": metrics["maximum_flat_region_p99_encoded_absolute_difference"] <= limits["maximum_flat_region_p99_encoded_absolute_difference"],
        "new_boundary": metrics["maximum_new_boundary_fraction_vs_baseline"] <= limits["maximum_new_boundary_fraction_vs_baseline"],
        "output_boundary": metrics["maximum_output_code_boundary_fraction"] <= limits["maximum_output_code_boundary_fraction"],
        "isolated_excursions": metrics["total_isolated_excursion_count"] <= limits["maximum_isolated_excursion_count"],
        "neutral_channels": metrics["maximum_channel_spread"] <= limits["maximum_channel_spread"],
        "positive_density": metrics["minimum_developed_density"] > limits["minimum_developed_density_exclusive"],
        "partition_exact": metrics["all_partition_exact"] is limits["require_partition_exact"],
        "finite": metrics["all_inputs_and_outputs_finite"] is limits["require_all_inputs_and_outputs_finite"],
        "rgb16_readback": metrics["rgb16_png_exact_readback"] is limits["require_rgb16_png_exact_readback"],
    }
    automatic_pass = all(gates.values())
    contact_path = output_dir / "contact_sheet.png"
    contact_sha = _contact(visual_rows, contact_path)
    stable = {
        "schema": "neuro_film.u6_p2bb_bw_structure_photographic_value_report.v1",
        "source_manifest_sha256": contract["parents"]["source_manifest"]["sha256"],
        "arms": contract["arms"],
        "rows": result_rows,
        "metrics": metrics,
        "gate_results": gates,
        "contact_sheet_sha256": contact_sha,
        "automatic_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "decision": contract["decision_if_pass" if automatic_pass else "decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = [
    "BWStructurePhotographicValueError",
    "evaluate",
    "load_contract",
    "write_report",
]
