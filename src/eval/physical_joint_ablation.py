"""U6.P7A1 colour and developed-spatial physical-path ablation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import uniform_filter
from skimage.color import rgb2lab

from src.eval.b0_real_film_residual_fresh_confirmation import (
    validate_contract as validate_ao7_contract,
)
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.dual_champion_composition import build_operators, compose_rgb
from src.eval.global_frontier import new_hard_clipping_fraction, sha256_file
from src.eval.physical_spatial_response import _profile
from src.eval.sensitometry_print_composition import build_composition
from src.real_film.gold_matrix_transplant import style_and_basic_residual
from src.roll2film.factorized_boundary_guard import (
    apply_factorized_boundary_guard,
)
from src.film_physics import (
    SpatialResponseProfile,
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_interpretation_bounded_development_adjacency,
    apply_scanner_mtf,
)


SCHEMA = "neuro_film.u6_p7a1_interpretation_bounded_ablation_contract.v1"


@dataclass(frozen=True)
class JointAblationRuntime:
    profile: SpatialResponseProfile
    print_operator: Any
    apply_colour: Callable[[np.ndarray], np.ndarray]
    apply_adjacency: Callable[[np.ndarray, SpatialResponseProfile], np.ndarray]
    eligible_ids: tuple[str, ...]
    source_rows: dict[str, dict[str, Any]]
    sample_budget: int


def _load_exact_json(
    root: Path, path: str, expected_sha256: str
) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_contracts(
    root: Path, correction: dict[str, Any]
) -> tuple[dict[str, Any], JointAblationRuntime]:
    if (
        correction.get("schema") != SCHEMA
        or correction["correction"].get("hard_clip_allowed")
        or correction["correction"].get("endpoint_or_gate_retuning_allowed")
    ):
        raise ValueError("unsupported U6.P7A1 correction contract")
    contract = _load_exact_json(
        root,
        correction["parent_contract"],
        correction["parent_contract_sha256"],
    )
    if contract.get("schema") != (
        "neuro_film.u6_p7a_colour_developed_spatial_ablation_contract.v1"
    ):
        raise ValueError("unsupported U6.P7A parent contract")
    parents = contract["parents"]
    ao7 = _load_exact_json(
        root, parents["ao7_config"], parents["ao7_config_sha256"]
    )
    p5a = _load_exact_json(
        root, parents["p5a_config"], parents["p5a_config_sha256"]
    )
    p5c = _load_exact_json(
        root, parents["p5c_config"], parents["p5c_config_sha256"]
    )
    sensitometry = _load_exact_json(
        root,
        parents["sensitometry_config"],
        parents["sensitometry_config_sha256"],
    )
    composition = _load_exact_json(
        root,
        parents["print_composition_config"],
        parents["print_composition_config_sha256"],
    )
    print_source = _load_exact_json(
        root,
        parents["print_source_config"],
        parents["print_source_config_sha256"],
    )
    validated = validate_ao7_contract(root, ao7)
    population = contract["population"]
    if (
        len(validated["eligible_ids"]) != int(population["expected_rows"])
        or len(
            {
                validated["source_rows"][sample_id]["make"]
                for sample_id in validated["eligible_ids"]
            }
        )
        != int(population["expected_camera_makes"])
    ):
        raise ValueError("U6.P7A population drift")
    for sample_id in validated["eligible_ids"]:
        row = validated["source_rows"][sample_id]
        if (
            row["allowed_use"] != population["allowed_use"]
            or row["decoded_color_state"] != population["decoded_color_state"]
        ):
            raise ValueError("U6.P7A source boundary drift")

    apply_anchor, apply_density = build_operators(
        validated["base_config"], validated["base_validated"]
    )
    base_spec = ao7["fixed_base"]
    candidate = ao7["fixed_candidate"]

    def apply_colour(encoded: np.ndarray) -> np.ndarray:
        base = compose_rgb(
            np.asarray(encoded, dtype=np.float32),
            order=base_spec["order"],
            density_strength=float(base_spec["density_strength"]),
            apply_anchor=apply_anchor,
            apply_density=apply_density,
            output_margin=int(base_spec["final_output_margin"]),
        )
        guard = candidate["factorization"]
        result = apply_factorized_boundary_guard(
            validated["operator"],
            encoded_srgb_to_linear(base),
            tone_strength=float(candidate["tone_strength"]),
            chroma_strength=float(candidate["chroma_strength"]),
            luma_weights=np.asarray(guard["luma_weights"], dtype=np.float64),
            hard_boundary_epsilon_encoded_srgb=float(
                guard["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                guard["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        return linear_srgb_to_encoded(result.output)

    profile = _profile(p5a)
    print_operator = build_composition(composition, sensitometry, print_source)
    adjacency = p5c["candidate"]
    apply_adjacency = partial(
        apply_interpretation_bounded_development_adjacency,
        maximum_absolute_transmittance_delta=float(
            adjacency["maximum_absolute_transmittance_delta"]
        ),
        maximum_absolute_density_delta=float(
            adjacency["maximum_absolute_density_delta"]
        ),
        black_reference_density=(
            print_operator.interpretation.black_reference_density
        ),
        white_reference_density=(
            print_operator.interpretation.white_reference_density
        ),
    )
    runtime = JointAblationRuntime(
        profile=profile,
        print_operator=print_operator,
        apply_colour=apply_colour,
        apply_adjacency=apply_adjacency,
        eligible_ids=tuple(validated["eligible_ids"]),
        source_rows=validated["source_rows"],
        sample_budget=int(ao7["metrics"]["maximum_pixels_per_image"]),
    )
    return contract, runtime


def apply_physical_display(
    linear: np.ndarray,
    runtime: JointAblationRuntime,
    *,
    spatial: bool,
) -> np.ndarray:
    values = np.asarray(linear, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("physical-display input must be finite linear RGB")
    if not spatial:
        return runtime.print_operator.apply(values)
    exposure = apply_forward_scatter(values, runtime.profile)
    density = runtime.print_operator.sensitometry.apply(exposure)
    density = runtime.apply_adjacency(density, runtime.profile)
    density = apply_dye_diffusion(density, runtime.profile)
    interpreted = runtime.print_operator.interpretation.apply(density)
    interpreted = _canonicalize_endpoint_roundoff(interpreted)
    return _canonicalize_endpoint_roundoff(
        apply_scanner_mtf(interpreted, runtime.profile)
    )


def _canonicalize_endpoint_roundoff(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    tolerance = 1e-12
    if (
        not np.all(np.isfinite(array))
        or np.any(array < -tolerance)
        or np.any(array > 1.0 + tolerance)
    ):
        raise RuntimeError("physical display exceeded endpoint tolerance")
    return np.where(array < 0.0, 0.0, np.where(array > 1.0, 1.0, array))


def render_arms(
    encoded: np.ndarray, runtime: JointAblationRuntime
) -> dict[str, np.ndarray]:
    source = np.asarray(encoded, dtype=np.float64)
    linear = encoded_srgb_to_linear(source)
    colour = runtime.apply_colour(source)
    physics_linear = apply_physical_display(linear, runtime, spatial=True)
    physics = linear_srgb_to_encoded(physics_linear)
    cheap_linear = apply_physical_display(linear, runtime, spatial=False)
    cheap = runtime.apply_colour(linear_srgb_to_encoded(cheap_linear))
    combined = runtime.apply_colour(physics)
    wrong_linear = apply_physical_display(
        encoded_srgb_to_linear(colour), runtime, spatial=True
    )
    wrong = linear_srgb_to_encoded(wrong_linear)
    outputs = {
        "colour_only": colour,
        "physics_only": physics,
        "combined": combined,
        "cheap": cheap,
        "wrong_order": wrong,
    }
    for name, output in outputs.items():
        if (
            output.shape != source.shape
            or not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError(f"{name} left encoded RGB domain")
    return outputs


def _sample(values: np.ndarray, maximum: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1, 3)
    if len(flat) <= maximum:
        return flat
    indices = np.linspace(0, len(flat) - 1, maximum, dtype=np.int64)
    return flat[indices]


def _median_delta_e76(first: np.ndarray, second: np.ndarray) -> float:
    left = rgb2lab(_sample(first, 65_536).reshape(-1, 1, 3))
    right = rgb2lab(_sample(second, 65_536).reshape(-1, 1, 3))
    return float(np.median(np.linalg.norm(left - right, axis=-1)))


def _isolated_excursions(
    difference: np.ndarray,
    *,
    threshold: float,
    radius: int,
    minimum_support: int,
) -> int:
    excursion = np.max(np.abs(difference), axis=-1) > threshold
    width = 2 * radius + 1
    support = uniform_filter(
        excursion.astype(np.float64),
        size=width,
        mode="constant",
        cval=0.0,
    ) * float(width * width)
    return int(np.count_nonzero(excursion & (support < minimum_support)))


def _save_rgb(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        path, format="PNG", compress_level=6
    )
    return sha256_file(path)


def _tile(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def _save_sheets(
    visual_rows: dict[str, dict[str, np.ndarray]],
    contract: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    protocol = contract["visual_protocol"]
    fixed_ids = protocol["fixed_ids"]
    columns = protocol["diagnostic_columns"]
    tile_size = (240, 160)
    header = 24
    diagnostic = Image.new(
        "RGB",
        (len(columns) * tile_size[0], len(fixed_ids) * (tile_size[1] + header)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(diagnostic)
    for row_index, sample_id in enumerate(fixed_ids):
        y = row_index * (tile_size[1] + header)
        for column_index, name in enumerate(columns):
            values = visual_rows[sample_id][name]
            diagnostic.paste(
                _tile(values, tile_size),
                (column_index * tile_size[0], y + header),
            )
            draw.text(
                (column_index * tile_size[0] + 4, y + 4),
                sample_id if column_index == 0 else name,
                fill=(235, 235, 235),
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output_dir / "diagnostic.png"
    diagnostic.save(diagnostic_path, format="PNG", compress_level=6)

    blind_hashes: list[str] = []
    mappings: dict[str, dict[str, str]] = {}
    seed = int(hashlib.sha256(contract["node"].encode()).hexdigest()[:8], 16)
    for round_index in range(1, int(protocol["blind_rounds"]) + 1):
        order = list(protocol["primary_comparison"])
        random.Random(seed + round_index).shuffle(order)
        mapping = {"A": order[0], "B": order[1]}
        mappings[f"round_{round_index}"] = mapping
        canvas = Image.new(
            "RGB",
            (3 * tile_size[0], len(fixed_ids) * (tile_size[1] + header)),
            (24, 24, 24),
        )
        blind_draw = ImageDraw.Draw(canvas)
        for row_index, sample_id in enumerate(fixed_ids):
            y = row_index * (tile_size[1] + header)
            row = visual_rows[sample_id]
            for column_index, (label, name) in enumerate(
                (("SOURCE", "source"), ("A", order[0]), ("B", order[1]))
            ):
                canvas.paste(
                    _tile(row[name], tile_size),
                    (column_index * tile_size[0], y + header),
                )
                blind_draw.text(
                    (column_index * tile_size[0] + 4, y + 4),
                    f"{label} {sample_id}" if column_index == 0 else label,
                    fill=(235, 235, 235),
                )
        path = output_dir / f"blind_round_{round_index}.png"
        canvas.save(path, format="PNG", compress_level=6)
        blind_hashes.append(sha256_file(path))
    mapping_raw = (
        json.dumps(mappings, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (output_dir / "private_mapping.json").write_bytes(mapping_raw)
    return {
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "blind_sha256": blind_hashes,
        "mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
    }


def evaluate_ablation(
    *,
    root: Path,
    correction: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    contract, runtime = load_contracts(root, correction)
    gates = contract["automatic_gates"]
    rows: list[dict[str, Any]] = []
    visual_rows: dict[str, dict[str, np.ndarray]] = {}
    for sample_id in runtime.eligible_ids:
        source_row = runtime.source_rows[sample_id]
        source_path = root / source_row["decoded_path"]
        if sha256_file(source_path) != source_row["decoded_sha256"]:
            raise ValueError(f"source hash drift: {sample_id}")
        with Image.open(source_path) as image:
            source = (
                np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    dtype=np.float64,
                )
                / 255.0
            )
        arms = render_arms(source, runtime)
        output_hashes = {
            name: _save_rgb(
                output_dir / "renders" / name / f"{sample_id}.png", values
            )
            for name, values in arms.items()
        }
        source_sample = _sample(source, runtime.sample_budget)
        arm_samples = {
            name: _sample(values, runtime.sample_budget)
            for name, values in arms.items()
        }
        metrics: dict[str, Any] = {}
        for name, values in arm_samples.items():
            style, non_basic = style_and_basic_residual(
                source_sample, values
            )
            metrics[name] = {
                "style_delta_e76": style,
                "non_basic_delta_e76": non_basic,
                "new_hard_boundary_fraction": new_hard_clipping_fraction(
                    source_sample,
                    values,
                    0.5 / 255.0,
                ),
            }
        metrics["combined_to_colour_delta_e76"] = _median_delta_e76(
            arms["colour_only"], arms["combined"]
        )
        metrics["full_to_cheap_delta_e76"] = _median_delta_e76(
            arms["combined"], arms["cheap"]
        )
        metrics["combined_to_wrong_order_delta_e76"] = _median_delta_e76(
            arms["combined"], arms["wrong_order"]
        )
        metrics["combined_isolated_excursions_from_colour"] = (
            _isolated_excursions(
                arms["combined"] - arms["colour_only"],
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            )
        )
        rows.append(
            {
                "sample_id": sample_id,
                "make": source_row["make"],
                "source_sha256": source_row["decoded_sha256"],
                "output_sha256": output_hashes,
                "metrics": metrics,
            }
        )
        if sample_id in contract["visual_protocol"]["fixed_ids"]:
            visual_rows[sample_id] = {"source": source, **arms}
    if set(visual_rows) != set(contract["visual_protocol"]["fixed_ids"]):
        raise ValueError("visual population drift")
    sheets = _save_sheets(visual_rows, contract, output_dir)

    combined_styles = [
        row["metrics"]["combined"]["style_delta_e76"] for row in rows
    ]
    combined_non_basic = [
        row["metrics"]["combined"]["non_basic_delta_e76"] for row in rows
    ]
    combined_delta = [
        row["metrics"]["combined_to_colour_delta_e76"] for row in rows
    ]
    full_cheap = [row["metrics"]["full_to_cheap_delta_e76"] for row in rows]
    wrong_order = [
        row["metrics"]["combined_to_wrong_order_delta_e76"] for row in rows
    ]
    decisions = {
        "population": len(rows) == int(contract["population"]["expected_rows"]),
        "finite_bounded": True,
        "new_boundaries": max(
            row["metrics"][name]["new_hard_boundary_fraction"]
            for row in rows
            for name in contract["arms"]
        )
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "style_envelope": max(combined_styles)
        <= float(gates["maximum_combined_per_image_style_delta_e76"]),
        "non_basic": float(np.median(combined_non_basic))
        >= float(gates["minimum_combined_median_non_basic_delta_e76"]),
        "physics_material": float(np.median(combined_delta))
        >= float(gates["minimum_combined_to_colour_median_delta_e76"]),
        "physics_population": sum(value >= 0.5 for value in combined_delta)
        >= int(
            gates[
                "minimum_images_with_combined_to_colour_delta_e76_at_least_0_5"
            ]
        ),
        "full_not_cheap": float(np.median(full_cheap))
        >= float(gates["minimum_full_to_cheap_median_delta_e76"]),
        "order_identifiable": float(np.median(wrong_order))
        >= float(gates["minimum_combined_to_wrong_order_median_delta_e76"]),
        "isolated_excursions": sum(
            row["metrics"]["combined_isolated_excursions_from_colour"]
            for row in rows
        )
        <= int(gates["maximum_isolated_excursion_count"]),
    }
    core = {
        "schema": "neuro_film.u6_p7a1_colour_physics_ablation_report.v1",
        "node": correction["node"],
        "claim_ceiling": correction["claim_ceiling"],
        "rows": rows,
        "summary": {
            "combined_median_style_delta_e76": float(
                np.median(combined_styles)
            ),
            "combined_maximum_style_delta_e76": float(max(combined_styles)),
            "combined_median_non_basic_delta_e76": float(
                np.median(combined_non_basic)
            ),
            "combined_to_colour_median_delta_e76": float(
                np.median(combined_delta)
            ),
            "images_combined_to_colour_at_least_0_5": int(
                sum(value >= 0.5 for value in combined_delta)
            ),
            "full_to_cheap_median_delta_e76": float(np.median(full_cheap)),
            "combined_to_wrong_order_median_delta_e76": float(
                np.median(wrong_order)
            ),
            "maximum_new_hard_boundary_fraction": float(
                max(
                    row["metrics"][name]["new_hard_boundary_fraction"]
                    for row in rows
                    for name in contract["arms"]
                )
            ),
            "combined_isolated_excursion_count": int(
                sum(
                    row["metrics"][
                        "combined_isolated_excursions_from_colour"
                    ]
                    for row in rows
                )
            ),
        },
        "visual_evidence": sheets,
        "decisions": decisions,
        "automatic_pass": all(decisions.values()),
        "branch": contract["branch_rule"][
            "complete_pass" if all(decisions.values()) else "automatic_fail"
        ],
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


__all__ = [
    "JointAblationRuntime",
    "apply_physical_display",
    "evaluate_ablation",
    "load_contracts",
    "render_arms",
    "write_report",
]
