"""U5.R2CB3 photographic development for the exact CB2 dye operators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.eval.filmmatch_validation_scene import _save_rgb16_png
from src.eval.fujifilm_e6_bounded_dye_operator import (
    _linear_lab,
    _load_source_curves,
    compile_operator,
    hash_file,
)
from src.eval.fujifilm_e6_bounded_dye_operator import (
    load_contract as load_compiler_contract,
)
from src.eval.fujifilm_e6_spectral_dye_signature import STOCKS
from src.eval.spectral_film_lut_bank import _linear_to_encoded, synthetic_cube
from src.eval.velvia_datasheet_witness import encoded_srgb_to_linear
from src.film_physics.spectral_scanner import synthetic_profile_from_contract
from src.roll2film.baselines import fit_joint_basic_adjustment
from src.roll2film.lut import DenseLUT3D

SCHEMA = "neuro_film.u5_r2cb3_fujifilm_e6_dye_operator_photographic_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb3_fujifilm_e6_dye_operator_photographic_report.v1"
EXPERIMENT_ID = "U5.R2CB3"
CONTRACT_SHA256 = "72a812082f62501decddaef7bf4564bc36be6b58aee905d6f460c52d4759b171"


class FujifilmDyePhotographicError(RuntimeError):
    """Raised when a CB3 contract, input or execution boundary drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmDyePhotographicError("CB3 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmDyePhotographicError("CB3 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmDyePhotographicError("CB3 frozen contract structure drift")
    for section, keys in (
        ("parent_operator", ("decision_path", "compiler_contract_path")),
        ("source_population", ("decision_path", "manifest_path", "visual_review_path")),
    ):
        for key in keys:
            _relative_path(payload[section][key])
    return payload


def _load_hashed_json(root: Path, relative: str, expected: str) -> Any:
    path = root / _relative_path(relative)
    if not path.is_file() or hash_file(path) != expected:
        raise FujifilmDyePhotographicError(f"CB3 input identity mismatch: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_inputs(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    parent = config["parent_operator"]
    decision = _load_hashed_json(
        root, parent["decision_path"], parent["decision_sha256"]
    )
    if (
        decision.get("decision") != parent["required_decision"]
        or decision.get("stable_evidence_id") != parent["required_stable_evidence_id"]
    ):
        raise FujifilmDyePhotographicError("CB3 parent operator is not retained")
    compiler_path = root / _relative_path(parent["compiler_contract_path"])
    if hash_file(compiler_path) != parent["compiler_contract_sha256"]:
        raise FujifilmDyePhotographicError("CB3 compiler contract identity mismatch")
    compiler = load_compiler_contract(compiler_path)

    source = config["source_population"]
    source_decision = _load_hashed_json(
        root, source["decision_path"], source["decision_sha256"]
    )
    manifest = _load_hashed_json(root, source["manifest_path"], source["manifest_sha256"])
    visual = _load_hashed_json(
        root, source["visual_review_path"], source["visual_review_sha256"]
    )
    if (
        source_decision.get("result", {}).get("decision") != source["required_decision"]
        or visual.get("decision") != source["required_visual_decision"]
    ):
        raise FujifilmDyePhotographicError("CB3 source population is not eligible")
    eligible = tuple(visual.get("eligible_ids", ()))
    excluded = set(source["exclude_ids"])
    rows = [dict(row) for row in manifest if row.get("id") in eligible and row.get("id") not in excluded]
    if (
        len(rows) != int(source["required_rows"])
        or len({row["make"] for row in rows}) != int(source["required_camera_makes"])
        or tuple(row["id"] for row in rows) != tuple(item for item in eligible if item not in excluded)
    ):
        raise FujifilmDyePhotographicError("CB3 eligible source population drift")
    wavelength, curves = _load_source_curves(compiler, root)
    return compiler, rows, wavelength, curves


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= 0 or maximum <= 0:
        raise ValueError("CB3 sample sizes must be positive")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _median_delta_e76(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.median(np.linalg.norm(_linear_lab(left) - _linear_lab(right), axis=-1))
    )


def _gradient_p999_ratio(source: np.ndarray, output: np.ndarray) -> float:
    def magnitudes(image: np.ndarray) -> np.ndarray:
        horizontal = np.linalg.norm(np.diff(image, axis=1), axis=-1).reshape(-1)
        vertical = np.linalg.norm(np.diff(image, axis=0), axis=-1).reshape(-1)
        return np.concatenate((horizontal, vertical))

    source_value = float(np.quantile(magnitudes(source), 0.999))
    output_value = float(np.quantile(magnitudes(output), 0.999))
    return output_value / max(source_value, 1e-12)


def _new_boundary_fraction(source: np.ndarray, output: np.ndarray, epsilon: float) -> float:
    boundary = ((output <= epsilon) & (source > epsilon)) | (
        (output >= 1.0 - epsilon) & (source < 1.0 - epsilon)
    )
    return float(np.mean(np.any(boundary, axis=-1)))


def _resize_rgb8(encoded: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(
        np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB"
    )
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    return image


def _stable_identity(report: Mapping[str, Any]) -> str:
    value = dict(report)
    value.pop("stable_evidence_id", None)
    return hashlib.sha256(canonical_json(value)).hexdigest()


def evaluate_photographic(
    config: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    compiler, population, wavelength, curves = _validate_inputs(config, root)
    scanner_row = next(
        row
        for row in compiler["scanner_profiles"]
        if row["profile_id"] == config["parent_operator"]["scanner_profile_id"]
    )
    scanner = synthetic_profile_from_contract(wavelength, scanner_row)
    cube17 = synthetic_cube(int(compiler["compiler"]["lut_size"]))
    pooled_curves = np.mean(np.stack(list(curves.values()), axis=0), axis=0)
    lut_values: dict[str, np.ndarray] = {}
    for stock in STOCKS:
        lut_values[stock], _ = compile_operator(cube17, curves[stock], scanner, compiler)
    pooled_lut, _ = compile_operator(cube17, pooled_curves, scanner, compiler)
    luts = {
        stock: DenseLUT3D(values, np.zeros(3), np.ones(3), "trilinear")
        for stock, values in lut_values.items()
    }
    pooled_operator = DenseLUT3D(pooled_lut, np.zeros(3), np.ones(3), "trilinear")

    cube33 = synthetic_cube(33)
    fidelity: dict[str, Any] = {}
    for stock in STOCKS:
        direct, _ = compile_operator(cube33, curves[stock], scanner, compiler)
        approximate = luts[stock].apply(cube33)
        difference = approximate - direct
        fidelity[stock] = {
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_width = int(config["execution"]["contact_sheet_cell_width"])
    cell_height = round(cell_width * 0.72)
    label_height = 28
    canvas = Image.new(
        "RGB",
        (cell_width * 4, len(population) * (cell_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    rows: list[dict[str, Any]] = []
    gates = config["gates"]
    maximum_samples = int(config["execution"]["basic_fit_max_samples"])
    for row_index, manifest_row in enumerate(population):
        source_path = root / _relative_path(manifest_row["decoded_path"])
        if not source_path.is_file() or hash_file(source_path) != manifest_row["decoded_sha256"]:
            raise FujifilmDyePhotographicError("CB3 decoded source identity drift")
        source_encoded = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float64) / 255.0
        source = encoded_srgb_to_linear(source_encoded)
        flat_source = source.reshape(-1, 3)
        sample_indices = _sample_indices(flat_source.shape[0], maximum_samples)
        source_sample = flat_source[sample_indices]
        pooled_output = pooled_operator.apply(source)
        pooled_sample = pooled_output.reshape(-1, 3)[sample_indices]
        pooled_name = f"{manifest_row['id']}__pooled.png"
        pooled_sha = _save_rgb16_png(
            output_dir / pooled_name, _linear_to_encoded(pooled_output)
        )
        stock_rows: dict[str, Any] = {}
        stock_outputs: dict[str, np.ndarray] = {}
        stock_samples: dict[str, np.ndarray] = {}
        for stock in STOCKS:
            output = luts[stock].apply(source)
            sample = output.reshape(-1, 3)[sample_indices]
            basic = fit_joint_basic_adjustment(source_sample, sample)
            basic_output = np.clip(basic.apply(source_sample), 0.0, 1.0)
            output_name = f"{manifest_row['id']}__{stock}.png"
            output_sha = _save_rgb16_png(
                output_dir / output_name, _linear_to_encoded(output)
            )
            stock_outputs[stock] = output
            stock_samples[stock] = sample
            stock_rows[stock] = {
                "output_path": output_name,
                "output_sha256": output_sha,
                "finite": bool(np.all(np.isfinite(output))),
                "output_minimum": float(np.min(output)),
                "output_maximum": float(np.max(output)),
                "new_boundary_fraction": _new_boundary_fraction(
                    source, output, float(gates["boundary_epsilon"])
                ),
                "p999_gradient_ratio_vs_source": _gradient_p999_ratio(source, output),
                "style_delta_e76_median": _median_delta_e76(source_sample, sample),
                "joint_basic_residual_delta_e76_median": _median_delta_e76(
                    basic_output, sample
                ),
                "vs_pooled_delta_e76_median": _median_delta_e76(sample, pooled_sample),
            }
        pair_rows: dict[str, Any] = {}
        for first, second in combinations(STOCKS, 2):
            pair_rows[f"{first}__{second}"] = {
                "delta_e76_median": _median_delta_e76(
                    stock_samples[first], stock_samples[second]
                )
            }
        y = row_index * (cell_height + label_height)
        columns = [("source", source_encoded)] + [
            (stock, _linear_to_encoded(stock_outputs[stock])) for stock in STOCKS
        ]
        for column, (label, image) in enumerate(columns):
            thumbnail = _resize_rgb8(image, cell_width, cell_height)
            x = column * cell_width + (cell_width - thumbnail.width) // 2
            offset_y = y + label_height + (cell_height - thumbnail.height) // 2
            canvas.paste(thumbnail, (x, offset_y))
            draw.text((column * cell_width + 6, y + 6), label, fill="black")
        rows.append(
            {
                "id": manifest_row["id"],
                "make": manifest_row["make"],
                "source_sha256": manifest_row["decoded_sha256"],
                "sample_count": int(sample_indices.size),
                "pooled_output_path": pooled_name,
                "pooled_output_sha256": pooled_sha,
                "stocks": stock_rows,
                "pairs": pair_rows,
            }
        )

    contact_sheet = output_dir / "contact_sheet.png"
    canvas.save(contact_sheet, format="PNG", compress_level=6)
    contact_sheet_sha256 = hash_file(contact_sheet)

    aggregates: dict[str, Any] = {"stocks": {}, "pairs": {}}
    for stock in STOCKS:
        values = [row["stocks"][stock] for row in rows]
        aggregates["stocks"][stock] = {
            "maximum_new_boundary_fraction": max(row["new_boundary_fraction"] for row in values),
            "maximum_p999_gradient_ratio_vs_source": max(
                row["p999_gradient_ratio_vs_source"] for row in values
            ),
            "median_style_delta_e76": float(
                np.median([row["style_delta_e76_median"] for row in values])
            ),
            "median_joint_basic_residual_delta_e76": float(
                np.median(
                    [row["joint_basic_residual_delta_e76_median"] for row in values]
                )
            ),
            "rows_joint_basic_residual_delta_e76_ge_0p25": sum(
                row["joint_basic_residual_delta_e76_median"] >= 0.25 for row in values
            ),
            "median_vs_pooled_delta_e76": float(
                np.median([row["vs_pooled_delta_e76_median"] for row in values])
            ),
            "rows_vs_pooled_delta_e76_ge_0p25": sum(
                row["vs_pooled_delta_e76_median"] >= 0.25 for row in values
            ),
        }
    for first, second in combinations(STOCKS, 2):
        key = f"{first}__{second}"
        values = [row["pairs"][key]["delta_e76_median"] for row in rows]
        aggregates["pairs"][key] = {
            "median_delta_e76": float(np.median(values)),
            "rows_delta_e76_ge_0p5": sum(value >= 0.5 for value in values),
        }

    stock_aggregates = list(aggregates["stocks"].values())
    pair_aggregates = list(aggregates["pairs"].values())
    all_stock_rows = [value for row in rows for value in row["stocks"].values()]
    gate_results = {
        "population": len(rows) == int(gates["required_rows"])
        and len({row["make"] for row in rows}) == int(gates["required_camera_makes"]),
        "finite_range": all(
            value["finite"] is gates["all_outputs_finite"]
            and value["output_minimum"] >= gates["output_minimum"]
            and value["output_maximum"] <= gates["output_maximum"]
            for value in all_stock_rows
        ),
        "boundary": all(
            value["maximum_new_boundary_fraction"] <= gates["maximum_new_boundary_fraction"]
            for value in stock_aggregates
        ),
        "gradient": all(
            value["maximum_p999_gradient_ratio_vs_source"]
            <= gates["maximum_p999_gradient_ratio_vs_source"]
            for value in stock_aggregates
        ),
        "style": all(
            value["median_style_delta_e76"] >= gates["minimum_each_stock_median_style_delta_e76"]
            for value in stock_aggregates
        ),
        "non_basic": all(
            value["median_joint_basic_residual_delta_e76"]
            >= gates["minimum_each_stock_median_joint_basic_residual_delta_e76"]
            and value["rows_joint_basic_residual_delta_e76_ge_0p25"]
            >= gates["minimum_rows_per_stock_joint_basic_residual_delta_e76_ge_0p25"]
            for value in stock_aggregates
        ),
        "stock_separation": all(
            value["median_delta_e76"] >= gates["minimum_each_pair_median_stock_delta_e76"]
            and value["rows_delta_e76_ge_0p5"]
            >= gates["minimum_rows_per_pair_stock_delta_e76_ge_0p5"]
            for value in pair_aggregates
        ),
        "pooled_separation": all(
            value["median_vs_pooled_delta_e76"]
            >= gates["minimum_each_stock_median_vs_pooled_delta_e76"]
            and value["rows_vs_pooled_delta_e76_ge_0p25"]
            >= gates["minimum_rows_per_stock_vs_pooled_delta_e76_ge_0p25"]
            for value in stock_aggregates
        ),
        "lut_fidelity": all(
            value["rmse"] <= gates["maximum_lut_direct_reference_rmse"]
            and value["maximum_absolute_error"]
            <= gates["maximum_lut_direct_reference_max_abs"]
            for value in fidelity.values()
        ),
    }
    automatic_passed = all(gate_results.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "parent_stable_evidence_id": config["parent_operator"][
            "required_stable_evidence_id"
        ],
        "source_manifest_sha256": config["source_population"]["manifest_sha256"],
        "scanner_profile_sha256": scanner.profile_sha256,
        "lut_fidelity": fidelity,
        "rows": rows,
        "aggregates": aggregates,
        "gate_results": gate_results,
        "failed_gates": sorted(key for key, value in gate_results.items() if not value),
        "automatic_passed": automatic_passed,
        "visual_review_opened": automatic_passed,
        "contact_sheet_path": "contact_sheet.png",
        "contact_sheet_sha256": contact_sheet_sha256,
        "decision": (
            "automatic_pass_visual_review_required"
            if automatic_passed
            else "close_exact_photographic_development_family"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _stable_identity(report)
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONTRACT_SHA256",
    "FujifilmDyePhotographicError",
    "_gradient_p999_ratio",
    "_sample_indices",
    "evaluate_photographic",
    "load_contract",
    "write_report",
]
