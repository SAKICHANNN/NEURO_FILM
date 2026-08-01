"""Independent visual product-value evaluation for the BN2 operator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab

from scripts.build_fivek_freeze_pack import resize_float
from src.eval.fivek_adaptive_lut_basis_development import (
    _array_sha256,
    _canonical_bytes,
    _sha256,
)
from src.eval.fivek_hard_case_medoid_development import _load_ay0_population
from src.eval.fivek_neutral_base_parameter_pilot import source_descriptor
from src.eval.fivek_projection_curve_basis_development import (
    _build_predictions,
    _fit_population,
    _safe_operator,
    validate_contract as validate_bn1_contract,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)
from src.eval.fivek_unseen_content_confirmation import _new_boundary_fraction
from src.preprocess import save_srgb16_png


ARMS = (
    "fixed_ao6_direct",
    "global_projection_curves_then_fixed_ao6",
    "adaptive_projection_curves_then_fixed_ao6",
)


class FiveKProjectionCurveVisualError(ValueError):
    """Raised when BN3 inputs, execution or review policy drift."""


def _load_exact_json(root: Path, path: str, expected_sha256: str) -> Any:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise FiveKProjectionCurveVisualError(f"evidence drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _load_display_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise FiveKProjectionCurveVisualError(
                f"expected RGB source: {path}"
            )
        encoded = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.clip(encoded, 0.0, 1.0)


def _median_delta_e76(first: np.ndarray, second: np.ndarray) -> float:
    first_lab = rgb2lab(np.asarray(first, dtype=np.float64))
    second_lab = rgb2lab(np.asarray(second, dtype=np.float64))
    return float(
        np.median(np.linalg.norm(second_lab - first_lab, axis=-1))
    )


def _p999_gradient(rgb: np.ndarray) -> float:
    values = np.asarray(rgb, dtype=np.float64)
    horizontal = np.linalg.norm(values[:, 1:] - values[:, :-1], axis=-1)
    vertical = np.linalg.norm(values[1:] - values[:-1], axis=-1)
    return float(
        np.quantile(np.concatenate((horizontal.ravel(), vertical.ravel())), 0.999)
    )


def _resolve_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") == 1:
        return dict(config)
    if config.get("schema_version") != 2:
        raise FiveKProjectionCurveVisualError("unsupported BN3 schema")
    base_ref = config["base_contract"]
    base = _load_exact_json(root, base_ref["path"], base_ref["sha256"])
    correction = config["correction"]
    if (
        base.get("status") != base_ref["required_status"]
        or correction.get("descriptor_resolution")
        != "unchanged maximum-side-256 AY0 source descriptor"
        or correction.get("operator_and_ao6_resolution")
        != "exact decoded source pixels"
        or correction.get("operator_or_ao6_resampling_allowed")
        or any(
            int(correction[key]) != 0
            for key in (
                "threshold_changes",
                "source_changes",
                "model_changes",
                "blind_protocol_changes",
            )
        )
    ):
        raise FiveKProjectionCurveVisualError("BN3 v2 correction drift")
    effective = deepcopy(base)
    effective["schema_version"] = 2
    effective["experiment_id"] = config["experiment_id"]
    effective["status"] = config["status"]
    effective["software_commit_at_freeze"] = config[
        "software_commit_at_freeze"
    ]
    effective["rendering"] = deepcopy(correction)
    effective["claim_ceiling"] = config["claim_ceiling"]
    return effective


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    effective = _resolve_contract(root, config)
    if (
        effective.get("status") != "contract_frozen_implementation_ready"
        or tuple(effective.get("render_arms", ())) != ARMS
        or effective.get("new_data_download_allowed")
        or effective.get("production_default_changed")
        or effective.get("film_or_stock_claim_allowed")
        or effective["frozen_model"].get("target_pixels_available")
        or effective["frozen_model"].get("operator_or_predictor_tuning_allowed")
        or effective["blind_protocol"].get("rounds") != 3
        or not effective["blind_protocol"].get(
            "mapping_hidden_until_all_observations_are_frozen"
        )
    ):
        raise FiveKProjectionCurveVisualError("BN3 boundary drift")

    parent = _load_exact_json(
        root,
        effective["parent"]["decision"],
        effective["parent"]["decision_sha256"],
    )
    if (
        parent.get("status") != effective["parent"]["required_status"]
        or parent.get("report_sha256")
        != effective["parent"]["required_report_sha256"]
        or not parent.get("automatic_pass")
    ):
        raise FiveKProjectionCurveVisualError("BN2 parent drift")

    model = effective["frozen_model"]
    bn1_config = _load_exact_json(root, model["config"], model["config_sha256"])
    bn1_decision = _load_exact_json(
        root, model["decision"], model["decision_sha256"]
    )
    if bn1_decision.get("status") != "development_pass_confirmation_required":
        raise FiveKProjectionCurveVisualError("BN1 model decision drift")
    bn1_validated = validate_bn1_contract(root, bn1_config)

    source = effective["independent_source"]
    source_decision = _load_exact_json(
        root, source["decision"], source["decision_sha256"]
    )
    manifest = _load_exact_json(
        root, source["manifest"], source["manifest_sha256"]
    )
    review = _load_exact_json(
        root, source["visual_review"], source["visual_review_sha256"]
    )
    eligible_ids = [str(value) for value in review["eligible_ids"]]
    if (
        not source_decision["result"].get("automatic_pass")
        or not source_decision["result"].get("visual_pass")
        or review.get("confirmed_severe_source_artifact_count")
        != source["required_confirmed_severe_source_artifact_count"]
        or len(eligible_ids) != int(source["expected_rows"])
        or not isinstance(manifest, list)
    ):
        raise FiveKProjectionCurveVisualError("independent source gate closed")
    rows = {str(row["id"]): row for row in manifest}
    if len(rows) != len(manifest) or not set(eligible_ids).issubset(rows):
        raise FiveKProjectionCurveVisualError("independent source drift")
    return {
        "parent": parent,
        "bn1_config": bn1_config,
        "bn1_validated": bn1_validated,
        "eligible_ids": eligible_ids,
        "source_rows": rows,
        "effective_config": effective,
    }


def _load_visual_population(
    root: Path,
    eligible_ids: list[str],
    source_rows: Mapping[str, Mapping[str, Any]],
    ay0_config: Mapping[str, Any],
) -> dict[str, Any]:
    maximum_side = int(ay0_config["decode"]["maximum_side"])
    rows = []
    for source_id in eligible_ids:
        evidence = source_rows[source_id]
        path = root / str(evidence["decoded_path"])
        if _sha256(path) != evidence["decoded_sha256"]:
            raise FiveKProjectionCurveVisualError(
                f"source image drift: {source_id}"
            )
        source = _load_display_image(path)
        descriptor_source = resize_float(source, maximum_side)
        rows.append(
            {
                "pair_id": source_id,
                "group": evidence["make"],
                "source": source,
                "descriptor": source_descriptor(descriptor_source, ay0_config),
                "source_shape": list(source.shape),
                "descriptor_shape": list(descriptor_source.shape),
                "decoded_path": evidence["decoded_path"],
                "decoded_sha256": evidence["decoded_sha256"],
            }
        )
    return {"name": "bh1s_independent_cc0_visual", "rows": rows}


def run_visual_product_value(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    effective = validated["effective_config"]
    if output_dir.exists():
        raise FileExistsError("BN3 output is create-only")
    output_dir.mkdir(parents=True)

    bn1_config = validated["bn1_config"]
    curve = validated["bn1_validated"]["bj0"]["curve_validated"]
    ay0 = _load_ay0_population(root, curve["ay0_config"], curve["ay0_report"])
    ay0["name"] = bn1_config["development_populations"][0]["name"]
    directions = validated["bn1_validated"]["directions"]
    _fit_population(ay0, directions=directions, operator=bn1_config["operator"])
    population = _load_visual_population(
        root,
        validated["eligible_ids"],
        validated["source_rows"],
        curve["ay0_config"],
    )
    prediction = _build_predictions(ay0, [population], bn1_config)[
        population["name"]
    ]
    renderer = build_fixed_ao6_renderer(
        curve["safe_validated"]["fixed_config"],
        curve["safe_validated"]["fixed_validated"],
    )

    operator_config = bn1_config["operator"]
    epsilon = float(operator_config["boundary_epsilon"])
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for index, row in enumerate(population["rows"]):
        global_operator, global_diagnostics = _safe_operator(
            directions, prediction["global"][index], operator_config
        )
        adaptive_operator, adaptive_diagnostics = _safe_operator(
            directions, prediction["adaptive"][index], operator_config
        )
        source = row["source"]
        global_base = global_operator.apply(source)
        adaptive_base = adaptive_operator.apply(source)
        direct, direct_guard = renderer(source)
        global_look, global_guard = renderer(global_base)
        adaptive_look, adaptive_guard = renderer(adaptive_base)
        outputs = {
            ARMS[0]: direct,
            ARMS[1]: global_look,
            ARMS[2]: adaptive_look,
        }
        direct_gradient = max(_p999_gradient(direct), 1.0e-12)
        arm_records: dict[str, Any] = {}
        for arm_id, output in outputs.items():
            if (
                output.shape != source.shape
                or not np.all(np.isfinite(output))
            ):
                raise FiveKProjectionCurveVisualError(
                    f"invalid output: {row['pair_id']}:{arm_id}"
                )
            path = output_dir / "renders" / arm_id / f"{row['pair_id']}.png"
            save_srgb16_png(output, path)
            arm_records[arm_id] = {
                "output": path.relative_to(output_dir).as_posix(),
                "output_sha256": _sha256(path),
                "new_boundary_fraction_vs_source": _new_boundary_fraction(
                    source, output, epsilon
                ),
                "out_of_cube_fraction": float(
                    np.mean((output < 0.0) | (output > 1.0))
                ),
                "p999_gradient_ratio_vs_direct_ao6": float(
                    _p999_gradient(output) / direct_gradient
                ),
            }
        delta = _median_delta_e76(global_look, adaptive_look)
        deltas.append(delta)
        rows.append(
            {
                "source_id": row["pair_id"],
                "source_path": row["decoded_path"],
                "source_sha256": row["decoded_sha256"],
                "global_coefficient_sha256": _array_sha256(
                    prediction["global"][index]
                ),
                "adaptive_coefficient_sha256": _array_sha256(
                    prediction["adaptive"][index]
                ),
                "global_safe_dose": float(global_operator.dose),
                "adaptive_safe_dose": float(adaptive_operator.dose),
                "global_minimum_jacobian_determinant": float(
                    global_diagnostics["minimum_determinant"]
                ),
                "adaptive_minimum_jacobian_determinant": float(
                    adaptive_diagnostics["minimum_determinant"]
                ),
                "global_nonpositive_jacobian_count": int(
                    global_diagnostics["nonpositive_determinant_count"]
                ),
                "adaptive_nonpositive_jacobian_count": int(
                    adaptive_diagnostics["nonpositive_determinant_count"]
                ),
                "global_maximum_jacobian_condition": float(
                    global_diagnostics["maximum_condition"]
                ),
                "adaptive_maximum_jacobian_condition": float(
                    adaptive_diagnostics["maximum_condition"]
                ),
                "adaptive_vs_global_delta_e76": delta,
                "guard_fractions": {
                    ARMS[0]: direct_guard,
                    ARMS[1]: global_guard,
                    ARMS[2]: adaptive_guard,
                },
                "arms": arm_records,
            }
        )

    gate = effective["automatic_gate"]
    all_arms = [arm for row in rows for arm in row["arms"].values()]
    all_diagnostics = [
        (row["global_minimum_jacobian_determinant"],
         row["global_maximum_jacobian_condition"],
         row["global_nonpositive_jacobian_count"])
        for row in rows
    ] + [
        (row["adaptive_minimum_jacobian_determinant"],
         row["adaptive_maximum_jacobian_condition"],
         row["adaptive_nonpositive_jacobian_count"])
        for row in rows
    ]
    summary = {
        "median_adaptive_vs_global_delta_e76": float(np.median(deltas)),
        "sources_with_adaptive_vs_global_delta_e76_ge_0p5": int(
            np.sum(np.asarray(deltas) >= 0.5)
        ),
        "maximum_new_boundary_fraction_vs_source": max(
            arm["new_boundary_fraction_vs_source"] for arm in all_arms
        ),
        "maximum_out_of_cube_fraction": max(
            arm["out_of_cube_fraction"] for arm in all_arms
        ),
        "maximum_p999_gradient_ratio_vs_direct_ao6": max(
            arm["p999_gradient_ratio_vs_direct_ao6"] for arm in all_arms
        ),
        "minimum_sampled_jacobian_determinant": min(
            item[0] for item in all_diagnostics
        ),
        "maximum_sampled_jacobian_condition": max(
            item[1] for item in all_diagnostics
        ),
        "maximum_nonpositive_jacobian_count": max(
            item[2] for item in all_diagnostics
        ),
    }
    gates = {
        "output_count": len(all_arms) == int(gate["expected_outputs"]),
        "boundary": summary["maximum_new_boundary_fraction_vs_source"]
        <= float(gate["maximum_new_boundary_fraction_vs_source"]),
        "cube": summary["maximum_out_of_cube_fraction"]
        <= float(gate["maximum_out_of_cube_fraction"]),
        "jacobian_sign": summary["maximum_nonpositive_jacobian_count"]
        <= int(gate["maximum_nonpositive_jacobian_count"]),
        "jacobian_condition": summary["maximum_sampled_jacobian_condition"]
        <= float(gate["maximum_sampled_jacobian_condition"]),
        "jacobian_determinant": summary["minimum_sampled_jacobian_determinant"]
        >= float(gate["minimum_sampled_jacobian_determinant"]),
        "gradient_tail": summary["maximum_p999_gradient_ratio_vs_direct_ao6"]
        <= float(gate["maximum_p999_gradient_ratio_vs_direct_ao6"]),
        "material_delta": summary["median_adaptive_vs_global_delta_e76"]
        >= float(gate["minimum_median_adaptive_vs_global_delta_e76"]),
        "material_source_count": summary[
            "sources_with_adaptive_vs_global_delta_e76_ge_0p5"
        ] >= int(gate["minimum_sources_with_adaptive_vs_global_delta_e76_ge_0p5"]),
    }
    stable = {"summary": summary, "gates": gates, "rows": rows}
    report = {
        "schema_version": 1,
        "experiment_id": effective["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": all(gates.values()),
        "blind_review_allowed": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "claim_ceiling": effective["claim_ceiling"],
    }
    report_path = output_dir / "report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return {
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
    }


def build_blind_round(
    *,
    root: Path,
    render_dir: Path,
    report: Mapping[str, Any],
    round_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    if not report.get("blind_review_allowed"):
        raise FiveKProjectionCurveVisualError("automatic gate forbids blind review")
    primary = (ARMS[1], ARMS[2])
    font = ImageFont.load_default()
    tiles: list[Image.Image] = []
    mapping: list[dict[str, str]] = []
    for row in report["rows"]:
        source_id = row["source_id"]
        order = list(primary)
        random.Random(
            hashlib.sha256(
                f"u5-r2bn3:{round_index}:{source_id}".encode("ascii")
            ).digest()
        ).shuffle(order)
        mapping.append({"source_id": source_id, "A": order[0], "B": order[1]})
        paths = [root / row["source_path"]] + [
            render_dir / row["arms"][arm]["output"] for arm in order
        ]
        images = []
        for path in paths:
            with Image.open(path) as image:
                item = image.convert("RGB")
                item.thumbnail((520, 330), Image.Resampling.LANCZOS)
                images.append(item.copy())
        tile = Image.new("RGB", (1580, 375), "white")
        draw = ImageDraw.Draw(tile)
        for column, (label, image) in enumerate(zip(("Source", "A", "B"), images)):
            x = 5 + column * 525
            draw.text((x, 3), label, fill="black", font=font)
            tile.paste(image, (x, 23))
        draw.text((5, 357), source_id, fill="black", font=font)
        tiles.append(tile)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for part_index, start in enumerate(range(0, len(tiles), 4), start=1):
        selected = tiles[start : start + 4]
        sheet = Image.new("RGB", (1580, len(selected) * 375 + 28), "white")
        ImageDraw.Draw(sheet).text(
            (5, 5),
            f"U5.R2BN3 blind round {round_index} part {part_index}",
            fill="black",
            font=font,
        )
        for index, tile in enumerate(selected):
            sheet.paste(tile, (0, 28 + index * 375))
        path = output_dir / f"blind_round_{round_index}_part_{part_index}.png"
        sheet.save(path, "PNG")
        parts.append(path)
    mapping_path = output_dir / f"blind_round_{round_index}_mapping.json"
    mapping_path.write_bytes(_canonical_bytes({"rows": mapping}))
    return {
        "parts": [path.relative_to(render_dir).as_posix() for path in parts],
        "part_sha256": [_sha256(path) for path in parts],
        "mapping_path": mapping_path.relative_to(render_dir).as_posix(),
        "mapping_sha256": _sha256(mapping_path),
    }


__all__ = [
    "ARMS",
    "FiveKProjectionCurveVisualError",
    "build_blind_round",
    "run_visual_product_value",
    "validate_contract",
]
