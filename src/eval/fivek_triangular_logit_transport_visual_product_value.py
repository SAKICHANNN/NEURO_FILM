"""Independent full-resolution visual value test for the BN4/BN5 operator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.eval.fivek_adaptive_lut_basis_development import (
    _array_sha256,
    _canonical_bytes,
    _sha256,
)
from src.eval.fivek_hard_case_medoid_development import _load_ay0_population
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)
from src.eval.fivek_projection_curve_visual_product_value import (
    _load_exact_json,
    _load_visual_population,
    _median_delta_e76,
    _p999_gradient,
)
from src.eval.fivek_triangular_logit_transport_development import (
    _build_predictions,
    _fit_population,
    _safe_operator,
    validate_contract as validate_bn4_contract,
)
from src.eval.fivek_unseen_content_confirmation import _new_boundary_fraction
from src.preprocess import save_srgb16_png


ARMS = (
    "fixed_ao6_direct",
    "global_triangular_transport_then_fixed_ao6",
    "adaptive_triangular_transport_then_fixed_ao6",
)


class FiveKTriangularVisualError(ValueError):
    """Raised when BN6 inputs, execution, or review policy drift."""


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    primary_pair = tuple(config["blind_protocol"].get("primary_pair", ()))
    if (
        config.get("status") != "contract_frozen_implementation_ready"
        or tuple(config.get("render_arms", ())) != ARMS
        or config.get("new_data_download_allowed")
        or config.get("production_default_changed")
        or config.get("film_or_stock_claim_allowed")
        or config["frozen_model"].get("target_pixels_available")
        or config["frozen_model"].get("operator_or_predictor_tuning_allowed")
        or config["rendering"].get("descriptor_resolution")
        != "unchanged maximum-side-256 AY0 source descriptor"
        or config["rendering"].get("operator_and_ao6_resolution")
        != "exact decoded source pixels"
        or config["rendering"].get("operator_or_ao6_resampling_allowed")
        or config["automatic_gate"].get("boundary_epsilon") != 1.0 / 510.0
        or config["blind_protocol"].get("rounds") != 3
        or not config["blind_protocol"].get(
            "mapping_hidden_until_all_observations_are_frozen"
        )
        or len(primary_pair) != 2
        or len(set(primary_pair)) != 2
        or not set(primary_pair).issubset(ARMS)
        or primary_pair[1] != ARMS[2]
    ):
        raise FiveKTriangularVisualError("BN6 boundary drift")

    parent = _load_exact_json(
        root,
        config["parent"]["decision"],
        config["parent"]["decision_sha256"],
    )
    if (
        parent.get("status") != config["parent"]["required_status"]
        or parent.get("report_sha256")
        != config["parent"]["required_report_sha256"]
        or not parent.get("automatic_pass")
    ):
        raise FiveKTriangularVisualError("BN5 parent drift")

    model = config["frozen_model"]
    bn4_config = _load_exact_json(root, model["config"], model["config_sha256"])
    bn4_decision = _load_exact_json(
        root, model["decision"], model["decision_sha256"]
    )
    if bn4_decision.get("status") != "development_pass_confirmation_required":
        raise FiveKTriangularVisualError("BN4 model decision drift")
    bn4_validated = validate_bn4_contract(root, bn4_config)

    source = config["independent_source"]
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
        raise FiveKTriangularVisualError("independent source gate closed")
    rows = {str(row["id"]): row for row in manifest}
    if len(rows) != len(manifest) or not set(eligible_ids).issubset(rows):
        raise FiveKTriangularVisualError("independent source drift")
    return {
        "parent": parent,
        "bn4_config": bn4_config,
        "bn4_validated": bn4_validated,
        "eligible_ids": eligible_ids,
        "source_rows": rows,
    }


def run_visual_product_value(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    if output_dir.exists():
        raise FileExistsError("BN6 output is create-only")
    output_dir.mkdir(parents=True)

    bn4_config = validated["bn4_config"]
    bn4_validated = validated["bn4_validated"]
    curve = bn4_validated["bj0"]["curve_validated"]
    ay0 = _load_ay0_population(root, curve["ay0_config"], curve["ay0_report"])
    ay0["name"] = bn4_config["development_populations"][0]["name"]
    operator_config = bn4_config["operator"]
    lower = bn4_validated["lower_bounds"]
    upper = bn4_validated["upper_bounds"]
    _fit_population(ay0, operator=operator_config, lower=lower, upper=upper)
    population = _load_visual_population(
        root,
        validated["eligible_ids"],
        validated["source_rows"],
        curve["ay0_config"],
    )
    prediction = _build_predictions(
        ay0,
        [population],
        bn4_config,
        lower=lower,
        upper=upper,
    )[population["name"]]
    renderer = build_fixed_ao6_renderer(
        curve["safe_validated"]["fixed_config"],
        curve["safe_validated"]["fixed_validated"],
    )

    epsilon = float(config["automatic_gate"]["boundary_epsilon"])
    comparison_arm = str(config.get("comparison_reference_arm", ARMS[1]))
    if comparison_arm not in {ARMS[0], ARMS[1]}:
        raise FiveKTriangularVisualError("invalid material comparison arm")
    comparison_name = (
        "direct_ao6" if comparison_arm == ARMS[0] else "global"
    )
    delta_key = f"adaptive_vs_{comparison_name}_delta_e76"
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for index, row in enumerate(population["rows"]):
        global_operator, global_diagnostics = _safe_operator(
            prediction["global"][index], operator_config
        )
        adaptive_operator, adaptive_diagnostics = _safe_operator(
            prediction["adaptive"][index], operator_config
        )
        source = row["source"]
        global_base = global_operator.apply(source)
        adaptive_base = adaptive_operator.apply(source)
        direct, direct_guard = renderer(source)
        global_look, global_guard = renderer(global_base)
        adaptive_look, adaptive_guard = renderer(adaptive_base)
        outputs = {ARMS[0]: direct, ARMS[1]: global_look, ARMS[2]: adaptive_look}
        direct_gradient = max(_p999_gradient(direct), 1.0e-12)
        arm_records: dict[str, Any] = {}
        for arm_id, output in outputs.items():
            if output.shape != source.shape or not np.all(np.isfinite(output)):
                raise FiveKTriangularVisualError(
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
        comparison_output = direct if comparison_arm == ARMS[0] else global_look
        delta = _median_delta_e76(comparison_output, adaptive_look)
        deltas.append(delta)
        rows.append(
            {
                "source_id": row["pair_id"],
                "source_path": row["decoded_path"],
                "source_sha256": row["decoded_sha256"],
                "source_shape": row["source_shape"],
                "descriptor_shape": row["descriptor_shape"],
                "global_parameter_sha256": _array_sha256(prediction["global"][index]),
                "adaptive_parameter_sha256": _array_sha256(
                    prediction["adaptive"][index]
                ),
                "global_safe_dose": float(global_operator.dose),
                "adaptive_safe_dose": float(adaptive_operator.dose),
                "global_diagnostics": global_diagnostics,
                "adaptive_diagnostics": adaptive_diagnostics,
                delta_key: delta,
                "guard_fractions": {
                    ARMS[0]: direct_guard,
                    ARMS[1]: global_guard,
                    ARMS[2]: adaptive_guard,
                },
                "arms": arm_records,
            }
        )

    gate = config["automatic_gate"]
    all_arms = [arm for row in rows for arm in row["arms"].values()]
    diagnostics = [
        item
        for row in rows
        for item in (row["global_diagnostics"], row["adaptive_diagnostics"])
    ]
    summary = {
        f"median_{delta_key}": float(np.median(deltas)),
        f"sources_with_{delta_key}_ge_0p5": int(
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
            float(item["minimum_determinant"]) for item in diagnostics
        ),
        "maximum_sampled_jacobian_condition": max(
            float(item["maximum_condition"]) for item in diagnostics
        ),
        "maximum_nonpositive_jacobian_count": max(
            int(item["nonpositive_determinant_count"]) for item in diagnostics
        ),
        "maximum_inverse_roundtrip_error": max(
            float(item["maximum_inverse_roundtrip_error"]) for item in diagnostics
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
        "inverse": summary["maximum_inverse_roundtrip_error"]
        <= float(gate["maximum_inverse_roundtrip_error"]),
        "gradient_tail": summary["maximum_p999_gradient_ratio_vs_direct_ao6"]
        <= float(gate["maximum_p999_gradient_ratio_vs_direct_ao6"]),
        "material_delta": summary[f"median_{delta_key}"]
        >= float(gate[f"minimum_median_{delta_key}"]),
        "material_source_count": summary[f"sources_with_{delta_key}_ge_0p5"]
        >= int(gate[f"minimum_sources_with_{delta_key}_ge_0p5"]),
    }
    stable = {"summary": summary, "gates": gates, "rows": rows}
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": all(gates.values()),
        "blind_review_allowed": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
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
    primary_arms: tuple[str, str] = (ARMS[1], ARMS[2]),
) -> dict[str, Any]:
    if not report.get("blind_review_allowed"):
        raise FiveKTriangularVisualError("automatic gate forbids blind review")
    primary = tuple(primary_arms)
    if len(primary) != 2 or primary[1] != ARMS[2] or primary[0] not in ARMS[:2]:
        raise FiveKTriangularVisualError("invalid blind primary pair")
    font = ImageFont.load_default()
    tiles: list[Image.Image] = []
    mapping: list[dict[str, str]] = []
    for row in report["rows"]:
        source_id = row["source_id"]
        order = list(primary)
        random.Random(
            hashlib.sha256(
                f"u5-r2bn6:{round_index}:{source_id}".encode("ascii")
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
            f"U5.R2BN6 blind round {round_index} part {part_index}",
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
    "FiveKTriangularVisualError",
    "build_blind_round",
    "run_visual_product_value",
    "validate_contract",
]
