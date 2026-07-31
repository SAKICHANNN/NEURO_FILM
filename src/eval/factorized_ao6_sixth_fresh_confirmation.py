"""Sixth-fresh-population confirmation for the fixed BK16 response."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.eval.factorized_ao6_perceptual_residual_regression import (
    _operator as _bk16_operator,
)
from src.eval.global_frontier import sha256_file
from src.eval.log_chroma_fresh_comparison import (
    LogChromaFreshComparisonError,
    _boundary_metrics,
    _canonical_sha256,
    _load_exact_json,
    _style_delta_e76,
)
from src.eval.smooth_perceptual_fresh_confirmation import (
    ARMS as CONTROL_ARMS,
    _operator as _bk7_operator,
    render_fixed_arms as render_control_arms,
)
from src.film_physics.profile_consumer import compile_standalone_profile_artifact
from src.preprocess import save_srgb16_png


ARMS = ("fixed_bk16_safe_base_factorized_ao6_residual", *CONTROL_ARMS)
SCHEMA = (
    "neuro_film.u5_r2bk18_factorized_ao6_sixth_fresh_confirmation_report.v1"
)


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rendering = config["rendering"]
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk18_factorized_ao6_sixth_fresh_confirmation.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or tuple(row["arm_id"] for row in config["fixed_arms"]) != ARMS
        or any(
            config.get(key)
            for key in (
                "training_allowed",
                "operator_fitting_allowed",
                "selector_training_allowed",
                "production_default_changed",
                "stock_or_authenticity_claim_allowed",
            )
        )
        or not rendering["create_only"]
        or any(
            rendering[key]
            for key in (
                "per_image_fit_allowed",
                "operator_refit_allowed",
                "strength_retuning_allowed",
                "routing_allowed",
                "dense_blending_allowed",
                "hard_clipping_allowed",
            )
        )
    ):
        raise LogChromaFreshComparisonError("BK18 frozen contract drift")

    source = config["source_preflight"]
    decision = _load_exact_json(
        root, source["decision"], source["decision_sha256"]
    )
    manifest = _load_exact_json(
        root, source["manifest"], source["manifest_sha256"]
    )
    result = decision["result"]
    if (
        decision.get("status") != source["required_status"]
        or decision.get("decision") != source["required_decision"]
        or not result["automatic_pass"]
        or not result["visual_pass"]
        or result["confirmed_severe_source_artifact_count"] != 0
        or result["operator_outputs_inspected"]
        or not isinstance(manifest, list)
        or len(manifest) != source["expected_eligible_rows"]
        or len({str(row["id"]) for row in manifest}) != len(manifest)
        or len({str(row["make"]) for row in manifest})
        != source["expected_camera_makes"]
    ):
        raise LogChromaFreshComparisonError("BK17S source gate is not open")

    by_arm = {str(row["arm_id"]): row for row in config["fixed_arms"]}
    bk16_config = _load_exact_json(
        root, by_arm[ARMS[0]]["config"], by_arm[ARMS[0]]["config_sha256"]
    )
    bk16 = _bk16_operator(bk16_config)
    if bk16_config.get("status") != "primitive_contract_frozen":
        raise LogChromaFreshComparisonError("BK16 operator drift")

    bk7_config = _load_exact_json(
        root, by_arm[ARMS[1]]["config"], by_arm[ARMS[1]]["config_sha256"]
    )
    bk7 = _bk7_operator(bk7_config)
    if (
        bk7_config.get("status") != "primitive_contract_frozen"
        or float(by_arm[ARMS[1]]["strength"]) != 1.0
    ):
        raise LogChromaFreshComparisonError("BK7 operator drift")

    ao6_config = _load_exact_json(
        root,
        by_arm[ARMS[2]]["profile_compiler_config"],
        by_arm[ARMS[2]]["profile_compiler_config_sha256"],
    )
    safe = by_arm[ARMS[3]]
    profile = _load_exact_json(root, safe["profile"], safe["profile_sha256"])
    statistics = _load_exact_json(
        root, safe["style_statistics"], safe["style_statistics_sha256"]
    )
    guardrails = _load_exact_json(
        root, safe["guardrails"], safe["guardrails_sha256"]
    )
    style = str(safe["style"])
    if (
        profile.get("profile_id") != "safe-rich-v1"
        or style not in profile["style_parameters"]
        or style not in statistics["styles"]
        or style not in guardrails["styles"]
    ):
        raise LogChromaFreshComparisonError("safe-rich control drift")
    return {
        "source_rows": manifest,
        "bk16": bk16,
        "bk7": bk7,
        "bk7_strength": 1.0,
        "ao6_config": ao6_config,
        "ao6_component": str(by_arm[ARMS[2]]["component"]),
        "safe_profile": profile["style_parameters"][style],
        "safe_statistics": statistics["styles"][style],
        "safe_guardrails": {
            **guardrails["defaults"],
            **guardrails["styles"][style],
        },
        "safe_style": style,
        "safe_seed": int(safe["seed"]),
    }


def render_fixed_arms(
    encoded: np.ndarray,
    *,
    validated: Mapping[str, Any],
    ao6_artifact: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    controls = render_control_arms(
        encoded, validated=validated, ao6_artifact=ao6_artifact
    )
    return {
        ARMS[0]: validated["bk16"].apply(
            np.asarray(encoded, dtype=np.float32),
            controls[ARMS[3]],
            controls[ARMS[2]],
        ),
        **controls,
    }


def _contact_sheets(
    root: Path,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, str]]:
    font = ImageFont.load_default()
    labels = ("SOURCE", "BK16", "BK7", "AO6", "SAFE-RICH")
    tiles: list[Image.Image] = []
    for row in rows:
        paths = [
            root / row["decoded_path"],
            *(output_dir / "renders" / arm / f"{row['id']}.png" for arm in ARMS),
        ]
        tile = Image.new("RGB", (1600, 254), "white")
        draw = ImageDraw.Draw(tile)
        for index, (path, label) in enumerate(zip(paths, labels)):
            with Image.open(path) as opened:
                image = ImageOps.contain(
                    ImageOps.exif_transpose(opened).convert("RGB"),
                    (310, 216),
                    method=Image.Resampling.LANCZOS,
                )
            x = index * 320 + (310 - image.width) // 2
            tile.paste(image, (x, 24 + (216 - image.height) // 2))
            draw.text((index * 320 + 5, 5), label, fill="black", font=font)
        draw.text(
            (5, 241),
            f"{row['id']} | {row['make']} {row['model']}",
            fill="black",
            font=font,
        )
        tiles.append(tile)
    evidence = []
    for part, start in enumerate(range(0, len(tiles), 4), start=1):
        selected = tiles[start : start + 4]
        sheet = Image.new("RGB", (1600, len(selected) * 254 + 28), "white")
        ImageDraw.Draw(sheet).text(
            (5, 6),
            f"U5.R2BK18 fixed four-arm visual review part {part}",
            fill="black",
            font=font,
        )
        for index, tile in enumerate(selected):
            sheet.paste(tile, (0, 28 + index * 254))
        path = output_dir / f"visual_review_part_{part}.png"
        sheet.save(path, "PNG")
        evidence.append(
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        )
    return evidence


def run_confirmation(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    if output_dir.exists():
        raise FileExistsError("BK18 confirmation is create-only")
    output_dir.mkdir(parents=True)
    artifact = compile_standalone_profile_artifact(
        root=root, config=validated["ao6_config"]
    )
    records = []
    for row in validated["source_rows"]:
        source_path = root / str(row["decoded_path"])
        if sha256_file(source_path) != row["decoded_sha256"]:
            raise LogChromaFreshComparisonError("source pixel drift")
        with Image.open(source_path) as opened:
            source = np.asarray(
                ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.float32
            )
        source /= 255.0
        outputs = render_fixed_arms(
            source, validated=validated, ao6_artifact=artifact
        )
        for arm_id in ARMS:
            path = output_dir / "renders" / arm_id / f"{row['id']}.png"
            save_srgb16_png(outputs[arm_id], path)
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.dtype != np.uint16 or decoded.shape != source.shape:
                raise LogChromaFreshComparisonError("RGB16 PNG verification failed")
            record = {
                "source_id": row["id"],
                "make": row["make"],
                "arm_id": arm_id,
                "source_decoded_sha256": row["decoded_sha256"],
                "output": path.relative_to(output_dir).as_posix(),
                "output_sha256": sha256_file(path),
                "median_style_delta_e76": _style_delta_e76(source, outputs[arm_id]),
                **_boundary_metrics(source, outputs[arm_id]),
            }
            if arm_id == ARMS[0]:
                record["median_increment_vs_safe_delta_e76"] = _style_delta_e76(
                    outputs[ARMS[3]], outputs[ARMS[0]]
                )
            records.append(record)
    medians = {
        arm: float(
            np.median(
                [r["median_style_delta_e76"] for r in records if r["arm_id"] == arm]
            )
        )
        for arm in ARMS
    }
    increment = float(
        np.median(
            [
                r["median_increment_vs_safe_delta_e76"]
                for r in records
                if r["arm_id"] == ARMS[0]
            ]
        )
    )
    max_boundary = max(r["new_code_boundary_fraction_vs_source"] for r in records)
    gate = config["automatic_gate"]
    gates = {
        "complete_outputs": len(records) == int(gate["expected_outputs"]),
        "bk16_style_salience": medians[ARMS[0]]
        >= float(gate["minimum_bk16_population_median_style_delta_e76"]),
        "bk16_increment_over_safe": increment
        >= float(gate["minimum_bk16_population_median_increment_vs_safe_delta_e76"]),
        "new_boundary": max_boundary
        <= float(gate["maximum_per_output_new_code_boundary_fraction_vs_source"]),
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "source_count": len(validated["source_rows"]),
        "arms": list(ARMS),
        "ao6_bundle_sha256": artifact["bundle_sha256"],
        "records": records,
        "population_median_style_delta_e76": medians,
        "bk16_population_median_increment_vs_safe_delta_e76": increment,
        "maximum_new_code_boundary_fraction_vs_source": max_boundary,
        "automatic_gates": gates,
        "automatic_pass": all(gates.values()),
        "visual_review_allowed": all(gates.values()),
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sheets = (
        _contact_sheets(root, validated["source_rows"], output_dir)
        if report["visual_review_allowed"]
        else []
    )
    return {
        "report": report,
        "report_path": report_path,
        "report_sha256": sha256_file(report_path),
        "visual_sheets": sheets,
    }


__all__ = ["ARMS", "render_fixed_arms", "run_confirmation", "validate_contract"]
