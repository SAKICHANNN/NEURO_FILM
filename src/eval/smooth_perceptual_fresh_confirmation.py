"""Fourth-fresh-population confirmation for the fixed BK7 response."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.eval.density_witness_frontier import encoded_srgb_to_linear
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.global_frontier import sha256_file
from src.eval.log_chroma_fresh_comparison import (
    LogChromaFreshComparisonError,
    _boundary_metrics,
    _canonical_sha256,
    _load_exact_json,
    _safe_rich,
    _style_delta_e76,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)
from src.preprocess import save_srgb16_png
from src.roll2film.smooth_perceptual_hue_density import (
    SmoothPerceptualHueDensityResponse,
)


ARMS = (
    "fixed_bk7_smooth_perceptual_hue_density",
    "fixed_ao6_colour_only_t15_c35",
    "safe_rich_velvia_50",
)
SCHEMA = (
    "neuro_film.u5_r2bk8_smooth_perceptual_fresh_confirmation_report.v1"
)


def _operator(
    config: Mapping[str, Any],
) -> SmoothPerceptualHueDensityResponse:
    params = config["operator"]
    return SmoothPerceptualHueDensityResponse(
        tone_power=float(params["tone_power"]),
        luminance_floor=float(params["luminance_floor"]),
        base_chroma_gain=float(params["base_chroma_gain"]),
        first_harmonic_gain=float(params["first_harmonic_gain"]),
        first_harmonic_center_degrees=float(
            params["first_harmonic_center_degrees"]
        ),
        second_harmonic_gain=float(params["second_harmonic_gain"]),
        second_harmonic_center_degrees=float(
            params["second_harmonic_center_degrees"]
        ),
        hue_warp=float(params["hue_warp"]),
        hue_warp_center_degrees=float(
            params["hue_warp_center_degrees"]
        ),
        luminance_hue_tilt=float(params["luminance_hue_tilt"]),
        split_tone_a=float(params["split_tone_a"]),
        split_tone_b=float(params["split_tone_b"]),
        gamut_iterations=int(params["gamut_iterations"]),
        encoded_margin=float(params["encoded_margin"]),
    )


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rendering = config["rendering"]
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk8_smooth_perceptual_fresh_confirmation.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or tuple(row["arm_id"] for row in config["fixed_arms"]) != ARMS
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("selector_training_allowed")
        or config.get("production_default_changed")
        or config.get("stock_or_authenticity_claim_allowed")
        or not rendering["create_only"]
        or rendering["per_image_fit_allowed"]
        or rendering["operator_refit_allowed"]
        or rendering["strength_retuning_allowed"]
        or rendering["routing_allowed"]
        or rendering["dense_blending_allowed"]
        or rendering["hard_clipping_allowed"]
    ):
        raise LogChromaFreshComparisonError("BK8 frozen contract drift")

    source = config["source_preflight"]
    decision = _load_exact_json(
        root, source["decision"], source["decision_sha256"]
    )
    manifest = _load_exact_json(
        root, source["manifest"], source["manifest_sha256"]
    )
    if (
        decision.get("status") != source["required_status"]
        or not decision["result"]["automatic_pass"]
        or not decision["result"]["visual_pass"]
        or decision["result"]["confirmed_severe_source_artifact_count"] != 0
        or not isinstance(manifest, list)
        or len(manifest) != source["expected_eligible_rows"]
        or len({str(row["id"]) for row in manifest}) != len(manifest)
        or len({str(row["make"]) for row in manifest})
        != source["expected_camera_makes"]
    ):
        raise LogChromaFreshComparisonError("BK8S source gate is not open")

    by_arm = {str(row["arm_id"]): row for row in config["fixed_arms"]}
    bk7_config = _load_exact_json(
        root,
        by_arm[ARMS[0]]["config"],
        by_arm[ARMS[0]]["config_sha256"],
    )
    operator = _operator(bk7_config)
    if (
        bk7_config.get("status") != "primitive_contract_frozen"
        or float(by_arm[ARMS[0]]["strength"]) != 1.0
        or operator != SmoothPerceptualHueDensityResponse()
    ):
        raise LogChromaFreshComparisonError("BK7 operator drift")

    ao6_config = _load_exact_json(
        root,
        by_arm[ARMS[1]]["profile_compiler_config"],
        by_arm[ARMS[1]]["profile_compiler_config_sha256"],
    )
    safe_profile = _load_exact_json(
        root,
        by_arm[ARMS[2]]["profile"],
        by_arm[ARMS[2]]["profile_sha256"],
    )
    statistics = _load_exact_json(
        root,
        by_arm[ARMS[2]]["style_statistics"],
        by_arm[ARMS[2]]["style_statistics_sha256"],
    )
    guardrails = _load_exact_json(
        root,
        by_arm[ARMS[2]]["guardrails"],
        by_arm[ARMS[2]]["guardrails_sha256"],
    )
    style = str(by_arm[ARMS[2]]["style"])
    if (
        safe_profile.get("profile_id") != "safe-rich-v1"
        or style not in safe_profile["style_parameters"]
        or style not in statistics["styles"]
        or style not in guardrails["styles"]
    ):
        raise LogChromaFreshComparisonError("safe-rich control drift")
    return {
        "source_rows": manifest,
        "bk7": operator,
        "bk7_strength": float(by_arm[ARMS[0]]["strength"]),
        "ao6_config": ao6_config,
        "ao6_component": str(by_arm[ARMS[1]]["component"]),
        "safe_profile": safe_profile["style_parameters"][style],
        "safe_statistics": statistics["styles"][style],
        "safe_guardrails": {
            **guardrails["defaults"],
            **guardrails["styles"][style],
        },
        "safe_style": style,
        "safe_seed": int(by_arm[ARMS[2]]["seed"]),
    }


def render_fixed_arms(
    encoded: np.ndarray,
    *,
    validated: Mapping[str, Any],
    ao6_artifact: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    source = np.asarray(encoded, dtype=np.float32)
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise LogChromaFreshComparisonError("expected finite display RGB")
    outputs = {
        ARMS[0]: validated["bk7"].apply(
            source, strength=validated["bk7_strength"]
        ),
        ARMS[1]: render_fixed_pair(
            encoded_srgb_to_linear(source).astype(np.float32),
            ao6_artifact,
            validated["ao6_component"],
        )[ARMS[1]],
        ARMS[2]: _safe_rich(source, validated),
    }
    for arm_id, output in outputs.items():
        if (
            output.shape != source.shape
            or not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise LogChromaFreshComparisonError(
                f"{arm_id} left display RGB"
            )
    return outputs


def _build_contact_sheets(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, str]]:
    font = ImageFont.load_default()
    row_tiles: list[Image.Image] = []
    labels = ("SOURCE", "BK7", "AO6", "SAFE-RICH")
    for row in rows:
        paths = [
            root / row["decoded_path"],
            *(
                output_dir / "renders" / arm / f"{row['id']}.png"
                for arm in ARMS
            ),
        ]
        tile = Image.new("RGB", (1440, 290), "white")
        draw = ImageDraw.Draw(tile)
        for index, (path, label) in enumerate(zip(paths, labels)):
            with Image.open(path) as opened:
                image = ImageOps.contain(
                    ImageOps.exif_transpose(opened).convert("RGB"),
                    (350, 250),
                    method=Image.Resampling.LANCZOS,
                )
            x = index * 360 + (350 - image.width) // 2
            tile.paste(image, (x, 26 + (250 - image.height) // 2))
            draw.text((index * 360 + 5, 5), label, fill="black", font=font)
        draw.text(
            (5, 276),
            f"{row['id']} | {row['make']} {row['model']}",
            fill="black",
            font=font,
        )
        row_tiles.append(tile)
    evidence: list[dict[str, str]] = []
    for part, start in enumerate(range(0, len(row_tiles), 4), start=1):
        selected = row_tiles[start : start + 4]
        sheet = Image.new("RGB", (1440, len(selected) * 290 + 28), "white")
        ImageDraw.Draw(sheet).text(
            (5, 6),
            f"U5.R2BK8 fixed three-arm visual review part {part}",
            fill="black",
            font=font,
        )
        for index, tile in enumerate(selected):
            sheet.paste(tile, (0, 28 + index * 290))
        path = output_dir / f"visual_review_part_{part}.png"
        sheet.save(path, "PNG")
        evidence.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
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
        raise FileExistsError("BK8 confirmation is create-only")
    output_dir.mkdir(parents=True)
    artifact = compile_standalone_profile_artifact(
        root=root, config=validated["ao6_config"]
    )
    records: list[dict[str, Any]] = []
    for row in validated["source_rows"]:
        source_path = root / str(row["decoded_path"])
        if sha256_file(source_path) != row["decoded_sha256"]:
            raise LogChromaFreshComparisonError("source pixel drift")
        with Image.open(source_path) as opened:
            source = np.asarray(
                ImageOps.exif_transpose(opened).convert("RGB"),
                dtype=np.float32,
            )
        source /= 255.0
        outputs = render_fixed_arms(
            source, validated=validated, ao6_artifact=artifact
        )
        for arm_id in ARMS:
            path = output_dir / "renders" / arm_id / f"{row['id']}.png"
            save_srgb16_png(outputs[arm_id], path)
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if (
                decoded is None
                or decoded.dtype != np.uint16
                or decoded.shape != source.shape
            ):
                raise LogChromaFreshComparisonError(
                    "RGB16 PNG verification failed"
                )
            records.append(
                {
                    "source_id": row["id"],
                    "make": row["make"],
                    "arm_id": arm_id,
                    "source_decoded_sha256": row["decoded_sha256"],
                    "output": path.relative_to(output_dir).as_posix(),
                    "output_sha256": sha256_file(path),
                    "median_style_delta_e76": _style_delta_e76(
                        source, outputs[arm_id]
                    ),
                    **_boundary_metrics(source, outputs[arm_id]),
                }
            )
    medians = {
        arm: float(
            np.median(
                [
                    row["median_style_delta_e76"]
                    for row in records
                    if row["arm_id"] == arm
                ]
            )
        )
        for arm in ARMS
    }
    maximum_new_boundary = max(
        row["new_code_boundary_fraction_vs_source"] for row in records
    )
    gate = config["automatic_gate"]
    automatic_gates = {
        "complete_outputs": len(records) == int(gate["expected_outputs"]),
        "bk7_style_salience": medians[ARMS[0]]
        >= float(gate["minimum_bk7_population_median_style_delta_e76"]),
        "new_boundary": maximum_new_boundary
        <= float(
            gate[
                "maximum_per_output_new_code_boundary_fraction_vs_source"
            ]
        ),
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
        "maximum_new_code_boundary_fraction_vs_source": maximum_new_boundary,
        "automatic_gates": automatic_gates,
        "automatic_pass": all(automatic_gates.values()),
        "visual_review_allowed": all(automatic_gates.values()),
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sheets = (
        _build_contact_sheets(
            root=root,
            rows=validated["source_rows"],
            output_dir=output_dir,
        )
        if report["visual_review_allowed"]
        else []
    )
    return {
        "report": report,
        "report_path": report_path,
        "report_sha256": sha256_file(report_path),
        "visual_sheets": sheets,
    }


__all__ = [
    "ARMS",
    "render_fixed_arms",
    "run_confirmation",
    "validate_contract",
]
