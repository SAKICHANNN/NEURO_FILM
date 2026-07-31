"""Fresh-population confirmation of fixed BL5 strict-interior versus AO6."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.eval.filmmatch_identity_residual_ood import _encode_scene_linear
from src.eval.filmmatch_strict_interior_ood import _apply_rows, _operator_from_report
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fresh_native_standard_confirmation import boundary_metrics
from src.eval.global_frontier import sha256_file
from src.film_physics.profile_consumer import compile_standalone_profile_artifact
from src.preprocess import save_srgb16_png
from src.preprocess.raw_decode import load_raw_working_image


SCHEMA = "neuro_film.u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_report.v1"
AO6_ARM = "fixed_ao6_colour_only_t15_c35"
CANDIDATE_ARM = "fixed_bl5_strict_interior_sigmoid"


class FreshStrictInteriorError(RuntimeError):
    """Raised when a frozen BL8 identity or execution invariant drifts."""


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise FreshStrictInteriorError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rendering = config["rendering"]
    if (
        config.get("schema")
        != "neuro_film.u5_r2bl8_filmmatch_strict_interior_fresh_confirmation.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("production_default_changed")
        or not rendering["create_only"]
        or rendering["per_image_fit_allowed"]
        or rendering["operator_refit_allowed"]
        or rendering["strength_retuning_allowed"]
        or rendering["routing_allowed"]
        or rendering["hard_clipping_allowed"]
    ):
        raise FreshStrictInteriorError("BL8 frozen contract drift")

    source = config["source_preflight"]
    decision = _load_exact_json(root, source["decision"], source["decision_sha256"])
    manifest = _load_exact_json(root, source["manifest"], source["manifest_sha256"])
    review = _load_exact_json(root, source["visual_review"], source["visual_review_sha256"])
    eligible_ids = [str(value) for value in review["eligible_ids"]]
    rows = {str(row["id"]): dict(row) for row in manifest}
    if (
        decision["decision"] != source["required_decision"]
        or not decision["automatic_pass"]
        or not decision["visual_pass"]
        or review["confirmed_severe_source_artifact_count"] != 0
        or len(eligible_ids) != source["expected_eligible_rows"]
        or len(rows) != len(manifest)
        or not set(eligible_ids).issubset(rows)
        or len({rows[key]["make"] for key in eligible_ids}) != source["expected_camera_makes"]
    ):
        raise FreshStrictInteriorError("BL8S source gate is not open")

    candidate = config["candidate"]
    operator_report = _load_exact_json(root, candidate["bl5_report"], candidate["bl5_report_sha256"])
    candidate_decision = _load_exact_json(root, candidate["bl6_decision"], candidate["bl6_decision_sha256"])
    if (
        operator_report["stable_evidence_id"] != candidate["bl5_stable_evidence_id"]
        or candidate_decision["decision"] != candidate["required_bl6_decision"]
    ):
        raise FreshStrictInteriorError("BL5/BL6 candidate provenance drift")

    ao6 = config["ao6_provenance"]
    compiler_config = _load_exact_json(root, ao6["profile_compiler_config"], ao6["profile_compiler_config_sha256"])
    protocol = config["blind_protocol"]
    if (
        protocol["rounds"] != 3
        or protocol["sources_per_round"] != len(eligible_ids)
        or protocol["aggregate_choice_denominator"] != 3 * len(eligible_ids)
        or protocol["minimum_candidate_round_wins"] != 2
        or protocol["ties_allowed"]
    ):
        raise FreshStrictInteriorError("BL8 blind protocol drift")
    return {
        "eligible_ids": eligible_ids,
        "source_rows": rows,
        "operator_report": operator_report,
        "compiler_config": compiler_config,
        "component": ao6["source_context_component"],
    }


def run_confirmation(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise FreshStrictInteriorError("BL8 requires OMP_NUM_THREADS=1")
    if output_dir.exists():
        raise FileExistsError("BL8 output is create-only")
    output_dir.mkdir(parents=True)
    artifact = compile_standalone_profile_artifact(root=root, config=validated["compiler_config"])
    operator = _operator_from_report(validated["operator_report"])
    rows: list[dict[str, Any]] = []
    styles: list[float] = []
    for source_id in validated["eligible_ids"]:
        source = validated["source_rows"][source_id]
        raw_path = root / source["raw_path"]
        if sha256_file(raw_path) != source["raw_sha256"]:
            raise FreshStrictInteriorError("live RAW identity drift")
        working = load_raw_working_image(raw_path)
        if (
            working.transfer_state != "scene_linear"
            or working.working_space not in {"linear_srgb", "linear_srgb_d65"}
            or not working.orientation_applied
        ):
            raise FreshStrictInteriorError("WorkingImage contract drift")
        encoded = _encode_scene_linear(working.pixels)
        candidate = _apply_rows(operator, encoded)
        ao6 = render_fixed_pair(working.pixels, artifact, validated["component"])[AO6_ARM]
        style = float(np.sqrt(np.mean(np.square(candidate - encoded))))
        styles.append(style)
        for arm_id, values in ((AO6_ARM, ao6), (CANDIDATE_ARM, candidate)):
            path = output_dir / "renders" / arm_id / f"{source_id}.png"
            save_srgb16_png(values, path)
            rows.append(
                {
                    "source_id": source_id,
                    "source_raw_sha256": source["raw_sha256"],
                    "make": source["make"],
                    "arm_id": arm_id,
                    "output": path.relative_to(output_dir).as_posix(),
                    "output_sha256": sha256_file(path),
                    "style_rgb_rmse_from_source": style if arm_id == CANDIDATE_ARM else None,
                    **boundary_metrics(values, ao6),
                }
            )
        del working, encoded, candidate, ao6
    gate = config["automatic_gate"]
    candidate_rows = [row for row in rows if row["arm_id"] == CANDIDATE_ARM]
    aggregate = {
        "source_count": len(validated["eligible_ids"]),
        "output_count": len(rows),
        "maximum_candidate_output_code_boundary_fraction": max(row["output_code_boundary_fraction"] for row in candidate_rows),
        "maximum_candidate_new_boundary_fraction_vs_ao6": max(row["new_boundary_fraction_vs_ao6"] for row in candidate_rows),
        "median_candidate_style_rgb_rmse_from_source": float(np.median(styles)),
    }
    passed = bool(
        aggregate["source_count"] == gate["expected_sources"]
        and aggregate["output_count"] == gate["expected_outputs"]
        and aggregate["maximum_candidate_output_code_boundary_fraction"] <= gate["maximum_output_code_boundary_fraction"]
        and aggregate["maximum_candidate_new_boundary_fraction_vs_ao6"] <= gate["maximum_new_boundary_fraction_vs_ao6"]
        and aggregate["median_candidate_style_rgb_rmse_from_source"] >= gate["minimum_median_style_rgb_rmse_from_source"]
    )
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "operator_stable_evidence_id": validated["operator_report"]["stable_evidence_id"],
        "bundle_sha256": artifact["bundle_sha256"],
        "rows": rows,
        "aggregate": aggregate,
        "automatic_gate_pass": passed,
        "blind_review_allowed": passed,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_blind_round(
    *,
    root: Path,
    render_dir: Path,
    source_rows: Mapping[str, Mapping[str, Any]],
    eligible_ids: list[str],
    round_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    mapping: list[dict[str, str]] = []
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    for source_id in eligible_ids:
        order = [AO6_ARM, CANDIDATE_ARM]
        random.Random(hashlib.sha256(f"u5-r2bl8:{round_index}:{source_id}".encode()).digest()).shuffle(order)
        mapping.append({"source_id": source_id, "A": order[0], "B": order[1]})
        with Image.open(root / source_rows[source_id]["decoded_path"]) as image:
            source = image.convert("RGB")
            source.thumbnail((520, 330), Image.Resampling.LANCZOS)
        compared: list[Image.Image] = []
        for arm_id in order:
            with Image.open(render_dir / "renders" / arm_id / f"{source_id}.png") as image:
                rendered = image.convert("RGB")
                rendered.thumbnail((520, 330), Image.Resampling.LANCZOS)
            compared.append(rendered)
        tile = Image.new("RGB", (1580, 375), "white")
        draw = ImageDraw.Draw(tile)
        for index, (label, image) in enumerate(zip(("Source", "A", "B"), (source, *compared))):
            x = 5 + index * 525
            draw.text((x, 3), label, fill="black", font=font)
            tile.paste(image, (x, 23))
        draw.text((5, 357), source_id, fill="black", font=font)
        tiles.append(tile)
    output_dir.mkdir(parents=True, exist_ok=True)
    part_hashes: list[str] = []
    for part_index, start in enumerate(range(0, len(tiles), 4), start=1):
        selected = tiles[start : start + 4]
        sheet = Image.new("RGB", (1580, len(selected) * 375 + 28), "white")
        ImageDraw.Draw(sheet).text((5, 5), f"U5.R2BL8 round {round_index} part {part_index}", fill="black", font=font)
        for index, tile in enumerate(selected):
            sheet.paste(tile, (0, 28 + index * 375))
        path = output_dir / f"blind_round_{round_index}_part_{part_index}.png"
        sheet.save(path, "PNG")
        part_hashes.append(sha256_file(path))
    mapping_path = output_dir / f"blind_round_{round_index}_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"part_sha256": part_hashes, "mapping_sha256": sha256_file(mapping_path)}


__all__ = ["build_blind_round", "run_confirmation", "validate_contract"]
