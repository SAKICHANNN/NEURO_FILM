"""Fixed BL1 versus AO6 on the bounded BH1 CC0 RAW population."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.eval.filmmatch_identity_residual_validation import _operator_from_report
from src.eval.fixed_global_policy_confirmation import (
    FixedGlobalPolicyError,
    render_fixed_pair,
    validate_contract,
)
from src.eval.fresh_native_standard_confirmation import boundary_metrics
from src.eval.global_frontier import sha256_file
from src.film_physics.profile_consumer import compile_standalone_profile_artifact
from src.preprocess import save_srgb16_png
from src.preprocess.raw_decode import load_raw_working_image


SCHEMA = "neuro_film.u5_r2bl3_filmmatch_identity_residual_ood_render.v1"
ARMS = (
    "fixed_ao6_colour_only_t15_c35",
    "fixed_bl1_identity_residual_sigmoid",
)


class IdentityResidualOODError(RuntimeError):
    """Raised when a fixed BL3 identity or execution invariant drifts."""


def _load_exact_json(root: Path, path: str, expected: str) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise IdentityResidualOODError(f"hash mismatch: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IdentityResidualOODError(f"expected JSON object: {path}")
    return payload


def validate_bl3_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rendering = config["rendering"]
    if (
        config.get("schema")
        != "neuro_film.u5_r2bl3_filmmatch_identity_residual_ood.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or tuple(config["fixed_arms"]) != ARMS
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or rendering["per_image_fit_allowed"]
        or rendering["operator_refit_allowed"]
        or rendering["strength_retuning_allowed"]
        or rendering["routing_allowed"]
        or rendering["hard_clipping_allowed"]
    ):
        raise IdentityResidualOODError("BL3 frozen contract drift")

    candidate = config["candidate"]
    bl1 = _load_exact_json(
        root, candidate["bl1_report"], candidate["bl1_report_sha256"]
    )
    bl2 = _load_exact_json(
        root, candidate["bl2_decision"], candidate["bl2_decision_sha256"]
    )
    if (
        bl1["stable_evidence_id"] != candidate["bl1_stable_evidence_id"]
        or not bl1["automatic_gate_passed"]
        or bl2["decision"]["status"] != candidate["required_bl2_status"]
    ):
        raise IdentityResidualOODError("BL1/BL2 candidate gate drift")

    population = config["population"]
    bh1_config = _load_exact_json(
        root, population["bh1_contract"], population["bh1_contract_sha256"]
    )
    bh1_decision = _load_exact_json(
        root, population["bh1_decision"], population["bh1_decision_sha256"]
    )
    validated = validate_contract(root, bh1_config)
    if (
        len(validated["eligible_ids"]) != population["expected_sources"]
        or len(
            {
                validated["source_rows"][key]["make"]
                for key in validated["eligible_ids"]
            }
        )
        != population["expected_camera_makes"]
        or bh1_decision["next_branch"]
        != "retain_ao6_and_continue_distinct_algorithm_leaf"
    ):
        raise IdentityResidualOODError("BH1 population/provenance drift")
    return {**validated, "bl1_report": bl1}


def _encode_scene_linear(scene_linear: np.ndarray, rows: int = 128) -> np.ndarray:
    encoded = np.empty_like(scene_linear, dtype=np.float32)
    for y0 in range(0, scene_linear.shape[0], rows):
        y1 = min(scene_linear.shape[0], y0 + rows)
        encoded[y0:y1] = linear_srgb_to_encoded(
            np.asarray(scene_linear[y0:y1], dtype=np.float64)
        ).astype(np.float32)
    return encoded


def _apply_bl1(operator: Any, encoded: np.ndarray, rows: int = 128) -> np.ndarray:
    output = np.empty_like(encoded, dtype=np.float32)
    for y0 in range(0, encoded.shape[0], rows):
        y1 = min(encoded.shape[0], y0 + rows)
        output[y0:y1] = operator.apply(
            np.asarray(encoded[y0:y1], dtype=np.float64)
        ).astype(np.float32)
    return output


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def run_bl3_render(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_bl3_contract(root, config)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise IdentityResidualOODError("BL3 requires OMP_NUM_THREADS=1")
    if output_dir.exists():
        raise FileExistsError("BL3 render is create-only")
    output_dir.mkdir(parents=True)

    operator = _operator_from_report(validated["bl1_report"])
    artifact = compile_standalone_profile_artifact(
        root=root, config=validated["compiler_config"]
    )
    rows: list[dict[str, Any]] = []
    bl1_style_values: list[float] = []
    for source_id in validated["eligible_ids"]:
        source = validated["source_rows"][source_id]
        raw_path = root / source["raw_path"]
        if sha256_file(raw_path) != source["raw_sha256"]:
            raise IdentityResidualOODError("live RAW identity drift")
        working = load_raw_working_image(raw_path)
        if (
            working.transfer_state != "scene_linear"
            or working.working_space not in {"linear_srgb", "linear_srgb_d65"}
            or not working.orientation_applied
        ):
            raise IdentityResidualOODError("WorkingImage contract drift")
        fixed = render_fixed_pair(
            working.pixels, artifact, validated["component"]
        )
        encoded = _encode_scene_linear(working.pixels)
        outputs = {
            ARMS[0]: fixed[ARMS[0]],
            ARMS[1]: _apply_bl1(operator, encoded),
        }
        ao6 = outputs[ARMS[0]]
        for arm_id in ARMS:
            values = outputs[arm_id]
            if (
                values.shape != encoded.shape
                or values.dtype != np.float32
                or not np.all(np.isfinite(values))
                or np.any(values < 0.0)
                or np.any(values > 1.0)
            ):
                raise IdentityResidualOODError(f"{arm_id} left display RGB")
            path = output_dir / "renders" / arm_id / f"{source_id}.png"
            save_srgb16_png(values, path)
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.dtype != np.uint16:
                raise IdentityResidualOODError("PNG16 verification failed")
            style = float(np.sqrt(np.mean(np.square(values - encoded))))
            if arm_id == ARMS[1]:
                bl1_style_values.append(style)
            rows.append(
                {
                    "source_id": source_id,
                    "source_raw_sha256": source["raw_sha256"],
                    "make": source["make"],
                    "arm_id": arm_id,
                    "output": path.relative_to(output_dir).as_posix(),
                    "output_sha256": sha256_file(path),
                    "style_rgb_rmse_from_source": style,
                    **boundary_metrics(values, ao6),
                }
            )
        del working, fixed, encoded, outputs, ao6

    gate = config["automatic_gate"]
    maximum_boundary = max(
        row["output_code_boundary_fraction"] for row in rows
    )
    maximum_new_boundary = max(
        row["new_boundary_fraction_vs_ao6"]
        for row in rows
        if row["arm_id"] == ARMS[1]
    )
    median_style = float(np.median(bl1_style_values))
    automatic_pass = bool(
        len(rows) == gate["expected_outputs"]
        and maximum_boundary <= gate["maximum_output_code_boundary_fraction"]
        and maximum_new_boundary
        <= gate["maximum_new_boundary_fraction_vs_ao6"]
        and median_style
        >= gate["minimum_median_bl1_style_rgb_rmse_from_source"]
    )
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "source_count": len(validated["eligible_ids"]),
        "arms": list(ARMS),
        "rows": rows,
        "maximum_output_code_boundary_fraction": maximum_boundary,
        "maximum_new_boundary_fraction_vs_ao6": maximum_new_boundary,
        "median_bl1_style_rgb_rmse_from_source": median_style,
        "automatic_gate_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_bl3_blind_round(
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
        order = list(ARMS)
        random.Random(
            hashlib.sha256(
                f"u5-r2bl3:{round_index}:{source_id}".encode()
            ).digest()
        ).shuffle(order)
        mapping.append({"source_id": source_id, "A": order[0], "B": order[1]})
        with Image.open(root / source_rows[source_id]["decoded_path"]) as image:
            original = image.convert("RGB")
            original.thumbnail((520, 330), Image.Resampling.LANCZOS)
        candidates: list[Image.Image] = []
        for arm_id in order:
            with Image.open(
                render_dir / "renders" / arm_id / f"{source_id}.png"
            ) as image:
                candidate = image.convert("RGB")
                candidate.thumbnail((520, 330), Image.Resampling.LANCZOS)
            candidates.append(candidate)
        tile = Image.new("RGB", (1580, 375), "white")
        draw = ImageDraw.Draw(tile)
        for index, (label, image) in enumerate(
            zip(("Source", "A", "B"), (original, *candidates))
        ):
            x = 5 + index * 525
            draw.text((x, 3), label, fill="black", font=font)
            tile.paste(image, (x, 23))
        draw.text((5, 357), source_id, fill="black", font=font)
        tiles.append(tile)

    output_dir.mkdir(parents=True, exist_ok=True)
    part_hashes: list[str] = []
    for part_index, start in enumerate((0, 4, 8), start=1):
        selected = tiles[start : start + 4]
        sheet = Image.new("RGB", (1580, len(selected) * 375 + 28), "white")
        ImageDraw.Draw(sheet).text(
            (5, 5),
            f"U5.R2BL3 round {round_index} part {part_index}",
            fill="black",
            font=font,
        )
        for index, tile in enumerate(selected):
            sheet.paste(tile, (0, 28 + index * 375))
        path = output_dir / f"blind_round_{round_index}_part_{part_index}.png"
        sheet.save(path, "PNG")
        part_hashes.append(sha256_file(path))
    mapping_path = output_dir / f"blind_round_{round_index}_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "part_sha256": part_hashes,
        "mapping_path": mapping_path.as_posix(),
        "mapping_sha256": sha256_file(mapping_path),
    }


__all__ = [
    "ARMS",
    "IdentityResidualOODError",
    "build_bl3_blind_round",
    "run_bl3_render",
    "validate_bl3_contract",
]
