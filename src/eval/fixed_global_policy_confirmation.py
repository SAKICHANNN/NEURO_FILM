"""Independent fixed B0-versus-AO6 global-policy confirmation."""

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
from src.eval.fresh_native_standard_confirmation import boundary_metrics
from src.eval.global_frontier import sha256_file
from src.film_physics.display_look import (
    build_source_context_display_look_row_stages,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)
from src.preprocess import save_srgb16_png
from src.preprocess.raw_decode import load_raw_working_image


SCHEMA = "neuro_film.u5_r2bh1_fixed_global_policy_render_report.v1"
ARMS = ("fixed_b0", "fixed_ao6_colour_only_t15_c35")


class FixedGlobalPolicyError(RuntimeError):
    """Raised when a frozen identity or two-arm invariant drifts."""


def _load_exact_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise FixedGlobalPolicyError(f"hash mismatch: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FixedGlobalPolicyError(f"expected JSON object: {path}")
    return payload


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rendering = config["rendering"]
    if (
        config.get("schema")
        != "neuro_film.u5_r2bh1_fixed_global_policy_confirmation.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or tuple(row["arm_id"] for row in config["fixed_arms"]) != ARMS
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("selector_training_allowed")
        or config.get("production_default_changed")
        or not rendering["create_only"]
        or rendering["per_image_fit_allowed"]
        or rendering["operator_refit_allowed"]
        or rendering["strength_retuning_allowed"]
        or rendering["routing_allowed"]
        or rendering["dense_blending_allowed"]
        or rendering["hard_clipping_allowed"]
    ):
        raise FixedGlobalPolicyError("BH1 frozen contract drift")

    source = config["source_preflight"]
    decision = _load_exact_json(
        root, source["decision"], source["decision_sha256"]
    )
    manifest_path = root / source["manifest"]
    review_path = root / source["visual_review"]
    if (
        sha256_file(manifest_path) != source["manifest_sha256"]
        or sha256_file(review_path) != source["visual_review_sha256"]
    ):
        raise FixedGlobalPolicyError("BH1S evidence drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    eligible_ids = [str(value) for value in review["eligible_ids"]]
    if (
        decision["result"]["decision"] != source["required_decision"]
        or not decision["result"]["automatic_pass"]
        or not decision["result"]["visual_pass"]
        or review["confirmed_severe_source_artifact_count"] != 0
        or len(eligible_ids) != source["expected_eligible_rows"]
    ):
        raise FixedGlobalPolicyError("BH1S source gate is not open")
    if not isinstance(manifest, list):
        raise FixedGlobalPolicyError("BH1S manifest schema drift")
    rows = {str(row["id"]): dict(row) for row in manifest}
    if (
        len(rows) != len(manifest)
        or not set(eligible_ids).issubset(rows)
        or len({rows[key]["make"] for key in eligible_ids})
        != source["expected_camera_makes"]
    ):
        raise FixedGlobalPolicyError("BH1S population drift")

    provenance = config["arm_provenance"]
    bh0_contract = _load_exact_json(
        root,
        provenance["bh0_contract"],
        provenance["bh0_contract_sha256"],
    )
    bh0_decision = _load_exact_json(
        root,
        provenance["bh0_decision"],
        provenance["bh0_decision_sha256"],
    )
    compiler_config = _load_exact_json(
        root,
        provenance["profile_compiler_config"],
        provenance["profile_compiler_config_sha256"],
    )
    if (
        tuple(row["arm_id"] for row in bh0_contract["fixed_arms"])[:2]
        != ARMS
        or bh0_decision["descriptive_all_round_oracle"][
            "selected_global_arm"
        ]
        != ARMS[0]
        or bh0_decision["status"]
        != "oracle_fail_close_selector_and_router"
    ):
        raise FixedGlobalPolicyError("BH0 fixed-arm provenance drift")

    protocol = config["blind_protocol"]
    if (
        protocol["rounds"] != 3
        or protocol["sources_per_round"] != len(eligible_ids)
        or protocol["arms_per_source"] != len(ARMS)
        or protocol["ties_allowed"]
        or protocol["minimum_b0_round_wins"] != 2
        or protocol["minimum_b0_aggregate_choices"] != 22
        or protocol["aggregate_choice_denominator"]
        != 3 * len(eligible_ids)
    ):
        raise FixedGlobalPolicyError("BH1 blind protocol drift")
    return {
        "eligible_ids": eligible_ids,
        "source_rows": rows,
        "compiler_config": compiler_config,
        "component": provenance["source_context_component"],
    }


def render_fixed_pair(
    scene_linear: np.ndarray,
    artifact: Mapping[str, Any],
    component: str,
) -> dict[str, np.ndarray]:
    source = np.ascontiguousarray(scene_linear, dtype=np.float32)
    encoded = np.empty_like(source)
    row_chunk = 128
    for y0 in range(0, source.shape[0], row_chunk):
        y1 = min(source.shape[0], y0 + row_chunk)
        encoded[y0:y1] = linear_srgb_to_encoded(
            np.asarray(source[y0:y1], dtype=np.float64)
        ).astype(np.float32)
    payload = artifact["component_payloads"][component]
    apply_base_rows, apply_residual_rows = (
        build_source_context_display_look_row_stages(
            payload, encoded, tile_rows=row_chunk
        )
    )
    base = np.empty_like(source)
    ao6 = np.empty_like(source)
    for y0 in range(0, source.shape[0], row_chunk):
        y1 = min(source.shape[0], y0 + row_chunk)
        base[y0:y1] = apply_base_rows(encoded[y0:y1])
        ao6[y0:y1] = apply_residual_rows(base[y0:y1])
    outputs = {ARMS[0]: base, ARMS[1]: ao6}
    for arm_id, values in outputs.items():
        if (
            values.shape != source.shape
            or values.dtype != np.float32
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise FixedGlobalPolicyError(f"{arm_id} left display RGB")
    return outputs


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def run_render(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise FixedGlobalPolicyError(
            "BH1 RAW decode requires OMP_NUM_THREADS=1 before rawpy import"
        )
    if output_dir.exists():
        raise FileExistsError("BH1 render is create-only")
    output_dir.mkdir(parents=True)
    artifact = compile_standalone_profile_artifact(
        root=root, config=validated["compiler_config"]
    )
    rows: list[dict[str, Any]] = []
    for source_id in validated["eligible_ids"]:
        source = validated["source_rows"][source_id]
        raw_path = root / source["raw_path"]
        if sha256_file(raw_path) != source["raw_sha256"]:
            raise FixedGlobalPolicyError("live RAW identity drift")
        working = load_raw_working_image(raw_path)
        if (
            working.transfer_state != "scene_linear"
            or working.working_space not in {"linear_srgb", "linear_srgb_d65"}
            or not working.orientation_applied
        ):
            raise FixedGlobalPolicyError("WorkingImage contract drift")
        input_array_sha256 = hashlib.sha256(
            working.pixels.tobytes()
        ).hexdigest()
        outputs = render_fixed_pair(
            working.pixels, artifact, validated["component"]
        )
        ao6 = outputs[ARMS[1]]
        for arm_id in ARMS:
            path = output_dir / "renders" / arm_id / f"{source_id}.png"
            save_srgb16_png(outputs[arm_id], path)
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if (
                decoded is None
                or decoded.dtype != np.uint16
                or decoded.shape != outputs[arm_id].shape
            ):
                raise FixedGlobalPolicyError("PNG16 verification failed")
            rows.append(
                {
                    "source_id": source_id,
                    "source_raw_sha256": source["raw_sha256"],
                    "input_array_sha256": input_array_sha256,
                    "make": source["make"],
                    "arm_id": arm_id,
                    "output": path.relative_to(output_dir).as_posix(),
                    "output_sha256": sha256_file(path),
                    **boundary_metrics(outputs[arm_id], ao6),
                }
            )
        del working, outputs, ao6
    gate = config["automatic_gate"]
    maximum_boundary = max(
        row["output_code_boundary_fraction"] for row in rows
    )
    maximum_new_boundary = max(
        row["new_boundary_fraction_vs_ao6"]
        for row in rows
        if row["arm_id"] != ARMS[1]
    )
    automatic_pass = (
        len(rows) == gate["expected_outputs"]
        and maximum_boundary
        <= gate["maximum_output_code_boundary_fraction"]
        and maximum_new_boundary
        <= gate["maximum_new_boundary_fraction_vs_ao6"]
    )
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "source_count": len(validated["eligible_ids"]),
        "arms": list(ARMS),
        "bundle_sha256": artifact["bundle_sha256"],
        "rows": rows,
        "maximum_output_code_boundary_fraction": maximum_boundary,
        "maximum_new_boundary_fraction_vs_ao6": maximum_new_boundary,
        "automatic_gate_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
        order = list(ARMS)
        random.Random(
            hashlib.sha256(
                f"u5-r2bh1:{round_index}:{source_id}".encode()
            ).digest()
        ).shuffle(order)
        mapping.append(
            {"source_id": source_id, "A": order[0], "B": order[1]}
        )
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
    part_paths: list[Path] = []
    for part_index, start in enumerate((0, 4, 8), start=1):
        selected = tiles[start : start + 4]
        sheet = Image.new("RGB", (1580, len(selected) * 375 + 28), "white")
        ImageDraw.Draw(sheet).text(
            (5, 5),
            f"U5.R2BH1 pair round {round_index} part {part_index}",
            fill="black",
            font=font,
        )
        for index, tile in enumerate(selected):
            sheet.paste(tile, (0, 28 + index * 375))
        path = output_dir / f"blind_round_{round_index}_part_{part_index}.png"
        sheet.save(path, "PNG")
        part_paths.append(path)
    mapping_path = output_dir / f"blind_round_{round_index}_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "parts": part_paths,
        "part_sha256": [sha256_file(path) for path in part_paths],
        "mapping_path": mapping_path,
        "mapping_sha256": sha256_file(mapping_path),
    }


__all__ = [
    "ARMS",
    "FixedGlobalPolicyError",
    "build_blind_round",
    "render_fixed_pair",
    "run_render",
    "validate_contract",
]
