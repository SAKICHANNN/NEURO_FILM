"""Complete-ranking evaluator Oracle over five fixed explicit look arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.eval.b0_real_film_residual_fresh_confirmation import (
    validate_contract as validate_ao7_contract,
)
from src.eval.fresh_native_standard_confirmation import (
    boundary_metrics,
    load_confirmation_working_image,
)
from src.film_physics.display_look import (
    build_source_context_display_look_row_stages,
)
from src.eval.global_frontier import sha256_file
from src.preprocess import save_srgb16_png
from src.roll2film.density_residual_guard import (
    apply_density_residual_guard,
)
from src.roll2film.factorized_boundary_guard import (
    apply_factorized_boundary_guard,
)
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.b0_real_film_residual_fresh_confirmation import (
    validate_contract as validate_ap4_contract,
)


SCHEMA = "neuro_film.u5_r2bh0_fixed_bank_render_report.v1"
ARMS = (
    "fixed_b0",
    "fixed_ao6_colour_only_t15_c35",
    "fixed_ap3_ektachrome_composition",
    "fixed_az0_optical_density_residual",
    "fixed_native_standard_full_strength_1_0",
)


class FixedBankOracleError(RuntimeError):
    """Raised when a frozen identity or fixed-bank invariant drifts."""


def _load_exact_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise FixedBankOracleError(f"hash mismatch: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FixedBankOracleError(f"expected JSON object: {path}")
    return payload


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        config.get("schema")
        != "neuro_film.u5_r2bh0_fixed_bank_complete_oracle.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or tuple(row["arm_id"] for row in config["fixed_arms"]) != ARMS
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("selector_training_allowed")
        or config.get("production_default_changed")
        or not config["rendering"]["create_only"]
        or config["rendering"]["per_image_fit_allowed"]
        or config["rendering"]["strength_retuning_allowed"]
        or config["rendering"]["hard_clipping_allowed"]
    ):
        raise FixedBankOracleError("BH0 frozen contract drift")

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
        raise FixedBankOracleError("BH0S evidence drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        decision["result"]["decision"] != source["required_decision"]
        or not decision["result"]["automatic_pass"]
        or not decision["result"]["visual_pass"]
        or review["confirmed_severe_source_artifact_count"] != 0
        or len(review["eligible_ids"]) != source["expected_eligible_rows"]
    ):
        raise FixedBankOracleError("BH0S source gate is not open")
    rows = {str(row["id"]): dict(row) for row in manifest}
    eligible_ids = [str(value) for value in review["eligible_ids"]]
    if (
        len(rows) != len(manifest)
        or not set(eligible_ids).issubset(rows)
        or len({rows[key]["make"] for key in eligible_ids})
        != source["expected_camera_makes"]
    ):
        raise FixedBankOracleError("BH0S eligible population drift")

    provenance = config["arm_provenance"]
    ao7_spec = provenance["ao6_and_b0"]
    ao7_config = _load_exact_json(
        root, ao7_spec["config"], ao7_spec["sha256"]
    )
    ao7 = validate_ao7_contract(root, ao7_config)

    ap3_spec = provenance["ap3"]
    ap4_config = _load_exact_json(
        root, ap3_spec["config"], ap3_spec["config_sha256"]
    )
    ap4_decision = _load_exact_json(
        root, ap3_spec["decision"], ap3_spec["decision_sha256"]
    )
    if (
        ap4_decision["automatic_evidence"]["all_gates_pass"] is not True
        or ap4_decision["visual_evidence"][
            "full_resolution_confirmed_severe_artifact_count"
        ]
        != 0
        or float(ap3_spec["tone_strength"]) != 0.1
        or float(ap3_spec["chroma_strength"]) != 0.25
    ):
        raise FixedBankOracleError("AP3/AP4 fixed-arm evidence drift")
    ap4 = validate_ap4_contract(root, ap4_config)

    az0_spec = provenance["az0"]
    az0_config = _load_exact_json(
        root, az0_spec["config"], az0_spec["config_sha256"]
    )
    az0_decision = _load_exact_json(
        root, az0_spec["decision"], az0_spec["decision_sha256"]
    )
    if (
        az0_decision["decision"]
        != "retain_density_factorization_as_development_look_approximation_champion"
        or az0_decision["visual_adjudication"][
            "confirmed_severe_artifact_count"
        ]
        != 0
        or float(az0_spec["neutral_strength"]) != 0.15
        or float(az0_spec["opponent_strength"]) != 0.35
    ):
        raise FixedBankOracleError("AZ0 fixed-arm evidence drift")
    az0_candidate = az0_config["candidate"]

    native = provenance["native_standard"]
    p8bp = _load_exact_json(
        root,
        native["comparison_config"],
        native["comparison_config_sha256"],
    )
    build_config = _load_exact_json(
        root, native["build_config"], native["build_config_sha256"]
    )
    if p8bp["comparison"]["arms"] != [
        ARMS[0],
        ARMS[1],
        ARMS[4],
    ]:
        raise FixedBankOracleError("native Standard fixed identity drift")

    protocol = config["ranking_protocol"]
    if (
        protocol["rounds"] != 3
        or protocol["sources_per_round"] != len(eligible_ids)
        or protocol["arms_per_source"] != len(ARMS)
        or not protocol["strict_complete_ranking_required"]
        or protocol["ties_allowed"]
    ):
        raise FixedBankOracleError("complete-ranking protocol drift")
    return {
        "eligible_ids": eligible_ids,
        "source_rows": rows,
        "source_contract": _load_exact_json(
            root,
            decision["contract"]["path"],
            decision["contract"]["sha256"],
        ),
        "ao6_operator": ao7["operator"],
        "ap3_operator": ap4["operator"],
        "ap3_config": ap4_config,
        "az0_candidate": az0_candidate,
        "build_config": build_config,
    }


def render_fixed_bank(
    *,
    scene_linear: np.ndarray,
    artifact: dict[str, Any],
    runtime: Any,
    ap3_operator: Any,
    ao6_operator: Any,
    ap3_config: Mapping[str, Any],
    az0_candidate: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = np.ascontiguousarray(scene_linear, dtype=np.float32)
    row_chunk = 128
    encoded = np.empty_like(source)
    for y0 in range(0, source.shape[0], row_chunk):
        y1 = min(source.shape[0], y0 + row_chunk)
        encoded[y0:y1] = linear_srgb_to_encoded(
            np.asarray(source[y0:y1], dtype=np.float64)
        ).astype(np.float32)
    payload = artifact["component_payloads"][
        "ao6-source-context-display-look"
    ]
    apply_base_rows, apply_residual_rows = (
        build_source_context_display_look_row_stages(
            payload,
            encoded,
            tile_rows=row_chunk,
        )
    )
    base = np.empty_like(source)
    ao6 = np.empty_like(source)
    for y0 in range(0, source.shape[0], row_chunk):
        y1 = min(source.shape[0], y0 + row_chunk)
        base[y0:y1] = apply_base_rows(encoded[y0:y1])
        ao6[y0:y1] = apply_residual_rows(base[y0:y1])
    del encoded
    native = np.empty_like(source)

    def sink(y0: int, y1: int, rows: np.ndarray) -> None:
        native[y0:y1] = rows

    receipt = runtime.render_to_sink(source, output_sink=sink)
    if (
        hashlib.sha256(native.tobytes()).hexdigest()
        != receipt["output"]["array_sha256"]
    ):
        raise FixedBankOracleError("native Standard receipt drift")
    ap3_spec = ap3_config["fixed_candidate"]
    ap3_controls = ap3_spec["factorization"]
    ap3 = np.empty_like(base)
    az0 = np.empty_like(base)
    row_chunk = 128
    pixel_count = base.shape[0] * base.shape[1]
    ap3_tone_limited = 0
    ap3_chroma_limited = 0
    az0_neutral_limited = 0
    az0_opponent_limited = 0
    for y0 in range(0, base.shape[0], row_chunk):
        y1 = min(base.shape[0], y0 + row_chunk)
        base_linear = encoded_srgb_to_linear(
            np.asarray(base[y0:y1], dtype=np.float64)
        )
        ap3_guarded = apply_factorized_boundary_guard(
            ap3_operator,
            base_linear,
            tone_strength=float(ap3_spec["tone_strength"]),
            chroma_strength=float(ap3_spec["chroma_strength"]),
            luma_weights=np.asarray(ap3_controls["luma_weights"]),
            hard_boundary_epsilon_encoded_srgb=float(
                ap3_controls["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                ap3_controls["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        az0_guarded = apply_density_residual_guard(
            ao6_operator,
            base_linear,
            neutral_strength=float(az0_candidate["neutral_strength"]),
            opponent_strength=float(az0_candidate["opponent_strength"]),
            neutral_weights=np.asarray(az0_candidate["neutral_weights"]),
            density_floor=float(az0_candidate["density_floor"]),
            hard_boundary_epsilon_encoded_srgb=float(
                az0_candidate["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                az0_candidate["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        ap3[y0:y1] = linear_srgb_to_encoded(
            ap3_guarded.output
        ).astype(np.float32)
        az0[y0:y1] = linear_srgb_to_encoded(
            az0_guarded.output
        ).astype(np.float32)
        ap3_tone_limited += int(
            np.count_nonzero(ap3_guarded.tone_scale < 1.0 - 1e-12)
        )
        ap3_chroma_limited += int(
            np.count_nonzero(ap3_guarded.chroma_scale < 1.0 - 1e-12)
        )
        az0_neutral_limited += int(
            np.count_nonzero(az0_guarded.neutral_scale < 1.0 - 1e-12)
        )
        az0_opponent_limited += int(
            np.count_nonzero(az0_guarded.opponent_scale < 1.0 - 1e-12)
        )
    outputs = {
        ARMS[0]: base,
        ARMS[1]: ao6,
        ARMS[2]: ap3,
        ARMS[3]: az0,
        ARMS[4]: native,
    }
    source_shape = np.asarray(scene_linear).shape
    for arm_id, values in outputs.items():
        if (
            values.shape != source_shape
            or values.dtype != np.float32
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise FixedBankOracleError(f"{arm_id} left display RGB")
    diagnostics = {
        "native_receipt_sha256": receipt["receipt_sha256"],
        "residual_row_chunk": row_chunk,
        "ap3_tone_limited_fraction": ap3_tone_limited / pixel_count,
        "ap3_chroma_limited_fraction": ap3_chroma_limited / pixel_count,
        "az0_neutral_limited_fraction": az0_neutral_limited / pixel_count,
        "az0_opponent_limited_fraction": az0_opponent_limited / pixel_count,
    }
    return outputs, diagnostics


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
    build_components: Any,
    patch_runtime: Any,
    compile_artifact: Any,
    create_runtime: Any,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    if output_dir.exists():
        raise FileExistsError("BH0 render is create-only")
    output_dir.mkdir(parents=True)
    patch_runtime()
    build_config = validated["build_config"]
    builds = build_components(build_config, output_dir / "binaries")
    package = json.loads((root / build_config["package"]).read_text())
    artifact = compile_artifact(
        root=root,
        config=json.loads(
            (root / build_config["profile_compiler_config"]).read_text()
        ),
    )
    runtime, factory = create_runtime(
        package=package,
        artifact=artifact,
        library_paths={
            name: Path(row["dll_path"]) for name, row in builds.items()
        },
    )
    source_contract = validated["source_contract"]
    candidate_by_id = {
        str(row["id"]): dict(row) for row in source_contract["candidates"]
    }
    rows: list[dict[str, Any]] = []
    for source_id in validated["eligible_ids"]:
        source = validated["source_rows"][source_id]
        candidate = candidate_by_id[source_id]
        raw_path = root / candidate["path"]
        if (
            sha256_file(raw_path) != candidate["sha256"]
            or source["raw_sha256"] != candidate["sha256"]
        ):
            raise FixedBankOracleError("live RAW identity drift")
        working = load_confirmation_working_image(raw_path)
        if (
            working.transfer_state != "scene_linear"
            or working.working_space not in {"linear_srgb", "linear_srgb_d65"}
            or not working.orientation_applied
        ):
            raise FixedBankOracleError("WorkingImage contract drift")
        outputs, diagnostics = render_fixed_bank(
            scene_linear=working.pixels,
            artifact=artifact,
            runtime=runtime,
            ap3_operator=validated["ap3_operator"],
            ao6_operator=validated["ao6_operator"],
            ap3_config=validated["ap3_config"],
            az0_candidate=validated["az0_candidate"],
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
                raise FixedBankOracleError("PNG16 verification failed")
            row = {
                "source_id": source_id,
                "source_raw_sha256": candidate["sha256"],
                "make": source["make"],
                "arm_id": arm_id,
                "output": path.relative_to(output_dir).as_posix(),
                "output_sha256": sha256_file(path),
                **boundary_metrics(outputs[arm_id], ao6),
            }
            if arm_id == ARMS[4]:
                row.update(diagnostics)
            rows.append(row)
    gates = config["automatic_gate"]
    maximum_boundary = max(
        row["output_code_boundary_fraction"] for row in rows
    )
    maximum_new_boundary = max(
        row["new_boundary_fraction_vs_ao6"]
        for row in rows
        if row["arm_id"] != ARMS[1]
    )
    automatic_pass = (
        len(rows) == int(gates["expected_outputs"])
        and maximum_boundary
        <= float(gates["maximum_output_code_boundary_fraction"])
        and maximum_new_boundary
        <= float(gates["maximum_new_boundary_fraction_vs_ao6"])
    )
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "source_count": len(validated["eligible_ids"]),
        "arms": list(ARMS),
        "factory_receipt_sha256": factory["receipt_sha256"],
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
    path = output_dir / "report.json"
    path.write_text(
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
    labels = ("A", "B", "C", "D", "E")
    mapping: list[dict[str, Any]] = []
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    for source_id in eligible_ids:
        order = list(ARMS)
        random.Random(
            hashlib.sha256(
                f"u5-r2bh0:{round_index}:{source_id}".encode()
            ).digest()
        ).shuffle(order)
        mapping.append(
            {"source_id": source_id, **dict(zip(labels, order))}
        )
        with Image.open(root / source_rows[source_id]["decoded_path"]) as image:
            original = image.convert("RGB")
            original.thumbnail((340, 220), Image.Resampling.LANCZOS)
        candidates: list[Image.Image] = []
        for arm_id in order:
            with Image.open(
                render_dir / "renders" / arm_id / f"{source_id}.png"
            ) as image:
                candidate = image.convert("RGB")
                candidate.thumbnail((340, 220), Image.Resampling.LANCZOS)
            candidates.append(candidate)
        tile = Image.new("RGB", (2070, 260), "white")
        draw = ImageDraw.Draw(tile)
        for index, (label, image) in enumerate(
            zip(("Source", *labels), (original, *candidates))
        ):
            x = 5 + index * 345
            draw.text((x, 3), label, fill="black", font=font)
            tile.paste(image, (x, 23))
        draw.text((5, 244), source_id, fill="black", font=font)
        tiles.append(tile)

    output_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    for part_index, start in enumerate((0, 5), start=1):
        selected = tiles[start : start + 5]
        sheet = Image.new("RGB", (2070, len(selected) * 260 + 28), "white")
        ImageDraw.Draw(sheet).text(
            (5, 5),
            f"U5.R2BH0 complete ranking round {round_index} part {part_index}",
            fill="black",
            font=font,
        )
        for index, tile in enumerate(selected):
            sheet.paste(tile, (0, 28 + index * 260))
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
    "FixedBankOracleError",
    "build_blind_round",
    "render_fixed_bank",
    "run_render",
    "validate_contract",
]
