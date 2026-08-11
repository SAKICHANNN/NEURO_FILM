"""CB13 locally available frozen gold/stress regression for fixed CB11."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _median_delta_e76,
    _new_boundary_fraction,
)
from src.eval.kci_velvia_tone_photographic_stress import _load_rgb

SCHEMA = "neuro_film.u5_r2cb13_characteristic_gold_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb13_characteristic_gold_stress_report.v1"
EXPERIMENT_ID = "U5.R2CB13"


class CharacteristicGoldStressError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicGoldStressError("CB13 contract structure drift")
    return payload


def _encode_png(linear: np.ndarray) -> tuple[bytes, np.ndarray]:
    encoded = linear_srgb_to_encoded(linear)
    if not np.isfinite(encoded).all() or np.min(encoded) < 0.0 or np.max(encoded) > 1.0:
        raise CharacteristicGoldStressError("encoded candidate escaped sRGB")
    quantized = np.rint(encoded * np.float32(255.0)).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(quantized, mode="RGB").save(stream, format="PNG", compress_level=6)
    return stream.getvalue(), quantized


def _encode_image(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def _build_sheet(
    rows: list[dict[str, Any]],
    thumbnails: Mapping[str, tuple[Image.Image, Image.Image]],
) -> bytes:
    cell_width, cell_height, header = 320, 220, 28
    canvas = Image.new(
        "RGB", (cell_width * 2, header + cell_height * len(rows)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "SOURCE", fill="black")
    draw.text((cell_width + 8, 8), "CB11", fill="black")
    for index, row in enumerate(rows):
        for column, image in enumerate(thumbnails[row["id"]]):
            tile = image.copy()
            tile.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            x = column * cell_width + (cell_width - tile.width) // 2
            y = header + index * cell_height + (cell_height - tile.height) // 2
            canvas.paste(tile, (x, y))
        draw.text(
            (4, header + index * cell_height + 4),
            row["id"],
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
    return _encode_image(canvas)


def _inputs(config: Mapping[str, Any], root: Path):
    parents = config["parents"]
    decision = _load_exact_json(
        root, parents["cb12_decision_path"], parents["cb12_decision_sha256"]
    )
    if decision.get("status") != parents["cb12_required_status"] or not decision.get(
        "pass"
    ):
        raise CharacteristicGoldStressError("CB12 did not open CB13")
    cb11 = _load_exact_json(
        root, parents["cb11_contract_path"], parents["cb11_contract_sha256"]
    )
    curve = _compiled_curve(load_cb6(root / cb11["parents"]["cb6_contract_path"]))
    population = config["population"]
    frozen = _load_exact_json(
        root, population["frozen_set_path"], population["frozen_set_sha256"]
    )["frozen_set"]["samples"]
    available: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for row in frozen:
        if row.get("availability") != "available" or row.get("split") not in {
            "gold",
            "stress",
        }:
            continue
        source = root / row["source_path"]
        if not source.exists():
            unavailable.append(row["id"])
            continue
        if hash_file(source) != row["source_sha256"]:
            raise CharacteristicGoldStressError(f"source hash drift: {row['id']}")
        available.append(row)
    if (
        sorted(unavailable) != sorted(population["required_unavailable_ids"])
        or len(available) != population["expected_available_source_count"]
        or sum(row["split"] == "gold" for row in available)
        != population["expected_available_gold_count"]
        or sum(row["split"] == "stress" for row in available)
        != population["expected_available_stress_count"]
    ):
        raise CharacteristicGoldStressError("local population boundary drift")
    return cb11, curve, available, unavailable


def evaluate(
    config: Mapping[str, Any], root: Path, output_dir: Path, *, persist_outputs: bool
) -> dict[str, Any]:
    cb11, curve, available, unavailable = _inputs(config, root)
    if persist_outputs and output_dir.exists():
        raise FileExistsError("CB13 output directory already exists")
    if persist_outputs:
        output_dir.mkdir(parents=True)
    operator = cb11["operator"]
    weights = np.asarray(operator["luminance_weights"], dtype=np.float64)
    strength = float(operator["nominal_strength"])
    epsilon = float(operator["boundary_epsilon"])
    thumbnails: dict[str, tuple[Image.Image, Image.Image]] = {}
    rows: list[dict[str, Any]] = []
    for source_row in available:
        source_path = root / source_row["source_path"]
        source = _load_rgb(
            source_path,
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        candidate, scale, luma_error = apply_characteristic_luma_chroma(
            source,
            curve,
            weights=weights,
            strength=strength,
            boundary_epsilon=epsilon,
        )
        candidate_bytes, candidate_u8 = _encode_png(candidate)
        output_rel = Path("renders") / f"{source_row['id']}.png"
        if persist_outputs:
            output_path = output_dir / output_rel
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(candidate_bytes)
        source_lab = linear_rgb_to_lab(source, working_space="linear_srgb")
        candidate_lab = linear_rgb_to_lab(candidate, working_space="linear_srgb")
        rows.append(
            {
                "id": source_row["id"],
                "split": source_row["split"],
                "source_sha256": source_row["source_sha256"],
                "output_path": output_rel.as_posix(),
                "output_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                "median_style_delta_e76": _median_delta_e76(source, candidate),
                "median_chroma_scale": float(np.median(scale)),
                "maximum_luminance_reconstruction_error": float(
                    np.max(np.abs(luma_error))
                ),
                "new_hard_boundary_fraction": _new_boundary_fraction(
                    source, candidate, epsilon
                ),
                "p999_gradient_ratio_vs_source": _gradient_p999_ratio(
                    source, candidate
                ),
                "adjacent_lstar_gradient_sign_inversion_fraction": _gradient_inversion_fraction(
                    source_lab[..., 0], candidate_lab[..., 0], epsilon=0.01
                ),
            }
        )
        with Image.open(source_path) as image:
            source_thumb = image.convert("RGB")
        thumbnails[source_row["id"]] = (
            source_thumb,
            Image.fromarray(candidate_u8, mode="RGB"),
        )

    sheets = []
    sheet_groups = [
        ("gold", [row for row in rows if row["split"] == "gold"]),
        ("stress_1", [row for row in rows if row["split"] == "stress"][:16]),
        ("stress_2", [row for row in rows if row["split"] == "stress"][16:]),
    ]
    for name, sheet_rows in sheet_groups:
        payload = _build_sheet(sheet_rows, thumbnails)
        relative = Path("contact_sheets") / f"{name}.png"
        if persist_outputs:
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        sheets.append(
            {
                "name": name,
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "row_ids": [row["id"] for row in sheet_rows],
            }
        )

    metrics = {
        "available_source_count": len(rows),
        "available_gold_count": sum(row["split"] == "gold" for row in rows),
        "available_stress_count": sum(row["split"] == "stress" for row in rows),
        "maximum_luminance_reconstruction_error": max(
            row["maximum_luminance_reconstruction_error"] for row in rows
        ),
        "maximum_new_hard_boundary_fraction": max(
            row["new_hard_boundary_fraction"] for row in rows
        ),
        "maximum_p999_gradient_ratio_vs_source": max(
            row["p999_gradient_ratio_vs_source"] for row in rows
        ),
        "maximum_adjacent_lstar_gradient_sign_inversion_fraction": max(
            row["adjacent_lstar_gradient_sign_inversion_fraction"] for row in rows
        ),
        "gold_median_style_delta_e76": float(
            np.median(
                [
                    row["median_style_delta_e76"]
                    for row in rows
                    if row["split"] == "gold"
                ]
            )
        ),
        "stress_median_style_delta_e76": float(
            np.median(
                [
                    row["median_style_delta_e76"]
                    for row in rows
                    if row["split"] == "stress"
                ]
            )
        ),
        "population_median_chroma_scale": float(
            np.median([row["median_chroma_scale"] for row in rows])
        ),
        "unavailable_ids": unavailable,
    }
    gates = config["automatic_gates"]
    checks = {
        "source_counts": metrics["available_source_count"]
        == config["population"]["expected_available_source_count"],
        "luminance_exact": metrics["maximum_luminance_reconstruction_error"]
        <= gates["maximum_luminance_reconstruction_error"],
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"]
        <= gates["maximum_new_hard_boundary_fraction"],
        "gradient_magnitude": metrics["maximum_p999_gradient_ratio_vs_source"]
        <= gates["maximum_p999_gradient_ratio_vs_source"],
        "gradient_order": metrics[
            "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
        ]
        <= gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"],
        "id11_present": ("11" in {row["id"] for row in rows})
        is bool(gates["require_id11_presence"]),
    }
    automatic = all(checks.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": hash_file(
            root / "configs/u5_r2cb13_characteristic_gold_stress_v1.json"
        ),
        "rows": rows,
        "contact_sheets": sheets,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic,
        "visual_review_status": "pending" if automatic else "forbidden",
        "complete_gold_set": False,
        "decision": (
            "open_partial_gold_stress_severe_review"
            if automatic
            else "close_cb11_before_visual_review"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def adjudicate_files(
    *,
    config_path: Path,
    report_paths: list[Path],
    review_path: Path,
    adjudicator_software_commit: str,
) -> dict[str, Any]:
    """Bind exact CB13 replay and autonomous severe review."""

    if len(report_paths) != 2:
        raise CharacteristicGoldStressError("two CB13 reports are required")
    if len(adjudicator_software_commit) != 40 or any(
        value not in "0123456789abcdef" for value in adjudicator_software_commit
    ):
        raise CharacteristicGoldStressError("invalid adjudicator commit")
    config = load_contract(config_path)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    hashes = [hash_file(path) for path in report_paths]
    if (
        reports[0] != reports[1]
        or len(set(hashes)) != 1
        or reports[0].get("automatic_pass") is not True
        or reports[0].get("complete_gold_set") is not False
    ):
        raise CharacteristicGoldStressError("CB13 replay is not exact and eligible")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("status") != "partial_gold_stress_severe_review_complete"
        or review.get("report_sha256") != hashes[0]
        or review.get("confirmed_severe_artifact_count") != 0
    ):
        raise CharacteristicGoldStressError("CB13 severe review is invalid")
    expected_sheets = {
        row["name"]: row["sha256"] for row in reports[0]["contact_sheets"]
    }
    if review.get("contact_sheet_sha256") != expected_sheets:
        raise CharacteristicGoldStressError("CB13 contact sheet identity drift")
    report_rows = {row["id"]: row for row in reports[0]["rows"]}
    reviewed_ids: set[str] = set()
    for row in review["original_resolution_outputs"]:
        source_id = row["id"]
        if (
            source_id in reviewed_ids
            or report_rows[source_id]["output_sha256"] != row["sha256"]
        ):
            raise CharacteristicGoldStressError("CB13 full-resolution identity drift")
        reviewed_ids.add(source_id)
    required_ids = set(config["visual"]["review_original_resolution_ids"])
    if reviewed_ids != required_ids:
        raise CharacteristicGoldStressError("CB13 full-resolution review is incomplete")

    payload: dict[str, Any] = {
        "schema": "neuro_film.u5_r2cb13_characteristic_gold_stress_decision.v1",
        "experiment_id": EXPERIMENT_ID,
        "adjudicator_software_commit": adjudicator_software_commit,
        "status": "retain_partial_gold_stress_safety_face_open",
        "inputs": {
            "contract_sha256": hash_file(config_path),
            "report_sha256": hashes[0],
            "report_stable_evidence_id": reports[0]["stable_evidence_id"],
            "review_sha256": hash_file(review_path),
        },
        "measurements": reports[0]["metrics"],
        "automatic_checks": reports[0]["checks"],
        "confirmed_severe_artifact_count": 0,
        "partial_available_cohort_pass": True,
        "complete_gold_set_pass": False,
        "production_default_changed": False,
        "next_branch": "restore_exact_fs_face_01_then_complete_gold_or_continue_independent_ood",
        "claim_ceiling": config["claim_ceiling"],
    }
    payload["stable_evidence_id"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


__all__ = [
    "CharacteristicGoldStressError",
    "adjudicate_files",
    "evaluate",
    "load_contract",
    "write_report",
]
