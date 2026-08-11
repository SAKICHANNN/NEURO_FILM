"""CB30 frozen gold/stress regression for the unchanged CB27 operator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.characteristic_gold_stress import _build_sheet, _encode_png
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
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
from src.eval.monotone_fraction_quantile_transport import (
    monotone_fraction_quantile_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import apply_safe_base_direction_target
from src.film_physics.profile_consumer import validate_standalone_profile_artifact

SCHEMA = "neuro_film.u5_r2cb30_monotone_fraction_gold_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb30_monotone_fraction_gold_stress_report.v1"
EXPERIMENT_ID = "U5.R2CB30"


class MonotoneFractionGoldStressError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise MonotoneFractionGoldStressError("CB30 contract structure drift")
    return payload


def _inputs(config: Mapping[str, Any], root: Path):
    parents = config["parents"]
    decision = _load_exact_json(
        root, parents["cb29_decision_path"], parents["cb29_decision_sha256"]
    )
    if decision.get("decision") != parents["cb29_required_decision"]:
        raise MonotoneFractionGoldStressError("CB29 decision drift")
    cb27 = _load_exact_json(
        root, parents["cb27_contract_path"], parents["cb27_contract_sha256"]
    )
    if float(cb27["operator"]["minimum_valid_fraction"]) != float(
        config["operator"]["minimum_valid_fraction"]
    ):
        raise MonotoneFractionGoldStressError("CB27 operator drift")
    cb11 = _load_exact_json(
        root, parents["cb11_contract_path"], parents["cb11_contract_sha256"]
    )
    cb12 = _load_exact_json(
        root, parents["cb12_contract_path"], parents["cb12_contract_sha256"]
    )
    curve = _compiled_curve(load_cb6(root / cb11["parents"]["cb6_contract_path"]))
    ao6_config = cb12["ao6"]
    artifact_report = _load_exact_json(
        root,
        ao6_config["frozen_artifact_report_path"],
        ao6_config["frozen_artifact_report_sha256"],
    )
    artifact = artifact_report["artifact"]
    if (
        artifact_report.get("artifact_canonical_sha256")
        != ao6_config["frozen_artifact_canonical_sha256"]
        or artifact.get("bundle_sha256") != ao6_config["frozen_bundle_sha256"]
    ):
        raise MonotoneFractionGoldStressError("AO6 artifact drift")
    validate_standalone_profile_artifact(artifact)

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
            raise MonotoneFractionGoldStressError(f"source hash drift: {row['id']}")
        available.append(row)
    if (
        sorted(unavailable) != sorted(population["required_unavailable_ids"])
        or len(available) != population["expected_available_source_count"]
        or sum(row["split"] == "gold" for row in available)
        != population["expected_available_gold_count"]
        or sum(row["split"] == "stress" for row in available)
        != population["expected_available_stress_count"]
    ):
        raise MonotoneFractionGoldStressError("local population boundary drift")
    return cb11, ao6_config, artifact, curve, available, unavailable


def evaluate(
    config: Mapping[str, Any],
    root: Path,
    output_dir: Path,
    *,
    target_builder: Any = monotone_fraction_quantile_transport_target,
    report_schema: str = REPORT_SCHEMA,
    experiment_id: str = EXPERIMENT_ID,
    contract_filename: str = "u5_r2cb30_monotone_fraction_gold_stress_v1.json",
    candidate_builder: Any | None = None,
) -> dict[str, Any]:
    cb11, ao6_config, artifact, curve, available, unavailable = _inputs(config, root)
    if output_dir.exists():
        raise FileExistsError("CB30 is create-only")
    output_dir.mkdir(parents=True)
    operator = cb11["operator"]
    weights = np.asarray(operator["luminance_weights"], dtype=np.float64)
    epsilon = float(operator["boundary_epsilon"])
    minimum_valid_fraction = float(config["operator"]["minimum_valid_fraction"])
    rows: list[dict[str, Any]] = []
    thumbnails: dict[str, tuple[Image.Image, Image.Image]] = {}
    for source_row in available:
        source_path = root / source_row["source_path"]
        source = _load_rgb(
            source_path,
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        ao6 = render_fixed_pair(source, artifact, ao6_config["component"])[
            ao6_config["arm_id"]
        ]
        safe_base, _, _ = apply_characteristic_luma_chroma(
            source,
            curve,
            weights=weights,
            strength=float(operator["nominal_strength"]),
            boundary_epsilon=epsilon,
        )
        target = target_builder(
            safe_base,
            ao6,
            weights=weights,
            boundary_epsilon=epsilon,
            minimum_valid_fraction=minimum_valid_fraction,
        )
        if candidate_builder is None:
            candidate, scale, luma_error = apply_safe_base_direction_target(
                source,
                safe_base,
                target,
                weights=weights,
                boundary_epsilon=epsilon,
            )
        else:
            candidate, scale, luma_error = candidate_builder(
                source,
                safe_base,
                target,
                weights=weights,
                boundary_epsilon=epsilon,
            )
        candidate_bytes, candidate_u8 = _encode_png(candidate)
        output_rel = Path("renders") / f"{source_row['id']}.png"
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
                "median_delta_e76_vs_cb11": _median_delta_e76(safe_base, candidate),
                "median_direction_scale": float(np.median(scale)),
                "fraction_direction_scale_below_0p5": float(np.mean(scale < 0.5)),
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

    sheets: list[dict[str, Any]] = []
    groups = [
        ("gold", [row for row in rows if row["split"] == "gold"]),
        ("stress_1", [row for row in rows if row["split"] == "stress"][:16]),
        ("stress_2", [row for row in rows if row["split"] == "stress"][16:]),
    ]
    for name, sheet_rows in groups:
        payload = _build_sheet(sheet_rows, thumbnails)
        relative = Path("contact_sheets") / f"{name}.png"
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
        "population_median_direction_scale": float(
            np.median([row["median_direction_scale"] for row in rows])
        ),
        "population_median_fraction_direction_scale_below_0p5": float(
            np.median([row["fraction_direction_scale_below_0p5"] for row in rows])
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
        "direction_scale": metrics["population_median_direction_scale"]
        >= gates["minimum_population_median_direction_scale"],
        "direction_scale_tail": metrics[
            "population_median_fraction_direction_scale_below_0p5"
        ]
        <= gates["maximum_population_median_fraction_direction_scale_below_0p5"],
        "id11_present": ("11" in {row["id"] for row in rows})
        is bool(gates["require_id11_presence"]),
    }
    automatic = all(checks.values())
    report: dict[str, Any] = {
        "schema": report_schema,
        "experiment_id": experiment_id,
        "contract_sha256": hash_file(root / "configs" / contract_filename),
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
            else "close_cb27_before_visual_review"
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


__all__ = [
    "MonotoneFractionGoldStressError",
    "evaluate",
    "load_contract",
    "write_report",
]
