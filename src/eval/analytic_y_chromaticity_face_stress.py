"""CB56 decoded-pixel face stress for the exact CB54 streamed operator."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.srgb_transfer import encoded_srgb_to_linear
from src.eval.analytic_y_chromaticity_streaming import (
    select_analytic_y_chromaticity_candidate_streamed,
)
from src.eval.characteristic_gold_stress import _encode_png
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
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.film_physics.profile_consumer import validate_standalone_profile_artifact

SCHEMA = "neuro_film.u5_r2cb56_analytic_y_chromaticity_face_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb56_analytic_y_chromaticity_face_stress_report.v1"
EXPERIMENT_ID = "U5.R2CB56"


class AnalyticYChromaticityFaceStressError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise AnalyticYChromaticityFaceStressError("CB56 contract drift")
    return payload


def _load_face(config: dict[str, Any], root: Path) -> np.ndarray:
    source = config["source"]
    rows: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    for manifest_rel, manifest_sha, array_rel in zip(
        source["manifest_paths"],
        source["manifest_sha256"],
        source["decoded_srgb_npy_paths"],
        strict=True,
    ):
        manifest = _load_exact_json(root, manifest_rel, manifest_sha)
        matches = [
            row
            for row in manifest["records"]
            if row["sample_id"] == source["sample_id"]
        ]
        if len(matches) != 1:
            raise AnalyticYChromaticityFaceStressError("CB56 manifest row drift")
        row = matches[0]
        if (
            row["source_path"] != source["original_logical_path"]
            or row["source_sha256"] != source["original_file_sha256"]
            or row["source_npy_path"] != array_rel
            or row["source_npy_sha256"] != source["decoded_srgb_npy_sha256"]
            or hash_file(root / array_rel) != source["decoded_srgb_npy_sha256"]
        ):
            raise AnalyticYChromaticityFaceStressError("CB56 source lineage drift")
        array = np.load(root / array_rel, allow_pickle=False)
        if (
            list(array.shape) != source["expected_shape"]
            or str(array.dtype) != source["expected_dtype"]
        ):
            raise AnalyticYChromaticityFaceStressError("CB56 decoded array drift")
        rows.append(row)
        arrays.append(array)
    if not np.array_equal(arrays[0], arrays[1]):
        raise AnalyticYChromaticityFaceStressError("CB56 decoded arrays disagree")
    encoded = np.asarray(arrays[0], dtype=np.float32)
    linear = np.asarray(encoded_srgb_to_linear(encoded), dtype=np.float32)
    if not np.isfinite(linear).all() or np.min(linear) < 0.0 or np.max(linear) > 1.0:
        raise AnalyticYChromaticityFaceStressError("CB56 linear source drift")
    return linear


def _operator_inputs(config: dict[str, Any], root: Path):
    parents = config["parents"]
    decision = _load_exact_json(
        root, parents["cb55_decision_path"], parents["cb55_decision_sha256"]
    )
    if decision.get("decision") != parents["cb55_required_decision"]:
        raise AnalyticYChromaticityFaceStressError("CB55 decision drift")
    cb52 = _load_exact_json(
        root, parents["cb52_contract_path"], parents["cb52_contract_sha256"]
    )
    cb11 = _load_exact_json(
        root,
        cb52["parents"]["cb11_contract_path"],
        cb52["parents"]["cb11_contract_sha256"],
    )
    cb12 = _load_exact_json(
        root,
        cb52["parents"]["cb12_contract_path"],
        cb52["parents"]["cb12_contract_sha256"],
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
        raise AnalyticYChromaticityFaceStressError("AO6 artifact drift")
    validate_standalone_profile_artifact(artifact)
    return cb52, cb11, ao6_config, artifact, curve


def _contact_sheet(source: np.ndarray, ao6: np.ndarray, candidate: np.ndarray) -> bytes:
    panels = []
    for image in (source, ao6, candidate):
        _, quantized = _encode_png(image)
        panels.append(Image.fromarray(quantized, mode="RGB"))
    canvas = Image.new("RGB", (1536, 548), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(
        zip(("SOURCE", "AO6", "CB56"), panels, strict=True)
    ):
        canvas.paste(panel, (512 * index, 36))
        draw.text((512 * index + 8, 10), label, fill="black")
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("CB56 is create-only")
    output_dir.mkdir(parents=True)
    source = _load_face(config, root)
    cb52, cb11, ao6_config, artifact, curve = _operator_inputs(config, root)
    operator = cb11["operator"]
    weights = np.asarray(operator["luminance_weights"], dtype=np.float64)
    epsilon = float(operator["boundary_epsilon"])
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
    target = nonexpansive_fraction_transport_target(
        safe_base,
        ao6,
        weights=weights,
        minimum_valid_fraction=float(cb52["operator"]["minimum_valid_fraction"]),
        fraction_knots=int(cb52["operator"]["fraction_knots"]),
        maximum_fraction_slope=float(cb52["operator"]["maximum_fraction_slope"]),
    )
    gates = config["automatic_gates"]
    scratch = output_dir / "scratch"
    scratch.mkdir()
    candidate, scale, luma_error, facts = (
        select_analytic_y_chromaticity_candidate_streamed(
            source,
            target,
            curve=curve,
            strength=float(operator["nominal_strength"]),
            boundary_epsilon=epsilon,
            dose_grid=cb52["operator"]["dose_grid"],
            maximum_gradient_ratio=float(
                gates["maximum_p999_gradient_ratio_vs_source"]
            ),
            maximum_lstar_inversion_fraction=float(
                gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"]
            ),
            lstar_order_epsilon=float(cb52["operator"]["lstar_order_epsilon"]),
            row_chunk=int(config["execution"]["row_chunk"]),
            scratch_root=scratch,
        )
    )
    if any(scratch.iterdir()):
        raise AnalyticYChromaticityFaceStressError("CB56 scratch residue")
    scratch.rmdir()
    source_lab = linear_rgb_to_lab(source, working_space="linear_srgb")
    candidate_lab = linear_rgb_to_lab(candidate, working_space="linear_srgb")
    metrics = {
        "style_delta_e76": _median_delta_e76(source, candidate),
        "delta_e76_vs_cb11": _median_delta_e76(safe_base, candidate),
        "median_direction_scale": float(np.median(scale)),
        "fraction_direction_scale_below_0p5": float(np.mean(scale < 0.5)),
        "global_dose": facts["global_dose"],
        "median_gamut_scale": facts["median_gamut_scale"],
        "fraction_gamut_scale_below_0p8": facts["fraction_gamut_scale_below_0p8"],
        "maximum_luminance_reconstruction_error": float(np.max(np.abs(luma_error))),
        "new_hard_boundary_fraction": _new_boundary_fraction(
            source, candidate, epsilon
        ),
        "p999_gradient_ratio_vs_source": _gradient_p999_ratio(source, candidate),
        "adjacent_lstar_gradient_sign_inversion_fraction": _gradient_inversion_fraction(
            source_lab[..., 0], candidate_lab[..., 0], epsilon=0.01
        ),
    }
    checks = {
        "luminance_exact": metrics["maximum_luminance_reconstruction_error"]
        <= gates["maximum_luminance_reconstruction_error"],
        "new_boundaries": metrics["new_hard_boundary_fraction"]
        <= gates["maximum_new_hard_boundary_fraction"],
        "gradient_magnitude": metrics["p999_gradient_ratio_vs_source"]
        <= gates["maximum_p999_gradient_ratio_vs_source"],
        "gradient_order": metrics["adjacent_lstar_gradient_sign_inversion_fraction"]
        <= gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"],
        "direction_scale": metrics["median_direction_scale"]
        >= gates["minimum_direction_scale"],
        "direction_scale_tail": metrics["fraction_direction_scale_below_0p5"]
        <= gates["maximum_fraction_direction_scale_below_0p5"],
        "global_dose": metrics["global_dose"] >= gates["minimum_global_dose"],
        "gamut_scale": metrics["median_gamut_scale"] >= gates["minimum_gamut_scale"],
        "gamut_scale_tail": metrics["fraction_gamut_scale_below_0p8"]
        <= gates["maximum_fraction_gamut_scale_below_0p8"],
        "visible_style": metrics["style_delta_e76"] >= gates["minimum_style_delta_e76"],
        "material_vs_cb11": metrics["delta_e76_vs_cb11"]
        >= gates["minimum_delta_e76_vs_cb11"],
    }
    outputs: dict[str, dict[str, str]] = {}
    for name, image in (("source", source), ("ao6", ao6), ("candidate", candidate)):
        payload, _ = _encode_png(image)
        path = output_dir / f"{name}.png"
        path.write_bytes(payload)
        outputs[name] = {
            "path": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    sheet = _contact_sheet(source, ao6, candidate)
    sheet_path = output_dir / "contact_sheet.png"
    sheet_path.write_bytes(sheet)
    outputs["contact_sheet"] = {
        "path": sheet_path.name,
        "sha256": hashlib.sha256(sheet).hexdigest(),
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": hash_file(
            root / "configs/u5_r2cb56_analytic_y_chromaticity_face_stress_v1.json"
        ),
        "source": {
            "sample_id": config["source"]["sample_id"],
            "original_logical_path": config["source"]["original_logical_path"],
            "original_file_sha256": config["source"]["original_file_sha256"],
            "decoded_srgb_npy_sha256": config["source"]["decoded_srgb_npy_sha256"],
            "shape": list(source.shape),
            "original_file_reopened": False,
        },
        "outputs": outputs,
        "selector_facts": facts,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "visual_review_status": "pending" if all(checks.values()) else "forbidden",
        "decision": "open_cb56_face_severe_review"
        if all(checks.values())
        else "close_cb56_without_rescue",
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = ["evaluate", "load_contract"]
