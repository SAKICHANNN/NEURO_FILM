"""CHAM8 one-pair Portra chart display-chain texture diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.eval.flickr_paired_texture_identifiability import analyze_pair
from src.eval.portra400_chart_operator_d1 import _canonical, _sha
from src.eval.portra400_same_scene_registration import _raw_rgb

SCHEMA = "neuro-film.u5-r2cham8-portra400-chart-texture-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham8-portra400-chart-texture-d0-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    analysis = value.get("analysis", {})
    if (
        value.get("schema") != SCHEMA
        or analysis.get("minimum_patches_per_scene", 0) < 24
        or analysis.get("patch_size_pixels") != 96
        or "cannot separate emulsion" not in value.get("claim_ceiling", "")
    ):
        raise ValueError("unsupported CHAM8 contract")
    return value


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_payloads: dict[str, dict[str, Any]] = {}
    for name, binding in contract["parents"].items():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise ValueError("CHAM8 parent drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        parent_payloads[name] = payload
        if (
            "required_decision" in binding
            and payload.get("decision") != binding["required_decision"]
        ):
            raise ValueError("CHAM8 parent decision drift")
        if (
            "required_stable_evidence_id" in binding
            and payload.get("stable_evidence_id")
            != binding["required_stable_evidence_id"]
        ):
            raise ValueError("CHAM8 parent stable identity drift")

    source = contract["source"]
    manifest_rows = {
        row["name"]: row for row in parent_payloads["source_manifest"]["files"]
    }
    for name, digest in (
        (source["digital_raw_name"], source["digital_raw_sha256"]),
        (source["film_tiff_name"], source["film_tiff_sha256"]),
    ):
        if manifest_rows[name]["sha256"] != digest:
            raise ValueError("CHAM8 source manifest drift")
    data_root = root / source["data_root"]
    raw_path = data_root / source["digital_raw_name"]
    film_path = data_root / source["film_tiff_name"]
    if (
        _sha(raw_path) != source["digital_raw_sha256"]
        or _sha(film_path) != source["film_tiff_sha256"]
    ):
        raise ValueError("CHAM8 source bytes drift")
    digital, digital_meta = _raw_rgb(
        raw_path,
        {"raw_use_camera_wb": True, "raw_no_auto_bright": False, "raw_half_size": True},
    )
    with Image.open(film_path) as image:
        film = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    registration = parent_payloads["registration_report"]
    row = next(
        pair
        for pair in registration["pairs"]
        if pair["id"] == "chart_portra400_fuji_dpii"
    )
    texture = analyze_pair(
        digital,
        film,
        np.asarray(row["diagnostics"]["homography_digital_to_film"], dtype=np.float64),
        contract["analysis"],
    )
    gates = contract["gates"]
    checks = {
        "patch_support": texture["patches"] >= gates["minimum_selected_patches"],
        "film_energy": texture["film_to_digital_luma_highpass_energy_ratio"]
        >= gates["minimum_film_to_digital_luma_highpass_energy_ratio"],
        "alignment_control": texture["correct_to_shifted_residual_rms_ratio"]
        <= gates["maximum_correct_to_shifted_residual_rms_ratio"],
        "radial_power": texture["positive_delta_psd_bin_fraction"]
        >= gates["minimum_positive_delta_psd_bin_fraction"],
        "edge_independence": abs(texture["residual_edge_correlation"])
        <= gates["maximum_residual_edge_correlation"],
        "block_control": texture["film_jpeg_block_ratio"]
        <= gates["maximum_film_block_ratio"],
        "finite": all(
            np.isfinite(value)
            for value in np.asarray(list(_numeric_values(texture)), dtype=np.float64)
        ),
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "digital_decoded_sha256": digital_meta["decoded_sha256"],
        "film_decoded_sha256": hashlib.sha256(film.tobytes()).hexdigest(),
        "texture": texture,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def _numeric_values(value: Any):
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _numeric_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _numeric_values(nested)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield value


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
