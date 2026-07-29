"""U6.P4Y physical/ACF/severe audit for the fixed P4X scale."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.real_uniform_grain_physical import (
    _acf_metrics,
    _physical_metrics,
    _save_diagnostics,
)
from src.eval.real_uniform_grain_source import hash_file


SCHEMA = "neuro_film.u6_p4y_scaled_material_physical_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4y_scaled_material_physical_report.v1"


class ScaledMaterialPhysicalError(RuntimeError):
    """Raised when the fixed P4Y candidate or evidence drifts."""


def _load_exact_json(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise ScaledMaterialPhysicalError(f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScaledMaterialPhysicalError("parent must be an object")
    return payload


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    candidate = config["candidate"]
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("photographic_render_allowed")
        or config["execution"].get("post_result_retuning_allowed")
        or config["execution"].get("production_integration_allowed")
        or candidate.get("display_rgb_noise_allowed")
        or candidate.get("hard_clipping_allowed")
        or candidate.get("signed_density_allowed")
    ):
        raise ScaledMaterialPhysicalError("unsupported P4Y contract")
    parents = config["parents"]
    decision = _load_exact_json(root, parents["p4x_decision"])
    report = _load_exact_json(root, parents["p4x_report"])
    p4t = _load_exact_json(root, parents["p4t_contract"])
    p4r_report = _load_exact_json(root, parents["p4r_report"])
    p4r_analysis = _load_exact_json(root, parents["p4r_analysis"])
    p4s = _load_exact_json(root, parents["p4s_contract"])
    p4d = _load_exact_json(root, parents["p4d_contract"])
    scale = float(candidate["source_amplitude_scale"])
    expected_od = [
        scale * float(value)
        for value in p4t["candidate"][
            "grain_optical_density_by_rgb_layer"
        ]
    ]
    if (
        decision["decision"]
        != parents["p4x_decision"]["required_decision"]
        or not report["automatic_pass"]
        or float(report["selected_shared_amplitude_scale"]) != scale
        or [float(value) for value in candidate["sigma_yx_pixels"]]
        != [
            float(value) for value in p4t["candidate"]["sigma_yx_pixels"]
        ]
        or [
            float(value)
            for value in candidate["grain_optical_density_by_rgb_layer"]
        ]
        != expected_od
        or p4s["model"]["amplitude_fit_allowed"]
        or p4d["model"]["display_rgb_noise_allowed"]
    ):
        raise ScaledMaterialPhysicalError("P4Y candidate drift")
    return p4r_report, p4r_analysis, p4s, p4d


def evaluate_scaled_material_physical(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    p4r_report, p4r_analysis, p4s, p4d = validate_contract(root, config)
    physical, physical_checks = _physical_metrics(contract=config)
    acf, acf_checks = _acf_metrics(
        p4r_report=p4r_report,
        p4r_analysis=p4r_analysis,
        p4s_contract=p4s,
        contract=config,
    )
    checks = {**physical_checks, **acf_checks}
    automatic_pass = all(checks.values())
    diagnostics = (
        _save_diagnostics(
            output_dir=output_dir,
            contract=config,
            p4d=p4d,
        )
        if automatic_pass
        else None
    )
    core = {
        "schema": REPORT_SCHEMA,
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "selected_shared_amplitude_scale": config["candidate"][
            "source_amplitude_scale"
        ],
        "selected_sigma_yx_pixels": config["candidate"][
            "sigma_yx_pixels"
        ],
        "held_acf": acf,
        "physical_metrics": physical,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "visual_review_allowed": automatic_pass,
        "diagnostics": diagnostics,
        "decision": (
            "automatic_pass_open_fixed_visual_severe_review"
            if automatic_pass
            else "close_scaled_material_candidate"
        ),
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_scaled_material_physical",
    "validate_contract",
    "write_report",
]
