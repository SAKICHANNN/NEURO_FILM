"""U6.P6AF 32x-to-64x presampling reference convergence evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_source import hash_file
from src.eval.presampling_reference_convergence import (
    build_presampling_convergence_fixture,
)
from src.film_physics.presampling_reference_convergence import (
    render_fft_presampling_reference,
)

SCHEMA = "neuro_film.u6_p6af_presampling_reference_convergence_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6af_presampling_reference_convergence_report.v1"


class PresamplingConvergence64xAuditError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PresamplingConvergence64xAuditError("P6AF paths must be relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    material = payload.get("material", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parent.get("decision_sha256")
        != "8014605b78b5c8a7ab20b0f6b60ca834a287a9da3d15afd0fb15a4e427caf011"
        or parent.get("stable_evidence_id")
        != "6fe3d8828dc243cc4e66263a5e959b9451d4a6593003b1be85b4244b6985c24a"
        or parent.get("provisional_32x_sha256")
        != "b399cada3cecf588ed3837c60757ec0ddc19d51effdbea53d87b1eff996af1aa"
        or material.get("input_shape") != [17, 19]
        or material.get("target_pixel_pitch_um") != 6.35
        or material.get("zooms") != [16, 32, 64]
        or material.get("aperture_diameter_um") != 12.5
        or material.get("aperture_subpixels_per_axis") != 256
        or material.get("radius_um_cmy") != [1.8, 2.2, 2.6]
        or material.get("mark_optical_density_cmy") != [0.16, 0.20, 0.24]
        or material.get("monte_carlo_samples") != 4
        or material.get("seed") != 26080263
        or gates.get("maximum_regenerated_32x_error") != 0.0
        or gates.get("maximum_32x_vs_64x_rmse") != 0.0015
        or gates.get("maximum_32x_vs_64x_p95_absolute_error") != 0.00375
        or gates.get("minimum_convergence_improvement_from_16x_to_32x") != 0.50
        or gates.get("minimum_transmittance") != 0.0
        or gates.get("maximum_transmittance") != 1.0
        or gates.get("maximum_repeat_error") != 0.0
    ):
        raise PresamplingConvergence64xAuditError("P6AF contract drift")
    _relative(parent.get("decision_path", ""))
    return payload


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b), dtype=np.float64)))


def evaluate_presampling_reference_convergence_64x(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    parent = config["parent"]
    decision_path = root / _relative(parent["decision_path"])
    if hash_file(decision_path) != parent["decision_sha256"]:
        raise PresamplingConvergence64xAuditError("P6AF parent decision drift")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("stable_evidence_id") != parent["stable_evidence_id"]
        or decision.get("decision") != "revoke_16x_presampling_reference"
    ):
        raise PresamplingConvergence64xAuditError("P6AF parent facts drift")

    material = config["material"]
    target = build_presampling_convergence_fixture(tuple(material["input_shape"]))
    common = {
        "target_pixel_pitch_um": material["target_pixel_pitch_um"],
        "aperture_diameter_um": material["aperture_diameter_um"],
        "aperture_subpixels_per_axis": material["aperture_subpixels_per_axis"],
        "radius_um_cmy": tuple(material["radius_um_cmy"]),
        "mark_optical_density_cmy": tuple(material["mark_optical_density_cmy"]),
        "monte_carlo_samples": material["monte_carlo_samples"],
        "seed": material["seed"],
    }
    render16 = render_fft_presampling_reference(target, zoom=16, **common)
    render32 = render_fft_presampling_reference(target, zoom=32, **common)
    render64 = render_fft_presampling_reference(target, zoom=64, **common)
    repeat64 = render_fft_presampling_reference(target, zoom=64, **common)

    crop = config["evaluation"]["interior_crop_target_pixels"]
    interior = np.s_[crop:-crop, crop:-crop, :]
    error32 = np.abs(render32[interior] - render64[interior])
    rmse16 = _rmse(render16[interior], render64[interior])
    rmse32 = _rmse(render32[interior], render64[interior])
    render32_sha = _sha(render32)
    outputs = np.stack((render16, render32, render64))
    metrics = {
        "regenerated_32x_error": (
            0.0 if render32_sha == parent["provisional_32x_sha256"] else 1.0
        ),
        "rmse_16x_vs_64x": rmse16,
        "rmse_32x_vs_64x": rmse32,
        "p95_32x_vs_64x": float(np.quantile(error32, 0.95)),
        "convergence_improvement_from_16x_to_32x": (rmse16 - rmse32) / rmse16,
        "minimum_transmittance": float(np.min(outputs)),
        "maximum_transmittance": float(np.max(outputs)),
        "repeat_error": float(np.max(np.abs(repeat64 - render64))),
    }
    gates = config["gates"]
    gate_results = {
        "regenerated_32x_identity": metrics["regenerated_32x_error"]
        <= gates["maximum_regenerated_32x_error"],
        "rmse": rmse32 <= gates["maximum_32x_vs_64x_rmse"],
        "p95": metrics["p95_32x_vs_64x"]
        <= gates["maximum_32x_vs_64x_p95_absolute_error"],
        "convergence": metrics["convergence_improvement_from_16x_to_32x"]
        >= gates["minimum_convergence_improvement_from_16x_to_32x"],
        "range": metrics["minimum_transmittance"] > gates["minimum_transmittance"]
        and metrics["maximum_transmittance"] <= gates["maximum_transmittance"],
        "repeat": metrics["repeat_error"] <= gates["maximum_repeat_error"],
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "render32_sha256": render32_sha,
        "render64_sha256": _sha(render64),
        "metrics": metrics,
        "gate_results": gate_results,
        "automatic_pass": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "validate_32x_presampling_reference"
            if passed
            else "close_raster_presampling_reference_family"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "PresamplingConvergence64xAuditError",
    "evaluate_presampling_reference_convergence_64x",
    "load_contract",
]
