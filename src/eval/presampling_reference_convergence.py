"""U6.P6AE presampling reference convergence evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_source import hash_file
from src.film_physics.presampling_dye_cloud_scan import (
    render_presampling_dye_cloud_scan,
)
from src.film_physics.presampling_reference_convergence import (
    render_fft_presampling_reference,
)

SCHEMA = "neuro_film.u6_p6ae_presampling_reference_convergence_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6ae_presampling_reference_convergence_report.v1"


class PresamplingConvergenceAuditError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PresamplingConvergenceAuditError("P6AE paths must be relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    material = payload.get("material", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parent.get("decision_sha256")
        != "0f3a5b5d05ffd242ade85cc7399a02b7d50e7bb06f547bec590d45fcb38cd6e2"
        or parent.get("stable_evidence_id")
        != "f596eed9eace63ad820d2fb9bfe5411c2a2587a6a9fc688f434d29050957c271"
        or parent.get("p6ac_reference_report_sha256")
        != "203c0676643c58d85bb627c9936bc559af88412682fe52ea99d5ffd4bb30f846"
        or material.get("input_shape") != [17, 19]
        or material.get("target_pixel_pitch_um") != 6.35
        or material.get("zooms") != [8, 16, 32]
        or material.get("aperture_diameter_um") != 12.5
        or material.get("aperture_subpixels_per_axis") != 256
        or material.get("radius_um_cmy") != [1.8, 2.2, 2.6]
        or material.get("mark_optical_density_cmy") != [0.16, 0.20, 0.24]
        or material.get("monte_carlo_samples") != 4
        or material.get("seed") != 26080263
        or gates.get("maximum_fft_vs_spatial_16x_error") != 1e-12
        or gates.get("maximum_16x_vs_32x_rmse") != 0.003
        or gates.get("maximum_16x_vs_32x_p95_absolute_error") != 0.0075
        or gates.get("minimum_convergence_improvement_from_8x_to_16x") != 0.50
        or gates.get("minimum_transmittance") != 0.0
        or gates.get("maximum_transmittance") != 1.0
        or gates.get("maximum_repeat_error") != 0.0
    ):
        raise PresamplingConvergenceAuditError("P6AE contract drift")
    _relative(parent.get("decision_path", ""))
    _relative(parent.get("p6ac_reference_report_path", ""))
    return payload


def _fixture(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    checker = ((np.indices(shape).sum(axis=0) // 3) % 2).astype(np.float64)
    sparse = ((x > 0.62) & (y > 0.47)).astype(np.float64)
    return np.stack(
        (
            0.08 + 0.44 * x + 0.08 * checker,
            0.10 + 0.36 * y + 0.10 * sparse,
            0.06 + 0.24 * (x + y) + 0.06 * checker * sparse,
        ),
        axis=-1,
    )


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b), dtype=np.float64)))


def evaluate_presampling_reference_convergence(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    parent = config["parent"]
    dpath = root / _relative(parent["decision_path"])
    rpath = root / _relative(parent["p6ac_reference_report_path"])
    if (
        hash_file(dpath) != parent["decision_sha256"]
        or hash_file(rpath) != parent["p6ac_reference_report_sha256"]
    ):
        raise PresamplingConvergenceAuditError("P6AE parent drift")
    if (
        json.loads(dpath.read_text(encoding="utf-8")).get("stable_evidence_id")
        != parent["stable_evidence_id"]
    ):
        raise PresamplingConvergenceAuditError("P6AE parent facts drift")
    material = config["material"]
    target = _fixture(tuple(material["input_shape"]))
    common = {
        "target_pixel_pitch_um": material["target_pixel_pitch_um"],
        "aperture_diameter_um": material["aperture_diameter_um"],
        "aperture_subpixels_per_axis": material["aperture_subpixels_per_axis"],
        "radius_um_cmy": tuple(material["radius_um_cmy"]),
        "mark_optical_density_cmy": tuple(material["mark_optical_density_cmy"]),
        "monte_carlo_samples": material["monte_carlo_samples"],
        "seed": material["seed"],
    }
    spatial = render_presampling_dye_cloud_scan(
        target, candidate_zoom=8, reference_zoom=16, **common
    )
    fft16 = render_fft_presampling_reference(target, zoom=16, **common)
    fft32 = render_fft_presampling_reference(target, zoom=32, **common)
    repeat32 = render_fft_presampling_reference(target, zoom=32, **common)
    crop = config["evaluation"]["interior_crop_target_pixels"]
    interior = np.s_[crop:-crop, crop:-crop, :]
    error16 = np.abs(fft16[interior] - fft32[interior])
    rmse8 = _rmse(spatial.candidate[interior], fft32[interior])
    rmse16 = _rmse(fft16[interior], fft32[interior])
    outputs = np.stack((spatial.candidate, fft16, fft32))
    metrics = {
        "fft_vs_spatial_16x_error": float(np.max(np.abs(fft16 - spatial.reference))),
        "rmse_8x_vs_32x": rmse8,
        "rmse_16x_vs_32x": rmse16,
        "p95_16x_vs_32x": float(np.quantile(error16, 0.95)),
        "convergence_improvement_from_8x_to_16x": (rmse8 - rmse16) / rmse8,
        "minimum_transmittance": float(np.min(outputs)),
        "maximum_transmittance": float(np.max(outputs)),
        "repeat_error": float(np.max(np.abs(repeat32 - fft32))),
    }
    gates = config["gates"]
    results = {
        "fft_equivalence": metrics["fft_vs_spatial_16x_error"]
        <= gates["maximum_fft_vs_spatial_16x_error"],
        "rmse": rmse16 <= gates["maximum_16x_vs_32x_rmse"],
        "p95": metrics["p95_16x_vs_32x"]
        <= gates["maximum_16x_vs_32x_p95_absolute_error"],
        "convergence": metrics["convergence_improvement_from_8x_to_16x"]
        >= gates["minimum_convergence_improvement_from_8x_to_16x"],
        "range": metrics["minimum_transmittance"] > 0
        and metrics["maximum_transmittance"] <= 1,
        "repeat": metrics["repeat_error"] == 0,
    }
    passed = all(results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "fft16_sha256": _sha(fft16),
        "fft32_sha256": _sha(fft32),
        "metrics": metrics,
        "gate_results": results,
        "automatic_pass": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "validate_16x_presampling_reference"
            if passed
            else "revoke_16x_presampling_reference"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "PresamplingConvergenceAuditError",
    "evaluate_presampling_reference_convergence",
    "load_contract",
]
