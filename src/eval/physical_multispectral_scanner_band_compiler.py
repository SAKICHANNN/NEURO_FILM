"""U6.P6X ideal-band multispectral scanner compiler evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, xyz2lab

from src.eval.cave_conditional_variability import audit_cave_snapshot
from src.eval.physical_measured_slide_spectral_scanner import (
    _load_measured_table,
)
from src.film_physics.multispectral_scanner import (
    IdealBandProfile,
    apply_xyz_compiler,
    fit_nonnegative_xyz_compiler,
    sample_ideal_bands,
)

SCHEMA = "neuro_film.u6_p6x_multispectral_scanner_band_compiler_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6X contract")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def _load_spectral_columns(path: Path, wavelength_nm: np.ndarray) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError(f"invalid spectral table: {path}")
    columns = [
        np.interp(wavelength_nm, rows[:, 0], rows[:, index])
        for index in range(1, rows.shape[1])
    ]
    return np.stack(columns, axis=1)


def _xyz_weights(
    wavelength_nm: np.ndarray,
    cmf_path: Path,
    d65_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cmf = _load_spectral_columns(cmf_path, wavelength_nm)
    d65 = _load_spectral_columns(d65_path, wavelength_nm)[:, 0]
    if cmf.shape != (wavelength_nm.size, 3):
        raise ValueError("CIE observer must contain XYZ columns")
    step = np.diff(wavelength_nm)
    if not np.allclose(step, step[0], rtol=0.0, atol=0.0):
        raise ValueError("U6.P6X requires a uniform wavelength grid")
    raw = d65[:, None] * cmf * float(step[0])
    normalizer = float(np.sum(raw[:, 1]))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("D65 observer normalization is invalid")
    weights = raw / normalizer
    return weights, np.sum(weights, axis=0)


def _error_summary(prediction_xyz: np.ndarray, target_xyz: np.ndarray) -> dict[str, float]:
    prediction_lab = xyz2lab(prediction_xyz, illuminant="D65")
    target_lab = xyz2lab(target_xyz, illuminant="D65")
    error = deltaE_ciede2000(target_lab, prediction_lab)
    return {
        "mean_delta_e00": float(np.mean(error)),
        "median_delta_e00": float(np.median(error)),
        "p95_delta_e00": float(np.percentile(error, 95.0)),
        "maximum_delta_e00": float(np.max(error)),
    }


def _load_cave_representatives(
    scenes: Mapping[str, Sequence[Path]],
    cave_contract: Mapping[str, Any],
    xyz_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    population = cave_contract["spectral_population"]
    rows = int(population["grid_rows"])
    columns = int(population["grid_columns"])
    height = int(population["image_height"])
    width = int(population["image_width"])
    cell_height = height // rows
    cell_width = width // columns
    if rows * cell_height != height or columns * cell_width != width:
        raise ValueError("CAVE representative grid does not partition images")
    scale = float(population["uint16_scale"])
    representatives: list[np.ndarray] = []
    scene_indices: list[int] = []
    for scene_index, (_, paths) in enumerate(sorted(scenes.items())):
        bands = []
        for path in paths:
            with Image.open(path) as image:
                image.load()
                bands.append(np.asarray(image, dtype=np.uint16))
        cube = np.stack(bands, axis=-1).astype(np.float64) / scale
        tiled = cube.reshape(
            rows,
            cell_height,
            columns,
            cell_width,
            cube.shape[-1],
        )
        medians = np.median(tiled, axis=(1, 3)).reshape(-1, cube.shape[-1])
        representatives.append(medians)
        scene_indices.extend([scene_index] * medians.shape[0])
    spectra = np.concatenate(representatives, axis=0)
    scene = np.asarray(scene_indices, dtype=np.int16)
    xyz = spectra @ xyz_weights
    y = xyz[:, 1]
    keep = (
        np.all(np.isfinite(spectra), axis=1)
        & np.all((spectra >= 0.0) & (spectra <= 1.0), axis=1)
        & (y >= float(population["relative_y_min"]))
        & (y <= float(population["relative_y_max"]))
    )
    return spectra[keep], scene[keep]


def _relative_improvement(candidate: float, baseline: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("relative improvement requires positive finite metrics")
    return float((baseline - candidate) / baseline)


def evaluate_multispectral_scanner_band_compiler(
    root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    parents = contract["parents"]
    hashed_paths = (
        ("p6f_contract_path", "p6f_contract_sha256"),
        ("p6f_decision_path", "p6f_decision_sha256"),
        ("measured_pair_table_path", "measured_pair_table_sha256"),
        ("cave_contract_path", "cave_contract_sha256"),
        ("cave_official_tail_path", "cave_official_tail_sha256"),
        ("cie_xyz_path", "cie_xyz_sha256"),
        ("cie_d65_path", "cie_d65_sha256"),
    )
    for path_key, hash_key in hashed_paths:
        if _sha256(root / str(parents[path_key])) != str(parents[hash_key]):
            raise ValueError(f"{path_key} hash mismatch")
    paper = contract["research_basis"]
    paper_path = root / str(paper["local_pdf_path"])
    if (
        paper_path.stat().st_size != int(paper["local_pdf_bytes"])
        or _sha256(paper_path) != str(paper["local_pdf_sha256"])
    ):
        raise ValueError("Balica-Trumpy paper bytes do not match contract")
    p6f_decision = json.loads(
        (root / str(parents["p6f_decision_path"])).read_text(encoding="utf-8")
    )
    if (
        p6f_decision.get("decision")
        != "close_rgb_scanner_mapping_on_measured_spectra_retain_spectral_reference"
    ):
        raise ValueError("P6F RGB-only closure is not exact")

    domain = contract["spectral_domain"]["wavelength_nm"]
    wavelength = np.arange(
        float(domain["start"]),
        float(domain["stop"]) + float(domain["step"]) * 0.5,
        float(domain["step"]),
        dtype=np.float64,
    )
    if wavelength.size != int(domain["count"]):
        raise ValueError("U6.P6X wavelength count mismatch")
    xyz_weights, clear_xyz = _xyz_weights(
        wavelength,
        root / str(parents["cie_xyz_path"]),
        root / str(parents["cie_d65_path"]),
    )

    test_sets, slides, spectral_pct, _ = _load_measured_table(
        root / str(parents["measured_pair_table_path"])
    )
    if spectral_pct.shape != (8640, 41):
        raise ValueError("P6X measured film inventory mismatch")
    source_wavelength = np.arange(380.0, 781.0, 10.0, dtype=np.float64)
    measured_spectra = np.stack(
        [np.interp(wavelength, source_wavelength, row) for row in spectral_pct],
        axis=0,
    ) / 100.0
    measured_xyz = measured_spectra @ xyz_weights
    measured_masks = {
        "development": np.isin(test_sets, [1, 2, 3, 4])
        & np.isin(slides, [1, 2, 3]),
        "held-set": np.isin(test_sets, [5, 9])
        & np.isin(slides, [1, 2, 3]),
        "held-slide": np.isin(test_sets, [1, 2, 3, 4])
        & np.isin(slides, [4, 5]),
        "joint-held": np.isin(test_sets, [5, 9])
        & np.isin(slides, [4, 5]),
    }
    development_mask = measured_masks["development"]

    profiles = {
        name: IdealBandProfile(
            profile_id=f"balica-trumpy-{name}-ideal-centres-v1",
            center_nm=tuple(float(value) for value in centers),
        )
        for name, centers in contract["candidate_profiles"].items()
    }
    measured_bands = {
        name: sample_ideal_bands(measured_spectra, wavelength, profile)
        for name, profile in profiles.items()
    }
    matrices = {
        name: fit_nonnegative_xyz_compiler(
            bands[development_mask], measured_xyz[development_mask], clear_xyz
        )
        for name, bands in measured_bands.items()
    }
    measured_predictions = {
        name: apply_xyz_compiler(measured_bands[name], matrices[name])
        for name in profiles
    }
    measured_summaries = {
        cell: {
            name: _error_summary(prediction[mask], measured_xyz[mask])
            for name, prediction in measured_predictions.items()
        }
        for cell, mask in measured_masks.items()
    }

    cave_contract = json.loads(
        (root / str(parents["cave_contract_path"])).read_text(encoding="utf-8")
    )
    cave_audit, cave_scenes = audit_cave_snapshot(
        root / str(parents["cave_snapshot_root"]),
        root / str(parents["cave_official_tail_path"]),
        cave_contract,
    )
    cave_spectra, cave_scene = _load_cave_representatives(
        cave_scenes, cave_contract, xyz_weights
    )
    cave_xyz = cave_spectra @ xyz_weights
    cave_bands = {
        name: sample_ideal_bands(cave_spectra, wavelength, profile)
        for name, profile in profiles.items()
    }
    cave_predictions = {
        name: apply_xyz_compiler(cave_bands[name], matrices[name])
        for name in profiles
    }
    cave_summaries = {
        name: _error_summary(prediction, cave_xyz)
        for name, prediction in cave_predictions.items()
    }
    reversed_prediction = apply_xyz_compiler(
        cave_bands["paper7"][:, ::-1], matrices["paper7"]
    )
    reversed_summary = _error_summary(reversed_prediction, cave_xyz)

    dev = measured_summaries["development"]
    paper7_dev_median_improvement = _relative_improvement(
        dev["paper7"]["median_delta_e00"],
        dev["rgb3_control"]["median_delta_e00"],
    )
    paper8_dev_median_improvement = _relative_improvement(
        dev["paper8"]["median_delta_e00"],
        dev["rgb3_control"]["median_delta_e00"],
    )
    paper7_cave_median_improvement = _relative_improvement(
        cave_summaries["paper7"]["median_delta_e00"],
        cave_summaries["rgb3_control"]["median_delta_e00"],
    )
    paper7_cave_p95_improvement = _relative_improvement(
        cave_summaries["paper7"]["p95_delta_e00"],
        cave_summaries["rgb3_control"]["p95_delta_e00"],
    )
    paper8_to_paper7_median_ratio = float(
        cave_summaries["paper8"]["median_delta_e00"]
        / cave_summaries["paper7"]["median_delta_e00"]
    )
    reversed_to_paper7_median_ratio = float(
        reversed_summary["median_delta_e00"]
        / cave_summaries["paper7"]["median_delta_e00"]
    )

    all_predictions = tuple(measured_predictions.values()) + tuple(
        cave_predictions.values()
    ) + (reversed_prediction,)
    output_minimum = min(float(np.min(value)) for value in all_predictions)
    output_maximum_over_white = max(
        float(np.max(value - clear_xyz[None, :])) for value in all_predictions
    )
    clear_error = max(
        float(np.max(np.abs(np.sum(matrix, axis=0) - clear_xyz)))
        for matrix in matrices.values()
    )
    gates = contract["automatic_gates"]
    checks = {
        "paper_pdf_exact": True,
        "measured_inventory": measured_spectra.shape == (8640, 31),
        "measured_split_coverage": bool(
            np.all(
                np.sum(
                    np.column_stack(tuple(measured_masks.values())), axis=1
                )
                == 1
            )
        ),
        "cave_audit": cave_audit["retained_scene_count"] == 31
        and cave_audit["official_crc_mismatch_count"] == 0
        and int(np.unique(cave_scene).size) == 31,
        "paper7_development_p95": dev["paper7"]["p95_delta_e00"]
        <= float(gates["paper7_development_delta_e00_p95_max"]),
        "paper8_development_p95": dev["paper8"]["p95_delta_e00"]
        <= float(gates["paper8_development_delta_e00_p95_max"]),
        "paper7_development_vs_rgb3": paper7_dev_median_improvement
        >= float(gates["paper7_development_median_improvement_vs_rgb3_min"]),
        "paper8_development_vs_rgb3": paper8_dev_median_improvement
        >= float(gates["paper8_development_median_improvement_vs_rgb3_min"]),
        "paper7_cave_median": cave_summaries["paper7"]["median_delta_e00"]
        <= float(gates["paper7_cave_delta_e00_median_max"]),
        "paper7_cave_p95": cave_summaries["paper7"]["p95_delta_e00"]
        <= float(gates["paper7_cave_delta_e00_p95_max"]),
        "paper7_cave_maximum": cave_summaries["paper7"]["maximum_delta_e00"]
        <= float(gates["paper7_cave_delta_e00_maximum_max"]),
        "paper8_cave_median": cave_summaries["paper8"]["median_delta_e00"]
        <= float(gates["paper8_cave_delta_e00_median_max"]),
        "paper8_cave_p95": cave_summaries["paper8"]["p95_delta_e00"]
        <= float(gates["paper8_cave_delta_e00_p95_max"]),
        "paper7_cave_median_vs_rgb3": paper7_cave_median_improvement
        >= float(gates["paper7_cave_median_improvement_vs_rgb3_min"]),
        "paper7_cave_p95_vs_rgb3": paper7_cave_p95_improvement
        >= float(gates["paper7_cave_p95_improvement_vs_rgb3_min"]),
        "paper8_cave_nondegradation": paper8_to_paper7_median_ratio
        <= float(gates["paper8_cave_median_ratio_to_paper7_max"]),
        "reversed_band_negative_control": reversed_to_paper7_median_ratio
        >= float(gates["reversed_band_cave_median_ratio_to_paper7_min"]),
        "clear_response": clear_error
        <= float(gates["clear_xyz_max_abs_error"]),
        "candidate_bounded": output_minimum
        >= float(gates["candidate_output_minimum"])
        and output_maximum_over_white
        <= float(gates["candidate_output_maximum_by_d65_white_tolerance"]),
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": "neuro_film.u6_p6x_multispectral_scanner_band_compiler_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "source_evidence": {
            "paper_sha256": paper["local_pdf_sha256"],
            "measured_pair_table_sha256": parents["measured_pair_table_sha256"],
            "cave_official_tail_sha256": parents["cave_official_tail_sha256"],
        },
        "profiles": {
            name: {
                "center_nm": list(profile.center_nm),
                "profile_sha256": profile.profile_sha256,
                "compiler_matrix": matrices[name].tolist(),
            }
            for name, profile in profiles.items()
        },
        "populations": {
            "measured_rows": int(measured_spectra.shape[0]),
            "measured_split_rows": {
                name: int(np.sum(mask)) for name, mask in measured_masks.items()
            },
            "cave_audit": cave_audit,
            "cave_representative_rows": int(cave_spectra.shape[0]),
            "cave_representative_scene_count": int(np.unique(cave_scene).size),
        },
        "measured_summaries": measured_summaries,
        "cave_summaries": cave_summaries,
        "negative_control": {"reversed_paper7": reversed_summary},
        "comparisons": {
            "paper7_development_median_improvement_vs_rgb3": paper7_dev_median_improvement,
            "paper8_development_median_improvement_vs_rgb3": paper8_dev_median_improvement,
            "paper7_cave_median_improvement_vs_rgb3": paper7_cave_median_improvement,
            "paper7_cave_p95_improvement_vs_rgb3": paper7_cave_p95_improvement,
            "paper8_to_paper7_cave_median_ratio": paper8_to_paper7_median_ratio,
            "reversed_to_paper7_cave_median_ratio": reversed_to_paper7_median_ratio,
            "uniform7_to_paper7_cave_median_ratio": float(
                cave_summaries["uniform7_diagnostic"]["median_delta_e00"]
                / cave_summaries["paper7"]["median_delta_e00"]
            ),
        },
        "numerics": {
            "clear_xyz": clear_xyz.tolist(),
            "clear_response_max_abs_error": clear_error,
            "output_minimum": output_minimum,
            "output_maximum_over_d65_white": output_maximum_over_white,
        },
        "checks": checks,
        "automatic_pass": automatic_pass,
        "branch": contract["branch_rule"]["pass" if automatic_pass else "fail"],
    }
    return {**core, "stable_evidence_id": _stable_id(core)}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "evaluate_multispectral_scanner_band_compiler",
    "load_contract",
    "write_report",
]
