"""RF3.D6 measured-reflectance observability of three stock sensitivity priors."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SCHEMA = "neuro-film.rf3-three-stock-layer-exposure-observability-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-three-stock-layer-exposure-observability-report.v1"
STOCKS = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]


class ThreeStockLayerExposureError(RuntimeError):
    """Raised when an RF3.D6 binding or numerical invariant fails."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ThreeStockLayerExposureError("RF3.D6 paths must be repository-relative")
    return path


def _numeric(payload: bytes) -> tuple[str, np.ndarray]:
    lines = payload.decode("utf-8", "replace").splitlines()
    if not lines:
        raise ThreeStockLayerExposureError("empty USGS record")
    try:
        values = np.asarray([float(row) for row in lines[1:] if row.strip()], dtype=np.float64)
    except ValueError as error:
        raise ThreeStockLayerExposureError("invalid USGS numeric record") from error
    return lines[0].strip(), values


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    gates = value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != "RF3.D6"
        or value.get("status") != "FROZEN_BEFORE_LAYER_EXPOSURE_SCORE"
        or value.get("response", {}).get("cohort_fitting_allowed") is not False
        or value.get("response", {}).get("rgb_render_allowed") is not False
        or gates.get("bootstrap_repeats") != 1000
        or gates.get("require_all_three_pairs") is not True
        or gates.get("require_exact_replay") is not True
    ):
        raise ThreeStockLayerExposureError("RF3.D6 frozen contract drift")
    for section, keys in {
        "parent": ("path",),
        "spectral_digitization": ("path",),
        "reflectance_source": ("source_decision", "archive"),
        "illuminant": ("path", "observer_path"),
    }.items():
        for key in keys:
            _relative(value[section][key])
    return value


def _load_csv(path: Path, wavelength: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    table = np.loadtxt(path, delimiter=",")
    return np.column_stack(
        [np.interp(wavelength, table[:, 0], table[:, index]) for index in columns]
    )


def _load_population(
    archive_path: Path,
    source: Mapping[str, Any],
    pattern: str,
    eligibility_wavelength: np.ndarray,
    d65: np.ndarray,
    cmf_y: np.ndarray,
    score_wavelength: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    rejection: Counter[str] = Counter()
    rows: list[np.ndarray] = []
    groups: list[str] = []
    chapters: list[str] = []
    eligibility = source["eligibility"]
    normalizer = float(np.sum(d65 * cmf_y))
    with zipfile.ZipFile(archive_path) as archive:
        wavelengths: dict[int, np.ndarray] = {}
        for member in archive.namelist():
            if "Wavelengths_" in member and member.endswith(".txt"):
                _, values = _numeric(archive.read(member))
                wavelengths[len(values)] = values * 1000.0
        for member in archive.namelist():
            if not member.endswith(".txt") or "Wavelengths_" in member:
                continue
            if "/errorbars/" in member:
                rejection["errorbars"] += 1
                continue
            header, values = _numeric(archive.read(member))
            if not header.endswith(" " + str(eligibility["measurement_type"])):
                rejection["not_aref"] += 1
                continue
            wavelength = wavelengths.get(len(values))
            if wavelength is None:
                rejection["unknown_wavelength_record"] += 1
                continue
            valid = np.isfinite(values) & (values > float(eligibility["missing_value_below"]))
            if np.count_nonzero(valid) < 2:
                rejection["insufficient_valid_samples"] += 1
                continue
            x = wavelength[valid]
            y = values[valid]
            order = np.argsort(x, kind="mergesort")
            x, unique = np.unique(x[order], return_index=True)
            y = y[order][unique]
            if x[0] > eligibility_wavelength[0] or x[-1] < eligibility_wavelength[-1]:
                rejection["insufficient_visible_overlap"] += 1
                continue
            eligibility_spectrum = np.interp(eligibility_wavelength, x, y)
            if (
                not np.all(np.isfinite(eligibility_spectrum))
                or np.any(eligibility_spectrum < float(eligibility["reflectance_min"]))
                or np.any(eligibility_spectrum > float(eligibility["reflectance_max"]))
            ):
                rejection["reflectance_out_of_bounds"] += 1
                continue
            relative_y = float(np.sum(eligibility_spectrum * d65 * cmf_y) / normalizer)
            if not (
                float(eligibility["relative_y_min"])
                <= relative_y
                <= float(eligibility["relative_y_max"])
            ):
                rejection["relative_y_out_of_bounds"] += 1
                continue
            stem = PurePosixPath(member).stem
            match = re.match(pattern, stem)
            parts = PurePosixPath(member).parts
            if match is None or not match.group(1) or len(parts) < 3:
                raise ThreeStockLayerExposureError("USGS sample identity drift")
            rows.append(np.interp(score_wavelength, x, y))
            groups.append(match.group(1))
            chapters.append(parts[1])
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(groups, dtype="U192"),
        np.asarray(chapters, dtype="U64"),
        dict(sorted(rejection.items())),
    )


def _sensitivities(digitization: Mapping[str, Any], wavelength: np.ndarray, floor: float) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for stock in digitization["stocks"]:
        curves = []
        for points in stock["curves"].values():
            values = np.asarray(points, dtype=np.float64)
            sampled = np.interp(wavelength, values[:, 0], values[:, 1], left=-np.inf, right=-np.inf)
            sampled = np.maximum(sampled - float(np.max(values[:, 1])), floor)
            curves.append(np.power(10.0, sampled))
        result[stock["stock_id"]] = np.asarray(curves, dtype=np.float64)
    if list(result) != STOCKS:
        raise ThreeStockLayerExposureError("RF3.D6 stock order drift")
    return result


def _bootstrap_lcb(values: np.ndarray, groups: np.ndarray, repeats: int, seed: int) -> float:
    unique = np.unique(groups)
    by_group = {group: values[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    medians = []
    for _ in range(repeats):
        draw = rng.choice(unique, size=len(unique), replace=True)
        medians.append(float(np.median(np.concatenate([by_group[group] for group in draw]))))
    return float(np.percentile(medians, 2.5))


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    def bound(section: str, key: str, hash_key: str) -> Path:
        path = root / _relative(contract[section][key])
        if _sha(path) != contract[section][hash_key]:
            raise ThreeStockLayerExposureError(f"RF3.D6 {section}/{key} hash drift")
        return path

    parent_path = bound("parent", "path", "sha256")
    if json.loads(parent_path.read_text(encoding="utf-8")).get("decision") != contract["parent"]["required_decision"]:
        raise ThreeStockLayerExposureError("RF3.D6 parent decision drift")
    digitization_path = bound("spectral_digitization", "path", "sha256")
    source_path = bound("reflectance_source", "source_decision", "source_decision_sha256")
    archive_path = bound("reflectance_source", "archive", "archive_sha256")
    d65_path = bound("illuminant", "path", "sha256")
    observer_path = bound("illuminant", "observer_path", "observer_sha256")
    digitization = json.loads(digitization_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    score_wavelength = np.asarray(contract["spectral_digitization"]["wavelengths_nm"], dtype=np.float64)
    eligibility_wavelength = np.asarray(contract["illuminant"]["source_eligibility_wavelengths_nm"], dtype=np.float64)
    d65 = _load_csv(d65_path, eligibility_wavelength, (1,))[:, 0]
    cmf_y = _load_csv(observer_path, eligibility_wavelength, (2,))[:, 0]
    spectra, groups, chapters, rejection = _load_population(
        archive_path,
        source,
        contract["reflectance_source"]["sample_group_regex"],
        eligibility_wavelength,
        d65,
        cmf_y,
        score_wavelength,
    )
    if (
        len(spectra) != int(contract["reflectance_source"]["expected_eligible_records"])
        or len(np.unique(chapters)) != int(contract["reflectance_source"]["expected_chapters"])
        or len(np.unique(groups)) < int(contract["reflectance_source"]["minimum_sample_groups"])
    ):
        raise ThreeStockLayerExposureError("RF3.D6 USGS population drift")
    d65_score = _load_csv(d65_path, score_wavelength, (1,))[:, 0]
    sensitivity = _sensitivities(
        digitization,
        score_wavelength,
        float(contract["spectral_digitization"]["outside_support_log10"]),
    )
    signatures: dict[str, np.ndarray] = {}
    epsilon = float(contract["response"]["epsilon"])
    for stock, curves in sensitivity.items():
        kernel = curves * d65_score[None, :]
        exposure = np.trapezoid(spectra[:, None, :] * kernel[None, :, :], score_wavelength, axis=2)
        white = np.trapezoid(kernel, score_wavelength, axis=1)
        log_response = np.log(np.maximum(exposure / white[None, :], epsilon))
        signatures[stock] = log_response - np.mean(log_response, axis=1, keepdims=True)
    gates = contract["gates"]
    comparisons = []
    for index, left in enumerate(STOCKS):
        for right in STOCKS[index + 1 :]:
            vector = signatures[left] - signatures[right]
            distance = np.linalg.norm(vector, axis=1)
            centered = vector - np.mean(vector, axis=0, keepdims=True)
            material_rms = float(np.sqrt(np.mean(np.sum(np.square(centered), axis=1))))
            median = float(np.median(distance))
            p10 = float(np.percentile(distance, 10))
            lcb = _bootstrap_lcb(
                distance,
                groups,
                int(gates["bootstrap_repeats"]),
                int(gates["bootstrap_seed"]),
            )
            comparisons.append({
                "left": left,
                "right": right,
                "median_log_response_l2": median,
                "p10_log_response_l2": p10,
                "material_delta_vector_rms": material_rms,
                "group_bootstrap_95_lcb_median_l2": lcb,
                "median_gate_pass": median >= float(gates["minimum_pairwise_median_log_response_l2"]),
                "p10_gate_pass": p10 >= float(gates["minimum_pairwise_p10_log_response_l2"]),
                "material_gate_pass": material_rms >= float(gates["minimum_pairwise_material_delta_vector_rms"]),
                "bootstrap_gate_pass": lcb >= float(gates["minimum_pairwise_group_bootstrap_95_lcb_median_l2"]),
            })
    automatic_pass = len(comparisons) == 3 and all(
        all(row[key] for key in ("median_gate_pass", "p10_gate_pass", "material_gate_pass", "bootstrap_gate_pass"))
        for row in comparisons
    )
    population_hash = hashlib.sha256(
        spectra.astype("<f8", copy=False).tobytes()
        + "\n".join(groups.tolist()).encode()
    ).hexdigest()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": "RF3.D6",
        "population": {
            "eligible_records": len(spectra),
            "sample_groups": len(np.unique(groups)),
            "chapters": len(np.unique(chapters)),
            "population_sha256": population_hash,
            "rejection_counts": rejection,
        },
        "pairwise_comparisons": comparisons,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass" if automatic_pass else "decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    data = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
