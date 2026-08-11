"""Fresh-source confirmation of a bounded compound-Thomas residual model."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy import ndimage

from src.eval.absolute_cinema_archival_grain import (
    _acf_lags,
    _decode_scalar,
    _features,
    _radial_signature,
    _select_patches,
)
from src.film_physics.thomas_cluster_nps import thomas_cluster_gaussian_mark_nps

SCHEMA = "neuro_film.u6_p4cl_compound_thomas_archival_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cl_compound_thomas_archival_confirmation_report.v1"


class CompoundThomasArchivalError(RuntimeError):
    """Raised when the frozen contract, source, or execution order drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contract(root: Path, contract: Mapping[str, Any]) -> None:
    dataset = contract.get("dataset", {})
    candidate = contract.get("candidate", {})
    parent = contract.get("parent", {})
    if (
        contract.get("schema") != SCHEMA
        or dataset.get("development_source_count") != 4
        or dataset.get("confirmation_source_count") != 4
        or dataset.get("frames_per_source") != 10
        or candidate.get("scale_sigma_pixels") != 12.0
        or candidate.get("scale_bounds") != [0.25, 4.0]
        or candidate.get("fit_sample_count") != 256
        or candidate.get("evaluation_sample_count") != 512
    ):
        raise CompoundThomasArchivalError("P4CL frozen contract drift")
    evidence = root / str(parent["p4cj_evidence_path"])
    if not evidence.is_file() or sha256_file(evidence) != parent["p4cj_evidence_sha256"]:
        raise CompoundThomasArchivalError("P4CJ parent identity drift")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    if payload.get("decision") != parent["required_decision"]:
        raise CompoundThomasArchivalError("P4CJ parent decision drift")


def enumerate_selected_metadata(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = contract["dataset"]
    params: dict[str, Any] = {"pageSize": 200}
    rows: list[dict[str, Any]] = []
    for _ in range(16):
        response = requests.get(
            str(dataset["api_url"]),
            params=params,
            headers={"User-Agent": "neuro-film-u6-p4cl/1.0"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("datasetFiles", []))
        token = payload.get("nextPageTokenNullable")
        if not token:
            break
        params = {"pageSize": 200, "pageToken": token}
    else:
        raise CompoundThomasArchivalError("dataset listing exceeded page bound")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        name = str(row.get("name", ""))
        parts = name.split("/")
        if len(parts) == 3 and name.endswith(".png"):
            grouped[parts[0]][parts[1]].append(row)
    eligible: list[tuple[str, str, list[dict[str, Any]]]] = []
    for source, clips in grouped.items():
        first_clip = min(clips)
        frames = sorted(clips[first_clip], key=lambda item: str(item["name"]))
        if len(frames) == int(dataset["frames_per_source"]):
            eligible.append((source, first_clip, frames))
    chosen = sorted(eligible, key=lambda item: item[0])[4:12]
    if [source for source, _clip, _frames in chosen] != dataset["selected_source_names"]:
        raise CompoundThomasArchivalError("selected source names drift")
    lock = {
        "dataset_ref": dataset["ref"],
        "version": dataset["version"],
        "selected": [
            {
                "source": source,
                "clip": clip,
                "files": [
                    {"name": str(row["name"]), "bytes": int(row["totalBytes"])}
                    for row in frames
                ],
            }
            for source, clip, frames in chosen
        ],
    }
    compact = (json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if sha256_bytes(compact) != dataset["selected_metadata_lock_sha256"]:
        raise CompoundThomasArchivalError("selected metadata lock drift")
    flattened = [row for _source, _clip, frames in chosen for row in frames]
    if sum(int(row["totalBytes"]) for row in flattened) != dataset["expected_total_bytes"]:
        raise CompoundThomasArchivalError("selected byte total drift")
    return flattened


def acquire_selected_files(
    contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], scratch: Path
) -> list[dict[str, Any]]:
    dataset = contract["dataset"]
    scratch.mkdir(parents=True, exist_ok=True)
    if len(rows) != 80:
        raise CompoundThomasArchivalError("selected file count drift")
    base = "https://www.kaggle.com/api/v1/datasets/download/" + str(dataset["ref"])
    acquired: list[dict[str, Any]] = []
    downloaded = 0
    for row in rows:
        name = str(row["name"])
        expected_bytes = int(row["totalBytes"])
        path = scratch.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size != expected_bytes:
            temporary = path.with_name(path.name + ".download")
            temporary.unlink(missing_ok=True)
            with requests.get(
                base,
                params={"filename": name, "datasetVersionNumber": int(dataset["version"])},
                headers={"User-Agent": "neuro-film-u6-p4cl/1.0"},
                timeout=120,
                stream=True,
            ) as response:
                response.raise_for_status()
                with temporary.open("xb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > int(dataset["maximum_download_bytes"]):
                            raise CompoundThomasArchivalError("download exceeded frozen cap")
                        handle.write(chunk)
            if temporary.stat().st_size != expected_bytes:
                raise CompoundThomasArchivalError("download size mismatch")
            temporary.replace(path)
        acquired.append(
            {
                "name": name,
                "source": name.split("/")[0],
                "bytes": expected_bytes,
                "sha256": sha256_file(path),
                "path": path,
            }
        )
    return acquired


def _thomas_power(contract: Mapping[str, Any], size: int) -> np.ndarray:
    parent = contract["parent"]
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    power = thomas_cluster_gaussian_mark_nps(
        frequency,
        float(parent["p4bs_particle_sigma_pixels"]),
        float(parent["p4bs_cluster_sigma_pixels"]),
        float(parent["p4bs_mean_offspring"]),
    ).astype(np.float64, copy=True)
    power[0, 0] = 0.0
    return power


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - float(values.mean())
    rms = float(np.sqrt(np.mean(np.square(values))))
    if not math.isfinite(rms) or rms <= 1e-12:
        raise CompoundThomasArchivalError("degenerate synthetic field")
    return np.ascontiguousarray(values / rms)


def _spectral_field(power: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(power.shape)
    spectrum = np.fft.fft2(white)
    magnitude = np.abs(spectrum)
    phase = np.divide(spectrum, magnitude, out=np.ones_like(spectrum), where=magnitude > 0)
    return _normalized(np.fft.ifft2(phase * np.sqrt(power)).real)


def _base_and_scale_fields(
    contract: Mapping[str, Any], count: int, *, seed_offset: int
) -> tuple[np.ndarray, np.ndarray]:
    candidate = contract["candidate"]
    size = int(contract["patch_observation"]["patch_size_pixels"])
    power = _thomas_power(contract, size)
    bases = np.empty((count, size, size), dtype=np.float32)
    scales = np.empty_like(bases)
    for index in range(count):
        rng = np.random.default_rng(int(candidate["candidate_seed"]) + seed_offset + index)
        bases[index] = _spectral_field(power, rng)
        scale = ndimage.gaussian_filter(
            rng.standard_normal((size, size)),
            sigma=float(candidate["scale_sigma_pixels"]),
            mode="wrap",
        )
        scales[index] = _normalized(scale)
    return bases, scales


def _compound_fields(
    bases: np.ndarray, scales: np.ndarray, log_scale_std: float, bounds: Sequence[float]
) -> np.ndarray:
    envelope = np.exp(log_scale_std * scales - 0.5 * log_scale_std**2)
    envelope = np.clip(envelope, float(bounds[0]), float(bounds[1]))
    values = bases.astype(np.float64) * envelope
    values -= values.mean(axis=(1, 2), keepdims=True)
    rms = np.sqrt(np.mean(np.square(values), axis=(1, 2), keepdims=True))
    if not np.isfinite(rms).all() or np.any(rms <= 1e-12):
        raise CompoundThomasArchivalError("compound field normalization failed")
    return np.ascontiguousarray(values / rms, dtype=np.float32)


def _feature_matrix(fields: np.ndarray) -> np.ndarray:
    return np.asarray([_features(field) for field in fields], dtype=np.float64)


def _fit_log_scale_std(
    contract: Mapping[str, Any], target_local_energy_cv: float
) -> tuple[float, dict[str, Any]]:
    candidate = contract["candidate"]
    bases, scales = _base_and_scale_fields(
        contract, int(candidate["fit_sample_count"]), seed_offset=0
    )

    def prediction(value: float) -> float:
        features = _feature_matrix(
            _compound_fields(bases, scales, value, candidate["scale_bounds"])
        )
        return float(np.median(features[:, 2]))

    low, high = 0.0, 1.5
    low_value, high_value = prediction(low), prediction(high)
    if not low_value <= target_local_energy_cv <= high_value:
        raise CompoundThomasArchivalError(
            "development local-energy target is outside frozen fit interval"
        )
    for _ in range(32):
        middle = (low + high) * 0.5
        if prediction(middle) < target_local_energy_cv:
            low = middle
        else:
            high = middle
    fitted = (low + high) * 0.5
    return fitted, {
        "target_local_energy_cv": target_local_energy_cv,
        "endpoint_predictions": [low_value, high_value],
        "fitted_prediction": prediction(fitted),
    }


def _decode_source_patches(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, list[np.ndarray]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        image = _decode_scalar(Path(row["path"]), contract)
        grouped[str(row["source"])].extend(_select_patches(image, contract))
    expected = int(contract["patch_observation"]["patches_per_frame"]) * int(
        contract["dataset"]["frames_per_source"]
    )
    if any(len(patches) != expected for patches in grouped.values()):
        raise CompoundThomasArchivalError("source patch count drift")
    return dict(grouped)


def _model_freeze(
    contract: Mapping[str, Any], fitted: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    candidate = contract["candidate"]
    count = int(candidate["evaluation_sample_count"])
    bases, scales = _base_and_scale_fields(contract, count, seed_offset=100000)
    gaussian_features = _feature_matrix(bases)
    compound = _compound_fields(bases, scales, fitted, candidate["scale_bounds"])
    compound_features = _feature_matrix(compound)
    gaussian_median = np.median(gaussian_features, axis=0)
    gaussian_mad = np.maximum(
        np.median(np.abs(gaussian_features - gaussian_median), axis=0), 1e-6
    )

    size = int(contract["patch_observation"]["patch_size_pixels"])
    power = _thomas_power(contract, size)
    target = _normalized(np.fft.ifft2(np.sqrt(power)).real)
    target_nps = _radial_signature(target, contract)
    target_acf = _acf_lags(target, contract["second_order"]["acf_lags_yx"])

    def mean_vectors(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.mean([_radial_signature(field, contract) for field in fields], axis=0),
            np.mean(
                [
                    _acf_lags(field, contract["second_order"]["acf_lags_yx"])
                    for field in fields
                ],
                axis=0,
            ),
        )

    gaussian_nps, gaussian_acf = mean_vectors(bases)
    compound_nps, compound_acf = mean_vectors(compound)
    gaussian_nps_error = float(np.sqrt(np.mean(np.square(gaussian_nps - target_nps))))
    gaussian_acf_error = float(np.sqrt(np.mean(np.square(gaussian_acf - target_acf))))
    compound_nps_error = float(np.sqrt(np.mean(np.square(compound_nps - target_nps))))
    compound_acf_error = float(np.sqrt(np.mean(np.square(compound_acf - target_acf))))
    second = contract["second_order"]
    freeze = {
        "fitted_log_scale_std": fitted,
        "gaussian_feature_median": gaussian_median.tolist(),
        "gaussian_feature_mad": gaussian_mad.tolist(),
        "compound_feature_median": np.median(compound_features, axis=0).tolist(),
        "second_order": {
            "gaussian_nps_error": gaussian_nps_error,
            "compound_nps_error": compound_nps_error,
            "candidate_to_gaussian_nps_error_ratio": compound_nps_error
            / max(gaussian_nps_error, 1e-12),
            "gaussian_acf_error": gaussian_acf_error,
            "compound_acf_error": compound_acf_error,
            "candidate_to_gaussian_acf_error_ratio": compound_acf_error
            / max(gaussian_acf_error, 1e-12),
        },
    }
    freeze["second_order"]["nps_retained"] = bool(
        freeze["second_order"]["candidate_to_gaussian_nps_error_ratio"]
        <= second["maximum_candidate_to_gaussian_nps_error_ratio"]
    )
    freeze["second_order"]["acf_retained"] = bool(
        freeze["second_order"]["candidate_to_gaussian_acf_error_ratio"]
        <= second["maximum_candidate_to_gaussian_acf_error_ratio"]
    )
    freeze["model_freeze_id"] = sha256_bytes(canonical_json(freeze))
    return freeze, gaussian_median, gaussian_mad, np.median(compound_features, axis=0)


def _score_sources(
    patches: Mapping[str, Sequence[np.ndarray]],
    gaussian_median: np.ndarray,
    gaussian_mad: np.ndarray,
    compound_median: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(patches):
        observed = np.median(_feature_matrix(np.asarray(patches[source])), axis=0)
        gaussian_distance = float(
            np.sqrt(np.mean(np.square((observed - gaussian_median) / gaussian_mad)))
        )
        compound_distance = float(
            np.sqrt(np.mean(np.square((observed - compound_median) / gaussian_mad)))
        )
        ratio = compound_distance / max(gaussian_distance, 1e-12)
        rows.append(
            {
                "source_id": source,
                "patch_count": len(patches[source]),
                "observed_feature_median": observed.tolist(),
                "gaussian_feature_distance": gaussian_distance,
                "compound_feature_distance": compound_distance,
                "compound_to_gaussian_distance_ratio": ratio,
                "improvement_over_gaussian": 1.0 - ratio,
            }
        )
    return rows


def evaluate_compound_thomas(root: Path, contract_path: Path, scratch: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(root, contract)
    metadata = enumerate_selected_metadata(contract)
    acquired = acquire_selected_files(contract, metadata, scratch)
    dataset = contract["dataset"]
    development_names = set(
        dataset["selected_source_names"][: int(dataset["development_source_count"])]
    )
    development_rows = [row for row in acquired if row["source"] in development_names]
    confirmation_rows = [row for row in acquired if row["source"] not in development_names]

    development = _decode_source_patches(development_rows, contract)
    development_features = np.concatenate(
        [_feature_matrix(np.asarray(development[source])) for source in sorted(development)],
        axis=0,
    )
    target = float(np.median(development_features[:, 2]))
    fitted, fit = _fit_log_scale_std(contract, target)
    freeze, gaussian_median, gaussian_mad, compound_median = _model_freeze(
        contract, fitted
    )
    confirmation_decode_count_at_freeze = 0
    development_scores = _score_sources(
        development, gaussian_median, gaussian_mad, compound_median
    )

    confirmation = _decode_source_patches(confirmation_rows, contract)
    confirmation_scores = _score_sources(
        confirmation, gaussian_median, gaussian_mad, compound_median
    )
    ratios = np.asarray(
        [row["compound_to_gaussian_distance_ratio"] for row in confirmation_scores],
        dtype=np.float64,
    )
    gates_contract = contract["confirmation_gates"]
    gates = {
        "source_wins": int(np.count_nonzero(ratios < 1.0))
        >= int(gates_contract["minimum_sources_beating_gaussian_feature_distance"]),
        "median_feature_distance_improvement": float(np.median(1.0 - ratios))
        >= float(gates_contract["minimum_median_feature_distance_improvement"]),
        "worst_feature_distance": float(np.max(ratios))
        <= float(gates_contract["maximum_worst_feature_distance_ratio"]),
        "second_order_nps_retention": bool(freeze["second_order"]["nps_retained"]),
        "second_order_acf_retention": bool(freeze["second_order"]["acf_retained"]),
        "finite_bounded_fields": math.isfinite(fitted) and 0.0 <= fitted <= 1.5,
        "confirmation_unread_at_freeze": confirmation_decode_count_at_freeze == 0,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_bytes(canonical_json(contract)),
        "source_metadata_lock_sha256": dataset["selected_metadata_lock_sha256"],
        "source_total_bytes": sum(int(row["bytes"]) for row in acquired),
        "source_sha256": [row["sha256"] for row in acquired],
        "development_source_count": len(development),
        "confirmation_source_count": len(confirmation),
        "confirmation_decode_count_at_freeze": confirmation_decode_count_at_freeze,
        "fit": fit,
        "model_freeze": freeze,
        "development_scores": development_scores,
        "confirmation_scores": confirmation_scores,
        "confirmation_summary": {
            "sources_beating_gaussian": int(np.count_nonzero(ratios < 1.0)),
            "median_feature_distance_improvement": float(np.median(1.0 - ratios)),
            "worst_feature_distance_ratio": float(np.max(ratios)),
        },
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "PASS_GENERIC_BOUNDED_COMPOUND_THOMAS_DEVELOPMENT"
            if automatic_pass
            else "FAIL_CLOSED_COMPOUND_THOMAS_HIGH_ORDER_TRANSFER"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = sha256_bytes(canonical_json(report))
    return report
