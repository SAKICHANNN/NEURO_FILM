"""Fresh-source audit of a phase-projected compound Thomas field."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.eval.absolute_cinema_archival_grain import (
    _acf_lags,
    _decode_scalar,
    _select_patches,
)
from src.eval.compound_thomas_archival_confirmation import (
    _base_and_scale_fields,
    _compound_fields,
    _feature_matrix,
    _normalized,
    _thomas_power,
)

SCHEMA = "neuro_film.u6_p4cm_phase_projected_thomas_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cm_phase_projected_thomas_confirmation_report.v1"


class PhaseProjectedThomasError(RuntimeError):
    """Raised when a frozen source, parent, or projection invariant drifts."""


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
    parent = contract.get("parent", {})
    candidate = contract.get("candidate", {})
    dataset = contract.get("dataset", {})
    if (
        contract.get("schema") != SCHEMA
        or parent.get("fixed_log_scale_std") != 0.2847407310619019
        or candidate.get("log_scale_std") != parent.get("fixed_log_scale_std")
        or candidate.get("evaluation_sample_count") != 512
        or dataset.get("source_count") != 4
        or dataset.get("frames_per_source") != 10
    ):
        raise PhaseProjectedThomasError("P4CM frozen contract drift")
    evidence_path = root / str(parent["p4cl_evidence_path"])
    if (
        not evidence_path.is_file()
        or sha256_file(evidence_path) != parent["p4cl_evidence_sha256"]
    ):
        raise PhaseProjectedThomasError("P4CL parent identity drift")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("decision") != parent["required_decision"]
        or evidence.get("model", {}).get("fitted_log_scale_std")
        != parent["fixed_log_scale_std"]
        or evidence.get("model", {}).get("model_freeze_id")
        != parent["fixed_model_freeze_id"]
    ):
        raise PhaseProjectedThomasError("P4CL parent facts drift")


def enumerate_selected_metadata(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = contract["dataset"]
    params: dict[str, Any] = {"pageSize": 200}
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    for _ in range(16):
        for attempt in range(3):
            try:
                response = session.get(
                    str(dataset["api_url"]),
                    params=params,
                    headers={"User-Agent": "neuro-film-u6-p4cm/1.0"},
                    timeout=60,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(float(attempt + 1))
        payload = response.json()
        rows.extend(payload.get("datasetFiles", []))
        token = payload.get("nextPageTokenNullable")
        if not token:
            break
        params = {"pageSize": 200, "pageToken": token}
    else:
        raise PhaseProjectedThomasError("dataset listing exceeded page bound")

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
    chosen = sorted(eligible, key=lambda item: item[0])[12:16]
    if [source for source, _clip, _frames in chosen] != dataset["selected_source_names"]:
        raise PhaseProjectedThomasError("selected source names drift")
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
        raise PhaseProjectedThomasError("selected metadata lock drift")
    flattened = [row for _source, _clip, frames in chosen for row in frames]
    if sum(int(row["totalBytes"]) for row in flattened) != dataset["expected_total_bytes"]:
        raise PhaseProjectedThomasError("selected byte total drift")
    return flattened


def acquire_selected_files(
    contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], scratch: Path
) -> list[dict[str, Any]]:
    dataset = contract["dataset"]
    if len(rows) != 40:
        raise PhaseProjectedThomasError("selected file count drift")
    scratch.mkdir(parents=True, exist_ok=True)
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
                headers={"User-Agent": "neuro-film-u6-p4cm/1.0"},
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
                            raise PhaseProjectedThomasError("download exceeded frozen cap")
                        handle.write(chunk)
            if temporary.stat().st_size != expected_bytes:
                raise PhaseProjectedThomasError("download size mismatch")
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


def _phase_projected_fields(
    contract: Mapping[str, Any], bases: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    candidate = contract["candidate"]
    compound = _compound_fields(
        bases,
        scales,
        float(candidate["log_scale_std"]),
        candidate["scale_bounds"],
    )
    power = _thomas_power(contract, compound.shape[1])
    # Preserve the analytical spectrum in float64.  Storing the projected
    # field in the float32 compound buffer destroys relative accuracy in the
    # deliberately tiny high-frequency tail.
    output = np.empty(compound.shape, dtype=np.float64)
    for index, field in enumerate(compound):
        # NumPy preserves float32 as complex64 on this runtime.  Promote before
        # FFT so the frozen analytical projection is not limited by the
        # compound buffer's storage precision.
        spectrum = np.fft.fft2(field.astype(np.float64))
        magnitude = np.abs(spectrum)
        phase = np.divide(
            spectrum, magnitude, out=np.ones_like(spectrum), where=magnitude > 0
        )
        phase[0, 0] = 0.0
        output[index] = _normalized(np.fft.ifft2(phase * np.sqrt(power)).real)
    return output


def _projection_errors(
    contract: Mapping[str, Any], gaussian: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    power = _thomas_power(contract, gaussian.shape[1])
    target = power / float(np.sum(power))
    reference = _normalized(np.fft.ifft2(np.sqrt(power)).real)
    target_acf = _acf_lags(reference, contract["second_order"]["acf_lags_yx"])
    max_power = 0.0
    max_acf = 0.0
    for field in candidate:
        observed = np.square(np.abs(np.fft.fft2(field)))
        observed /= float(np.sum(observed))
        mask = target > 0
        max_power = max(
            max_power,
            float(np.max(np.abs(observed[mask] - target[mask]) / target[mask])),
        )
        acf = _acf_lags(field, contract["second_order"]["acf_lags_yx"])
        max_acf = max(max_acf, float(np.max(np.abs(acf - target_acf))))
    return {
        "maximum_per_sample_power_relative_error": max_power,
        "maximum_per_sample_acf_absolute_error": max_acf,
    }


def _freeze_models(
    contract: Mapping[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    candidate = contract["candidate"]
    count = int(candidate["evaluation_sample_count"])
    bases, scales = _base_and_scale_fields(contract, count, seed_offset=0)
    projected = _phase_projected_fields(contract, bases, scales)
    gaussian_features = _feature_matrix(bases)
    projected_features = _feature_matrix(projected)
    gaussian_median = np.median(gaussian_features, axis=0)
    gaussian_mad = np.maximum(
        np.median(np.abs(gaussian_features - gaussian_median), axis=0), 1e-6
    )
    projection = _projection_errors(contract, bases, projected)
    freeze = {
        "gaussian_feature_median": gaussian_median.tolist(),
        "gaussian_feature_mad": gaussian_mad.tolist(),
        "projected_feature_median": np.median(projected_features, axis=0).tolist(),
        "projection": projection,
    }
    freeze["model_freeze_id"] = sha256_bytes(canonical_json(freeze))
    return freeze, gaussian_median, gaussian_mad, np.median(projected_features, axis=0)


def _decode_source_patches(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, list[np.ndarray]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].extend(
            _select_patches(_decode_scalar(Path(row["path"]), contract), contract)
        )
    expected = int(contract["patch_observation"]["patches_per_frame"]) * int(
        contract["dataset"]["frames_per_source"]
    )
    if any(len(patches) != expected for patches in grouped.values()):
        raise PhaseProjectedThomasError("source patch count drift")
    return dict(grouped)


def _score_sources(
    patches: Mapping[str, Sequence[np.ndarray]],
    gaussian_median: np.ndarray,
    gaussian_mad: np.ndarray,
    projected_median: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(patches):
        observed = np.median(_feature_matrix(np.asarray(patches[source])), axis=0)
        gaussian_distance = float(
            np.sqrt(np.mean(np.square((observed - gaussian_median) / gaussian_mad)))
        )
        projected_distance = float(
            np.sqrt(np.mean(np.square((observed - projected_median) / gaussian_mad)))
        )
        ratio = projected_distance / max(gaussian_distance, 1e-12)
        rows.append(
            {
                "source_id": source,
                "patch_count": len(patches[source]),
                "observed_feature_median": observed.tolist(),
                "gaussian_feature_distance": gaussian_distance,
                "projected_feature_distance": projected_distance,
                "projected_to_gaussian_distance_ratio": ratio,
                "improvement_over_gaussian": 1.0 - ratio,
            }
        )
    return rows


def evaluate_phase_projected_thomas(
    root: Path, contract_path: Path, scratch: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(root, contract)
    freeze, gaussian_median, gaussian_mad, projected_median = _freeze_models(contract)
    source_decode_count_at_freeze = 0
    metadata = enumerate_selected_metadata(contract)
    acquired = acquire_selected_files(contract, metadata, scratch)
    patches = _decode_source_patches(acquired, contract)
    scores = _score_sources(patches, gaussian_median, gaussian_mad, projected_median)
    ratios = np.asarray(
        [row["projected_to_gaussian_distance_ratio"] for row in scores],
        dtype=np.float64,
    )
    projection = freeze["projection"]
    second = contract["second_order"]
    confirmation = contract["confirmation_gates"]
    gates = {
        "source_wins": int(np.count_nonzero(ratios < 1.0))
        >= int(confirmation["minimum_sources_beating_gaussian_feature_distance"]),
        "median_feature_distance_improvement": float(np.median(1.0 - ratios))
        >= float(confirmation["minimum_median_feature_distance_improvement"]),
        "worst_feature_distance": float(np.max(ratios))
        <= float(confirmation["maximum_worst_feature_distance_ratio"]),
        "exact_power_projection": projection["maximum_per_sample_power_relative_error"]
        <= float(second["maximum_per_sample_power_relative_error"]),
        "exact_acf_projection": projection["maximum_per_sample_acf_absolute_error"]
        <= float(second["maximum_per_sample_acf_absolute_error"]),
        "finite_fields": all(math.isfinite(float(value)) for value in ratios),
        "source_pixels_unread_at_freeze": source_decode_count_at_freeze == 0,
    }
    automatic_pass = all(gates.values())
    dataset = contract["dataset"]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_bytes(canonical_json(contract)),
        "source_metadata_lock_sha256": dataset["selected_metadata_lock_sha256"],
        "source_total_bytes": sum(int(row["bytes"]) for row in acquired),
        "source_sha256": [row["sha256"] for row in acquired],
        "source_decode_count_at_model_freeze": source_decode_count_at_freeze,
        "model_freeze": freeze,
        "source_scores": scores,
        "summary": {
            "sources_beating_gaussian": int(np.count_nonzero(ratios < 1.0)),
            "median_feature_distance_improvement": float(np.median(1.0 - ratios)),
            "worst_feature_distance_ratio": float(np.max(ratios)),
        },
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "PASS_PHASE_PROJECTED_THOMAS_HIGH_ORDER_DEVELOPMENT"
            if automatic_pass
            else "FAIL_CLOSED_PHASE_PROJECTED_THOMAS_TRANSFER"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = sha256_bytes(canonical_json(report))
    return report
