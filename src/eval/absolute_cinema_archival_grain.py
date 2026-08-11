"""Independent archival-film residual challenge for the retained P4BS spectrum."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

from src.film_physics.thomas_cluster_nps import thomas_cluster_gaussian_mark_nps

SCHEMA = "neuro_film.u6_p4cj_absolute_cinema_archival_grain_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cj_absolute_cinema_archival_grain_report.v1"


class AbsoluteCinemaArchivalGrainError(RuntimeError):
    """Raised when the frozen source, inputs, or experiment contract drift."""


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


def _validate_contract(contract: Mapping[str, Any]) -> None:
    selection = contract.get("selection", {})
    patch = contract.get("patch_observation", {})
    second = contract.get("second_order", {})
    phase = contract.get("phase_sensitive", {})
    if (
        contract.get("schema") != SCHEMA
        or selection.get("expected_source_count") != 4
        or selection.get("expected_frames_per_source") != 10
        or selection.get("expected_total_bytes") != 33111370
        or patch.get("patch_size_pixels") != 96
        or patch.get("patches_per_frame") != 8
        or second.get("radial_bin_count") != 24
        or second.get("error_metric")
        != "root mean square error over the frozen signature or lag vector"
        or phase.get("phase_scrambles_per_patch") != 128
        or phase.get("features")
        != [
            "skew",
            "excess_kurtosis",
            "local_energy_cv_8x8",
            "absolute_excursion_component_density_z2p5",
        ]
        or contract.get("automatic_gates", {}).get(
            "require_two_byte_identical_reports"
        )
        is not True
    ):
        raise AbsoluteCinemaArchivalGrainError("P4CJ frozen contract drift")


def _load_parent(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    parent = contract["parent"]
    path = root / str(parent["p4bs_decision_path"])
    if not path.is_file() or sha256_file(path) != parent["p4bs_decision_sha256"]:
        raise AbsoluteCinemaArchivalGrainError("P4BS parent identity drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if (
        payload.get("decision") != parent["required_decision"]
        or results.get("thomas_particle_sigma_pixels")
        != parent["particle_sigma_pixels"]
        or results.get("thomas_cluster_sigma_pixels")
        != parent["cluster_sigma_pixels"]
        or results.get("thomas_mean_offspring") != parent["mean_offspring"]
        or results.get("gaussian_particle_sigma_pixels")
        != parent["gaussian_control_sigma_pixels"]
    ):
        raise AbsoluteCinemaArchivalGrainError("P4BS parent facts drift")
    return payload


def enumerate_selected_metadata(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct the exact metadata-only source selection from Kaggle."""
    primary = contract["primary_source"]
    url = str(primary["dataset_api_url"])
    params: dict[str, Any] = {"pageSize": 200}
    rows: list[dict[str, Any]] = []
    for _ in range(16):
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "neuro-film-u6-p4cj/1.0"},
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
        raise AbsoluteCinemaArchivalGrainError("dataset listing exceeded page bound")

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
        if len(frames) == int(contract["selection"]["expected_frames_per_source"]):
            eligible.append((source, first_clip, frames))
    chosen = sorted(eligible, key=lambda item: item[0])[
        : int(contract["selection"]["expected_source_count"])
    ]
    observed_names = [item[0] for item in chosen]
    if observed_names != contract["selection"]["selected_source_names"]:
        raise AbsoluteCinemaArchivalGrainError("selected source names drift")

    lock = {
        "dataset_ref": primary["dataset_ref"],
        "version": primary["dataset_version"],
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
    if sha256_bytes(compact) != contract["selection"]["selected_metadata_lock_sha256"]:
        raise AbsoluteCinemaArchivalGrainError("selected metadata lock drift")
    flattened = [row for _source, _clip, frames in chosen for row in frames]
    if sum(int(row["totalBytes"]) for row in flattened) != contract["selection"][
        "expected_total_bytes"
    ]:
        raise AbsoluteCinemaArchivalGrainError("selected byte total drift")
    return flattened


def acquire_selected_files(
    contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], scratch: Path
) -> list[dict[str, Any]]:
    scratch = scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    if len(rows) != 40:
        raise AbsoluteCinemaArchivalGrainError("selected file count drift")
    primary = contract["primary_source"]
    base = "https://www.kaggle.com/api/v1/datasets/download/" + str(
        primary["dataset_ref"]
    )
    acquired: list[dict[str, Any]] = []
    downloaded = 0
    for row in rows:
        name = str(row["name"])
        expected_bytes = int(row["totalBytes"])
        path = scratch.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size != expected_bytes:
            temporary = path.with_name(path.name + ".download")
            if temporary.exists():
                temporary.unlink()
            with requests.get(
                base,
                params={
                    "filename": name,
                    "datasetVersionNumber": int(primary["dataset_version"]),
                },
                headers={"User-Agent": "neuro-film-u6-p4cj/1.0"},
                timeout=120,
                stream=True,
            ) as response:
                response.raise_for_status()
                with temporary.open("xb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            downloaded += len(chunk)
                            if downloaded > int(
                                contract["selection"]["maximum_download_bytes"]
                            ):
                                raise AbsoluteCinemaArchivalGrainError(
                                    "download exceeded frozen byte cap"
                                )
            if temporary.stat().st_size != expected_bytes:
                raise AbsoluteCinemaArchivalGrainError("download size mismatch")
            temporary.replace(path)
        if path.stat().st_size != expected_bytes:
            raise AbsoluteCinemaArchivalGrainError("local source size mismatch")
        acquired.append(
            {
                "name": name,
                "bytes": expected_bytes,
                "sha256": sha256_file(path),
                "path": path,
            }
        )
    return acquired


def _decode_scalar(path: Path, contract: Mapping[str, Any]) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        if mode not in contract["decode"]["accepted_png_modes"]:
            raise AbsoluteCinemaArchivalGrainError(f"unsupported PNG mode: {mode}")
        values = np.asarray(image)
    if values.dtype != np.uint8:
        raise AbsoluteCinemaArchivalGrainError("decoded source is not uint8")
    if mode == "L":
        scalar = values.astype(np.float64) / 255.0
    else:
        if mode == "RGBA":
            if np.any(values[..., 3] != 255):
                raise AbsoluteCinemaArchivalGrainError("RGBA source is not opaque")
            values = values[..., :3]
        scalar = np.tensordot(
            values.astype(np.float64) / 255.0,
            np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64),
            axes=([-1], [0]),
        )
    if scalar.ndim != 2 or not np.all(np.isfinite(scalar)):
        raise AbsoluteCinemaArchivalGrainError("invalid decoded scalar image")
    return np.ascontiguousarray(scalar)


def _select_patches(
    scalar: np.ndarray, contract: Mapping[str, Any]
) -> list[np.ndarray]:
    rule = contract["patch_observation"]
    size = int(rule["patch_size_pixels"])
    border = int(rule["border_pixels"])
    low = ndimage.gaussian_filter(
        scalar, sigma=float(rule["lowpass_gaussian_sigma_pixels"]), mode="reflect"
    )
    gy, gx = np.gradient(low)
    minimum, maximum = map(float, rule["accepted_patch_mean_interval"])
    candidates: list[tuple[float, int, int]] = []
    for y in range(border, scalar.shape[0] - border - size + 1, size):
        for x in range(border, scalar.shape[1] - border - size + 1, size):
            low_patch = low[y : y + size, x : x + size]
            mean = float(np.mean(low_patch, dtype=np.float64))
            if minimum <= mean <= maximum:
                score = math.sqrt(
                    float(
                        np.mean(
                            np.square(gy[y : y + size, x : x + size])
                            + np.square(gx[y : y + size, x : x + size]),
                            dtype=np.float64,
                        )
                    )
                )
                candidates.append((score, y, x))
    candidates.sort()
    count = int(rule["patches_per_frame"])
    if len(candidates) < count:
        raise AbsoluteCinemaArchivalGrainError("insufficient flat patch candidates")
    output: list[np.ndarray] = []
    for _score, y, x in candidates[:count]:
        residual = scalar[y : y + size, x : x + size] - low[
            y : y + size, x : x + size
        ]
        residual = residual - float(np.mean(residual, dtype=np.float64))
        rms = math.sqrt(float(np.mean(np.square(residual), dtype=np.float64)))
        if rms < 1e-6:
            raise AbsoluteCinemaArchivalGrainError("selected residual is degenerate")
        output.append(np.ascontiguousarray(residual / rms))
    return output


def _radial_signature(values: np.ndarray, contract: Mapping[str, Any]) -> np.ndarray:
    second = contract["second_order"]
    height, width = values.shape
    power = np.square(np.abs(np.fft.fft2(values))) / float(height * width)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    edges = np.linspace(
        float(second["minimum_frequency_cycles_per_pixel"]),
        float(second["maximum_frequency_cycles_per_pixel"]),
        int(second["radial_bin_count"]) + 1,
    )
    bands = []
    for lower, upper in pairwise(edges):
        mask = (frequency >= lower) & (frequency < upper)
        bands.append(float(np.mean(power[mask], dtype=np.float64)))
    bands_array = np.asarray(bands, dtype=np.float64)
    if np.any(bands_array <= 0.0) or not np.all(np.isfinite(bands_array)):
        raise AbsoluteCinemaArchivalGrainError("invalid radial signature")
    signature = np.log(bands_array / float(np.sum(bands_array)))
    signature -= float(np.mean(signature))
    norm = float(np.linalg.norm(signature))
    return np.ascontiguousarray(signature / norm)


def _acf_lags(values: np.ndarray, lags: Sequence[Sequence[int]]) -> np.ndarray:
    power = np.square(np.abs(np.fft.fft2(values)))
    covariance = np.fft.ifft2(power).real
    covariance /= float(covariance[0, 0])
    return np.asarray([covariance[int(y), int(x)] for y, x in lags])


def _model_vectors(
    contract: Mapping[str, Any], *, thomas: bool
) -> tuple[np.ndarray, np.ndarray]:
    size = int(contract["patch_observation"]["patch_size_pixels"])
    parent = contract["parent"]
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    if thomas:
        power = thomas_cluster_gaussian_mark_nps(
            frequency,
            float(parent["particle_sigma_pixels"]),
            float(parent["cluster_sigma_pixels"]),
            float(parent["mean_offspring"]),
        )
    else:
        sigma = float(parent["gaussian_control_sigma_pixels"])
        power = np.exp(-4.0 * math.pi**2 * sigma * sigma * np.square(frequency))
    # Observed patch residuals are explicitly zero-mean.  Remove the model DC
    # term before deriving either comparison vector so ACF error measures only
    # spatial covariance rather than an artificial mean offset.
    power = np.array(power, dtype=np.float64, copy=True)
    power[0, 0] = 0.0
    field = np.fft.ifft2(np.sqrt(power)).real
    # The signature and ACF depend only on magnitude power.  Constructing this
    # deterministic zero-phase field reuses exactly the same analysis path.
    signature = _radial_signature(field, contract)
    acf = _acf_lags(field, contract["second_order"]["acf_lags_yx"])
    return signature, acf


def _features(values: np.ndarray) -> np.ndarray:
    flattened = values.ravel()
    skew = float(np.mean(np.power(flattened, 3), dtype=np.float64))
    kurtosis = float(np.mean(np.power(flattened, 4), dtype=np.float64) - 3.0)
    energies = np.mean(
        np.square(values.reshape(12, 8, 12, 8)).transpose(0, 2, 1, 3),
        axis=(2, 3),
    )
    energy_cv = float(np.std(energies) / np.mean(energies))
    labels, count = ndimage.label(
        np.abs(values) >= 2.5, structure=np.ones((3, 3), dtype=np.uint8)
    )
    del labels
    component_density = float(count / values.size)
    return np.asarray([skew, kurtosis, energy_cv, component_density], dtype=np.float64)


def _phase_scramble(
    values: np.ndarray, *, seed: int, patch_index: int, scramble_index: int
) -> np.ndarray:
    key = hashlib.sha256(
        f"{seed}:{patch_index}:{scramble_index}".encode("ascii")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    observed = np.fft.rfft2(values)
    random_phase = np.fft.rfft2(rng.standard_normal(values.shape))
    phase = np.exp(1j * np.angle(random_phase))
    surrogate = np.fft.irfft2(np.abs(observed) * phase, s=values.shape)
    surrogate -= float(np.mean(surrogate))
    surrogate /= math.sqrt(float(np.mean(np.square(surrogate))))
    return np.ascontiguousarray(surrogate)


def _phase_source_result(
    patches: Sequence[tuple[int, np.ndarray]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    phase = contract["phase_sensitive"]
    observed = np.asarray([_features(values) for _frame, values in patches])
    observed_full = np.median(observed, axis=0)
    frame_indices = np.asarray([frame for frame, _values in patches])
    masks = [frame_indices % 2 == 0, frame_indices % 2 == 1]
    observed_halves = [np.median(observed[mask], axis=0) for mask in masks]
    scramble_count = int(phase["phase_scrambles_per_patch"])
    surrogate_full = np.empty((scramble_count, observed.shape[1]), dtype=np.float64)
    surrogate_halves = [np.empty_like(surrogate_full), np.empty_like(surrogate_full)]
    for scramble_index in range(scramble_count):
        generated = np.asarray(
            [
                _features(
                    _phase_scramble(
                        values,
                        seed=int(phase["counter_seed"]),
                        patch_index=index,
                        scramble_index=scramble_index,
                    )
                )
                for index, (_frame, values) in enumerate(patches)
            ]
        )
        surrogate_full[scramble_index] = np.median(generated, axis=0)
        for half, mask in enumerate(masks):
            surrogate_halves[half][scramble_index] = np.median(
                generated[mask], axis=0
            )

    def zscore(observed_value: np.ndarray, controls: np.ndarray) -> np.ndarray:
        center = np.mean(controls, axis=0)
        scale = np.std(controls, axis=0, ddof=1)
        if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
            raise AbsoluteCinemaArchivalGrainError("degenerate phase control")
        return (observed_value - center) / scale

    full_z = zscore(observed_full, surrogate_full)
    half_z = [zscore(value, control) for value, control in zip(observed_halves, surrogate_halves, strict=True)]
    material = []
    for index, name in enumerate(phase["features"]):
        stable_sign = (
            np.sign(full_z[index]) == np.sign(half_z[0][index])
            and np.sign(full_z[index]) == np.sign(half_z[1][index])
        )
        passed = bool(
            abs(full_z[index]) >= float(phase["material_z_threshold"])
            and abs(half_z[0][index]) >= float(phase["half_split_z_threshold"])
            and abs(half_z[1][index]) >= float(phase["half_split_z_threshold"])
            and stable_sign
        )
        material.append(passed)
    return {
        "feature_names": list(phase["features"]),
        "observed_medians": observed_full.tolist(),
        "full_z": full_z.tolist(),
        "even_frame_z": half_z[0].tolist(),
        "odd_frame_z": half_z[1].tolist(),
        "material_features": material,
        "material_feature_names": [
            name for name, passed in zip(phase["features"], material, strict=True) if passed
        ],
        "source_has_material_phase_evidence": sum(material)
        >= int(phase["minimum_features_per_source"]),
    }


def evaluate_absolute_cinema_archival_grain(
    root: Path, contract_path: Path, scratch: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract)
    parent = _load_parent(root, contract)
    metadata = enumerate_selected_metadata(contract)
    acquired = acquire_selected_files(contract, metadata, scratch)
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in acquired:
        sources[str(row["name"]).split("/")[0]].append(row)

    thomas_signature, thomas_acf = _model_vectors(contract, thomas=True)
    gaussian_signature, gaussian_acf = _model_vectors(contract, thomas=False)
    source_results: list[dict[str, Any]] = []
    nps_ratios: list[float] = []
    acf_ratios: list[float] = []
    for source in contract["selection"]["selected_source_names"]:
        patches: list[tuple[int, np.ndarray]] = []
        files = sorted(sources[source], key=lambda row: row["name"])
        for frame_index, row in enumerate(files):
            scalar = _decode_scalar(row["path"], contract)
            patches.extend(
                (frame_index, values)
                for values in _select_patches(scalar, contract)
            )
        expected_patches = int(contract["selection"]["expected_frames_per_source"]) * int(
            contract["patch_observation"]["patches_per_frame"]
        )
        if len(patches) != expected_patches:
            raise AbsoluteCinemaArchivalGrainError("observed patch count drift")
        observed_signature = np.median(
            np.asarray([_radial_signature(values, contract) for _frame, values in patches]),
            axis=0,
        )
        observed_signature /= float(np.linalg.norm(observed_signature))
        observed_acf = np.median(
            np.asarray(
                [
                    _acf_lags(values, contract["second_order"]["acf_lags_yx"])
                    for _frame, values in patches
                ]
            ),
            axis=0,
        )
        nps_thomas = math.sqrt(float(np.mean(np.square(observed_signature - thomas_signature))))
        nps_gaussian = math.sqrt(float(np.mean(np.square(observed_signature - gaussian_signature))))
        acf_thomas = math.sqrt(float(np.mean(np.square(observed_acf - thomas_acf))))
        acf_gaussian = math.sqrt(float(np.mean(np.square(observed_acf - gaussian_acf))))
        nps_ratio = nps_thomas / nps_gaussian
        acf_ratio = acf_thomas / acf_gaussian
        nps_ratios.append(nps_ratio)
        acf_ratios.append(acf_ratio)
        source_results.append(
            {
                "source_id": source,
                "frame_count": len(files),
                "patch_count": len(patches),
                "file_sha256": [row["sha256"] for row in files],
                "nps_thomas_rmse": nps_thomas,
                "nps_gaussian_rmse": nps_gaussian,
                "nps_error_ratio": nps_ratio,
                "acf_thomas_rmse": acf_thomas,
                "acf_gaussian_rmse": acf_gaussian,
                "acf_error_ratio": acf_ratio,
                "phase_sensitive": _phase_source_result(patches, contract),
            }
        )

    second = contract["second_order"]
    nps_wins = sum(ratio < 1.0 for ratio in nps_ratios)
    acf_wins = sum(ratio < 1.0 for ratio in acf_ratios)
    median_nps_improvement = 1.0 - float(np.median(nps_ratios))
    median_acf_improvement = 1.0 - float(np.median(acf_ratios))
    second_order_checks = {
        "nps_source_wins": nps_wins
        >= int(second["minimum_sources_where_thomas_beats_gaussian_nps"]),
        "acf_source_wins": acf_wins
        >= int(second["minimum_sources_where_thomas_beats_gaussian_acf"]),
        "nps_median_improvement": median_nps_improvement
        >= float(second["minimum_median_nps_improvement_over_gaussian"]),
        "acf_median_improvement": median_acf_improvement
        >= float(second["minimum_median_acf_improvement_over_gaussian"]),
        "nps_worst": max(nps_ratios)
        <= float(second["maximum_worst_nps_error_ratio"]),
        "acf_worst": max(acf_ratios)
        <= float(second["maximum_worst_acf_error_ratio"]),
    }
    second_order_pass = all(second_order_checks.values())
    material_sources = sum(
        bool(row["phase_sensitive"]["source_has_material_phase_evidence"])
        for row in source_results
    )
    phase_material = material_sources >= int(
        contract["phase_sensitive"]["minimum_sources_for_material_phase_evidence"]
    )
    if not second_order_pass:
        decision = "FAIL_CLOSE_P4BS_CROSS_SOURCE_SECOND_ORDER_TRANSFER"
    elif phase_material:
        decision = "PASS_SECOND_ORDER_ONLY_REJECT_GAUSSIAN_COMPLETE_STRUCTURE"
    else:
        decision = "PASS_GENERIC_ARCHIVAL_SECOND_ORDER_PHASE_NULL"

    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "source_metadata_lock_sha256": contract["selection"][
            "selected_metadata_lock_sha256"
        ],
        "source_count": len(source_results),
        "frame_count": len(acquired),
        "source_total_bytes": sum(row["bytes"] for row in acquired),
        "source_results": source_results,
        "second_order": {
            "nps_sources_beating_gaussian": nps_wins,
            "acf_sources_beating_gaussian": acf_wins,
            "median_nps_improvement_over_gaussian": median_nps_improvement,
            "median_acf_improvement_over_gaussian": median_acf_improvement,
            "worst_nps_error_ratio": max(nps_ratios),
            "worst_acf_error_ratio": max(acf_ratios),
            "checks": second_order_checks,
            "automatic_pass": second_order_pass,
        },
        "phase_sensitive": {
            "sources_with_material_phase_evidence": material_sources,
            "material_phase_evidence": phase_material,
        },
        "decision": decision,
        "automatic_pass": second_order_pass,
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_payload = dict(report)
    stable_payload.pop("source_results")
    stable_payload["source_result_identities"] = [
        sha256_bytes(canonical_json(row)) for row in source_results
    ]
    report["stable_evidence_id"] = sha256_bytes(canonical_json(stable_payload))
    return report


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "AbsoluteCinemaArchivalGrainError",
    "acquire_selected_files",
    "canonical_json",
    "enumerate_selected_metadata",
    "evaluate_absolute_cinema_archival_grain",
    "sha256_file",
]
