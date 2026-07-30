"""Deterministic paired-patch extraction for the bounded FilmMatch source."""

from __future__ import annotations

import colorsys
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from src.eval.filmmatch_paired_source import (
    FilmMatchSourceAuditError,
    canonical_sha256,
    sequence_number,
    sha256_file,
    validate_download_manifest,
)


HUE_LABELS = ("red", "yellow", "green", "cyan", "blue", "magenta")


def slog3_to_linear_reflection(values: np.ndarray) -> np.ndarray:
    """Decode full-range S-Log3 using Sony's published piecewise formula."""

    encoded = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(encoded)):
        raise ValueError("S-Log3 values must be finite")
    threshold = 171.2102946929 / 1023.0
    return np.where(
        encoded >= threshold,
        np.power(10.0, (encoded * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01,
        (encoded * 1023.0 - 95.0)
        * 0.01125
        / (171.2102946929 - 95.0),
    )


def _hue_sector(representative_slog3: np.ndarray) -> str:
    linear = np.maximum(slog3_to_linear_reflection(representative_slog3), 0.0)
    maximum = float(np.max(linear))
    if maximum <= 0.0:
        raise ValueError("emissive representative has no positive signal")
    hue = colorsys.rgb_to_hsv(*(linear / maximum))[0]
    return HUE_LABELS[int(np.floor(hue * 6.0 + 0.5)) % 6]


def _sample_rectangle(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    half_x: int,
    half_y: int,
) -> np.ndarray:
    x = int(round(center_x))
    y = int(round(center_y))
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or x - half_x < 0
        or y - half_y < 0
        or x + half_x >= image.shape[1]
        or y + half_y >= image.shape[0]
    ):
        raise ValueError("sample rectangle escapes RGB image")
    return np.median(
        image[y - half_y : y + half_y + 1, x - half_x : x + half_x + 1],
        axis=(0, 1),
    )


def chart_sample_geometry(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    reflective = config["sampling"]["reflective"]
    output: list[dict[str, Any]] = []
    for chart in ("sg", "classic"):
        geometry = reflective[chart]
        origin_x, origin_y = map(float, geometry["center_origin_xy"])
        step_x, step_y = map(float, geometry["center_step_xy"])
        half_x, half_y = map(int, geometry["half_window_xy"])
        for row in range(int(geometry["rows"])):
            for column in range(int(geometry["columns"])):
                output.append(
                    {
                        "chart": chart,
                        "patch_row": row,
                        "patch_column": column,
                        "center_x": origin_x + column * step_x,
                        "center_y": origin_y + row * step_y,
                        "half_x": half_x,
                        "half_y": half_y,
                    }
                )
    if len(output) != int(reflective["expected_patches_per_pair"]):
        raise ValueError("reflective patch geometry count drift")
    return output


def linear_reflection_to_slog3(values: np.ndarray) -> np.ndarray:
    """Encode scene-linear reflection with Sony's published S-Log3 formula."""
    linear = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(linear)):
        raise ValueError("linear reflection values must be finite")
    threshold = 0.01125
    output = np.empty_like(linear)
    upper = linear >= threshold
    output[upper] = (
        420.0
        + np.log10((linear[upper] + 0.01) / 0.19) * 261.5
    ) / 1023.0
    output[~upper] = (
        linear[~upper] * (171.2102946929 - 95.0) / threshold + 95.0
    ) / 1023.0
    return output


def _load_rgb16(path: Path, *, stride: int) -> np.ndarray:
    pixels = tifffile.memmap(path)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint16:
        raise FilmMatchSourceAuditError(f"unexpected RGB16 TIFF: {path.name}")
    return np.asarray(pixels[::stride, ::stride], dtype=np.float64) / 65535.0


def _sample_reflective_pair(
    film_path: Path,
    digital_path: Path,
    *,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    sampling = config["sampling"]
    reflective = sampling["reflective"]
    height, width = map(int, reflective["canvas"])
    film = _load_rgb16(film_path, stride=int(reflective["film_stride"]))
    digital = _load_rgb16(digital_path, stride=int(reflective["digital_stride"]))
    if digital.shape != (height, width, 3):
        raise FilmMatchSourceAuditError("digital reflective canvas drift")
    homography = np.asarray(
        sampling["film_to_digital_reflective_homography_at_960x540"],
        dtype=np.float64,
    )
    target = cv2.warpPerspective(
        film, homography, (width, height), flags=cv2.INTER_LINEAR
    )
    geometry = chart_sample_geometry(config)
    def sample(image: np.ndarray, row: Mapping[str, Any]) -> np.ndarray:
        return _sample_rectangle(
            image,
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            half_x=int(row["half_x"]),
            half_y=int(row["half_y"]),
        )

    source_samples = np.stack(
        [sample(digital, row) for row in geometry]
    )
    target_samples = np.stack(
        [sample(target, row) for row in geometry]
    )
    return source_samples, target_samples


def _wedge_centers(start: float, end: float, count: int) -> np.ndarray:
    if count < 2 or not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("invalid wedge center range")
    return np.linspace(start, end, count, dtype=np.float64)


def _sample_emissive_pair(
    film_path: Path,
    digital_path: Path,
    *,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    emissive = config["sampling"]["emissive"]
    count = int(emissive["steps"])
    half_x, half_y = map(int, emissive["half_window_xy"])
    film = _load_rgb16(film_path, stride=int(emissive["film_stride"]))
    digital = _load_rgb16(digital_path, stride=int(emissive["digital_stride"]))
    film_x = _wedge_centers(*map(float, emissive["film_center_x_range"]), count)
    digital_x = _wedge_centers(
        *map(float, emissive["digital_center_x_range"]), count
    )
    source = np.stack(
        [
            _sample_rectangle(
                digital,
                center_x=x,
                center_y=float(emissive["digital_center_y"]),
                half_x=half_x,
                half_y=half_y,
            )
            for x in digital_x
        ]
    )
    target = np.stack(
        [
            _sample_rectangle(
                film,
                center_x=x,
                center_y=float(emissive["film_center_y"]),
                half_x=half_x,
                half_y=half_y,
            )
            for x in film_x
        ]
    )
    return source, target


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_parent(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent = config["parent"]
    report_path = root / str(parent["integrity_report"])
    if (
        not report_path.is_file()
        or sha256_file(report_path) != parent["integrity_report_sha256"]
    ):
        raise FilmMatchSourceAuditError("parent integrity report hash drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("stable_evidence_id") != parent["stable_evidence_id"]
        or not report.get("source_integrity_passed")
        or not report.get("ordered_mapping_passed")
    ):
        raise FilmMatchSourceAuditError("parent integrity decision drift")
    parent_config = json.loads(
        (root / str(parent["config"])).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / str(parent["download_manifest"])).read_text(encoding="utf-8")
    )
    rows = validate_download_manifest(
        manifest,
        root=root / parent_config["acquisition"]["root"],
        config=parent_config,
    )
    return parent_config, rows


def extract_pair_datasets(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    parent_config, rows = _load_parent(root, config)
    data_root = root / parent_config["acquisition"]["root"]
    by_lane = {
        lane: sorted(
            (row for row in rows if row["lane"] == lane),
            key=lambda row: sequence_number(str(row["name"])),
        )
        for lane in (
            "ektachrome_reflective",
            "sony_reflective",
            "ektachrome_emissive",
            "sony_emissive",
        )
    }

    reflective_source: list[np.ndarray] = []
    reflective_target: list[np.ndarray] = []
    reflective_records: list[dict[str, Any]] = []
    geometry = chart_sample_geometry(config)
    exposure_order = list(config["sampling"]["reflective"]["exposure_order_ev"])
    illuminants = config["sampling"]["reflective"]["illuminant_groups"]
    for pair_index, (film, digital) in enumerate(
        zip(
            by_lane["ektachrome_reflective"],
            by_lane["sony_reflective"],
            strict=True,
        )
    ):
        source, target = _sample_reflective_pair(
            data_root / film["relative_path"],
            data_root / digital["relative_path"],
            config=config,
        )
        reflective_source.append(source)
        reflective_target.append(target)
        illuminant_index, within_group = divmod(pair_index, len(exposure_order))
        for patch_index, patch in enumerate(geometry):
            reflective_records.append(
                {
                    "sample_index": pair_index * len(geometry) + patch_index,
                    "pair_index": pair_index,
                    "film_sequence": sequence_number(str(film["name"])),
                    "digital_sequence": sequence_number(str(digital["name"])),
                    "illuminant": illuminants[illuminant_index]["name"],
                    "exposure_ev": exposure_order[within_group],
                    "chart": patch["chart"],
                    "patch_row": patch["patch_row"],
                    "patch_column": patch["patch_column"],
                }
            )

    emissive_source: list[np.ndarray] = []
    emissive_target: list[np.ndarray] = []
    emissive_records: list[dict[str, Any]] = []
    steps = int(config["sampling"]["emissive"]["steps"])
    for stimulus_index, (film, digital) in enumerate(
        zip(
            by_lane["ektachrome_emissive"],
            by_lane["sony_emissive"],
            strict=True,
        )
    ):
        source, target = _sample_emissive_pair(
            data_root / film["relative_path"],
            data_root / digital["relative_path"],
            config=config,
        )
        hue_sector = _hue_sector(np.median(source[-5:], axis=0))
        emissive_source.append(source)
        emissive_target.append(target)
        for step in range(steps):
            emissive_records.append(
                {
                    "sample_index": stimulus_index * steps + step,
                    "stimulus_index": stimulus_index,
                    "film_sequence": sequence_number(str(film["name"])),
                    "digital_sequence": sequence_number(str(digital["name"])),
                    "hue_sector": hue_sector,
                    "wedge_step": step,
                }
            )

    arrays = {
        "reflective_source": np.concatenate(reflective_source),
        "reflective_target": np.concatenate(reflective_target),
        "emissive_source": np.concatenate(emissive_source),
        "emissive_target": np.concatenate(emissive_target),
    }
    if any(
        value.ndim != 2
        or value.shape[1] != 3
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or np.any(value > 1.0)
        for value in arrays.values()
    ):
        raise FilmMatchSourceAuditError("sampled paired arrays escaped finite RGB cube")
    expected_reflective = 33 * int(
        config["sampling"]["reflective"]["expected_patches_per_pair"]
    )
    expected_emissive = int(
        config["sampling"]["emissive"]["expected_patches"]
    )
    if (
        arrays["reflective_source"].shape != (expected_reflective, 3)
        or arrays["reflective_target"].shape != (expected_reflective, 3)
        or arrays["emissive_source"].shape != (expected_emissive, 3)
        or arrays["emissive_target"].shape != (expected_emissive, 3)
    ):
        raise FilmMatchSourceAuditError("sample population drift")
    return {
        **arrays,
        "reflective_records": reflective_records,
        "emissive_records": emissive_records,
    }


def sample_audit_report(
    datasets: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    arrays = {
        name: np.asarray(datasets[name], dtype=np.float64)
        for name in (
            "reflective_source",
            "reflective_target",
            "emissive_source",
            "emissive_target",
        )
    }
    hue_counts: dict[str, int] = {}
    for record in datasets["emissive_records"]:
        hue = str(record["hue_sector"])
        hue_counts[hue] = hue_counts.get(hue, 0) + 1
    report = {
        "schema_version": "u5-r2aw1-filmmatch-paired-sample-audit-v1",
        "experiment_id": config["experiment_id"],
        "array_contract": {
            name: {
                "shape": list(value.shape),
                "minimum": float(np.min(value)),
                "maximum": float(np.max(value)),
                "float64_le_sha256": _array_sha256(value),
            }
            for name, value in arrays.items()
        },
        "reflective_illuminant_counts": {
            name: sum(
                record["illuminant"] == name
                for record in datasets["reflective_records"]
            )
            for name in ("5600K", "3200K", "5600K_CTB")
        },
        "emissive_hue_sector_sample_counts": dict(sorted(hue_counts.items())),
        "validation_fit_forbidden": True,
        "colour_state": config["colour_state"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = [
    "HUE_LABELS",
    "chart_sample_geometry",
    "extract_pair_datasets",
    "sample_audit_report",
    "slog3_to_linear_reflection",
    "linear_reflection_to_slog3",
]
