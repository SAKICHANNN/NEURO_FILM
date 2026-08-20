"""Frozen small-pair PPISP member, EXIF and geometry preflight."""

from __future__ import annotations

import binascii
import hashlib
import io
import json
import math
import struct
import zlib
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import ExifTags, Image

from src.real_film.ppisp_capture_pair_source_lock import (
    ZipMember,
    _normalized_pair_key,
    _variant,
    canonical_sha256,
    http_range_get,
    locate_central_directory,
    parse_central_directory,
    sha256_bytes,
)


class PPISPPixelPreflightError(ValueError):
    """Raised when the frozen pixel preflight cannot execute exactly."""


def _stable_exif(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256_bytes(value), "length": len(value)}
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, tuple):
        return [_stable_exif(item) for item in value]
    if isinstance(value, (str, int, float)) or value is None:
        return value
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(numerator, int) and isinstance(denominator, int):
        return [numerator, denominator]
    return str(value)


def _exif_facts(image: Image.Image) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in image.getexif().items():
        name = ExifTags.TAGS.get(key, str(key))
        values[name] = _stable_exif(value)
    return values


def _downsample_luma(rgb: np.ndarray, maximum_side: int = 512) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB")
    width, height = image.size
    scale = max(1, math.ceil(max(width, height) / maximum_side))
    if scale > 1:
        image = image.resize(
            (max(1, width // scale), max(1, height // scale)), Image.Resampling.BOX
        )
    value = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return (
        np.float32(0.2126) * value[..., 0]
        + np.float32(0.7152) * value[..., 1]
        + np.float32(0.0722) * value[..., 2]
    )


def census_agreement(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2 or min(left.shape) < 3:
        raise PPISPPixelPreflightError("invalid census inputs")
    agreements = []
    for dy, dx in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        left_neighbor = left[1 + dy : left.shape[0] - 1 + dy, 1 + dx : left.shape[1] - 1 + dx]
        right_neighbor = right[
            1 + dy : right.shape[0] - 1 + dy,
            1 + dx : right.shape[1] - 1 + dx,
        ]
        left_bit = left_neighbor >= left[1:-1, 1:-1]
        right_bit = right_neighbor >= right[1:-1, 1:-1]
        agreements.append(np.mean(left_bit == right_bit, dtype=np.float64))
    return float(np.mean(agreements, dtype=np.float64))


def _gradient_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_gradient = np.concatenate(
        (np.diff(left, axis=0).ravel(), np.diff(left, axis=1).ravel())
    ).astype(np.float64)
    right_gradient = np.concatenate(
        (np.diff(right, axis=0).ravel(), np.diff(right, axis=1).ravel())
    ).astype(np.float64)
    left_gradient -= np.mean(left_gradient)
    right_gradient -= np.mean(right_gradient)
    denominator = np.linalg.norm(left_gradient) * np.linalg.norm(right_gradient)
    return float(np.dot(left_gradient, right_gradient) / denominator) if denominator else 0.0


def _range_get(url: str, start: int, end: int, archive_size: int) -> bytes:
    return http_range_get(url, start, end, archive_size)


def extract_member(
    url: str,
    member: ZipMember,
    *,
    archive_size: int,
) -> tuple[bytes, int]:
    header = _range_get(url, member.local_offset, member.local_offset + 29, archive_size)
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != b"PK\x03\x04":
        raise PPISPPixelPreflightError(f"invalid local header: {member.name}")
    flags, method = values[2], values[3]
    name_length, extra_length = values[-2], values[-1]
    if flags != member.flags or method != member.method:
        raise PPISPPixelPreflightError(f"local/central mismatch: {member.name}")
    variable_size = name_length + extra_length
    variable = _range_get(
        url,
        member.local_offset + 30,
        member.local_offset + 30 + variable_size - 1,
        archive_size,
    )
    name = variable[:name_length].decode("utf-8" if flags & 0x800 else "cp437")
    if name != member.name:
        raise PPISPPixelPreflightError(f"local filename differs: {member.name}")
    data_start = member.local_offset + 30 + variable_size
    compressed = _range_get(
        url,
        data_start,
        data_start + member.compressed_size - 1,
        archive_size,
    )
    if method == 0:
        raw = compressed
    elif method == 8:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise PPISPPixelPreflightError(f"unsupported method: {method}")
    if len(raw) != member.uncompressed_size:
        raise PPISPPixelPreflightError(f"member size differs: {member.name}")
    if binascii.crc32(raw) & 0xFFFFFFFF != member.crc32:
        raise PPISPPixelPreflightError(f"member CRC differs: {member.name}")
    return raw, 30 + variable_size + member.compressed_size


def _camera_group(key: str) -> str:
    parts = PurePosixPath(key).parts
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise PPISPPixelPreflightError(f"pair path lacks images group: {key}") from exc
    if index + 1 >= len(parts):
        raise PPISPPixelPreflightError(f"pair path lacks camera group: {key}")
    return parts[index + 1]


def select_pairs(
    members: list[ZipMember],
    *,
    scene_id: str,
    camera_groups: list[str],
    count_per_camera: int,
    seed: str,
) -> list[dict[str, Any]]:
    standard: dict[str, ZipMember] = {}
    auto: dict[str, ZipMember] = {}
    for member in members:
        if PurePosixPath(member.name).suffix.lower() not in (".jpg", ".jpeg"):
            continue
        key = _normalized_pair_key(member.name)
        target = auto if _variant(member.name) == "auto" else standard
        if key in target:
            raise PPISPPixelPreflightError(f"ambiguous normalized pair key: {key}")
        target[key] = member
    pair_keys = set(standard) & set(auto)
    selected = []
    for camera in camera_groups:
        eligible = [key for key in pair_keys if _camera_group(key) == camera]
        ranked = sorted(
            eligible,
            key=lambda key: hashlib.sha256(
                f"{seed}|{scene_id}|{camera}|{key}".encode()
            ).hexdigest(),
        )
        if len(ranked) < count_per_camera:
            raise PPISPPixelPreflightError(
                f"insufficient {scene_id}/{camera} pairs: {len(ranked)}"
            )
        for key in ranked[:count_per_camera]:
            selected.append(
                {
                    "scene_id": scene_id,
                    "camera_group": camera,
                    "pair_key": key,
                    "standard": standard[key],
                    "auto": auto[key],
                }
            )
    return selected


def _decode_jpeg(raw: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.format != "JPEG":
            raise PPISPPixelPreflightError(f"decoded format differs: {image.format}")
        facts = {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "icc_sha256": (
                sha256_bytes(image.info["icc_profile"])
                if isinstance(image.info.get("icc_profile"), bytes)
                else None
            ),
            "exif": _exif_facts(image),
        }
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return rgb, facts


def run_preflight(config_path: Path, *, root: Path) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema") != "neuro-film.sf3-a0q-ppisp-capture-pair-pixel-preflight-contract.v1":
        raise PPISPPixelPreflightError("contract schema differs")
    evidence_path = root / config["source_lock"]["evidence_path"]
    if sha256_bytes(evidence_path.read_bytes()) != config["source_lock"]["evidence_sha256"]:
        raise PPISPPixelPreflightError("source-lock evidence identity differs")
    source_path = root / config["source_config_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    selection = config["selection"]
    source_info = source["source"]
    archive_reports = []
    planned_archives: list[tuple[str, int, list[dict[str, Any]]]] = []
    selected_rows: list[dict[str, Any]] = []
    total_network_bytes = 0
    for archive in source["archives"]:
        size = int(archive["size"])
        url = (
            "https://huggingface.co/datasets/"
            f"{source_info['dataset_id']}/resolve/{source_info['revision']}/{archive['path']}"
        )
        tail_size = min(source["network_limits"]["tail_probe_bytes_per_archive"], size)
        tail_start = size - tail_size
        tail = _range_get(url, tail_start, size - 1, size)
        total_network_bytes += len(tail)
        central_offset, central_size, expected_members = locate_central_directory(
            tail, archive_size=size
        )
        if central_offset < tail_start:
            central = _range_get(
                url, central_offset, central_offset + central_size - 1, size
            )
            total_network_bytes += len(central)
        else:
            central = tail[
                central_offset - tail_start : central_offset - tail_start + central_size
            ]
        members = parse_central_directory(central)
        if len(members) != expected_members:
            raise PPISPPixelPreflightError("central member count differs")
        pairs = select_pairs(
            members,
            scene_id=archive["scene_id"],
            camera_groups=selection["required_camera_groups"],
            count_per_camera=selection["pairs_per_scene_camera"],
            seed=selection["seed"],
        )
        archive_reports.append(
            {
                "scene_id": archive["scene_id"],
                "central_sha256": sha256_bytes(central),
                "selected_pair_keys_sha256": canonical_sha256(
                    [row["pair_key"] for row in pairs]
                ),
                "selected_pair_count": len(pairs),
            }
        )
        planned_archives.append((url, size, pairs))

    frozen_selection = [
        [pair["scene_id"], pair["camera_group"], pair["pair_key"]]
        for _, _, pairs in planned_archives
        for pair in pairs
    ]
    if len(frozen_selection) != selection["expected_pair_count"]:
        raise PPISPPixelPreflightError("frozen selection count differs")

    for url, size, pairs in planned_archives:
        for pair in pairs:
            standard_raw, standard_network = extract_member(
                url, pair["standard"], archive_size=size
            )
            auto_raw, auto_network = extract_member(url, pair["auto"], archive_size=size)
            total_network_bytes += standard_network + auto_network
            if max(len(standard_raw), len(auto_raw)) > config["decode"][
                "maximum_member_uncompressed_bytes"
            ]:
                raise PPISPPixelPreflightError("selected member exceeds byte ceiling")
            standard_rgb, standard_facts = _decode_jpeg(standard_raw)
            auto_rgb, auto_facts = _decode_jpeg(auto_raw)
            same_dimensions = standard_rgb.shape == auto_rgb.shape
            if same_dimensions:
                standard_luma = _downsample_luma(standard_rgb)
                auto_luma = _downsample_luma(auto_rgb)
                census = census_agreement(standard_luma, auto_luma)
                gradient = _gradient_correlation(standard_luma, auto_luma)
                mean_abs = float(
                    np.mean(
                        np.abs(
                            standard_rgb.astype(np.float32)
                            - auto_rgb.astype(np.float32)
                        ),
                        dtype=np.float64,
                    )
                    / 255.0
                )
            else:
                census = gradient = mean_abs = 0.0
            exif_fields = config["pair_gates"]["same_capture_exif_fields"]
            present = [
                field
                for field in exif_fields
                if field in standard_facts["exif"] and field in auto_facts["exif"]
            ]
            equal = [
                field
                for field in present
                if standard_facts["exif"][field] == auto_facts["exif"][field]
            ]
            selected_rows.append(
                {
                    "scene_id": pair["scene_id"],
                    "camera_group": pair["camera_group"],
                    "pair_key": pair["pair_key"],
                    "standard_member": pair["standard"].name,
                    "auto_member": pair["auto"].name,
                    "standard_sha256": sha256_bytes(standard_raw),
                    "auto_sha256": sha256_bytes(auto_raw),
                    "standard_size": len(standard_raw),
                    "auto_size": len(auto_raw),
                    "standard_facts": standard_facts,
                    "auto_facts": auto_facts,
                    "same_dimensions": same_dimensions,
                    "present_capture_exif_fields": present,
                    "equal_capture_exif_fields": equal,
                    "census_agreement": census,
                    "gradient_correlation": gradient,
                    "mean_absolute_rgb_difference": mean_abs,
                }
            )

    pair_gates = config["pair_gates"]
    expected_pairs = selection["expected_pair_count"]
    gates = {
        "scene_set_exact": sorted({row["scene_id"] for row in selected_rows})
        == sorted(selection["required_scenes"]),
        "camera_groups_each_scene_exact": all(
            {row["camera_group"] for row in selected_rows if row["scene_id"] == scene}
            == set(selection["required_camera_groups"])
            for scene in selection["required_scenes"]
        ),
        "pair_count_exact": len(selected_rows) == expected_pairs,
        "unique_pair_keys": len({row["pair_key"] for row in selected_rows})
        == expected_pairs,
        "jpeg_decode_and_minimum_geometry": all(
            row["standard_facts"]["format"] == row["auto_facts"]["format"] == "JPEG"
            and min(row["standard_facts"]["width"], row["standard_facts"]["height"])
            >= config["decode"]["minimum_short_side"]
            for row in selected_rows
        ),
        "same_dimensions_each_pair": all(row["same_dimensions"] for row in selected_rows),
        "capture_exif_support_and_identity": all(
            len(row["present_capture_exif_fields"])
            >= pair_gates["minimum_present_capture_exif_fields"]
            and row["present_capture_exif_fields"] == row["equal_capture_exif_fields"]
            for row in selected_rows
        ),
        "census_registration_each_pair": all(
            row["census_agreement"] >= pair_gates["minimum_census_agreement"]
            for row in selected_rows
        ),
        "processing_difference_each_pair": all(
            row["standard_sha256"] != row["auto_sha256"]
            and row["mean_absolute_rgb_difference"]
            >= pair_gates["minimum_mean_absolute_rgb_difference"]
            for row in selected_rows
        ),
        "network_byte_ceiling": total_network_bytes
        <= config["decode"]["maximum_total_network_bytes"],
        "finite_metrics": all(
            math.isfinite(row[key])
            for row in selected_rows
            for key in (
                "census_agreement",
                "gradient_correlation",
                "mean_absolute_rgb_difference",
            )
        ),
        "zero_operator_fit_render_score": True,
    }
    passed = all(gates.values())
    report = {
        "schema": "neuro-film.sf3-a0q-ppisp-capture-pair-pixel-preflight-report.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "source_lock_evidence_sha256": config["source_lock"]["evidence_sha256"],
        "source_revision": source_info["revision"],
        "selection_frozen_before_member_reads": True,
        "selection_manifest_sha256": canonical_sha256(frozen_selection),
        "archive_reports": archive_reports,
        "rows": selected_rows,
        "total_network_bytes": total_network_bytes,
        "operator_fits": 0,
        "renders": 0,
        "scores": 0,
        "bounded_final_candidate_counter_before": 0,
        "bounded_final_candidate_counter_after": 0,
        "gates": gates,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report
