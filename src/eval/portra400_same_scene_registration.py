"""CHAM5 exact source integrity and geometric registration audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from PIL import Image, ImageOps

from src.eval.flickr_single_author_pair_registration import register_pair

SCHEMA = "neuro-film.u5-r2cham5-portra400-registration-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham5-portra400-registration-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or len(value.get("pairs", [])) != 3:
        raise ValueError("unsupported CHAM5 contract")
    return value


def _raw_rgb(
    path: Path, contract: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    def decode() -> tuple[np.ndarray, dict[str, Any]]:
        with rawpy.imread(str(path)) as source:
            sizes = source.sizes
            metadata = {
                "raw_type": str(source.raw_type),
                "raw_shape": [int(sizes.raw_height), int(sizes.raw_width)],
                "visible_shape": [int(sizes.height), int(sizes.width)],
                "flip": int(sizes.flip),
                "black_level_per_channel": [
                    int(v) for v in source.black_level_per_channel
                ],
                "white_level": int(source.white_level),
            }
            rgb = source.postprocess(
                use_camera_wb=bool(contract["raw_use_camera_wb"]),
                no_auto_bright=bool(contract["raw_no_auto_bright"]),
                half_size=bool(contract["raw_half_size"]),
                output_bps=8,
                output_color=rawpy.ColorSpace.sRGB,
                gamma=(2.222, 4.5),
                user_flip=None,
            )
        return np.ascontiguousarray(rgb), metadata

    first, metadata = decode()
    second, second_metadata = decode()
    if metadata != second_metadata or not np.array_equal(first, second):
        raise ValueError(f"CHAM5 RAW decode replay drift: {path.name}")
    return first, {
        **metadata,
        "decoded_shape": list(first.shape),
        "decoded_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
    }


def _raster_rgb(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as source:
        frames = int(getattr(source, "n_frames", 1))
        info = {
            "format": str(source.format),
            "mode": str(source.mode),
            "stored_size": list(source.size),
            "frames": frames,
            "embedded_icc": "icc_profile" in source.info,
        }
        if frames != 1:
            raise ValueError(f"CHAM5 multi-frame raster: {path.name}")
        pixels = np.ascontiguousarray(
            np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
        )
    return pixels, {
        **info,
        "decoded_shape": list(pixels.shape),
        "decoded_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        if _sha(root / binding["path"]) != binding["sha256"]:
            raise ValueError("CHAM5 parent drift")
    source_contract = json.loads(
        (root / contract["parents"]["source_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (root / contract["parents"]["download_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if manifest.get("file_count") != 6 or manifest.get("bytes") != 183875041:
        raise ValueError("CHAM5 manifest drift")
    data_root = root / source_contract["acquisition"]["root"]
    decoded: dict[str, np.ndarray] = {}
    files: list[dict[str, Any]] = []
    for row in manifest["files"]:
        path = data_root / row["name"]
        if path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
            raise ValueError(f"CHAM5 payload drift: {row['role']}")
        suffix = path.suffix.lower()
        if suffix == ".rw2":
            pixels, details = _raw_rgb(path, contract["decode"])
            decoded[row["role"]] = pixels
        elif suffix in {".jpg", ".jpeg", ".tif", ".tiff"}:
            pixels, details = _raster_rgb(path)
            decoded[row["role"]] = pixels
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            details = {"json_top_level_keys": sorted(payload), "json_decode": "valid"}
        else:
            raise ValueError(f"CHAM5 unsupported payload: {path.name}")
        files.append({**dict(row), **details})
    pairs: list[dict[str, Any]] = []
    required_results: list[bool] = []
    for pair in contract["pairs"]:
        _homography, diagnostics = register_pair(
            decoded[pair["digital_role"]],
            decoded[pair["target_role"]],
            contract["registration"],
        )
        pairs.append({**dict(pair), "diagnostics": diagnostics})
        if pair["registration_required"]:
            required_results.append(bool(diagnostics["registration_gate_passed"]))
    passed = bool(required_results) and all(required_results)
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "manifest_sha256": contract["parents"]["download_manifest"]["sha256"],
        "rawpy_version": rawpy.__version__,
        "files": files,
        "pairs": pairs,
        "automatic_pass": passed,
        "operator_fit_allowed": False,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
