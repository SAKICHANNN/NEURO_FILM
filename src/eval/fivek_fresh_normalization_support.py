"""Normalize and audit the frozen fresh FiveK pairs without fitting."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import ExifTags, Image

from scripts.build_fivek_freeze_pack import (
    filtered_target,
    load_expert_icc_srgb,
    load_raw_default,
    resize_to_shape,
    save_preview,
    save_tiff16,
)


class FiveKFreshNormalizationError(ValueError):
    """Raised when frozen normalization or source evidence drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise FiveKFreshNormalizationError(f"evidence drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKFreshNormalizationError("contract is not frozen")
    if (
        config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("candidate_rendering_allowed")
        or config.get("production_integration_allowed")
    ):
        raise FiveKFreshNormalizationError("data-only boundary drift")
    parent = config["parent_acquisition"]
    acquisition = _load_hashed_json(
        root, parent["manifest"], parent["manifest_sha256"]
    )
    report = _load_hashed_json(
        root, parent["report"], parent["report_sha256"]
    )
    decision = (
        _load_hashed_json(
            root, parent["decision"], parent["decision_sha256"]
        )
        if "decision" in parent
        else None
    )
    development = config["development_candidate"]
    candidate = _load_hashed_json(
        root, development["decision"], development["decision_sha256"]
    )
    support = config["support_and_leakage"]
    existing_manifest = root / support["existing_128_manifest"]
    if (
        not existing_manifest.is_file()
        or _sha256(existing_manifest)
        != support["existing_128_manifest_sha256"]
    ):
        raise FiveKFreshNormalizationError("existing 128 manifest drift")
    for item in support.get("additional_existing_manifests", []):
        path = root / str(item["path"])
        if not path.is_file() or _sha256(path) != str(
            item["sha256"]
        ).lower():
            raise FiveKFreshNormalizationError(
                f"additional existing manifest drift: {item['path']}"
            )
    parent_eligible = (
        report.get("automatic_pass")
        is parent["required_automatic_pass"]
        if "required_automatic_pass" in parent
        else (
            report.get("gates", {}).get("dimension_mismatches") is False
            and decision is not None
            and decision.get("automatic_pass") is False
        )
    )
    if (
        report.get("observed", {}).get("assets")
        != parent["required_assets"]
        or report.get("observed", {}).get("decode_failures")
        != parent["required_decode_failures"]
        or not parent_eligible
        or candidate.get("status") != development["required_status"]
        or development.get("use_during_this_leaf") is not False
    ):
        raise FiveKFreshNormalizationError("parent branch boundary drift")
    return {
        "acquisition": acquisition,
        "existing_manifest": existing_manifest,
    }


def center_crop_to_aspect(
    rgb: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Crop symmetrically to target aspect using geometry alone."""

    array = np.asarray(rgb)
    if (
        array.ndim != 3
        or array.shape[-1] != 3
        or target_height <= 0
        or target_width <= 0
    ):
        raise FiveKFreshNormalizationError("invalid crop geometry")
    height, width = array.shape[:2]
    source_aspect = width / height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        crop_width = min(width, max(1, int(round(height * target_aspect))))
        left = (width - crop_width) // 2
        return np.ascontiguousarray(array[:, left : left + crop_width])
    crop_height = min(
        height, max(1, int(round(width / target_aspect)))
    )
    top = (height - crop_height) // 2
    return np.ascontiguousarray(array[top : top + crop_height, :])


def _alignment(
    source: np.ndarray,
    target: np.ndarray,
    audit_side: int,
    decimal_places: int,
) -> tuple[float, float]:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cv2.setUseOptimized(False)
    def gradient(rgb: np.ndarray) -> np.ndarray:
        luma = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        ).astype(np.float32)
        height, width = luma.shape
        scale = audit_side / max(height, width)
        resized = cv2.resize(
            luma,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    first = gradient(source)
    second = gradient(target)
    if first.shape != second.shape:
        second = cv2.resize(
            second,
            (first.shape[1], first.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    left = first.ravel().astype(np.float64)
    right = second.ravel().astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    correlation = (
        float(np.dot(left, right) / denominator)
        if denominator > 0.0
        else -1.0
    )
    shift, _ = cv2.phaseCorrelate(
        first.astype(np.float32), second.astype(np.float32)
    )
    return (
        round(correlation, decimal_places),
        round(float(np.hypot(shift[0], shift[1])), decimal_places),
    )


def _dhash64(rgb: np.ndarray) -> str:
    pixels = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray = Image.fromarray(pixels, mode="RGB").convert("L").resize(
        (9, 8), Image.Resampling.LANCZOS
    )
    values = np.asarray(gray, dtype=np.uint8)
    result = 0
    for bit in (values[:, 1:] > values[:, :-1]).ravel():
        result = (result << 1) | int(bit)
    return f"{result:016x}"


def _dhash64_path(path: Path) -> int:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return int(_dhash64(rgb), 16)


def _camera_model(path: Path) -> str:
    with Image.open(path) as image:
        exif = image.getexif()
    facts = {
        ExifTags.TAGS.get(tag, tag): value for tag, value in exif.items()
    }
    make = " ".join(str(facts.get("Make", "unknown")).split())
    model = " ".join(str(facts.get("Model", "unknown")).split())
    return f"{make}/{model}".casefold()


def _save_exact(
    array: np.ndarray, path: Path, saver: Any
) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    saver(array, temporary)
    temporary.replace(path)
    return _sha256(path), path.stat().st_size


def run_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    normalization = config["normalization"]
    alignment = config["alignment_audit"]
    support = config["support_and_leakage"]
    external_root = Path(normalization["external_root"])
    external_root.mkdir(parents=True, exist_ok=True)
    marker = external_root.parent / ".neuro_film_owner.json"
    if not marker.is_file():
        raise FiveKFreshNormalizationError("owned root marker missing")
    target_values = normalization["target_policy"]
    target_policy = type(
        "TargetPolicy",
        (),
        {
            "luma_strength": target_values["luma_strength"],
            "chroma_strength": target_values["chroma_strength"],
            "chroma_headroom": target_values["chroma_headroom"],
            "wb_anchor_strength": target_values[
                "white_balance_anchor_strength"
            ],
        },
    )()
    maximum_side = int(normalization["maximum_side"])
    with validated["existing_manifest"].open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        development_rows = list(csv.DictReader(handle))
    if len(development_rows) != support["existing_rows"]:
        raise FiveKFreshNormalizationError("existing split row drift")
    existing_source_hashes = {
        _sha256(root / row["raw_default_srgb16"])
        for row in development_rows
    }
    existing_target_hashes = {
        _sha256(root / row["filtered_target_srgb16"])
        for row in development_rows
    }
    existing_dhashes = {
        row["id"]: _dhash64_path(root / row["raw_preview"])
        for row in development_rows
    }
    for item in support.get("additional_existing_manifests", []):
        payload = json.loads(
            (root / str(item["path"])).read_text(encoding="utf-8")
        )
        additional_rows = payload.get("rows", [])
        if len(additional_rows) != int(item["expected_rows"]):
            raise FiveKFreshNormalizationError(
                f"additional existing row drift: {item['path']}"
            )
        for row in additional_rows:
            row_id = str(row[item["id_field"]])
            existing_source_hashes.add(
                str(row[item["source_hash_field"]])
            )
            existing_target_hashes.add(
                str(row[item["target_hash_field"]])
            )
            existing_dhashes[row_id] = int(
                str(row[item["dhash_field"]]), 16
            )
    expected_total = support.get("existing_total_rows")
    if (
        expected_total is not None
        and len(existing_dhashes) != int(expected_total)
    ):
        raise FiveKFreshNormalizationError(
            "combined existing split row drift"
        )
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    perceptual_pairs: list[dict[str, Any]] = []
    for acquired in validated["acquisition"]["rows"]:
        source_shape = acquired["source_shape"]
        target_shape = acquired["target_shape"]
        source_name = str(acquired["source_name"])
        if (
            not source_shape
            or not target_shape
            or (source_shape[0] > source_shape[1])
            != (target_shape[0] > target_shape[1])
        ):
            rejected.append(
                {
                    "source_name": source_name,
                    "reason": "orientation-mismatch",
                }
            )
            continue
        source_path = Path(acquired["dng_path"])
        expert_path = Path(acquired["expert_c_path"])
        if (
            _sha256(source_path) != acquired["dng_sha256"]
            or _sha256(expert_path) != acquired["expert_c_sha256"]
        ):
            raise FiveKFreshNormalizationError(
                f"input hash drift: {source_name}"
            )
        source, _ = load_raw_default(source_path, maximum_side)
        expert = load_expert_icc_srgb(expert_path, maximum_side)
        source = center_crop_to_aspect(
            source, int(target_shape[0]), int(target_shape[1])
        )
        expert = resize_to_shape(expert, source.shape[:2])
        target = filtered_target(source, expert, target_policy)
        correlation, phase_shift = _alignment(
            source,
            expert,
            int(alignment["audit_side"]),
            int(alignment["reported_decimal_places"]),
        )
        if (
            correlation
            < alignment["minimum_per_pair_gradient_correlation"]
            or phase_shift
            > alignment["maximum_absolute_phase_shift_pixels"]
        ):
            rejected.append(
                {
                    "source_name": source_name,
                    "reason": "content-alignment-failed",
                }
            )
            continue
        pair_id = f"fresh_{len(rows) + 1:04d}_{source_name}"
        source_output = external_root / "source" / f"{source_name}.tif"
        target_output = external_root / "target" / f"{source_name}.tif"
        preview_output = external_root / "preview" / f"{source_name}.jpg"
        source_hash, source_bytes = _save_exact(
            source, source_output, save_tiff16
        )
        target_hash, target_bytes = _save_exact(
            target, target_output, save_tiff16
        )
        preview_hash, preview_bytes = _save_exact(
            source, preview_output, save_preview
        )
        dhash = _dhash64(source)
        for development_id, value in existing_dhashes.items():
            distance = (int(dhash, 16) ^ value).bit_count()
            if distance <= support["maximum_cross_split_dhash_hamming"]:
                perceptual_pairs.append(
                    {
                        "fresh_pair_id": pair_id,
                        "existing_pair_id": development_id,
                        "hamming_distance": distance,
                    }
                )
        rows.append(
            {
                "pair_id": pair_id,
                "source_name": source_name,
                "camera_model": _camera_model(source_path),
                "source_path": str(source_output.as_posix()),
                "source_sha256": source_hash,
                "source_bytes": source_bytes,
                "target_path": str(target_output.as_posix()),
                "target_sha256": target_hash,
                "target_bytes": target_bytes,
                "preview_path": str(preview_output.as_posix()),
                "preview_sha256": preview_hash,
                "preview_bytes": preview_bytes,
                "dhash64": dhash,
                "shape": [int(value) for value in source.shape],
                "dtype": "uint16",
                "gradient_correlation": correlation,
                "phase_shift_pixels_at_audit_scale": phase_shift,
                "source_dng_sha256": acquired["dng_sha256"],
                "expert_c_sha256": acquired["expert_c_sha256"],
            }
        )
    cameras = Counter(row["camera_model"] for row in rows)
    eligible = len(rows)
    largest_share = max(cameras.values(), default=0) / max(eligible, 1)
    exact_cross_split = sum(
        row["source_sha256"] in existing_source_hashes
        or row["target_sha256"] in existing_target_hashes
        for row in rows
    )
    correlations = [row["gradient_correlation"] for row in rows]
    shifts = [row["phase_shift_pixels_at_audit_scale"] for row in rows]
    observed = {
        "eligible_pairs": eligible,
        "rejected_pairs": len(rejected),
        "camera_models": len(cameras),
        "largest_camera_share": largest_share,
        "minimum_gradient_correlation": min(correlations, default=-1.0),
        "median_gradient_correlation": float(
            np.median(correlations) if correlations else -1.0
        ),
        "maximum_phase_shift_pixels": max(shifts, default=float("inf")),
        "exact_cross_split_duplicates": exact_cross_split,
        "perceptual_cross_split_duplicates": len(perceptual_pairs),
    }
    gates = {
        "eligible_pairs": eligible >= support["minimum_eligible_pairs"],
        "camera_models": len(cameras) >= support["minimum_camera_models"],
        "largest_camera_share": largest_share
        <= support["maximum_largest_camera_share"],
        "minimum_alignment": observed["minimum_gradient_correlation"]
        >= alignment["minimum_per_pair_gradient_correlation"],
        "median_alignment": observed["median_gradient_correlation"]
        >= alignment["minimum_median_gradient_correlation"],
        "phase_shift": observed["maximum_phase_shift_pixels"]
        <= alignment["maximum_absolute_phase_shift_pixels"],
        "exact_leakage": exact_cross_split == 0,
        "perceptual_leakage": len(perceptual_pairs) == 0,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "rows": rows,
        "rejected": rejected,
        "camera_counts": dict(sorted(cameras.items())),
        "claim_ceiling": config["claim_ceiling"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest))
    stable = {
        "observed": observed,
        "gates": gates,
        "manifest_sha256": _sha256(manifest_path),
        "perceptual_pairs": perceptual_pairs,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }
    report_path = output_dir / "report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
    }


__all__ = [
    "FiveKFreshNormalizationError",
    "center_crop_to_aspect",
    "run_audit",
    "validate_contract",
]
