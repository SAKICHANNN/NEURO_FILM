"""Source audit for the untouched FiveK high-precision confirmation rows."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tifffile
from PIL import Image, ImageOps


class FiveKUnseenContentSourceError(ValueError):
    """Raised when the retained confirmation source drifts."""


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


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L").resize(
            (9, 8), Image.Resampling.LANCZOS
        )
        values = np.asarray(grayscale, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def _decode_facts(path: Path) -> tuple[list[int], str]:
    data = tifffile.imread(path)
    if data.ndim != 3:
        raise FiveKUnseenContentSourceError(
            f"unexpected TIFF dimensions: {path}"
        )
    return [int(value) for value in data.shape], str(data.dtype)


def validate_contract(root: Path, config: Mapping[str, Any]) -> None:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKUnseenContentSourceError("source contract is not frozen")
    for section, keys in (
        ("freeze", ("manifest", "summary")),
        ("development_evidence", ("manifest",)),
    ):
        payload = config[section]
        for key in keys:
            path = root / str(payload[key])
            if not path.is_file() or _sha256(path) != payload[f"{key}_sha256"]:
                raise FiveKUnseenContentSourceError(
                    f"frozen evidence drift: {section}.{key}"
                )
    if (
        config.get("training_allowed")
        or config.get("pixel_rendering_allowed")
        or config.get("production_integration_allowed")
    ):
        raise FiveKUnseenContentSourceError("source-only boundary drift")


def build_source_evidence(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validate_contract(root, config)
    freeze = config["freeze"]
    with (root / str(freeze["manifest"])).open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(freeze["expected_total_rows"]):
        raise FiveKUnseenContentSourceError("freeze row count drift")
    development = rows[
        int(freeze["development_row_start_one_based"]) - 1 :
        int(freeze["development_row_end_one_based"])
    ]
    confirmation = rows[
        int(freeze["confirmation_row_start_one_based"]) - 1 :
        int(freeze["confirmation_row_end_one_based"])
    ]
    assets = config["confirmation_assets"]
    if (
        len(development) != config["development_evidence"]["expected_rows"]
        or len(confirmation) != assets["expected_rows"]
    ):
        raise FiveKUnseenContentSourceError("split row count drift")
    source_column = str(assets["source_column"])
    target_column = str(assets["target_column"])
    preview_column = str(assets["preview_column"])
    development_ids = {str(row["id"]) for row in development}
    confirmation_ids = {str(row["id"]) for row in confirmation}
    duplicate_ids = development_ids & confirmation_ids
    audited: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "confirmation": [],
    }
    missing = 0
    decode_failures = 0
    dimension_mismatches = 0
    for split, split_rows in (
        ("development", development),
        ("confirmation", confirmation),
    ):
        for row in split_rows:
            source_path = root / str(row[source_column])
            target_path = root / str(row[target_column])
            preview_path = root / str(row[preview_column])
            paths = (source_path, target_path, preview_path)
            if not all(path.is_file() for path in paths):
                missing += 1
                continue
            try:
                source_shape, source_dtype = _decode_facts(source_path)
                target_shape, target_dtype = _decode_facts(target_path)
            except (OSError, ValueError, tifffile.TiffFileError):
                decode_failures += 1
                continue
            if (
                source_dtype != assets["expected_dtype"]
                or target_dtype != assets["expected_dtype"]
                or source_shape[-1] != assets["expected_channels"]
                or target_shape[-1] != assets["expected_channels"]
            ):
                decode_failures += 1
            if source_shape != target_shape:
                dimension_mismatches += 1
            audited[split].append(
                {
                    "pair_id": str(row["id"]),
                    "source_name": str(row["source_name"]),
                    "source_path": str(row[source_column]),
                    "source_sha256": _sha256(source_path),
                    "source_bytes": source_path.stat().st_size,
                    "target_path": str(row[target_column]),
                    "target_sha256": _sha256(target_path),
                    "target_bytes": target_path.stat().st_size,
                    "preview_path": str(row[preview_column]),
                    "preview_sha256": _sha256(preview_path),
                    "shape": source_shape,
                    "dtype": source_dtype,
                    "dhash64": f"{_dhash(preview_path):016x}",
                }
            )
    development_source_hashes = {
        row["source_sha256"] for row in audited["development"]
    }
    development_target_hashes = {
        row["target_sha256"] for row in audited["development"]
    }
    exact_source = sum(
        row["source_sha256"] in development_source_hashes
        for row in audited["confirmation"]
    )
    exact_target = sum(
        row["target_sha256"] in development_target_hashes
        for row in audited["confirmation"]
    )
    threshold = int(assets["maximum_cross_split_hamming_distance"])
    perceptual_pairs = []
    for confirmation_row in audited["confirmation"]:
        confirmation_hash = int(confirmation_row["dhash64"], 16)
        for development_row in audited["development"]:
            distance = (
                confirmation_hash
                ^ int(development_row["dhash64"], 16)
            ).bit_count()
            if distance <= threshold:
                perceptual_pairs.append(
                    {
                        "development_pair_id": development_row["pair_id"],
                        "confirmation_pair_id": confirmation_row["pair_id"],
                        "hamming_distance": distance,
                    }
                )
    confirmation_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "rows": audited["confirmation"],
        "claim_ceiling": config["claim_ceiling"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(_canonical_bytes(confirmation_manifest))
    observed = {
        "confirmation_rows": len(audited["confirmation"]),
        "missing_assets": missing,
        "decode_or_dtype_failures": decode_failures,
        "dimension_mismatches": dimension_mismatches,
        "duplicate_pair_ids_across_splits": len(duplicate_ids),
        "exact_source_or_target_duplicates_across_splits": exact_source
        + exact_target,
        "perceptual_source_duplicates_across_splits_at_or_below_hamming_4": len(
            perceptual_pairs
        ),
    }
    gates = {
        key: (
            observed[key] == required
            if key != "confirmation_rows"
            else observed[key] == required
        )
        for key, required in config["pass_gates"].items()
        if key != "repeat_manifest_byte_identity"
    }
    stable_payload = {
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
        "manifest_sha256": _sha256(manifest_path),
        "observed": observed,
        "gates": gates,
        "automatic_pass": all(gates.values()),
        "perceptual_pairs": perceptual_pairs,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_payload)
        ).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }
    report_path = output_dir / "report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": report["manifest_sha256"],
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "report": report,
    }


__all__ = [
    "FiveKUnseenContentSourceError",
    "build_source_evidence",
    "validate_contract",
]
