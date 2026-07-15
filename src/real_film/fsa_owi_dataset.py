"""Phase-C pixel acquisition helpers for the grouped FSA/OWI corpus."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.real_film.fsa_owi import FsaOwiAcquisitionError


def prepare_download_records(
    grouped_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt stable grouped rows to the generic bounded image downloader."""
    ordered = sorted(grouped_records, key=lambda row: str(row["loc_fsac_id"]))
    if len(ordered) != len({str(row["loc_fsac_id"]) for row in ordered}):
        raise FsaOwiAcquisitionError("grouped corpus contains duplicate LOC ids")
    prepared: list[dict[str, Any]] = []
    for index, source in enumerate(ordered):
        row = dict(source)
        row["pilot_index"] = index
        row["pilot_creator_group"] = str(row["evaluation_creator_group"])
        prepared.append(row)
    return prepared


def seed_verified_pilot(
    prepared: Sequence[Mapping[str, Any]],
    pilot_rows: Sequence[Mapping[str, Any]],
    image_dir: Path,
) -> dict[str, int]:
    """Reuse exact pilot payloads after verifying their recorded SHA-256."""
    by_identifier = {str(row["loc_fsac_id"]): row for row in prepared}
    image_dir.mkdir(parents=True, exist_ok=True)
    seeded = 0
    seeded_bytes = 0
    for pilot in pilot_rows:
        identifier = str(pilot["loc_fsac_id"])
        if identifier not in by_identifier:
            continue
        source_path = Path(str(pilot["local_path"]))
        payload = source_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != str(pilot["sha256"]):
            raise FsaOwiAcquisitionError("pilot seed hash mismatch")
        target_record = by_identifier[identifier]
        target_prefix = (
            f"{int(target_record['pilot_index']):03d}_"
            f"{identifier.replace('.', '_')}"
        )
        existing = sorted(image_dir.glob(target_prefix + ".*"))
        if existing:
            if len(existing) != 1 or hashlib.sha256(existing[0].read_bytes()).hexdigest() != str(
                pilot["sha256"]
            ):
                raise FsaOwiAcquisitionError("existing Phase-C seed disagrees with pilot")
        else:
            target = image_dir / (target_prefix + source_path.suffix.lower())
            shutil.copy2(source_path, target)
        seeded += 1
        seeded_bytes += len(payload)
    return {"seeded_records": seeded, "seeded_bytes": seeded_bytes}


def merge_pixel_evidence(
    grouped_records: Sequence[Mapping[str, Any]],
    pixel_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join decoded pixel evidence back onto every grouped metadata row."""
    pixel_by_id = {str(row["loc_fsac_id"]): row for row in pixel_rows}
    if len(pixel_by_id) != len(pixel_rows):
        raise FsaOwiAcquisitionError("pixel evidence contains duplicate LOC ids")
    output: list[dict[str, Any]] = []
    for source in sorted(grouped_records, key=lambda row: str(row["loc_fsac_id"])):
        identifier = str(source["loc_fsac_id"])
        if identifier not in pixel_by_id:
            raise FsaOwiAcquisitionError(f"missing pixel evidence for {identifier}")
        pixel = pixel_by_id[identifier]
        row = dict(source)
        row["pixel_evidence"] = {
            key: pixel[key]
            for key in (
                "local_path", "download_url", "bytes", "sha256", "dhash64",
                "width", "height", "mode", "format", "content_type",
                "icc_profile_bytes", "icc_profile_sha256", "exif_tags",
            )
        }
        output.append(row)
    return output
