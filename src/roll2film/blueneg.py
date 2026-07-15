"""Metadata-first BlueNeg whole-roll evidence and bounded acquisition contract."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BLUENEG_FRAME_SCHEMA = "roll2film.blueneg_frame.v1"
BLUENEG_ROLL_SCHEMA = "roll2film.blueneg_roll.v1"


class BlueNegContractError(ValueError):
    """Raised when BlueNeg metadata, split, or remote inventory fails closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def _stable_order(values: list[str], seed: int, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{seed}:{namespace}:{value}".encode()
        ).hexdigest(),
    )


@dataclass(frozen=True)
class BlueNegEvidenceConfig:
    root: Path
    output_dir: Path
    repo_id: str
    revision: str
    metadata_sha256: dict[str, str]
    expected: dict[str, int]
    split_seed: int
    minimum_paired_frames: int = 4
    confirm_if_at_least_four: int = 2
    confirm_if_at_least_two: int = 1
    required_credit: str = "Copyrighted by Tien-Tsin Wong"
    software_commit: str = "unknown"


@dataclass(frozen=True)
class BlueNegEvidenceResult:
    report: dict[str, Any]
    paths: dict[str, Path]


def _load_metadata(config: BlueNegEvidenceConfig) -> list[dict[str, Any]]:
    for filename, expected_hash in config.metadata_sha256.items():
        path = config.root / filename
        if not path.is_file():
            raise BlueNegContractError(f"missing metadata snapshot: {path}")
        if _sha256(path) != expected_hash:
            raise BlueNegContractError(f"metadata hash mismatch: {filename}")
    try:
        rows = json.loads((config.root / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlueNegContractError("invalid BlueNeg meta.json") from exc
    if not isinstance(rows, list) or len(rows) != config.expected["metadata_rows"]:
        raise BlueNegContractError("unexpected BlueNeg metadata row count")
    required = {
        "filename",
        "partition",
        "is_testset",
        "date",
        "roll_id",
        "film_type",
        "preview_path",
        "pseudogt_path",
        "scene_property",
    }
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or required - set(row):
            raise BlueNegContractError(f"metadata row {index} has missing fields")
        filename = str(row["filename"])
        if not filename or filename in seen:
            raise BlueNegContractError(f"duplicate or empty filename: {filename!r}")
        seen.add(filename)
        if not str(row["preview_path"]).startswith("negative-preview-8bit/"):
            raise BlueNegContractError(f"invalid preview path for {filename}")
    return rows


def _load_inventory(config: BlueNegEvidenceConfig) -> dict[str, dict[str, Any]]:
    path = config.root / "remote_inventory.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlueNegContractError("missing or invalid remote_inventory.json") from exc
    if payload.get("repo_id") != config.repo_id or payload.get("revision") != config.revision:
        raise BlueNegContractError("remote inventory identity/revision mismatch")
    inventory = {str(row["path"]): row for row in payload.get("files", [])}
    for prefix, count_key, bytes_key in (
        ("negative-preview-8bit/", "preview_files", "preview_bytes"),
        ("pseudogt-8bit/", "pseudogt_files", "pseudogt_bytes"),
    ):
        rows = [row for name, row in inventory.items() if name.startswith(prefix)]
        if len(rows) != config.expected[count_key]:
            raise BlueNegContractError(f"remote {prefix} file count mismatch")
        if sum(int(row["size"]) for row in rows) != config.expected[bytes_key]:
            raise BlueNegContractError(f"remote {prefix} byte count mismatch")
    return inventory


def build_blueneg_metadata_evidence(
    config: BlueNegEvidenceConfig,
) -> BlueNegEvidenceResult:
    """Freeze whole-roll pools without decoding or downloading image payloads."""
    metadata = _load_metadata(config)
    inventory = _load_inventory(config)
    by_roll: dict[str, list[dict[str, Any]]] = {}
    for row in metadata:
        by_roll.setdefault(str(row["roll_id"]), []).append(row)
    film_types = {str(row["film_type"]) for row in metadata}
    if len(by_roll) != config.expected["rolls"]:
        raise BlueNegContractError("unexpected BlueNeg roll count")
    if len(film_types) != config.expected["film_types"]:
        raise BlueNegContractError("unexpected BlueNeg film-type count")

    sealed_rolls = {
        roll_id
        for roll_id, rows in by_roll.items()
        if any(bool(row["is_testset"]) for row in rows)
    }
    paired_by_roll = {
        roll_id: [
            row
            for row in rows
            if row.get("pseudogt_path") in inventory and not bool(row["is_testset"])
        ]
        for roll_id, rows in by_roll.items()
    }
    operator_rolls = {
        roll_id
        for roll_id, rows in paired_by_roll.items()
        if roll_id not in sealed_rolls and len(rows) >= config.minimum_paired_frames
    }
    operator_by_film: dict[str, list[str]] = {}
    for roll_id in operator_rolls:
        film_type = str(by_roll[roll_id][0]["film_type"])
        if {str(row["film_type"]) for row in by_roll[roll_id]} != {film_type}:
            raise BlueNegContractError(f"roll spans multiple film types: {roll_id}")
        operator_by_film.setdefault(film_type, []).append(roll_id)

    pool_by_roll: dict[str, str] = {}
    for film_type, roll_ids in sorted(operator_by_film.items()):
        ordered = _stable_order(roll_ids, config.split_seed, f"roll:{film_type}")
        if len(ordered) >= 4:
            confirm_count = config.confirm_if_at_least_four
        elif len(ordered) >= 2:
            confirm_count = config.confirm_if_at_least_two
        else:
            confirm_count = 0
        for roll_id in ordered[:confirm_count]:
            pool_by_roll[roll_id] = "confirmatory_roll"
        for roll_id in ordered[confirm_count:]:
            pool_by_roll[roll_id] = "development_roll"

    frame_role: dict[str, str] = {}
    for roll_id in operator_rolls:
        paired = paired_by_roll[roll_id]
        ordered_names = _stable_order(
            [str(row["filename"]) for row in paired],
            config.split_seed,
            f"frame:{roll_id}",
        )
        support_count = len(ordered_names) // 2
        support = set(ordered_names[:support_count])
        for row in by_roll[roll_id]:
            filename = str(row["filename"])
            if row.get("pseudogt_path") not in inventory:
                frame_role[filename] = "support_source_only"
            elif filename in support:
                frame_role[filename] = "unpaired_support"
            else:
                frame_role[filename] = "hidden_aligned_query"

    frame_rows: list[dict[str, Any]] = []
    roll_rows: list[dict[str, Any]] = []
    acquisition_paths: set[str] = set()
    for roll_id, rows in sorted(by_roll.items()):
        film_type = str(rows[0]["film_type"])
        same_film_operator_rolls = sorted(set(operator_by_film.get(film_type, [])) - {roll_id})
        if roll_id in sealed_rolls:
            pool = "official_test_roll_lockbox"
        elif roll_id in operator_rolls:
            pool = pool_by_roll[roll_id]
        else:
            pool = "metadata_auxiliary"
        matched_control = roll_id in operator_rolls and bool(same_film_operator_rolls)
        paired_count = len(paired_by_roll[roll_id])
        roll_rows.append(
            {
                "schema_version": BLUENEG_ROLL_SCHEMA,
                "roll_id": roll_id,
                "film_type": film_type,
                "frames": len(rows),
                "public_non_test_pseudogt_frames": paired_count,
                "research_pool": pool,
                "operator_eligible": roll_id in operator_rolls,
                "matched_same_film_wrong_roll_control_eligible": matched_control,
                "same_film_operator_control_rolls": same_film_operator_rolls,
                "contains_official_test_frame": roll_id in sealed_rolls,
            }
        )
        for row in sorted(rows, key=lambda value: str(value["filename"])):
            filename = str(row["filename"])
            role = frame_role.get(filename, "metadata_only")
            if roll_id in operator_rolls:
                acquisition_paths.add(str(row["preview_path"]))
                if (
                    row.get("pseudogt_path") in inventory
                    and not bool(row["is_testset"])
                ):
                    acquisition_paths.add(str(row["pseudogt_path"]))
            frame_rows.append(
                {
                    "schema_version": BLUENEG_FRAME_SCHEMA,
                    "filename": filename,
                    "roll_id": roll_id,
                    "film_type": film_type,
                    "partition": row["partition"],
                    "date": row["date"],
                    "location": row["location"],
                    "scene_property": row["scene_property"],
                    "preview_path": row["preview_path"],
                    "pseudogt_path": row["pseudogt_path"],
                    "research_pool": pool,
                    "frame_role": role,
                    "matched_control_eligible": matched_control,
                    "allowed_use": "internal_research_with_required_attribution",
                    "redistributable_without_credit": False,
                    "required_credit": config.required_credit,
                }
            )

    missing = sorted(path for path in acquisition_paths if path not in inventory)
    if missing:
        raise BlueNegContractError(f"acquisition paths absent remotely: {missing[:3]}")
    acquisition_rows = [
        {
            "path": path,
            "size": int(inventory[path]["size"]),
            "sha256": inventory[path].get("sha256"),
        }
        for path in sorted(acquisition_paths)
    ]
    paths = {
        "frames": config.output_dir / "frames.jsonl",
        "rolls": config.output_dir / "rolls.jsonl",
        "acquisition": config.output_dir / "acquisition.json",
        "report": config.output_dir / "report.json",
    }
    hashes = {
        "frames.jsonl": _atomic_jsonl(paths["frames"], frame_rows),
        "rolls.jsonl": _atomic_jsonl(paths["rolls"], roll_rows),
    }
    acquisition_payload = {
        "schema_version": 1,
        "repo_id": config.repo_id,
        "revision": config.revision,
        "files": acquisition_rows,
        "file_count": len(acquisition_rows),
        "bytes": sum(row["size"] for row in acquisition_rows),
        "full_archive_forbidden": True,
        "required_credit": config.required_credit,
    }
    hashes["acquisition.json"] = _atomic_json(paths["acquisition"], acquisition_payload)
    pool_counts: dict[str, int] = {}
    for row in roll_rows:
        pool_counts[row["research_pool"]] = pool_counts.get(row["research_pool"], 0) + 1
    report = {
        "schema_version": 1,
        "repo_id": config.repo_id,
        "revision": config.revision,
        "software_commit": config.software_commit,
        "metadata": {
            "frames": len(metadata),
            "rolls": len(by_roll),
            "film_types": len(film_types),
            "sealed_official_test_rolls": len(sealed_rolls),
            "usable_nonsealed_frames": sum(
                len(rows) for roll_id, rows in by_roll.items() if roll_id not in sealed_rolls
            ),
        },
        "operator_rolls": len(operator_rolls),
        "matched_control_operator_rolls": sum(
            bool(row["matched_same_film_wrong_roll_control_eligible"])
            for row in roll_rows
        ),
        "roll_pool_counts": pool_counts,
        "acquisition": {
            "files": len(acquisition_rows),
            "bytes": acquisition_payload["bytes"],
            "full_two_lane_bytes": config.expected["preview_bytes"]
            + config.expected["pseudogt_bytes"],
            "pixels_downloaded_or_decoded": False,
        },
        "manifest_sha256": hashes,
        "whole_roll_overlap": 0,
        "claim_boundary": "archive/scanner-specific grouped-negative mechanism pilot only",
    }
    _atomic_json(paths["report"], report)
    return BlueNegEvidenceResult(report=report, paths=paths)
