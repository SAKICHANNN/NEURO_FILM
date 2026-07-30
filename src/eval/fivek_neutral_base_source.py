"""Exact source audit for the retained FiveK neutral-base paired pilot."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


class FiveKNeutralBaseSourceError(ValueError):
    """Raised when the frozen FiveK source contract or retained bytes drift."""


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


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _camera_group(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        exif = image.getexif()
        make = str(exif.get(271, "")).strip()
        model = str(exif.get(272, "")).strip()
    if not make or not model:
        raise FiveKNeutralBaseSourceError(
            f"missing EXIF make/model: {path.name}"
        )
    return " ".join(make.split()), " ".join(model.split())


def _freeze_rows(root: Path, path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        source_name = str(row["source_name"])
        if source_name in result:
            raise FiveKNeutralBaseSourceError(
                f"duplicate freeze source: {source_name}"
            )
        result[source_name] = row
    if not result:
        raise FiveKNeutralBaseSourceError("empty freeze manifest")
    return result


def validate_contract(root: Path, config: Mapping[str, Any]) -> None:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKNeutralBaseSourceError("source contract is not frozen")
    source = config["source"]
    for key in ("freeze_manifest", "freeze_summary"):
        path = root / str(source[key])
        if not path.is_file():
            raise FiveKNeutralBaseSourceError(f"missing source file: {key}")
        if _sha256(path) != str(source[f"{key}_sha256"]):
            raise FiveKNeutralBaseSourceError(f"source hash drift: {key}")
    if int(source["expected_pair_count"]) != 64:
        raise FiveKNeutralBaseSourceError("pair-count contract drift")
    if config["grouping"]["primary_group"] != "normalized EXIF make + model":
        raise FiveKNeutralBaseSourceError("camera grouping contract drift")
    if "train a network or model that directly emits final RGB" not in set(
        config["forbidden"]
    ):
        raise FiveKNeutralBaseSourceError("final-RGB prohibition missing")


def build_source_evidence(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validate_contract(root, config)
    source = config["source"]
    raw_root = root / str(source["raw_root"])
    expert_root = root / str(source["expert_root"])
    freeze = _freeze_rows(root, root / str(source["freeze_manifest"]))
    raw_paths = sorted(path for path in raw_root.iterdir() if path.is_file())
    expert_by_stem = {
        path.stem: path
        for path in expert_root.iterdir()
        if path.is_file()
    }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    groups: Counter[str] = Counter()
    for raw_path in raw_paths:
        expert_path = expert_by_stem.get(raw_path.stem)
        freeze_row = freeze.get(raw_path.stem)
        if expert_path is None or freeze_row is None:
            missing.append(raw_path.stem)
            continue
        make, model = _camera_group(raw_path)
        group_id = f"{make}::{model}"
        groups[group_id] += 1
        rows.append(
            {
                "pair_id": str(freeze_row["id"]),
                "source_name": raw_path.stem,
                "raw_path": _relative(root, raw_path),
                "raw_sha256": _sha256(raw_path),
                "raw_bytes": raw_path.stat().st_size,
                "expert_path": _relative(root, expert_path),
                "expert_sha256": _sha256(expert_path),
                "expert_bytes": expert_path.stat().st_size,
                "camera_make": make,
                "camera_model": model,
                "camera_group_id": group_id,
                "visible_width": int(freeze_row["raw_visible_width"]),
                "visible_height": int(freeze_row["raw_visible_height"]),
            }
        )
    rows.sort(key=lambda row: row["pair_id"])
    pair_count = len(rows)
    largest_group_count = max(groups.values(), default=0)
    largest_group_share = (
        float(largest_group_count / pair_count) if pair_count else 1.0
    )
    gates = {
        "pair_count": pair_count == int(source["expected_pair_count"]),
        "camera_model_groups": (
            len(groups)
            >= int(config["pass_gates"]["camera_model_groups_minimum"])
        ),
        "largest_group_share": (
            largest_group_share
            <= float(config["pass_gates"]["largest_group_share_maximum"])
        ),
        "missing_pairs": not missing,
        "unknown_camera_groups": all(
            row["camera_make"] and row["camera_model"] for row in rows
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "rows": rows,
    }
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    stable_payload = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_path),
        "manifest_sha256": manifest_sha256,
        "pair_count": pair_count,
        "camera_model_group_count": len(groups),
        "largest_group_count": largest_group_count,
        "largest_group_share": largest_group_share,
        "group_counts": dict(sorted(groups.items())),
        "missing": sorted(missing),
        "gates": gates,
        "automatic_pass": all(gates.values()),
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_id = hashlib.sha256(_canonical_bytes(stable_payload)).hexdigest()
    report = {
        "schema_version": 1,
        "software_commit": software_commit,
        **stable_payload,
        "stable_evidence_id": stable_id,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / str(config["output"]["manifest"])).write_bytes(
        manifest_bytes
    )
    report_bytes = _canonical_bytes(report)
    (output_dir / str(config["output"]["report"])).write_bytes(report_bytes)
    return {
        "manifest": manifest,
        "report": report,
        "manifest_sha256": manifest_sha256,
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }


__all__ = [
    "FiveKNeutralBaseSourceError",
    "build_source_evidence",
    "validate_contract",
]
