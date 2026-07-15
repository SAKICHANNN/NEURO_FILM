"""Deterministic, stock-scoped acquisition contracts for real-film pilots."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .stock_registry import StockRegistryError, crosscheck_blueneg, validate_registry


class StockAcquisitionError(ValueError):
    """Raised when a stock pilot would leak, infer, or exceed frozen evidence."""


def _atomic_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def build_stock_pilot_acquisition(
    registry: Mapping[str, Any],
    frame_rows: Sequence[Mapping[str, Any]],
    roll_rows: Sequence[Mapping[str, Any]],
    remote_inventory: Mapping[str, Any],
    *,
    software_commit: str,
    input_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze preview/proxy paths for selected stocks with whole-test-roll sealing."""
    try:
        validation = validate_registry(registry)
        source_check = crosscheck_blueneg(registry, frame_rows, roll_rows)
    except StockRegistryError as exc:
        raise StockAcquisitionError(str(exc)) from exc
    if not source_check["passed"]:
        raise StockAcquisitionError("registry/BlueNeg metadata mismatch")

    source = registry["source_datasets"]["blueneg"]
    if remote_inventory.get("repo_id") != "ttgroup/blueneg-release":
        raise StockAcquisitionError("unexpected BlueNeg repository")
    if remote_inventory.get("revision") != source["revision"]:
        raise StockAcquisitionError("BlueNeg inventory revision mismatch")
    inventory = {str(row["path"]): row for row in remote_inventory.get("files", [])}
    if not inventory:
        raise StockAcquisitionError("empty remote inventory")
    required_hashes = {"registry", "frames", "rolls", "remote_inventory"}
    if set(input_sha256) != required_hashes or any(
        not isinstance(value, str) or len(value) != 64 for value in input_sha256.values()
    ):
        raise StockAcquisitionError("input_sha256 must freeze all four physical inputs")

    selected = {
        str(row["source_label"]): row
        for row in registry["stocks"]
        if row["source_dataset"] == "blueneg" and row["selected_first_pilot"]
    }
    selected_ids = {str(row["film_stock_id"]) for row in selected.values()}
    if set(validation["selected_pilots"]) != selected_ids:
        raise StockAcquisitionError("selected pilot identity mismatch")

    roll_by_id = {str(row["roll_id"]): row for row in roll_rows}
    sealed_rolls = {
        roll_id
        for roll_id, row in roll_by_id.items()
        if bool(row["contains_official_test_frame"])
    }
    files: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    frames_by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        frames_by_label[str(row["film_type"])].append(row)

    for label, stock in sorted(selected.items(), key=lambda item: item[1]["film_stock_id"]):
        rows = frames_by_label[label]
        all_rolls = {str(row["roll_id"]) for row in rows}
        selected_sealed = all_rolls & sealed_rolls
        eligible_rows = [row for row in rows if str(row["roll_id"]) not in sealed_rolls]
        eligible_rolls = {str(row["roll_id"]) for row in eligible_rows}
        expected_public_proxies = sum(
            int(roll_by_id[roll_id]["public_non_test_pseudogt_frames"])
            for roll_id in eligible_rolls
        )
        aligned_rolls: set[str] = set()
        unavailable_aligned_proxy_frames = 0
        preview_count = 0
        proxy_count = 0
        for frame in sorted(eligible_rows, key=lambda row: str(row["filename"])):
            preview_path = str(frame["preview_path"])
            paths = [("negative_preview", preview_path)]
            proxy_path = frame.get("pseudogt_path")
            if bool(frame["alignment_available"]) and proxy_path and str(proxy_path) in inventory:
                paths.append(("display_proxy", str(proxy_path)))
                aligned_rolls.add(str(frame["roll_id"]))
            elif bool(frame["alignment_available"]) and proxy_path:
                unavailable_aligned_proxy_frames += 1
            for lane, remote_path in paths:
                item = inventory.get(remote_path)
                if item is None:
                    raise StockAcquisitionError(f"missing remote file: {remote_path}")
                digest = item.get("sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    raise StockAcquisitionError(f"missing LFS SHA-256: {remote_path}")
                files.append(
                    {
                        "film_stock_id": stock["film_stock_id"],
                        "source_label": label,
                        "roll_id": frame["roll_id"],
                        "frame_id": frame["filename"],
                        "lane": lane,
                        "path": remote_path,
                        "size": int(item["size"]),
                        "sha256": digest,
                    }
                )
                preview_count += lane == "negative_preview"
                proxy_count += lane == "display_proxy"
        if proxy_count != expected_public_proxies:
            raise StockAcquisitionError(
                f"public proxy inventory/count mismatch for {label}: "
                f"{proxy_count} != {expected_public_proxies}"
            )
        summaries[str(stock["film_stock_id"])] = {
            "source_label": label,
            "total_rolls": len(all_rolls),
            "sealed_official_test_rolls": len(selected_sealed),
            "eligible_rolls": len(eligible_rolls),
            "eligible_preview_frames": preview_count,
            "display_proxy_frames": proxy_count,
            "display_proxy_rolls": len(aligned_rolls),
            "aligned_but_proxy_unavailable_frames": unavailable_aligned_proxy_frames,
            "metadata_identifiability_candidate": len(eligible_rolls) >= 3,
            "display_operator_candidate": len(aligned_rolls) >= 3 and proxy_count >= 12,
            "current_claim_ceiling": stock["claim_ceiling"],
        }

    files.sort(key=lambda row: (row["film_stock_id"], row["roll_id"], row["frame_id"], row["lane"]))
    paths = [row["path"] for row in files]
    if len(paths) != len(set(paths)):
        raise StockAcquisitionError("duplicate remote path across stock pilots")
    if any(str(row["roll_id"]) in sealed_rolls for row in files):
        raise StockAcquisitionError("official test roll leaked into acquisition")

    manifest = {
        "schema_version": 1,
        "manifest_id": "stock-first-blueneg-four-pilot-v1",
        "repo_id": remote_inventory["repo_id"],
        "revision": remote_inventory["revision"],
        "software_commit": software_commit,
        "input_sha256": dict(sorted(input_sha256.items())),
        "selected_film_stock_ids": sorted(selected_ids),
        "required_credit": source["required_credit"],
        "official_test_roll_policy": "seal_entire_physical_roll",
        "full_16bit_archive_forbidden": True,
        "files": files,
        "file_count": len(files),
        "bytes": sum(row["size"] for row in files),
    }
    report = {
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "software_commit": software_commit,
        "input_sha256": dict(sorted(input_sha256.items())),
        "registry_id": registry["registry_id"],
        "registry_validation": validation,
        "source_crosscheck": source_check,
        "stock_summaries": summaries,
        "sealed_official_test_rolls_global": len(sealed_rolls),
        "file_count": manifest["file_count"],
        "bytes": manifest["bytes"],
        "passed": True,
        "claim_boundary": "bounded acquisition eligibility only; no S2 stock expert or calibration",
    }
    return manifest, report


def write_stock_pilot_acquisition(
    manifest_path: Path,
    report_path: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, str]:
    """Write deterministic acquisition evidence and return physical hashes."""
    manifest_hash = _atomic_json(manifest_path, manifest)
    report_payload = dict(report)
    report_payload["manifest_sha256"] = manifest_hash
    report_hash = _atomic_json(report_path, report_payload)
    return {"manifest_sha256": manifest_hash, "report_sha256": report_hash}
