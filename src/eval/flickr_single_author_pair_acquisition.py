"""Bounded acquisition and integrity audit for the BO0 weak-pair source."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

from src.real_film.fsa_owi_pilot import dhash64, hamming64


SCHEMA = "neuro-film.u5-r2bo1-flickr-single-author-pair-acquisition.v1"


class FlickrPairAcquisitionError(ValueError):
    """Raised when the frozen acquisition contract or pixels drift."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return sha256_bytes(payload)


def selected_rows(parent: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Materialize only the 53 complete BO0 pairs and their two exact roles."""

    if config.get("schema") != SCHEMA:
        raise FlickrPairAcquisitionError("invalid BO1 schema")
    if parent.get("stable_evidence_id") != config["parent"]["stable_evidence_id"]:
        raise FlickrPairAcquisitionError("parent evidence identity drift")
    if not parent.get("automatic_pass"):
        raise FlickrPairAcquisitionError("parent source gate did not pass")
    records = {str(row["photo_id"]): row for row in parent.get("records", [])}
    limits = config["acquisition"]
    allowed = {int(value) for value in limits["allowed_license_ids"]}
    rows: list[dict[str, Any]] = []
    pairs = parent.get("complete_pairs", [])
    if len(pairs) != int(limits["expected_pairs"]):
        raise FlickrPairAcquisitionError("parent pair count drift")
    for pair_index, pair in enumerate(pairs):
        if not all(
            pair.get(key)
            for key in (
                "both_allowed_license",
                "both_public_photo_media",
                "both_bounded_derivatives",
            )
        ):
            raise FlickrPairAcquisitionError("parent contains an ineligible complete pair")
        for role, id_key in (("digital", "digital_photo_id"), ("film", "film_photo_id")):
            source = records.get(str(pair[id_key]))
            if source is None or source.get("role") != role:
                raise FlickrPairAcquisitionError("parent pair role lookup failed")
            width, height = source.get("derivative_width_l"), source.get("derivative_height_l")
            if (
                int(source.get("license_id", -1)) not in allowed
                or not source.get("public_photo_media")
                or not isinstance(source.get("derivative_url_l"), str)
                or not isinstance(width, int)
                or not isinstance(height, int)
                or max(width, height) > int(limits["maximum_long_edge"])
                or width * height < int(limits["minimum_pixels"])
            ):
                raise FlickrPairAcquisitionError("selected derivative violates frozen metadata limits")
            rows.append(
                {
                    "selection_index": len(rows),
                    "pair_index": pair_index,
                    "pair_id": str(pair["pair_id"]),
                    "family_id": str(pair["family_id"]),
                    "scene_id": int(pair["scene_id"]),
                    "role": role,
                    "photo_id": str(source["photo_id"]),
                    "page_url": str(source["page_url"]),
                    "derivative_url": str(source["derivative_url_l"]),
                    "expected_width": width,
                    "expected_height": height,
                    "license_id": int(source["license_id"]),
                    "stock_label": str(pair["stock_label"]),
                }
            )
    if len(rows) != int(limits["expected_files"]):
        raise FlickrPairAcquisitionError("selected file count drift")
    return rows


def _decode(payload: bytes, row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    limits = config["acquisition"]
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            frames = int(getattr(image, "n_frames", 1))
            if image.format != limits["required_format"]:
                raise FlickrPairAcquisitionError("decoded format is not frozen JPEG")
            if image.mode not in set(limits["allowed_modes"]):
                raise FlickrPairAcquisitionError("decoded mode is not allowed")
            if limits["require_single_frame"] and frames != 1:
                raise FlickrPairAcquisitionError("multi-frame derivative is forbidden")
            tolerance = int(limits["maximum_metadata_dimension_delta_per_axis"])
            if (
                abs(width - int(row["expected_width"])) > tolerance
                or abs(height - int(row["expected_height"])) > tolerance
            ):
                raise FlickrPairAcquisitionError("decoded dimensions drift from BO0 metadata")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            return {
                "width": width,
                "height": height,
                "mode": image.mode,
                "format": image.format,
                "frames": frames,
                "dhash64": dhash64(rgb),
            }
    except FlickrPairAcquisitionError:
        raise
    except Exception as exc:
        raise FlickrPairAcquisitionError("derivative failed Pillow decode") from exc


def _download_payload(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    session: requests.Session,
) -> tuple[bytes, str]:
    limits = config["acquisition"]
    error: Exception | None = None
    for attempt in range(int(limits["retries"])):
        response = None
        try:
            response = session.get(
                str(row["derivative_url"]),
                timeout=float(limits["timeout_seconds"]),
                stream=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type not in {"image/jpeg", "image/jpg"}:
                raise FlickrPairAcquisitionError("response is not JPEG")
            buffer = bytearray()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    buffer.extend(chunk)
                if len(buffer) > int(limits["maximum_bytes_per_file"]):
                    raise FlickrPairAcquisitionError("per-file byte cap exceeded")
            if not buffer:
                raise FlickrPairAcquisitionError("empty derivative payload")
            return bytes(buffer), content_type
        except (requests.RequestException, FlickrPairAcquisitionError) as exc:
            error = exc
            if attempt + 1 < int(limits["retries"]):
                time.sleep(float(limits["retry_backoff_seconds"]) * (2**attempt))
        finally:
            if response is not None:
                response.close()
    raise FlickrPairAcquisitionError(f"download failed for {row['photo_id']}: {error}")


def acquire(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    config: Mapping[str, Any],
    prior_manifest: Mapping[str, Any] | None = None,
    session: requests.Session | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Acquire exact derivatives atomically and support hash-bound restart."""

    limits = config["acquisition"]
    client = session or requests.Session()
    client.headers["User-Agent"] = "K-MCFM-research/BO1 bounded weak-pair audit"
    prior = {
        str(item["photo_id"]): item for item in (prior_manifest or {}).get("rows", [])
    }
    records: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        relative = Path(str(row["family_id"])) / f"scene-{int(row['scene_id']):02d}-{row['role']}-{row['photo_id']}.jpg"
        target = root / relative
        previous = prior.get(str(row["photo_id"]))
        content_type = "image/jpeg"
        if target.is_file():
            if previous is None:
                raise FlickrPairAcquisitionError(f"untracked existing derivative: {target}")
            payload = target.read_bytes()
            if sha256_bytes(payload) != previous["sha256"] or len(payload) != previous["bytes"]:
                raise FlickrPairAcquisitionError("resumed derivative identity drift")
            content_type = str(previous["content_type"])
        else:
            payload, content_type = _download_payload(row, config, client)
        if total + len(payload) > int(limits["maximum_bytes_total"]):
            raise FlickrPairAcquisitionError("aggregate byte cap exceeded")
        decoded = _decode(payload, row, config)
        record = {
            **dict(row),
            "local_path": relative.as_posix(),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "content_type": content_type,
            **decoded,
        }
        if previous is not None and any(previous.get(key) != record.get(key) for key in record):
            raise FlickrPairAcquisitionError("resumed manifest row drift")
        records.append(record)
        total += len(payload)
        if checkpoint_path is not None:
            atomic_json(checkpoint_path, _manifest(records, total, config, complete=False))
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        if previous is None and float(limits["request_interval_seconds"]) > 0:
            time.sleep(float(limits["request_interval_seconds"]))
    return _manifest(records, total, config, complete=True)


def _manifest(
    rows: Sequence[Mapping[str, Any]], total: int, config: Mapping[str, Any], *, complete: bool
) -> dict[str, Any]:
    return {
        "schema": "neuro-film.u5-r2bo1-flickr-single-author-pair-download-manifest.v1",
        "node": config["node"],
        "parent_report_sha256": config["parent"]["report_sha256"],
        "files": len(rows),
        "bytes": total,
        "complete": complete,
        "rows": list(rows),
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def audit(manifest: Mapping[str, Any], *, root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read every byte, decode every image, and evaluate frozen integrity gates."""

    limits = config["acquisition"]
    if not manifest.get("complete") or len(manifest.get("rows", [])) != int(limits["expected_files"]):
        raise FlickrPairAcquisitionError("download manifest is incomplete")
    records: list[dict[str, Any]] = []
    sha_groups: dict[str, list[str]] = defaultdict(list)
    for source in manifest["rows"]:
        target = root / Path(str(source["local_path"]))
        if not target.is_file():
            raise FlickrPairAcquisitionError(f"missing derivative: {target}")
        payload = target.read_bytes()
        if len(payload) != int(source["bytes"]) or sha256_bytes(payload) != source["sha256"]:
            raise FlickrPairAcquisitionError("persisted derivative identity drift")
        decoded = _decode(payload, source, config)
        if any(decoded[key] != source[key] for key in decoded):
            raise FlickrPairAcquisitionError("persisted decode facts drift")
        records.append(dict(source))
        sha_groups[str(source["sha256"])].append(str(source["local_path"]))
    exact = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(sha_groups.items())
        if len(paths) > 1
    ]
    near: list[dict[str, Any]] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            distance = hamming64(str(left["dhash64"]), str(right["dhash64"]))
            if distance <= 4:
                near.append(
                    {
                        "left": left["local_path"],
                        "right": right["local_path"],
                        "left_pair_id": left["pair_id"],
                        "right_pair_id": right["pair_id"],
                        "hamming": distance,
                    }
                )
    cross_pair_near = [item for item in near if item["left_pair_id"] != item["right_pair_id"]]
    pair_roles: dict[str, set[str]] = defaultdict(set)
    for row in records:
        pair_roles[str(row["pair_id"])].add(str(row["role"]))
    complete_pairs = sum(roles == {"digital", "film"} for roles in pair_roles.values())
    gates = config["integrity_gates"]
    checks = {
        "required_files": len(records) == int(gates["required_files"]),
        "required_complete_pairs": complete_pairs == int(gates["required_complete_pairs"]),
        "exact_duplicate_groups": len(exact) <= int(gates["maximum_exact_duplicate_groups"]),
        "cross_pair_dhash_le_4": len(cross_pair_near) <= int(gates["maximum_cross_pair_dhash_le_4"]),
    }
    stable = {
        "schema": "neuro-film.u5-r2bo1-flickr-single-author-pair-integrity-report.v1",
        "node": config["node"],
        "parent_report_sha256": config["parent"]["report_sha256"],
        "metrics": {
            "files": len(records),
            "bytes": sum(int(row["bytes"]) for row in records),
            "complete_pairs": complete_pairs,
            "families": len({row["family_id"] for row in records}),
            "exact_duplicate_groups": len(exact),
            "within_pair_dhash_le_4": len(near) - len(cross_pair_near),
            "cross_pair_dhash_le_4": len(cross_pair_near),
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "branch": config["branches"]["pass" if all(checks.values()) else "fail"],
        "exact_duplicate_groups": exact,
        "dhash_pairs_le_4": near,
        "records": records,
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": sha256_bytes(canonical_bytes(stable))}


__all__ = [
    "FlickrPairAcquisitionError",
    "SCHEMA",
    "acquire",
    "atomic_json",
    "audit",
    "canonical_bytes",
    "selected_rows",
    "sha256_file",
]
