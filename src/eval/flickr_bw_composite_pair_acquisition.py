"""Bounded acquisition and deterministic split of BO9 B&W composites."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageOps

from src.eval.flickr_single_author_pair_acquisition import atomic_json, canonical_bytes, sha256_file


SCHEMA = "neuro-film.u5-r2bp0-flickr-bw-composite-pair-acquisition.v1"


class FlickrBwCompositeAcquisitionError(ValueError):
    """Raised when a bounded composite or its deterministic split drifts."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            result.append((start, end))
            start = None
    return result


def split_composite(rgb: np.ndarray, contract: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split top-film/bottom-digital views using the explicit white separator."""

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise FlickrBwCompositeAcquisitionError("composite must be uint8 RGB")
    white = np.all(rgb >= int(contract["white_threshold"]), axis=2)
    row_white = np.mean(white, axis=1)
    start = int(contract["central_search_start"])
    end = min(int(contract["central_search_end"]), rgb.shape[0])
    candidates = [
        run
        for run in _runs(row_white[start:end] >= float(contract["minimum_white_row_fraction"]))
        if run[1] - run[0] + 1 >= int(contract["minimum_separator_rows"])
    ]
    if len(candidates) != 1:
        raise FlickrBwCompositeAcquisitionError("expected one central white separator")
    separator = (candidates[0][0] + start, candidates[0][1] + start)

    def crop_between(y0: int, y1: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        region = rgb[y0:y1]
        nonwhite = np.any(region < int(contract["white_threshold"]), axis=2)
        row_support = np.mean(nonwhite, axis=1) >= float(contract["minimum_content_row_fraction"])
        if not np.any(row_support):
            raise FlickrBwCompositeAcquisitionError("empty composite role")
        ys = np.flatnonzero(row_support)
        local_y0, local_y1 = int(ys[0]), int(ys[-1]) + 1
        content_rows = nonwhite[local_y0:local_y1]
        col_support = np.mean(content_rows, axis=0) >= float(contract["minimum_content_column_fraction"])
        if not np.any(col_support):
            raise FlickrBwCompositeAcquisitionError("empty composite columns")
        xs = np.flatnonzero(col_support)
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        crop = np.ascontiguousarray(region[local_y0:local_y1, x0:x1])
        if crop.shape[1] < int(contract["minimum_crop_width"]) or crop.shape[0] < int(
            contract["minimum_crop_height"]
        ):
            raise FlickrBwCompositeAcquisitionError("derived crop is below frozen dimensions")
        return crop, (x0, y0 + local_y0, x1, y0 + local_y1)

    film, film_box = crop_between(0, separator[0])
    digital, digital_box = crop_between(separator[1] + 1, rgb.shape[0])
    return film, digital, {
        "separator_rows_inclusive": list(separator),
        "film_box_xyxy": list(film_box),
        "digital_box_xyxy": list(digital_box),
        "film_shape": list(film.shape),
        "digital_shape": list(digital.shape),
    }


def _png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", compress_level=9)
    return buffer.getvalue()


def _write_once_or_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FlickrBwCompositeAcquisitionError(f"existing file drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _download(url: str, config: Mapping[str, Any], session: requests.Session) -> bytes:
    limits = config["acquisition"]
    error: Exception | None = None
    for attempt in range(int(limits["retries"])):
        response = None
        try:
            response = session.get(url, timeout=float(limits["timeout_seconds"]), stream=True)
            response.raise_for_status()
            if response.headers.get("content-type", "").split(";", 1)[0].lower() not in {
                "image/jpeg",
                "image/jpg",
            }:
                raise FlickrBwCompositeAcquisitionError("response is not JPEG")
            payload = bytearray()
            for chunk in response.iter_content(1024 * 1024):
                payload.extend(chunk)
                if len(payload) > int(limits["maximum_bytes_per_file"]):
                    raise FlickrBwCompositeAcquisitionError("per-file byte cap exceeded")
            return bytes(payload)
        except (requests.RequestException, FlickrBwCompositeAcquisitionError) as exc:
            error = exc
            if attempt + 1 < int(limits["retries"]):
                time.sleep(float(limits["retry_backoff_seconds"]) * (2**attempt))
        finally:
            if response is not None:
                response.close()
    raise FlickrBwCompositeAcquisitionError(f"download failed: {error}")


def acquire(parent: Mapping[str, Any], config: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema") != SCHEMA:
        raise FlickrBwCompositeAcquisitionError("invalid BP0 contract")
    if re.fullmatch(r"[0-9a-f]{40}", str(config.get("software_commit", ""))) is None:
        raise FlickrBwCompositeAcquisitionError("software commit is not frozen")
    if (
        parent.get("stable_evidence_id") != config["parent"]["stable_evidence_id"]
        or not parent.get("automatic_pass")
    ):
        raise FlickrBwCompositeAcquisitionError("BO9 parent identity or gate drift")
    records = parent.get("records", [])
    if len(records) != int(config["acquisition"]["expected_composites"]):
        raise FlickrBwCompositeAcquisitionError("parent composite count drift")
    data_root = root / str(config["acquisition"]["data_root"])
    session = requests.Session()
    session.headers["User-Agent"] = "K-MCFM-research/BP0 bounded composite audit"
    rows: list[dict[str, Any]] = []
    total = 0
    for record in sorted(records, key=lambda value: int(value["scene_id"])):
        scene = int(record["scene_id"])
        photo_id = str(record["photo_id"])
        composite_rel = Path("composites") / f"scene-{scene:02d}-{photo_id}.jpg"
        composite_path = data_root / composite_rel
        payload = composite_path.read_bytes() if composite_path.is_file() else _download(
            str(record["derivative_url_l"]), config, session
        )
        total += len(payload)
        if total > int(config["acquisition"]["maximum_bytes_total"]):
            raise FlickrBwCompositeAcquisitionError("aggregate byte cap exceeded")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                if (
                    image.format != config["acquisition"]["required_format"]
                    or image.size
                    != (
                        int(config["acquisition"]["required_width"]),
                        int(config["acquisition"]["required_height"]),
                    )
                    or int(getattr(image, "n_frames", 1)) != 1
                ):
                    raise FlickrBwCompositeAcquisitionError("composite decode contract failed")
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
        except FlickrBwCompositeAcquisitionError:
            raise
        except Exception as exc:
            raise FlickrBwCompositeAcquisitionError("composite decode failed") from exc
        film, digital, split = split_composite(rgb, config["split"])
        film_payload, digital_payload = _png_bytes(film), _png_bytes(digital)
        film_rel = Path("pairs") / f"scene-{scene:02d}-film.png"
        digital_rel = Path("pairs") / f"scene-{scene:02d}-digital.png"
        _write_once_or_exact(composite_path, payload)
        _write_once_or_exact(data_root / film_rel, film_payload)
        _write_once_or_exact(data_root / digital_rel, digital_payload)
        rows.append(
            {
                "scene_id": scene,
                "photo_id": photo_id,
                "page_url": record["page_url"],
                "license_id": record["license_id"],
                "composite_local_path": composite_rel.as_posix(),
                "composite_bytes": len(payload),
                "composite_sha256": sha256_bytes(payload),
                "split": split,
                "film_local_path": film_rel.as_posix(),
                "film_bytes": len(film_payload),
                "film_sha256": sha256_bytes(film_payload),
                "digital_local_path": digital_rel.as_posix(),
                "digital_bytes": len(digital_payload),
                "digital_sha256": sha256_bytes(digital_payload),
            }
        )
        time.sleep(float(config["acquisition"]["request_interval_seconds"]))
    manifest = {
        "schema": "neuro-film.u5-r2bp0-flickr-bw-composite-pair-manifest.v1",
        "node": config["node"],
        "software_commit": config["software_commit"],
        "parent_report_sha256": config["parent"]["report_sha256"],
        "data_root": config["acquisition"]["data_root"],
        "files": len(rows) * 3,
        "bytes_downloaded": total,
        "rows": rows,
    }
    composite_hashes = [row["composite_sha256"] for row in rows]
    derived_hashes = [value for row in rows for value in (row["film_sha256"], row["digital_sha256"])]
    gates = config["integrity_gates"]
    checks = {
        "required_composites": len(rows) == int(gates["required_composites"]),
        "required_derived_files": len(derived_hashes) == int(gates["required_derived_files"]),
        "exact_composite_duplicates": len(composite_hashes) - len(set(composite_hashes))
        <= int(gates["maximum_exact_duplicate_composites"]),
        "exact_derived_duplicates": len(derived_hashes) - len(set(derived_hashes))
        <= int(gates["maximum_exact_duplicate_derived_files"]),
    }
    stable = {
        "schema": "neuro-film.u5-r2bp0-flickr-bw-composite-pair-acquisition-report.v1",
        "node": config["node"],
        "parent_report_sha256": config["parent"]["report_sha256"],
        "metrics": {
            "composites": len(rows),
            "derived_pairs": len(rows),
            "downloaded_bytes": total,
            "film_shapes": dict(sorted(Counter(str(row["split"]["film_shape"]) for row in rows).items())),
            "digital_shapes": dict(
                sorted(Counter(str(row["split"]["digital_shape"]) for row in rows).items())
            ),
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "branch": config["branches"]["pass" if all(checks.values()) else "fail"],
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return manifest, {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}


__all__ = [
    "FlickrBwCompositeAcquisitionError",
    "SCHEMA",
    "acquire",
    "split_composite",
]
