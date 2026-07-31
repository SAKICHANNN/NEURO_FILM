"""Bounded acquisition and source audit for U6.P6L repeat scans."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
import tifffile


SCHEMA = "neuro_film.u6_p6l_uchicago_repeat_scan_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6l_uchicago_repeat_scan_source_report.v1"
_ALLOWED_HOST = "knowledge.uchicago.edu"


class RepeatScanSourceError(RuntimeError):
    """Raised when the frozen P6L source contract or evidence is invalid."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_file(path: Path, name: str) -> str:
    digest = hashlib.new(name)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_destination(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RepeatScanSourceError("destination must be a bounded relative path")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise RepeatScanSourceError("unsupported P6L contract")
    acquisition = payload["acquisition"]
    files = acquisition["files"]
    if (
        len(files) != 4
        or sum(int(row["expected_bytes"]) for row in files)
        != int(acquisition["maximum_total_bytes"])
        or int(acquisition["maximum_total_bytes"]) != 88_475_062
        or {row["role"] for row in files}
        != {
            "same_plate_repeat_1",
            "same_plate_repeat_2",
            "same_plate_repeat_3",
            "source_metadata",
        }
        or not acquisition["network_policy"]["bounded_exact_files_only"]
        or not acquisition["network_policy"]["no_other_record_files"]
        or payload["source_gate"]["required_tiff_count"] != 3
        or not payload["source_gate"]["two_independent_audits"]
    ):
        raise RepeatScanSourceError("P6L frozen contract drift")
    _validate_relative_destination(acquisition["destination"])
    for row in files:
        parsed = urlparse(row["url"])
        if (
            parsed.scheme != "https"
            or parsed.hostname != _ALLOWED_HOST
            or Path(parsed.path).name != "content"
            or int(row["expected_bytes"]) <= 0
            or len(row["expected_md5"]) != 32
        ):
            raise RepeatScanSourceError("P6L source row is invalid")
    return payload


def _download_one(row: dict[str, Any], destination: Path) -> None:
    expected_bytes = int(row["expected_bytes"])
    part = destination.with_name(destination.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_bytes:
        raise RepeatScanSourceError("partial file exceeds frozen size")
    headers = {"User-Agent": "neuro-film-p6l/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(row["url"], headers=headers)
    with urlopen(request, timeout=60) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
            raise RepeatScanSourceError("download redirected off frozen host")
        status = int(getattr(response, "status", 200))
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset and status == 206 else "wb"
        written = offset
        with part.open(mode) as stream:
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > expected_bytes:
                    raise RepeatScanSourceError("download exceeds frozen size")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if written != expected_bytes:
        raise RepeatScanSourceError("download ended before frozen size")
    if _hash_file(part, "md5") != row["expected_md5"]:
        raise RepeatScanSourceError("download MD5 mismatch")
    os.replace(part, destination)


def acquire_frozen_subset(config: dict[str, Any], root: Path) -> Path:
    relative = _validate_relative_destination(
        config["acquisition"]["destination"]
    )
    destination = root / relative
    destination.mkdir(parents=True, exist_ok=True)
    for row in config["acquisition"]["files"]:
        target = destination / row["name"]
        if target.exists():
            if (
                target.is_file()
                and target.stat().st_size == int(row["expected_bytes"])
                and _hash_file(target, "md5") == row["expected_md5"]
            ):
                continue
            raise RepeatScanSourceError("existing source file does not match")
        _download_one(row, target)
    return destination


def _normalise_image(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.size == 0:
        raise RepeatScanSourceError("repeat scan must decode as one 2D plane")
    if array.dtype.kind not in "ui" or array.dtype.itemsize > 2:
        raise RepeatScanSourceError("repeat scan dtype must be uint8 or uint16")
    maximum = float(np.iinfo(array.dtype).max)
    return np.asarray(array, dtype=np.float32) / maximum


def _overlap(
    reference: np.ndarray, moving: np.ndarray, dx: int, dy: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference.shape
    ref_x0 = max(0, dx)
    ref_x1 = min(width, width + dx)
    ref_y0 = max(0, dy)
    ref_y1 = min(height, height + dy)
    mov_x0 = max(0, -dx)
    mov_x1 = min(width, width - dx)
    mov_y0 = max(0, -dy)
    mov_y1 = min(height, height - dy)
    return (
        reference[ref_y0:ref_y1, ref_x0:ref_x1],
        moving[mov_y0:mov_y1, mov_x0:mov_x1],
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    stride = max(1, int(math.ceil(max(left.shape) / 1024)))
    a = np.asarray(left[::stride, ::stride], dtype=np.float64).reshape(-1)
    b = np.asarray(right[::stride, ::stride], dtype=np.float64).reshape(-1)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def _integer_registration(
    reference: np.ndarray, moving: np.ndarray, maximum_shift: int
) -> tuple[int, int, float, float]:
    stride = max(1, int(math.ceil(max(reference.shape) / 960)))
    ref_small = np.ascontiguousarray(reference[::stride, ::stride])
    mov_small = np.ascontiguousarray(moving[::stride, ::stride])
    phase_shift, response = cv2.phaseCorrelate(ref_small, mov_small)
    estimates = {
        (
            int(round(sign * phase_shift[0] * stride)),
            int(round(sign * phase_shift[1] * stride)),
        )
        for sign in (-1.0, 1.0)
    }
    candidates: set[tuple[int, int]] = {(0, 0)}
    for base_x, base_y in estimates:
        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                dx, dy = base_x + delta_x, base_y + delta_y
                if abs(dx) <= maximum_shift and abs(dy) <= maximum_shift:
                    candidates.add((dx, dy))
    scored = []
    for dx, dy in sorted(candidates):
        left, right = _overlap(reference, moving, dx, dy)
        scored.append((_correlation(left, right), dx, dy))
    correlation, dx, dy = max(scored)
    return dx, dy, correlation, float(response)


def audit_repeat_scan_source(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    destination = root / _validate_relative_destination(
        config["acquisition"]["destination"]
    )
    file_rows = []
    images = []
    for row in config["acquisition"]["files"]:
        path = destination / row["name"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["expected_bytes"])
            or _hash_file(path, "md5") != row["expected_md5"]
        ):
            raise RepeatScanSourceError("P6L local source integrity mismatch")
        item = {
            "name": row["name"],
            "role": row["role"],
            "bytes": path.stat().st_size,
            "md5": row["expected_md5"],
            "sha256": _hash_file(path, "sha256"),
        }
        if row["role"].startswith("same_plate_repeat"):
            with tifffile.TiffFile(path) as document:
                if len(document.pages) != 1:
                    raise RepeatScanSourceError("repeat scan must be single-frame")
                decoded = document.asarray()
            item.update(
                {
                    "shape": list(decoded.shape),
                    "dtype": str(decoded.dtype),
                    "minimum_code": int(np.min(decoded)),
                    "maximum_code": int(np.max(decoded)),
                }
            )
            images.append(_normalise_image(decoded))
        file_rows.append(item)

    if len(images) != int(config["source_gate"]["required_tiff_count"]):
        raise RepeatScanSourceError("repeat scan inventory mismatch")
    shapes = {image.shape for image in images}
    dtypes = {
        row["dtype"]
        for row in file_rows
        if row["role"].startswith("same_plate_repeat")
    }
    if (
        config["source_gate"]["required_equal_shape"]
        and len(shapes) != 1
    ) or (
        config["source_gate"]["required_equal_dtype"]
        and len(dtypes) != 1
    ):
        raise RepeatScanSourceError("repeat scan decode identity mismatch")

    maximum_shift = int(
        config["source_gate"]["maximum_integer_registration_shift_pixels"]
    )
    registration = []
    for index in range(1, len(images)):
        dx, dy, correlation, phase_response = _integer_registration(
            images[0], images[index], maximum_shift
        )
        registration.append(
            {
                "reference_repeat": 1,
                "moving_repeat": index + 1,
                "integer_shift_xy": [dx, dy],
                "registered_correlation": correlation,
                "phase_response": phase_response,
            }
        )

    minimum_correlation = min(
        row["registered_correlation"] for row in registration
    )
    source_pass = (
        max(
            max(abs(value) for value in row["integer_shift_xy"])
            for row in registration
        )
        <= maximum_shift
        and minimum_correlation
        >= float(config["source_gate"]["minimum_registered_pair_correlation"])
    )
    stable = {
        "experiment_id": config["experiment_id"],
        "files": file_rows,
        "registration": registration,
        "minimum_registered_pair_correlation": minimum_correlation,
        "source_pass": source_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_json(stable)
        ).hexdigest(),
        "decision": (
            "open_repeat_residual_analysis"
            if source_pass
            else "close_source_before_residual_analysis"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "RepeatScanSourceError",
    "acquire_frozen_subset",
    "audit_repeat_scan_source",
    "load_contract",
]
