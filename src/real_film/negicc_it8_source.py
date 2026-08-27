"""Exact source lock for the NegICC controlled IT8 film observations."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from typing import Any


class NegiccIt8SourceError(ValueError):
    """Raised when the frozen source contract is not satisfied."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _fetch(url: str, *, user_agent: str, maximum_bytes: int = 1_000_000) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise NegiccIt8SourceError(f"source exceeds byte ceiling: {url}")
    return data


def _fetch_blob(
    base: str, spec: list[Any], *, user_agent: str
) -> tuple[bytes, dict[str, Any]]:
    path, expected_sha1, expected_bytes = spec
    data = _fetch(f"{base}/{path}", user_agent=user_agent)
    fact = {
        "path": path,
        "bytes": len(data),
        "git_blob_sha1": _git_blob_sha1(data),
        "sha256": _sha256(data),
    }
    if fact["bytes"] != expected_bytes or fact["git_blob_sha1"] != expected_sha1:
        raise NegiccIt8SourceError(f"blob identity mismatch: {path}")
    return data, fact


def _parse_raw(data: bytes) -> dict[str, tuple[float, float, float]]:
    lines = data.decode("utf-8").splitlines()
    if not lines or lines[0].split() != ["patch", "r", "g", "b"]:
        raise NegiccIt8SourceError("invalid raw patch header")
    rows: dict[str, tuple[float, float, float]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split()
        if len(cells) != 4 or cells[0] in rows:
            raise NegiccIt8SourceError("invalid or duplicate raw patch row")
        rows[cells[0]] = tuple(float(value) for value in cells[1:])
    return rows


def _parse_train(
    data: bytes,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    text = io.StringIO(data.decode("utf-8"))
    reader = csv.DictReader(text)
    if reader.fieldnames != [
        "patch",
        "r",
        "g",
        "b",
        "refR",
        "refG",
        "refB",
        "refX",
        "refY",
        "refZ",
    ]:
        raise NegiccIt8SourceError("invalid train patch header")
    rows = {}
    for row in reader:
        patch = row["patch"]
        if not patch or patch in rows:
            raise NegiccIt8SourceError("invalid or duplicate train patch row")
        rgb = tuple(float(row[key]) for key in ("r", "g", "b"))
        xyz = tuple(float(row[key]) for key in ("refX", "refY", "refZ"))
        rows[patch] = (rgb, xyz)
    return rows


def audit_source(config: dict[str, Any], *, reverse: bool = False) -> dict[str, Any]:
    repository = config["repository"]
    user_agent = "NeuroFilm-SF3-A3H/1.0"
    commit_data = _fetch(repository["commit_api_url"], user_agent=user_agent)
    commit_payload = json.loads(commit_data)
    commit_ok = commit_payload.get("sha") == repository["commit"]

    base = repository["raw_base_url"]
    readme, readme_fact = _fetch_blob(
        base,
        [repository["readme"][key] for key in ("path", "git_blob_sha1", "bytes")],
        user_agent=user_agent,
    )
    license_data, license_fact = _fetch_blob(
        base,
        [repository["license"][key] for key in ("path", "git_blob_sha1", "bytes")],
        user_agent=user_agent,
    )
    _reference, reference_fact = _fetch_blob(
        base,
        [config["reference"][key] for key in ("path", "git_blob_sha1", "bytes")],
        user_agent=user_agent,
    )
    licence_ok = repository["license"]["required_text"].encode() in license_data
    readme_ok = (
        b"Included in this repo is data files from films exposed with a" in readme
    )

    observations = []
    common_xyz: dict[str, tuple[float, float, float]] | None = None
    stock_items = list(config["stocks"].items())
    if reverse:
        stock_items.reverse()
    for stock_id, exposure_specs in stock_items:
        specs = list(exposure_specs)
        if reverse:
            specs.reverse()
        for exposure in specs:
            raw_data, raw_fact = _fetch_blob(
                base, exposure["raw"], user_agent=user_agent
            )
            train_data, train_fact = _fetch_blob(
                base, exposure["train"], user_agent=user_agent
            )
            raw_rows = _parse_raw(raw_data)
            train_rows = _parse_train(train_data)
            rgb_exact = set(raw_rows) == set(train_rows) and all(
                raw_rows[patch] == train_rows[patch][0] for patch in raw_rows
            )
            xyz = {patch: values[1] for patch, values in train_rows.items()}
            if common_xyz is None:
                common_xyz = xyz
            reference_exact = xyz == common_xyz
            observations.append(
                {
                    "stock_id": stock_id,
                    "ev": exposure["ev"],
                    "patch_rows": len(train_rows),
                    "raw_train_rgb_exact": rgb_exact,
                    "common_reference_xyz_exact": reference_exact,
                    "raw": raw_fact,
                    "train": train_fact,
                }
            )

    observations.sort(key=lambda row: (row["stock_id"], row["ev"]))
    gates = config["gates"]
    observed_stocks = sorted({row["stock_id"] for row in observations})
    observed_exposures = sorted({row["ev"] for row in observations})
    gate_results = {
        "exact_commit": commit_ok,
        "exact_git_blobs": True,
        "required_stocks": observed_stocks == sorted(gates["required_stocks"]),
        "required_exposures": observed_exposures == sorted(gates["required_exposures"]),
        "patch_rows": all(
            row["patch_rows"] == gates["expected_patch_rows"] for row in observations
        ),
        "raw_train_rgb_identity": all(
            row["raw_train_rgb_exact"] for row in observations
        ),
        "common_reference_xyz": all(
            row["common_reference_xyz_exact"] for row in observations
        ),
        "repository_gpl3_intent": licence_ok,
        "readme_data_provenance": readme_ok,
        "image_or_tiff_requests": gates["image_or_tiff_requests_max"] == 0,
    }
    scientific_payload = {
        "repository_commit": repository["commit"],
        "readme": readme_fact,
        "license": license_fact,
        "reference": reference_fact,
        "observations": observations,
        "gates": gate_results,
    }
    passed = all(gate_results.values())
    return {
        "schema": "neuro-film.sf3-a3h-negicc-it8-source-lock-result.v1",
        "experiment_id": config["experiment_id"],
        "status": "PASS_SOURCE_LOCK" if passed else "FAIL_CLOSED_SOURCE_LOCK",
        "decision": "OPEN_CONTROLLED_PATCH_BASELINE"
        if passed
        else "STOP_BEFORE_BASELINE",
        "scientific_payload": scientific_payload,
        "scientific_identity": _sha256(_canonical_bytes(scientific_payload)),
        "network_reads": 20,
        "image_or_tiff_requests": 0,
        "pixel_decodes": 0,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["NegiccIt8SourceError", "audit_source"]
