"""Bounded Internet Archive fixed-author/camera stock-identifiability audit."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.real_film.connected_stock_identifiability import (
    extract_descriptors,
    group_loo_centroid,
    paired_group_bootstrap_delta,
)


class InternetArchiveStockError(ValueError):
    """Raised when the frozen source or evaluation contract drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(url: str) -> dict[str, Any]:
    for attempt in range(3):
        request = urllib.request.Request(url, headers={"User-Agent": "K-MCFM-SF3.A3M/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise InternetArchiveStockError(f"metadata HTTP status {response.status}")
                return json.load(response)
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _plain_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(value or ""))).strip()


def _rank(experiment_id: str, identifier: str, name: str, md5: str) -> str:
    payload = f"{experiment_id}|{identifier}|{name}|{md5}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_source_lock(config: Mapping[str, Any], *, reverse: bool = False) -> dict[str, Any]:
    """Fetch metadata only and freeze exact original/thumbnail identities."""
    items = list(config["items"])
    if reverse:
        items.reverse()
    locked: list[dict[str, Any]] = []
    for item in items:
        identifier = str(item["identifier"])
        url = str(config["source"]["metadata_endpoint"]).format(identifier=identifier)
        payload = _request_json(url)
        metadata = payload.get("metadata", {})
        description = _plain_text(metadata.get("description"))
        subject = _plain_text(metadata.get("subject"))
        combined = f"{description} {subject}"
        if str(metadata.get("creator")) != str(config["source"]["creator"]):
            raise InternetArchiveStockError(f"creator drift: {identifier}")
        if str(metadata.get("licenseurl")) != str(config["source"]["license_url"]):
            raise InternetArchiveStockError(f"licence drift: {identifier}")
        if str(config["source"]["camera"]).casefold() not in combined.casefold():
            raise InternetArchiveStockError(f"camera drift: {identifier}")
        if str(item["film_text"]).casefold() not in combined.casefold():
            raise InternetArchiveStockError(f"stock drift: {identifier}")
        if str(metadata.get("date", ""))[:10] != str(item["capture_date"]):
            raise InternetArchiveStockError(f"capture date drift: {identifier}")
        files = list(payload.get("files", []))
        derivatives = {str(row.get("name")): row for row in files if row.get("source") == "derivative"}
        candidates: list[dict[str, Any]] = []
        for original in files:
            if original.get("source") != "original" or original.get("format") != config["selection"]["allowed_original_format"]:
                continue
            original_name = str(original.get("name", ""))
            if original_name.casefold().endswith("_thumb.jpg"):
                continue
            stem = str(Path(original_name).with_suffix(""))
            derivative_name = f"{stem}_thumb.jpg"
            derivative = derivatives.get(derivative_name)
            if derivative is None or derivative.get("format") != config["selection"]["allowed_derivative_format"]:
                continue
            record = {
                "original_name": original_name,
                "original_size": int(original["size"]),
                "original_md5": str(original["md5"]),
                "derivative_name": derivative_name,
                "derivative_size": int(derivative["size"]),
                "derivative_md5": str(derivative["md5"]),
            }
            record["rank"] = _rank(config["experiment_id"], identifier, original_name, record["original_md5"])
            candidates.append(record)
        candidates.sort(key=lambda row: (row["rank"], row["original_name"]))
        count = int(config["selection"]["frames_per_item"])
        if len(candidates) < count:
            raise InternetArchiveStockError(f"insufficient thumbnail pairs: {identifier}")
        locked.append({
            "identifier": identifier,
            "film_stock_id": str(item["film_stock_id"]),
            "film_text": str(item["film_text"]),
            "role": str(item["role"]),
            "capture_date": str(item["capture_date"]),
            "title": str(metadata.get("title")),
            "creator": str(metadata.get("creator")),
            "license_url": str(metadata.get("licenseurl")),
            "description": description,
            "selected_files": candidates[:count],
            "eligible_original_thumbnail_pairs": len(candidates),
        })
    locked.sort(key=lambda row: (row["film_stock_id"], row["role"], row["identifier"]))
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "metadata_requests": len(locked),
        "image_requests": 0,
        "pixel_decodes": 0,
        "items": locked,
    }


def _download(url: str, destination: Path, *, expected_size: int, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + ".partial")
    if stage.exists():
        stage.unlink()
    for attempt in range(3):
        digest = hashlib.md5()
        total = 0
        request = urllib.request.Request(url, headers={"User-Agent": "K-MCFM-SF3.A3M/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, stage.open("xb") as output:
                if response.status != 200:
                    raise InternetArchiveStockError(f"image HTTP status {response.status}")
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
            if total != expected_size or digest.hexdigest() != expected_md5:
                raise InternetArchiveStockError(f"download identity mismatch: {destination.name}")
            os.replace(stage, destination)
            return
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
        finally:
            if stage.exists():
                stage.unlink()


def acquire(config: Mapping[str, Any], root: Path, *, reverse: bool = False) -> dict[str, Any]:
    lock_path = root / str(config["source"]["source_lock"])
    if sha256_file(lock_path) != str(config["source"]["source_lock_sha256"]):
        raise InternetArchiveStockError("source lock hash drift")
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    items = list(source_lock["items"])
    if reverse:
        items.reverse()
    output_rows: list[dict[str, Any]] = []
    download_bytes = 0
    for item in items:
        files = list(item["selected_files"])
        if reverse:
            files.reverse()
        for selected in files:
            identifier = str(item["identifier"])
            name = str(selected["derivative_name"])
            encoded = urllib.parse.quote(name, safe="")
            url = str(config["source"]["download_endpoint"]).format(identifier=identifier, name=encoded)
            local_name = hashlib.sha256(f"{identifier}|{name}".encode()).hexdigest()[:20] + ".jpg"
            relative = Path(str(config["source"]["pixel_root"])) / str(item["film_stock_id"]) / str(item["role"]) / local_name
            destination = root / relative
            if destination.is_file():
                if destination.stat().st_size != int(selected["derivative_size"]):
                    raise InternetArchiveStockError(f"existing size drift: {relative}")
                digest = hashlib.md5(destination.read_bytes()).hexdigest()
                if digest != str(selected["derivative_md5"]):
                    raise InternetArchiveStockError(f"existing md5 drift: {relative}")
            else:
                _download(url, destination, expected_size=int(selected["derivative_size"]), expected_md5=str(selected["derivative_md5"]))
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                width, height = image.size
                if image.format != "JPEG" or width < 32 or height < 32:
                    raise InternetArchiveStockError(f"invalid thumbnail: {relative}")
            download_bytes += int(selected["derivative_size"])
            output_rows.append({
                "identifier": identifier,
                "group_id": identifier,
                "film_stock_id": str(item["film_stock_id"]),
                "binary_label": "ektar" if item["film_stock_id"] == "kodak_ektar_100" else "control",
                "role": str(item["role"]),
                "capture_date": str(item["capture_date"]),
                "original_name": str(selected["original_name"]),
                "original_md5": str(selected["original_md5"]),
                "derivative_name": name,
                "derivative_md5": str(selected["derivative_md5"]),
                "bytes": int(selected["derivative_size"]),
                "width": int(width),
                "height": int(height),
                "local_path": relative.as_posix(),
                "sha256": sha256_file(destination),
            })
    if download_bytes > int(config["selection"]["maximum_total_download_bytes"]):
        raise InternetArchiveStockError("download byte budget exceeded")
    output_rows.sort(key=lambda row: (row["film_stock_id"], row["role"], row["identifier"], row["original_name"]))
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "source_lock_sha256": sha256_file(lock_path),
        "network_requests_maximum": len(output_rows),
        "download_bytes": download_bytes,
        "rows": output_rows,
    }
    manifest_path = root / str(config["source"]["acquisition_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json(report))
    return report


def _exact_binary_permutation(
    features: np.ndarray, groups: Sequence[str], labels: Sequence[str], observed: float
) -> dict[str, Any]:
    unique_groups = sorted(set(groups))
    if len(unique_groups) != 6 or Counter(labels)["ektar"] == 0:
        raise InternetArchiveStockError("unexpected exact-permutation shape")
    results: list[float] = []
    for positives in itertools.combinations(unique_groups, 2):
        positive_set = set(positives)
        permuted = ["ektar" if group in positive_set else "control" for group in groups]
        results.append(group_loo_centroid(features, groups, permuted)["balanced_accuracy"])
    values = np.asarray(results, dtype=np.float64)
    return {
        "assignments": len(results),
        "p_value_greater_equal": float(np.mean(values >= observed)),
        "null_mean": float(values.mean()),
        "null_max": float(values.max()),
    }


def evaluate(config: Mapping[str, Any], root: Path, *, reverse: bool = False) -> dict[str, Any]:
    lock_path = root / str(config["source"]["source_lock"])
    manifest_path = root / str(config["source"]["acquisition_manifest"])
    if sha256_file(lock_path) != str(config["source"]["source_lock_sha256"]):
        raise InternetArchiveStockError("source lock hash drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_lock_sha256"] != str(config["source"]["source_lock_sha256"]):
        raise InternetArchiveStockError("manifest parent drift")
    rows = list(manifest["rows"])
    if reverse:
        rows.reverse()
    rows.sort(key=lambda row: (row["identifier"], row["original_name"]))
    for row in rows:
        path = root / str(row["local_path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
            raise InternetArchiveStockError(f"local pixel drift: {row['local_path']}")
        row["path"] = path
    groups = [str(row["group_id"]) for row in rows]
    labels = [str(row["binary_label"]) for row in rows]
    descriptors = [extract_descriptors(row) for row in rows]
    ordinal = {
        str(row["identifier"]): float((date.fromisoformat(str(row["capture_date"])) - date(2018, 1, 1)).days)
        for row in rows
    }
    results: dict[str, Any] = {}
    matrices: dict[str, np.ndarray] = {}
    for name in config["descriptors"]:
        if name == "capture_date_ordinal":
            matrix = np.asarray([[ordinal[str(row["identifier"])]] for row in rows], dtype=np.float64)
        else:
            matrix = np.stack([descriptor[str(name)] for descriptor in descriptors])
        matrices[str(name)] = matrix
        results[str(name)] = group_loo_centroid(matrix, groups, labels)
    primary_name = str(config["gates"]["primary_descriptor"])
    primary = results[primary_name]
    primary["exact_permutation"] = _exact_binary_permutation(
        matrices[primary_name], groups, labels, float(primary["balanced_accuracy"])
    )
    nuisance_names = [str(name) for name in config["gates"]["nuisance_control_descriptors"]]
    best_nuisance_name = max(nuisance_names, key=lambda name: (results[name]["balanced_accuracy"], name))
    delta = paired_group_bootstrap_delta(
        primary,
        results[best_nuisance_name],
        iterations=int(config["statistics"]["bootstrap_iterations"]),
        seed=int(config["statistics"]["bootstrap_seed"]),
    )
    counts = Counter(groups)
    gates = config["gates"]
    checks = {
        "source_lock_exact": sha256_file(lock_path) == str(config["source"]["source_lock_sha256"]),
        "item_groups_exact": len(counts) == int(gates["minimum_item_groups"]),
        "frames_per_item_exact": all(count == int(gates["minimum_frames_per_item"]) for count in counts.values()),
        "primary_balanced_accuracy": primary["balanced_accuracy"] >= float(gates["minimum_primary_balanced_accuracy"]),
        "ektar_recall": primary["per_class_group_recall"]["ektar"] >= float(gates["minimum_ektar_recall"]),
        "control_recall": primary["per_class_group_recall"]["control"] >= float(gates["minimum_control_recall"]),
        "exact_permutation": primary["exact_permutation"]["p_value_greater_equal"] <= float(gates["maximum_exact_permutation_p"]),
        "delta_vs_nuisance": delta["observed_delta"] >= float(gates["minimum_delta_vs_nuisance"]),
        "date_control_ceiling": results["capture_date_ordinal"]["balanced_accuracy"] <= float(gates["maximum_date_control_accuracy"]),
        "hog_control_ceiling": results["hog_grayscale"]["balanced_accuracy"] <= float(gates["maximum_hog_control_accuracy"]),
        "low_frequency_control_ceiling": results["low_frequency_rgb_4x4"]["balanced_accuracy"] <= float(gates["maximum_low_frequency_control_accuracy"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "source_lock_sha256": sha256_file(lock_path),
        "acquisition_manifest_sha256": sha256_file(manifest_path),
        "rows": len(rows),
        "item_groups": len(counts),
        "stock_counts": dict(sorted(Counter(str(row["film_stock_id"]) for row in rows).items())),
        "item_counts": dict(sorted(counts.items())),
        "descriptor_results": results,
        "best_nuisance_control": best_nuisance_name,
        "primary_minus_best_nuisance": delta,
        "checks": checks,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
