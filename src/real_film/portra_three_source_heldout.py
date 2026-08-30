"""Prospective source-held-out Portra appearance identifiability diagnostic."""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.film_physics.create_only_file import publish_create_only
from src.real_film.connected_stock_identifiability import (
    extract_descriptors,
    load_connected_rows,
)
from src.real_film.fsa_owi_pilot import dhash64, hamming64


class PortraThreeSourceError(ValueError):
    """Raised when the frozen source, data or evaluation contract drifts."""


FetchResult = tuple[int, Mapping[str, str], bytes]
Fetcher = Callable[[str], FetchResult]

_LUMINANT_ALT_PATTERN = re.compile(
    r'alt="\#(?P<id>\d{5})\s+-\s+(?P<date>[^"]+?)\s+-\s+'
    r'(?P<film>[^"]+?)\s+-\s+(?P<place>[^"]*)"'
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _json_sha256(value: object) -> str:
    return _sha256_bytes(_json_bytes(value))


def _upstream_inventory_sha256(value: object) -> str:
    """Reproduce the non-sorted compact JSON binding used by SF3.A3S."""
    payload = json.dumps(
        value,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256_bytes(payload)


def _default_fetcher(url: str) -> FetchResult:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "K-MCFM-SF3-A3U/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.status), dict(response.headers.items()), response.read()


def _header(headers: Mapping[str, str], name: str) -> str:
    lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
    return lowered.get(name.casefold(), "")


def _verify_local_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = root / str(binding["path"])
    if not path.is_file() or _sha256_file(path) != str(binding["sha256"]):
        raise PortraThreeSourceError(f"upstream binding drift: {path}")


def verify_upstream_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    """Verify local evidence and the exact producer evidence Git object."""
    bindings = config["upstream_bindings"]
    for key in (
        "commons_metadata_evidence",
        "commons_pixel_evidence",
        "commons_identifiability_evidence",
        "luminant_source_evidence",
    ):
        _verify_local_binding(root, bindings[key])

    producer = bindings["r1hg_source_evidence"]
    producer_root = root.parent / "追色"
    if not (producer_root / ".git").exists():
        raise PortraThreeSourceError("producer repository unavailable")
    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(producer_root),
            "rev-parse",
            f"{producer['evidence_commit']}^{{commit}}",
        ],
        text=True,
    ).strip()
    payload = subprocess.check_output(
        [
            "git",
            "-C",
            str(producer_root),
            "show",
            f"{commit}:{producer['evidence_path']}",
        ]
    )
    if _sha256_bytes(payload) != str(producer["evidence_sha256"]):
        raise PortraThreeSourceError("R1HG evidence Git object drift")
    parsed = json.loads(payload)
    if (
        parsed["formal_report"]["forward_sha256"] != producer["formal_report_sha256"]
        or parsed["source_facts"]["work_manifest_canonical_sha256"]
        != producer["work_manifest_canonical_sha256"]
        or parsed["status"] != "PASS_PRIVATE_R1HG_NICKNICK_PORTRA_SOURCE_ADMISSION"
    ):
        raise PortraThreeSourceError("R1HG scientific handoff drift")
    return {
        "producer_evidence_commit": commit,
        "producer_evidence_sha256": _sha256_bytes(payload),
    }


def primary_selection(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact preregistered nicknick confirmation rows."""
    source = config["primary_confirmation"]
    template = str(source["work_url_template"])
    rows = [
        {
            "index": int(index),
            "label": "portra",
            "source_label": source["portra"]["required_label"],
            "url": template.format(index=int(index)),
        }
        for index in source["portra"]["indices"]
    ]
    wrong_stock_indices = [int(index) for index in source["wrong_stock"]["indices"]]
    candidate_indices = range(11, 39)
    expected_wrong_stock = [
        index
        for _, index in sorted(
            (
                _sha256_bytes(template.format(index=index).encode("ascii")),
                index,
            )
            for index in candidate_indices
        )[:12]
    ]
    if wrong_stock_indices != expected_wrong_stock:
        raise PortraThreeSourceError("wrong-stock SHA-ranked selection drift")
    rows.extend(
        {
            "index": int(index),
            "label": "wrong_stock",
            "source_label": source["wrong_stock"]["required_label"],
            "url": template.format(index=int(index)),
        }
        for index in wrong_stock_indices
    )
    if len(rows) != 24 or len({row["index"] for row in rows}) != 24:
        raise PortraThreeSourceError("primary role cardinality drift")
    excluded = {int(index) for index in source["excluded"]["indices"]}
    if excluded & {row["index"] for row in rows}:
        raise PortraThreeSourceError("excluded Portra generation entered primary roles")
    return sorted(rows, key=lambda row: row["index"])


def _decode_identity(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            if image.format != "JPEG" or getattr(image, "n_frames", 1) != 1:
                raise PortraThreeSourceError(f"unsupported media container: {path}")
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            width, height = oriented.size
            if width < 256 or height < 256:
                raise PortraThreeSourceError(f"undersized media: {path}")
            difference_hash = dhash64(oriented)
    except (OSError, ValueError) as error:
        raise PortraThreeSourceError(f"media decode failed: {path}") from error
    return int(width), int(height), difference_hash


def _materialize_rows(
    rows: Sequence[Mapping[str, Any]],
    destination_root: Path,
    *,
    filename_prefix: str,
    max_single_bytes: int,
    max_total_bytes: int,
    reverse: bool,
    fetcher: Fetcher,
) -> list[dict[str, Any]]:
    destination_root.mkdir(parents=True, exist_ok=True)
    requested = list(reversed(rows)) if reverse else list(rows)
    downloaded: dict[int | str, dict[str, Any]] = {}
    total = 0
    with tempfile.TemporaryDirectory(
        prefix=".sf3-a3u-stage-", dir=destination_root
    ) as stage_dir:
        stage_root = Path(stage_dir)
        for row in requested:
            status, headers, payload = fetcher(str(row["url"]))
            if status != 200:
                raise PortraThreeSourceError(f"image request failed: {row['url']}")
            content_type = _header(headers, "Content-Type").split(";", 1)[0].strip()
            if content_type != "image/jpeg":
                raise PortraThreeSourceError(f"unexpected content type: {row['url']}")
            if not payload or len(payload) > max_single_bytes:
                raise PortraThreeSourceError(f"image byte bound failed: {row['url']}")
            declared = _header(headers, "Content-Length")
            if declared and int(declared) != len(payload):
                raise PortraThreeSourceError(f"content length drift: {row['url']}")
            total += len(payload)
            if total > max_total_bytes:
                raise PortraThreeSourceError("source transfer total byte bound failed")
            identity = row.get("index", row.get("id"))
            name = f"{filename_prefix}_{identity}.jpg"
            stage_path = stage_root / name
            stage_path.write_bytes(payload)
            width, height, difference_hash = _decode_identity(stage_path)
            digest = _sha256_bytes(payload)
            final_path = destination_root / name
            if final_path.exists():
                if (
                    final_path.stat().st_size != len(payload)
                    or _sha256_file(final_path) != digest
                ):
                    raise PortraThreeSourceError(f"existing media drift: {final_path}")
                stage_path.unlink()
            else:
                publish_create_only(stage_path, final_path)
            downloaded[identity] = {
                **dict(row),
                "local_path": name,
                "bytes": len(payload),
                "sha256": digest,
                "width": width,
                "height": height,
                "dhash64": difference_hash,
                "content_type": content_type,
            }
    return [downloaded[row.get("index", row.get("id"))] for row in rows]


def _write_or_verify_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = _json_bytes(
        {"schema": "neuro-film.sf3-a3u-media-manifest.v1", "rows": list(rows)}
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise PortraThreeSourceError(f"media manifest drift: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".sf3-a3u-manifest-", dir=path.parent
        ) as stage_dir:
            stage = Path(stage_dir) / path.name
            stage.write_bytes(payload)
            publish_create_only(stage, path)
    return _sha256_bytes(payload)


def acquire_primary(
    root: Path,
    config: Mapping[str, Any],
    *,
    reverse: bool,
    fetcher: Fetcher,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-fetch and materialize the exact 24-row primary confirmation."""
    source = config["primary_confirmation"]
    status, _, page = fetcher(str(source["page_url"]))
    if status != 200 or _sha256_bytes(page) != str(source["page_sha256"]):
        raise PortraThreeSourceError("nicknick source page drift")
    rows = primary_selection(config)
    destination = root / str(source["pixel_root"])
    limits = config["operation_limits"]
    manifest_rows = _materialize_rows(
        rows,
        destination,
        filename_prefix="nicknick",
        max_single_bytes=int(limits["nicknick_max_single_image_bytes"]),
        max_total_bytes=int(limits["nicknick_max_total_image_bytes"]),
        reverse=reverse,
        fetcher=fetcher,
    )
    manifest_path = destination / "manifest.json"
    manifest_sha = _write_or_verify_manifest(manifest_path, manifest_rows)
    return manifest_rows, {
        "page_sha256": _sha256_bytes(page),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha,
        "files": len(manifest_rows),
        "bytes": sum(int(row["bytes"]) for row in manifest_rows),
    }


def _luminant_inventory(
    config: Mapping[str, Any], *, reverse: bool, fetcher: Fetcher
) -> list[dict[str, str]]:
    source = config["conditional_positive_confirmation"]
    first, last = (int(value) for value in source["page_range"])
    page_numbers = list(range(first, last + 1))
    request_pages = list(reversed(page_numbers)) if reverse else page_numbers
    payloads: dict[int, bytes] = {}
    for page in request_pages:
        status, _, payload = fetcher(str(source["page_url_template"]).format(page=page))
        if status != 200:
            raise PortraThreeSourceError(f"Luminant page request failed: {page}")
        payloads[page] = payload
    items: dict[str, dict[str, str]] = {}
    for page in page_numbers:
        text = payloads[page].decode("utf-8")
        for match in _LUMINANT_ALT_PATTERN.finditer(text):
            image_id = match.group("id")
            item = {
                "id": image_id,
                "date": html.unescape(match.group("date")).strip(),
                "film": html.unescape(match.group("film")).strip(),
                "place": html.unescape(match.group("place")).strip(),
                "url": f"{source['base_url']}/img/{image_id}/{image_id}_full.jpg",
            }
            if image_id in items and items[image_id] != item:
                raise PortraThreeSourceError(f"Luminant duplicate drift: {image_id}")
            items[image_id] = item
    inventory = [items[key] for key in sorted(items)]
    accepted = [
        {
            "id": row["id"],
            "date": row["date"],
            "film": row["film"],
            "place": row["place"],
            "full_url": row["url"],
        }
        for row in inventory
        if row["film"] in {source["required_label"], *source["excluded_labels"]}
    ]
    accepted.sort(key=lambda row: row["id"])
    if len(accepted) != int(
        source["required_inventory_count"]
    ) or _upstream_inventory_sha256(accepted) != str(
        source["required_inventory_sha256"]
    ):
        raise PortraThreeSourceError("Luminant inventory drift")
    return inventory


def acquire_conditional_luminant(
    root: Path,
    config: Mapping[str, Any],
    *,
    reverse: bool,
    fetcher: Fetcher,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = config["conditional_positive_confirmation"]
    inventory = _luminant_inventory(config, reverse=reverse, fetcher=fetcher)
    eligible = [row for row in inventory if row["film"] == source["required_label"]]
    ranked = sorted(
        eligible,
        key=lambda row: (_sha256_bytes(row["id"].encode("ascii")), row["id"]),
    )
    selected = [
        {
            "id": row["id"],
            "label": "portra",
            "source_label": row["film"],
            "url": row["url"],
        }
        for row in ranked[: int(source["count"])]
    ]
    destination = root / str(source["pixel_root"])
    limits = config["operation_limits"]
    manifest_rows = _materialize_rows(
        selected,
        destination,
        filename_prefix="luminant",
        max_single_bytes=int(limits["luminant_max_single_image_bytes"]),
        max_total_bytes=int(limits["luminant_max_total_image_bytes"]),
        reverse=reverse,
        fetcher=fetcher,
    )
    manifest_path = destination / "manifest.json"
    manifest_sha = _write_or_verify_manifest(manifest_path, manifest_rows)
    return manifest_rows, {
        "inventory_rows": len(inventory),
        "selected_ids": [row["id"] for row in manifest_rows],
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha,
        "files": len(manifest_rows),
        "bytes": sum(int(row["bytes"]) for row in manifest_rows),
    }


def _development_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    binding = config["upstream_bindings"]["commons_pixel_evidence"]
    development = config["development"]
    cells = []
    for role in ("portra", "wrong_stock"):
        row = development[role]
        cells.append(
            {
                "cell_id": f"commons_{role}",
                "source_id": development["source_id"],
                "film_stock_id": row["film_stock_id"],
                "manifest": binding["download_manifest"],
                "manifest_sha256": binding["download_manifest_sha256"],
                "pixel_root": development["pixel_root"],
                "expected_files": row["expected_files"],
                "included_page_ids": row["included_page_ids"],
            }
        )
    rows = load_connected_rows(root, {"cells": cells})
    labels = {
        development["portra"]["film_stock_id"]: "portra",
        development["wrong_stock"]["film_stock_id"]: "wrong_stock",
    }
    for row in rows:
        row["label"] = labels[row["film_stock_id"]]
    return rows


def _new_rows(
    root: Path, source_root: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    base = root / source_root
    return [
        {
            "path": base / str(row["local_path"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "bytes": int(row["bytes"]),
            "label": str(row["label"]),
            "source_record": dict(row),
        }
        for row in rows
    ]


def fit_centroid_predict(
    development: np.ndarray,
    development_labels: Sequence[str],
    confirmation: np.ndarray,
    confirmation_labels: Sequence[str],
) -> dict[str, Any]:
    """Fit development-only standardized centroids and score confirmation rows."""
    train = np.asarray(development, dtype=np.float64)
    test = np.asarray(confirmation, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise PortraThreeSourceError("descriptor matrix shape drift")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise PortraThreeSourceError("non-finite descriptor")
    labels = sorted({str(label) for label in development_labels})
    if labels != ["portra", "wrong_stock"]:
        raise PortraThreeSourceError("development label set drift")
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-8)
    standardized_train = (train - mean) / scale
    standardized_test = (test - mean) / scale
    centroids = {
        label: standardized_train[
            [str(value) == label for value in development_labels]
        ].mean(axis=0)
        for label in labels
    }
    predictions: list[dict[str, Any]] = []
    for index, (vector, truth) in enumerate(
        zip(standardized_test, confirmation_labels, strict=True)
    ):
        distances = {
            label: float(np.linalg.norm(vector - centroid))
            for label, centroid in centroids.items()
        }
        predicted = min(distances, key=lambda label: (distances[label], label))
        predictions.append(
            {
                "index": index,
                "true_label": str(truth),
                "predicted_label": predicted,
                "correct": predicted == str(truth),
                "distances": dict(sorted(distances.items())),
            }
        )
    recalls = {
        label: float(
            np.mean(
                [row["correct"] for row in predictions if row["true_label"] == label]
            )
        )
        for label in sorted({str(label) for label in confirmation_labels})
    }
    return {
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_class_recall": recalls,
        "predictions": predictions,
    }


def _permutation_p(
    development: np.ndarray,
    development_labels: Sequence[str],
    confirmation: np.ndarray,
    confirmation_labels: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    observed = fit_centroid_predict(
        development, development_labels, confirmation, confirmation_labels
    )["balanced_accuracy"]
    labels = np.asarray(development_labels, dtype=object)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(permutations):
        shuffled = labels[rng.permutation(len(labels))]
        values.append(
            fit_centroid_predict(
                development, shuffled, confirmation, confirmation_labels
            )["balanced_accuracy"]
        )
    null = np.asarray(values, dtype=np.float64)
    return {
        "permutations": permutations,
        "p_value_greater_equal": float(
            (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
        ),
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
    }


def _bootstrap_delta(
    primary: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if len(primary["predictions"]) != len(control["predictions"]):
        raise PortraThreeSourceError("paired prediction length drift")
    labels = sorted({row["true_label"] for row in primary["predictions"]})
    indices = {
        label: [
            index
            for index, row in enumerate(primary["predictions"])
            if row["true_label"] == label
        ]
        for label in labels
    }
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        class_deltas = []
        for label in labels:
            sampled = rng.choice(indices[label], size=len(indices[label]), replace=True)
            class_deltas.append(
                float(
                    np.mean(
                        [
                            int(primary["predictions"][index]["correct"])
                            - int(control["predictions"][index]["correct"])
                            for index in sampled
                        ]
                    )
                )
            )
        deltas.append(float(np.mean(class_deltas)))
    values = np.asarray(deltas, dtype=np.float64)
    observed = float(primary["balanced_accuracy"] - control["balanced_accuracy"])
    return {
        "observed_delta": observed,
        "iterations": iterations,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def _duplicate_audit(
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dev = []
    for row in development:
        with Image.open(Path(row["path"])) as image:
            dev.append(
                (
                    str(row["path"]),
                    _sha256_file(Path(row["path"])),
                    dhash64(ImageOps.exif_transpose(image).convert("RGB")),
                )
            )
    confirm = [
        (
            str(row["path"]),
            str(row["source_record"]["sha256"]),
            str(row["source_record"]["dhash64"]),
        )
        for row in confirmation
    ]
    exact = []
    near = []
    for left_path, left_sha, left_hash in dev:
        for right_path, right_sha, right_hash in confirm:
            if left_sha == right_sha:
                exact.append([left_path, right_path])
            distance = hamming64(left_hash, right_hash)
            if distance <= 4:
                near.append([left_path, right_path, distance])
    return {
        "cross_split_exact_pairs": exact,
        "cross_split_dhash_le_4_pairs": near,
        "passed": not exact and not near,
    }


def evaluate_primary(
    development_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    descriptors = list(config["descriptors"])
    development_features = [extract_descriptors(row) for row in development_rows]
    confirmation_features = [extract_descriptors(row) for row in confirmation_rows]
    development_labels = [str(row["label"]) for row in development_rows]
    confirmation_labels = [str(row["label"]) for row in confirmation_rows]
    results = {}
    for descriptor in descriptors:
        train = np.stack([row[descriptor] for row in development_features])
        test = np.stack([row[descriptor] for row in confirmation_features])
        results[descriptor] = fit_centroid_predict(
            train, development_labels, test, confirmation_labels
        )
    primary_name = str(config["statistics"]["primary_descriptor"])
    control_names = [name for name in descriptors if name != primary_name]
    best_control = max(
        control_names,
        key=lambda name: (results[name]["balanced_accuracy"], name),
    )
    primary_train = np.stack([row[primary_name] for row in development_features])
    primary_test = np.stack([row[primary_name] for row in confirmation_features])
    permutation = _permutation_p(
        primary_train,
        development_labels,
        primary_test,
        confirmation_labels,
        permutations=int(config["statistics"]["permutations"]),
        seed=int(config["statistics"]["permutation_seed"]),
    )
    delta = _bootstrap_delta(
        results[primary_name],
        results[best_control],
        iterations=int(config["statistics"]["bootstrap_iterations"]),
        seed=int(config["statistics"]["bootstrap_seed"]),
    )
    duplicate = _duplicate_audit(development_rows, confirmation_rows)
    gates = config["primary_gates"]
    primary = results[primary_name]
    checks = {
        "development_rows_exact": len(development_rows)
        == int(gates["expected_development_rows"]),
        "confirmation_rows_exact": len(confirmation_rows)
        == int(gates["expected_confirmation_rows"]),
        "primary_balanced_accuracy": primary["balanced_accuracy"]
        >= float(gates["minimum_primary_balanced_accuracy"]),
        "portra_recall": primary["per_class_recall"]["portra"]
        >= float(gates["minimum_portra_recall"]),
        "wrong_stock_specificity": primary["per_class_recall"]["wrong_stock"]
        >= float(gates["minimum_wrong_stock_specificity"]),
        "primary_permutation": permutation["p_value_greater_equal"]
        <= float(gates["maximum_primary_permutation_p"]),
        "delta_vs_best_nuisance": delta["observed_delta"]
        >= float(gates["minimum_delta_vs_best_nuisance"]),
        "hog_content_ceiling": results["hog_grayscale"]["balanced_accuracy"]
        <= float(gates["maximum_hog_control_accuracy"]),
        "reversed_centroid_control": 1.0 - primary["balanced_accuracy"]
        <= float(gates["maximum_reversed_centroid_accuracy"]),
        "cross_split_duplicate_gate": duplicate["passed"],
    }
    return {
        "descriptor_results": results,
        "best_nuisance_control": best_control,
        "primary_permutation": permutation,
        "primary_minus_best_nuisance": delta,
        "reversed_centroid_accuracy": 1.0 - primary["balanced_accuracy"],
        "duplicate_audit": duplicate,
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_positive_confirmation(
    development_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    development_features = [extract_descriptors(row) for row in development_rows]
    confirmation_features = [extract_descriptors(row) for row in confirmation_rows]
    labels = [str(row["label"]) for row in development_rows]
    truth = ["portra"] * len(confirmation_rows)
    recalls = {}
    for descriptor in config["descriptors"]:
        result = fit_centroid_predict(
            np.stack([row[descriptor] for row in development_features]),
            labels,
            np.stack([row[descriptor] for row in confirmation_features]),
            truth,
        )
        recalls[str(descriptor)] = result["per_class_recall"]["portra"]
    primary = str(config["statistics"]["primary_descriptor"])
    best_control = max(
        (name for name in recalls if name != primary),
        key=lambda name: (recalls[name], name),
    )
    gates = config["conditional_positive_gates"]
    checks = {
        "rows_exact": len(confirmation_rows) == int(gates["expected_rows"]),
        "primary_recall": recalls[primary] >= float(gates["minimum_primary_recall"]),
        "delta_vs_best_nuisance": recalls[primary] - recalls[best_control]
        >= float(gates["minimum_delta_vs_best_nuisance_recall"]),
    }
    return {
        "descriptor_recalls": recalls,
        "best_nuisance_control": best_control,
        "primary_minus_best_nuisance_recall": recalls[primary] - recalls[best_control],
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_portra_three_source_audit(
    root: Path,
    config_path: Path,
    *,
    reverse: bool = False,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Run the committed SF3.A3U staged source-held-out diagnostic."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fetch = fetcher or _default_fetcher
    upstream = verify_upstream_bindings(root, config)
    primary_manifest, primary_source = acquire_primary(
        root, config, reverse=reverse, fetcher=fetch
    )
    development = _development_rows(root, config)
    primary_rows = _new_rows(
        root, config["primary_confirmation"]["pixel_root"], primary_manifest
    )
    primary = evaluate_primary(development, primary_rows, config)

    luminant_source = None
    luminant_result = None
    if primary["passed"]:
        luminant_manifest, luminant_source = acquire_conditional_luminant(
            root, config, reverse=reverse, fetcher=fetch
        )
        luminant_rows = _new_rows(
            root,
            config["conditional_positive_confirmation"]["pixel_root"],
            luminant_manifest,
        )
        luminant_result = evaluate_positive_confirmation(
            development, luminant_rows, config
        )

    if not primary["passed"]:
        decision = config["decision_if_primary_fails"]
    elif not luminant_result or not luminant_result["passed"]:
        decision = config["decision_if_primary_passes_and_luminant_fails"]
    else:
        decision = config["decision_if_all_pass"]
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3u-portra-three-source-heldout-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "upstream": upstream,
        "bindings": {
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": _sha256_file(config_path),
            "implementation_path": Path(__file__).relative_to(root).as_posix(),
            "implementation_sha256": _sha256_file(Path(__file__)),
        },
        "primary_source": primary_source,
        "primary_result": primary,
        "luminant_source": luminant_source,
        "luminant_result": luminant_result,
        "operation_counts": {
            "nicknick_page_get_requests": 1,
            "nicknick_image_body_requests": len(primary_manifest),
            "nicknick_pixel_decodes": len(primary_manifest),
            "excluded_nicknick_image_body_requests": 0,
            "luminant_gallery_page_get_requests": 74 if primary["passed"] else 0,
            "luminant_image_body_requests": (
                int(luminant_source["files"]) if luminant_source else 0
            ),
            "operator_fit_calls": 0,
            "render_calls": 0,
            "product_writes": 0,
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _json_sha256(report)
    return report


__all__ = [
    "PortraThreeSourceError",
    "acquire_conditional_luminant",
    "acquire_primary",
    "evaluate_positive_confirmation",
    "evaluate_primary",
    "fit_centroid_predict",
    "primary_selection",
    "run_portra_three_source_audit",
    "verify_upstream_bindings",
]
