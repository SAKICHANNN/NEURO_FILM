"""Page-bag scanner-nuisance audit for the Color Precision chart archive.

The PDFs do not identify which embedded image corresponds to which advertised
exposure.  This module therefore treats each page as an unordered bag and
never assigns physical EV labels to extracted pixels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.eval.color_precision_chart_source import canonical_sha256, sha256_file


SCHEMA = "neuro-film.u5-r2bd1-color-precision-page-identifiability-contract.v1"
REPORT_SCHEMA = (
    "neuro-film.u5-r2bd1-color-precision-page-identifiability-report.v1"
)


class ColorPrecisionPageIdentifiabilityError(RuntimeError):
    """Raised when the frozen page-bag audit cannot be executed faithfully."""


def validate_contract(config: Mapping[str, Any], *, root: Path) -> None:
    if config.get("schema") != SCHEMA:
        raise ColorPrecisionPageIdentifiabilityError("unsupported contract schema")
    if config.get("status") != "contract_frozen_before_full_pixel_extraction":
        raise ColorPrecisionPageIdentifiabilityError("contract is not frozen")
    parent = config.get("parent")
    source = config.get("source")
    extractor = config.get("extractor")
    sample_unit = config.get("sample_unit")
    if not all(
        isinstance(value, Mapping)
        for value in (parent, source, extractor, sample_unit)
    ):
        raise ColorPrecisionPageIdentifiabilityError(
            "contract sections are missing"
        )
    parent_path = root / str(parent["decision"])
    if (
        not parent_path.is_file()
        or sha256_file(parent_path) != parent["decision_sha256"]
    ):
        raise ColorPrecisionPageIdentifiabilityError(
            "parent decision identity mismatch"
        )
    if (
        source.get("rights_status")
        != "unknown_all_rights_reserved_no_training_or_redistribution_grant"
        or source.get("allowed_use")
        != "internal_source_specific_scanner_nuisance_identifiability_audit_only"
    ):
        raise ColorPrecisionPageIdentifiabilityError("rights boundary drift")
    pdfs = source.get("pdfs")
    if not isinstance(pdfs, list) or len(pdfs) != 2:
        raise ColorPrecisionPageIdentifiabilityError("two exact PDFs required")
    if {row.get("scanner_id") for row in pdfs} != {"frontier", "noritsu"}:
        raise ColorPrecisionPageIdentifiabilityError("scanner identities drifted")
    for row in pdfs:
        path = Path(str(row["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ColorPrecisionPageIdentifiabilityError(
                f"PDF identity mismatch: {row['scanner_id']}"
            )
    tool = Path(str(extractor["executable"]))
    if not tool.is_file() or sha256_file(tool) != extractor["sha256"]:
        raise ColorPrecisionPageIdentifiabilityError(
            "extractor executable identity mismatch"
        )
    if extractor.get("embedded_profile_semantics") != (
        "not_preserved_or_proven_by_pdfimages_png_output"
    ):
        raise ColorPrecisionPageIdentifiabilityError(
            "embedded-profile claim drift"
        )
    if sample_unit.get("exposure_assignment_allowed") is not False:
        raise ColorPrecisionPageIdentifiabilityError(
            "exposure assignment must remain forbidden"
        )
    forbidden = set(config.get("forbidden_actions", ()))
    if not {
        "assigning_exposure_ev_to_images",
        "operator_fitting",
        "training",
        "latent_mode_clustering",
        "redistribution",
    } <= forbidden:
        raise ColorPrecisionPageIdentifiabilityError(
            "forbidden-action boundary drift"
        )


def parse_pdfimages_objects(
    text: str,
    *,
    scanner_id: str,
    minimum_width: int,
    minimum_height: int,
) -> list[dict[str, Any]]:
    """Return full-photo placements with extraction numbers and object IDs."""

    if scanner_id not in {"frontier", "noritsu"}:
        raise ColorPrecisionPageIdentifiabilityError("unsupported scanner")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 16 or not fields[0].isdigit():
            continue
        page = int(fields[0])
        number = int(fields[1])
        object_type = fields[2]
        width = int(fields[3])
        height = int(fields[4])
        object_id = int(fields[10])
        generation = int(fields[11])
        if (
            object_type != "image"
            or width < minimum_width
            or height < minimum_height
        ):
            continue
        rows.append(
            {
                "scanner_id": scanner_id,
                "page_index": page,
                "image_number": number,
                "object_id": object_id,
                "object_generation": generation,
                "width": width,
                "height": height,
            }
        )
    if not rows:
        raise ColorPrecisionPageIdentifiabilityError(
            "no full-photo objects found"
        )
    return rows


def bind_extracted_images(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: Path,
) -> list[dict[str, Any]]:
    """Bind object listings to Poppler outputs and collapse object duplicates."""

    selected: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in rows:
        path = prefix.parent / f"{prefix.name}-{int(row['image_number']):03d}.png"
        if not path.is_file():
            raise ColorPrecisionPageIdentifiabilityError(
                f"missing extracted image {path.name}"
            )
        key = (
            str(row["scanner_id"]),
            int(row["page_index"]),
            int(row["object_id"]),
            int(row["object_generation"]),
        )
        bound = {
            **dict(row),
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        previous = selected.get(key)
        if previous is not None:
            if previous["sha256"] != bound["sha256"]:
                raise ColorPrecisionPageIdentifiabilityError(
                    "one PDF object identity decoded to different pixels"
                )
            continue
        selected[key] = bound
    return sorted(
        selected.values(),
        key=lambda row: (
            str(row["scanner_id"]),
            int(row["page_index"]),
            int(row["image_number"]),
        ),
    )


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    )


def _block_means(channels: np.ndarray) -> np.ndarray:
    height, width, count = channels.shape
    if count != 3 or height % 4 or width % 4:
        raise ColorPrecisionPageIdentifiabilityError(
            "descriptor image must be RGB with dimensions divisible by four"
        )
    return (
        channels.reshape(4, height // 4, 4, width // 4, 3)
        .mean(axis=(1, 3))
        .reshape(-1)
    )


def image_descriptors(
    path: Path, *, resize: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen raw and basic-normalized handcrafted descriptors."""

    try:
        with Image.open(path) as image:
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            elif image.mode == "L":
                image = image.convert("RGB")
            resized = image.resize(resize, Image.Resampling.BOX)
            encoded = np.asarray(resized, dtype=np.float64) / 255.0
    except (OSError, ValueError) as exc:
        raise ColorPrecisionPageIdentifiabilityError(
            f"cannot decode extracted image {path.name}"
        ) from exc
    if encoded.shape != (resize[1], resize[0], 3):
        raise ColorPrecisionPageIdentifiabilityError("decoded image shape drift")
    linear = _srgb_to_linear(encoded)
    luma = (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )
    raw = np.concatenate(
        (
            linear.mean(axis=(0, 1)),
            linear.std(axis=(0, 1)),
            np.quantile(luma, (0.05, 0.25, 0.5, 0.75, 0.95)),
            _block_means(linear),
        )
    )

    epsilon = 1.0 / 65535.0
    log_luma = np.log(np.maximum(luma, epsilon))
    luma_center = float(np.median(log_luma))
    luma_scale = float(
        np.quantile(log_luma, 0.75) - np.quantile(log_luma, 0.25)
    )
    if not math.isfinite(luma_scale) or luma_scale < 1e-6:
        luma_scale = 1.0
    luma_normalized = np.clip((log_luma - luma_center) / luma_scale, -4.0, 4.0)

    log_rg = np.log(np.maximum(linear[..., 0], epsilon)) - np.log(
        np.maximum(linear[..., 1], epsilon)
    )
    log_bg = np.log(np.maximum(linear[..., 2], epsilon)) - np.log(
        np.maximum(linear[..., 1], epsilon)
    )
    log_rg -= np.median(log_rg)
    log_bg -= np.median(log_bg)
    chroma_scale = float(np.sqrt(np.mean(log_rg**2 + log_bg**2)))
    if not math.isfinite(chroma_scale) or chroma_scale < 1e-6:
        chroma_scale = 1.0
    normalized_channels = np.stack(
        (
            luma_normalized,
            np.clip(log_rg / chroma_scale, -4.0, 4.0),
            np.clip(log_bg / chroma_scale, -4.0, 4.0),
        ),
        axis=-1,
    )
    normalized = np.concatenate(
        (
            normalized_channels.mean(axis=(0, 1)),
            normalized_channels.std(axis=(0, 1)),
            np.quantile(
                luma_normalized, (0.05, 0.25, 0.5, 0.75, 0.95)
            ),
            _block_means(normalized_channels),
        )
    )
    if (
        raw.shape != (59,)
        or normalized.shape != (59,)
        or not np.isfinite(raw).all()
        or not np.isfinite(normalized).all()
    ):
        raise ColorPrecisionPageIdentifiabilityError(
            "descriptor is nonfinite or has unexpected shape"
        )
    return raw, normalized


def aggregate_page_bags(
    image_rows: Sequence[Mapping[str, Any]],
    page_labels: Sequence[Mapping[str, Any]],
    *,
    resize: tuple[int, int],
) -> list[dict[str, Any]]:
    """Aggregate unordered image descriptors into one row per PDF page."""

    label_by_key = {
        (str(row["scanner_id"]), int(row["page_index"])): row
        for row in page_labels
    }
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in image_rows:
        grouped[(str(row["scanner_id"]), int(row["page_index"]))].append(row)
    if set(grouped) != set(label_by_key):
        raise ColorPrecisionPageIdentifiabilityError(
            "extracted page identities do not match source labels"
        )
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        raw_rows: list[np.ndarray] = []
        normalized_rows: list[np.ndarray] = []
        image_bindings: list[dict[str, Any]] = []
        for row in grouped[key]:
            raw, normalized = image_descriptors(
                Path(str(row["path"])), resize=resize
            )
            raw_rows.append(raw)
            normalized_rows.append(normalized)
            image_bindings.append(
                {
                    "image_number": int(row["image_number"]),
                    "object_id": int(row["object_id"]),
                    "object_generation": int(row["object_generation"]),
                    "bytes": int(row["bytes"]),
                    "sha256": str(row["sha256"]),
                }
            )
        raw_matrix = np.stack(raw_rows)
        normalized_matrix = np.stack(normalized_rows)
        label = label_by_key[key]
        output.append(
            {
                "scanner_id": key[0],
                "page_index": key[1],
                "stock_id": str(label["stock_id"]),
                "process_variant": str(label["process_variant"]),
                "variant_id": (
                    f"{label['stock_id']}/{label['process_variant']}"
                ),
                "bag_size": len(image_bindings),
                "images": image_bindings,
                "raw_descriptor": np.quantile(
                    raw_matrix, (0.25, 0.5, 0.75), axis=0
                )
                .reshape(-1)
                .tolist(),
                "basic_normalized_descriptor": np.quantile(
                    normalized_matrix, (0.25, 0.5, 0.75), axis=0
                )
                .reshape(-1)
                .tolist(),
            }
        )
    return output


def _distance_matrix(
    frontier: np.ndarray, noritsu: np.ndarray
) -> np.ndarray:
    combined = np.concatenate((frontier, noritsu), axis=0)
    center = combined.mean(axis=0)
    scale = combined.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    left = (frontier - center) / scale
    right = (noritsu - center) / scale
    return np.sqrt(np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2))


def _evaluate_descriptor(
    rows: Sequence[Mapping[str, Any]],
    *,
    descriptor_key: str,
    seed: int,
    permutations: int,
) -> dict[str, Any]:
    by_scanner = {
        scanner: sorted(
            (row for row in rows if row["scanner_id"] == scanner),
            key=lambda row: str(row["variant_id"]),
        )
        for scanner in ("frontier", "noritsu")
    }
    frontier = by_scanner["frontier"]
    noritsu = by_scanner["noritsu"]
    frontier_ids = [str(row["variant_id"]) for row in frontier]
    noritsu_ids = [str(row["variant_id"]) for row in noritsu]
    if (
        len(frontier) != 12
        or frontier_ids != noritsu_ids
        or len(set(frontier_ids)) != 12
    ):
        raise ColorPrecisionPageIdentifiabilityError(
            "expected 12 exactly paired stock/process variants"
        )
    frontier_matrix = np.asarray(
        [row[descriptor_key] for row in frontier], dtype=np.float64
    )
    noritsu_matrix = np.asarray(
        [row[descriptor_key] for row in noritsu], dtype=np.float64
    )
    distances = _distance_matrix(frontier_matrix, noritsu_matrix)
    count = len(frontier_ids)
    correct = np.arange(count)
    forward_order = np.argsort(distances, axis=1, kind="stable")
    reverse_order = np.argsort(distances, axis=0, kind="stable")
    forward_ranks = [
        int(np.flatnonzero(forward_order[index] == index)[0]) + 1
        for index in range(count)
    ]
    reverse_ranks = [
        int(np.flatnonzero(reverse_order[:, index] == index)[0]) + 1
        for index in range(count)
    ]
    same = distances[correct, correct]
    wrong_rows = distances[~np.eye(count, dtype=bool)]
    within_wrong_median = float(np.median(wrong_rows))
    ratio = float(np.median(same) / within_wrong_median)
    observed = float(np.mean(same))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        null[index] = float(
            np.mean(distances[correct, rng.permutation(count)])
        )
    p_value = float((1 + np.count_nonzero(null <= observed)) / (permutations + 1))
    return {
        "frontier_to_noritsu_top1_accuracy": float(
            np.mean(forward_order[:, 0] == correct)
        ),
        "noritsu_to_frontier_top1_accuracy": float(
            np.mean(reverse_order[0, :] == correct)
        ),
        "symmetric_cross_scanner_top1_accuracy": float(
            (
                np.count_nonzero(forward_order[:, 0] == correct)
                + np.count_nonzero(reverse_order[0, :] == correct)
            )
            / (2 * count)
        ),
        "median_correct_cross_scanner_rank": float(
            np.median(forward_ranks + reverse_ranks)
        ),
        "same_variant_scanner_distance_median": float(np.median(same)),
        "wrong_variant_cross_scanner_distance_median": within_wrong_median,
        "same_over_wrong_distance_ratio": ratio,
        "paired_same_distance_mean": observed,
        "permutation_p": p_value,
        "permutation_draws": permutations,
        "frontier_to_noritsu_ranks": forward_ranks,
        "noritsu_to_frontier_ranks": reverse_ranks,
        "frontier_to_noritsu_top1_variant_ids": [
            noritsu_ids[int(index)] for index in forward_order[:, 0]
        ],
        "noritsu_to_frontier_top1_variant_ids": [
            frontier_ids[int(index)] for index in reverse_order[0, :]
        ],
    }


def evaluate_page_identifiability(
    page_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    permutation = evaluation["permutation"]
    raw = _evaluate_descriptor(
        page_rows,
        descriptor_key="raw_descriptor",
        seed=int(permutation["seed"]),
        permutations=int(permutation["draws"]),
    )
    normalized = _evaluate_descriptor(
        page_rows,
        descriptor_key="basic_normalized_descriptor",
        seed=int(permutation["seed"]),
        permutations=int(permutation["draws"]),
    )
    gates = evaluation["frozen_gates"]

    def passed(prefix: str, metrics: Mapping[str, Any]) -> bool:
        return bool(
            metrics["symmetric_cross_scanner_top1_accuracy"]
            >= float(gates[f"{prefix}_symmetric_top1_accuracy_minimum"])
            and metrics["median_correct_cross_scanner_rank"]
            <= float(gates[f"{prefix}_median_correct_rank_maximum"])
            and metrics["same_over_wrong_distance_ratio"]
            <= float(gates[f"{prefix}_same_over_wrong_distance_ratio_maximum"])
            and metrics["permutation_p"]
            <= float(gates[f"{prefix}_permutation_p_maximum"])
        )

    raw_pass = passed("raw", raw)
    normalized_pass = passed("basic_normalized", normalized)
    if raw_pass and normalized_pass:
        branch = "raw_and_normalized_pass"
    elif raw_pass:
        branch = "raw_pass_normalized_fail"
    elif normalized_pass:
        branch = "normalized_pass_raw_fail"
    else:
        branch = "both_fail"
    stable = {
        "source_rows": list(page_rows),
        "metrics": {"raw": raw, "basic_normalized": normalized},
        "gates": {
            "raw_passed": raw_pass,
            "basic_normalized_passed": normalized_pass,
            "all_passed": raw_pass and normalized_pass,
        },
        "branch": branch,
    }
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        **stable,
        "stable_evidence_id": canonical_sha256(stable),
        "decision": evaluation["branching"][branch],
        "exposure_assignment_performed": False,
        "learned_features_used": False,
        "operator_fitting_performed": False,
        "training_performed": False,
        "latent_mode_clustering_performed": False,
        "decoded_pixel_label": config["extractor"]["decoded_pixel_label"],
        "claim_ceiling": config["claim_ceiling"],
    }


def extraction_inventory_id(rows: Sequence[Mapping[str, Any]]) -> str:
    stable = [
        {
            "scanner_id": row["scanner_id"],
            "page_index": row["page_index"],
            "image_number": row["image_number"],
            "object_id": row["object_id"],
            "object_generation": row["object_generation"],
            "width": row["width"],
            "height": row["height"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ColorPrecisionPageIdentifiabilityError",
    "aggregate_page_bags",
    "bind_extracted_images",
    "evaluate_page_identifiability",
    "extraction_inventory_id",
    "image_descriptors",
    "parse_pdfimages_objects",
    "validate_contract",
]
