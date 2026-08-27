"""Frozen same-input population separation audit for three fixed K=1 looks."""

from __future__ import annotations

import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


class ThreeStockPopulationSeparationError(ValueError):
    """Raised when an RF3.D14 input or frozen identity drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_exact(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> bytes:
    data = path.read_bytes()
    if _sha256(data) != expected_sha256:
        raise ThreeStockPopulationSeparationError(f"input hash mismatch: {path}")
    if expected_bytes is not None and len(data) != expected_bytes:
        raise ThreeStockPopulationSeparationError(f"input byte count mismatch: {path}")
    return data


def _read_rgb8(path: Path, expected_sha256: str) -> np.ndarray:
    data = _read_exact(path, expected_sha256)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB":
            raise ThreeStockPopulationSeparationError(f"expected RGB8: {path}")
        return np.asarray(image, dtype=np.uint8)


def delta_e76_summary(left_rgb8: np.ndarray, right_rgb8: np.ndarray) -> dict[str, float]:
    """Return deterministic CIELAB DeltaE76 statistics in bounded row chunks."""
    if left_rgb8.shape != right_rgb8.shape or left_rgb8.ndim != 3:
        raise ThreeStockPopulationSeparationError("pair geometry mismatch")
    values: list[np.ndarray] = []
    for y0 in range(0, left_rgb8.shape[0], 128):
        left = left_rgb8[y0 : y0 + 128].astype(np.float32) / np.float32(255.0)
        right = right_rgb8[y0 : y0 + 128].astype(np.float32) / np.float32(255.0)
        left_lab = rgb2lab(left).astype(np.float32)
        right_lab = rgb2lab(right).astype(np.float32)
        values.append(
            np.sqrt(np.sum(np.square(left_lab - right_lab), axis=2)).reshape(-1)
        )
    delta = np.concatenate(values)
    return {
        "median_delta_e76": float(np.median(delta)),
        "p95_delta_e76": float(np.quantile(delta, 0.95)),
        "maximum_delta_e76": float(np.max(delta)),
    }


def run_population_separation(
    config: dict[str, Any], root: Path, *, reverse: bool = False
) -> dict[str, Any]:
    if config.get("schema") != (
        "neuro-film.rf3-d14-three-stock-population-separation-contract.v1"
    ):
        raise ThreeStockPopulationSeparationError("unsupported RF3.D14 contract")

    report_spec = config["input_report"]
    report_data = _read_exact(
        root / report_spec["path"], report_spec["sha256"], int(report_spec["bytes"])
    )
    diagnostic_report = json.loads(report_data)
    if diagnostic_report["scientific_identity"] != report_spec["scientific_identity"]:
        raise ThreeStockPopulationSeparationError("diagnostic identity drift")

    severe_spec = config["severe_review_evidence"]
    severe_data = _read_exact(
        root / severe_spec["path"], severe_spec["sha256"], int(severe_spec["bytes"])
    )
    severe = json.loads(severe_data)
    if severe["status"] != severe_spec["required_status"]:
        raise ThreeStockPopulationSeparationError("severe review is not closed-pass")

    parent_spec = config["parent_separation_contract"]
    parent_data = _read_exact(
        root / parent_spec["path"], parent_spec["sha256"], int(parent_spec["bytes"])
    )
    parent = json.loads(parent_data)
    inherited_threshold = float(
        parent["gates"]["minimum_pairwise_full_output_population_median_delta_e76"]
    )
    threshold = float(config["gates"]["per_source_pair_median_delta_e76_threshold"])
    if inherited_threshold != threshold:
        raise ThreeStockPopulationSeparationError("U7.2C threshold was not inherited")

    arm_ids = tuple(str(value) for value in config["arm_ids"])
    if len(set(arm_ids)) != 3 or any("ao6" in arm_id for arm_id in arm_ids):
        raise ThreeStockPopulationSeparationError("invalid three-arm selection")
    diagnostic_rows = {
        (row["source_id"], row["arm_id"]): row
        for row in diagnostic_report["scientific_payload"]["rows"]
        if row["arm_id"] in arm_ids
    }
    source_ids = sorted({key[0] for key in diagnostic_rows})
    expected_pairs = {(source_id, arm_id) for source_id in source_ids for arm_id in arm_ids}
    if (
        len(source_ids) != int(config["required_source_count"])
        or set(diagnostic_rows) != expected_pairs
    ):
        raise ThreeStockPopulationSeparationError("population selection drift")

    processing_sources = list(reversed(source_ids)) if reverse else source_ids
    render_root = root / config["render_root"]
    rows: list[dict[str, Any]] = []
    for source_id in processing_sources:
        outputs: dict[str, np.ndarray] = {}
        for arm_id in arm_ids:
            row = diagnostic_rows[(source_id, arm_id)]
            outputs[arm_id] = _read_rgb8(
                render_root / row["output"]["relative_path"], row["output"]["sha256"]
            )
        for left_id, right_id in combinations(arm_ids, 2):
            rows.append(
                {
                    "source_id": source_id,
                    "left_arm_id": left_id,
                    "right_arm_id": right_id,
                    **delta_e76_summary(outputs[left_id], outputs[right_id]),
                }
            )
    rows.sort(key=lambda row: (row["source_id"], row["left_arm_id"], row["right_arm_id"]))

    aggregates: list[dict[str, Any]] = []
    for left_id, right_id in combinations(arm_ids, 2):
        pair_rows = [
            row
            for row in rows
            if row["left_arm_id"] == left_id and row["right_arm_id"] == right_id
        ]
        medians = np.asarray(
            [float(row["median_delta_e76"]) for row in pair_rows], dtype=np.float64
        )
        aggregates.append(
            {
                "left_arm_id": left_id,
                "right_arm_id": right_id,
                "source_count": len(pair_rows),
                "source_pass_count": int(np.sum(medians >= threshold)),
                "population_minimum_median_delta_e76": float(np.min(medians)),
                "population_q25_median_delta_e76": float(np.quantile(medians, 0.25)),
                "population_median_delta_e76": float(np.median(medians)),
            }
        )

    gates = {
        "exact_16_source_three_arm_population": len(rows) == 48,
        "u4_3d_severe_review_pass": severe["decision_counts"] == {
            "PASS_NO_CONFIRMED_SEVERE_ARTIFACT": 48,
            "FAIL_CONFIRMED_SEVERE_ARTIFACT": 0,
            "REVIEW_UNRESOLVED": 0,
        },
        "parent_threshold_inherited": inherited_threshold == threshold,
        "every_pair_population_median": all(
            float(row["population_median_delta_e76"])
            >= float(config["gates"]["minimum_pairwise_population_median_delta_e76"])
            for row in aggregates
        ),
        "every_pair_source_pass_count": all(
            int(row["source_pass_count"])
            >= int(config["gates"]["minimum_per_pair_source_pass_count"])
            for row in aggregates
        ),
        "ao6_excluded": all("ao6" not in arm_id for arm_id in arm_ids),
    }
    automatic_pass = all(gates.values())
    payload = {
        "input_report_sha256": _sha256(report_data),
        "severe_review_evidence_sha256": _sha256(severe_data),
        "parent_separation_contract_sha256": _sha256(parent_data),
        "source_count": len(source_ids),
        "arm_ids": list(arm_ids),
        "pair_source_rows": rows,
        "pair_aggregates": aggregates,
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            config["decision_if_pass"] if automatic_pass else config["decision_if_fail"]
        ),
        "render_calls": 0,
        "network_reads": 0,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        "schema": "neuro-film.rf3-d14-three-stock-population-separation-report.v1",
        "experiment_id": config["experiment_id"],
        "status": "PASS" if automatic_pass else "FAIL_CLOSED",
        "scientific_payload": payload,
        "scientific_identity": _sha256(_canonical_bytes(payload)),
    }


__all__ = [
    "ThreeStockPopulationSeparationError",
    "delta_e76_summary",
    "run_population_separation",
]
