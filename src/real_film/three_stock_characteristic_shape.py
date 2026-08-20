"""RF3.D4 comparison of normalized first-party characteristic-curve shapes."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

SCHEMA = "neuro-film.rf3-three-stock-characteristic-shape-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-three-stock-characteristic-shape-report.v1"
STOCKS = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]


class ThreeStockCharacteristicShapeError(RuntimeError):
    """Raised when the frozen contract or a bound source drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ThreeStockCharacteristicShapeError("RF3.D4 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources", [])
    gates = payload.get("gates", {})
    digitization = payload.get("digitization", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "RF3.D4"
        or [row.get("stock_id") for row in sources] != STOCKS
        or len(payload.get("normalized_exposure_positions", [])) != 7
        or digitization.get("minimum_endpoint_pixel_span") != 150.0
        or digitization.get("require_strict_monotonicity") is not True
        or gates.get("minimum_pairwise_median_absolute_normalized_shape_difference") != 0.03
        or gates.get("minimum_pairwise_rmse_normalized_shape_difference") != 0.04
        or not all(
            gates.get(key) is True
            for key in (
                "require_all_pairwise_gates",
                "require_all_source_hashes",
                "require_exact_replay",
            )
        )
    ):
        raise ThreeStockCharacteristicShapeError("RF3.D4 frozen contract drift")
    _relative(str(payload.get("parent", {}).get("path", "")))
    for source in sources:
        _relative(str(source.get("page_image", "")))
        if (
            source.get("density_direction_with_exposure") not in {"increasing", "decreasing"}
            or len(source.get("sample_x_pixels", [])) != 7
            or len(source.get("sample_midpoint_y_pixels", [])) != 7
        ):
            raise ThreeStockCharacteristicShapeError("RF3.D4 source digitization drift")
    return payload


def _trace_source(
    source: Mapping[str, Any], root: Path, overlay_dir: Path, minimum_span: float
) -> tuple[dict[str, Any], str]:
    path = root / _relative(str(source["page_image"]))
    if not path.is_file() or _hash_file(path) != source["page_image_sha256"]:
        raise ThreeStockCharacteristicShapeError(f"RF3.D4 page drift: {source['stock_id']}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or [image.shape[1], image.shape[0]] != source["page_image_size"]:
        raise ThreeStockCharacteristicShapeError(f"RF3.D4 page size drift: {source['stock_id']}")
    xs = np.asarray(source["sample_x_pixels"], dtype=np.float64)
    ys = np.asarray(source["sample_midpoint_y_pixels"], dtype=np.float64)
    if not np.all(np.diff(xs) > 0):
        raise ThreeStockCharacteristicShapeError("RF3.D4 sample x positions are not strict")
    span = float(abs(ys[-1] - ys[0]))
    if not math.isfinite(span) or span < minimum_span:
        raise ThreeStockCharacteristicShapeError("RF3.D4 endpoint span is insufficient")
    if source["density_direction_with_exposure"] == "decreasing":
        oriented = ys - ys[0]
    else:
        oriented = ys[0] - ys
    normalized = oriented / oriented[-1]
    if not np.all(np.isfinite(normalized)) or not np.all(np.diff(normalized) > 0):
        raise ThreeStockCharacteristicShapeError("RF3.D4 normalized shape is not strict")
    for x, y in zip(xs, ys, strict=True):
        cv2.circle(image, (round(float(x)), round(float(y))), 7, (0, 0, 255), 2)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{source['stock_id']}_overlay.png"
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(overlay_path)
    return (
        {
            "stock_id": source["stock_id"],
            "film_family": source["film_family"],
            "published_curve_semantics": source["published_curve_semantics"],
            "density_direction_with_exposure": source["density_direction_with_exposure"],
            "endpoint_pixel_span": span,
            "normalized_shape": [float(value) for value in normalized],
        },
        _hash_file(overlay_path),
    )


def evaluate(contract: Mapping[str, Any], root: Path, *, overlay_dir: Path) -> dict[str, Any]:
    parent = root / _relative(str(contract["parent"]["path"]))
    if not parent.is_file() or _hash_file(parent) != contract["parent"]["sha256"]:
        raise ThreeStockCharacteristicShapeError("RF3.D4 parent evidence drift")
    parent_payload = json.loads(parent.read_text(encoding="utf-8"))
    if parent_payload.get("decision") != contract["parent"]["required_decision"]:
        raise ThreeStockCharacteristicShapeError("RF3.D4 parent decision drift")

    traces: list[dict[str, Any]] = []
    overlays: dict[str, str] = {}
    for source in contract["sources"]:
        trace, overlay_hash = _trace_source(
            source,
            root,
            overlay_dir,
            float(contract["digitization"]["minimum_endpoint_pixel_span"]),
        )
        traces.append(trace)
        overlays[trace["stock_id"]] = overlay_hash

    median_gate = float(
        contract["gates"]["minimum_pairwise_median_absolute_normalized_shape_difference"]
    )
    rmse_gate = float(
        contract["gates"]["minimum_pairwise_rmse_normalized_shape_difference"]
    )
    comparisons: list[dict[str, Any]] = []
    for left, right in itertools.combinations(traces, 2):
        delta = np.abs(
            np.asarray(left["normalized_shape"], dtype=np.float64)
            - np.asarray(right["normalized_shape"], dtype=np.float64)
        )
        median = float(np.median(delta))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        comparisons.append(
            {
                "left": left["stock_id"],
                "right": right["stock_id"],
                "median_absolute_normalized_shape_difference": median,
                "rmse_normalized_shape_difference": rmse,
                "median_gate_pass": median >= median_gate,
                "rmse_gate_pass": rmse >= rmse_gate,
            }
        )
    gate_results = {
        "all_source_hashes_and_shapes_valid": len(traces) == 3,
        "all_pairwise_median_differences_material": all(
            row["median_gate_pass"] for row in comparisons
        ),
        "all_pairwise_rmse_differences_material": all(
            row["rmse_gate_pass"] for row in comparisons
        ),
    }
    automatic_pass = all(gate_results.values())
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "normalized_exposure_positions": contract["normalized_exposure_positions"],
        "traces": traces,
        "pairwise_comparisons": comparisons,
        "overlay_sha256": overlays,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass" if automatic_pass else "decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable = dict(result)
    stable.pop("overlay_sha256")
    result["stable_evidence_id"] = hashlib.sha256(_canonical_json(stable)).hexdigest()
    return result


def write_report(report: Mapping[str, Any], path: Path) -> str:
    data = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
