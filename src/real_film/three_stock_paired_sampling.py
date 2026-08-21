"""Common-coordinate sampling for aligned controlled three-stock scans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.real_film.three_stock_k1_baseline import StockFrameSamples


class ThreeStockPairedSamplingError(ValueError):
    """Raised when controlled rows cannot share an identical source sample grid."""


@dataclass(frozen=True)
class AlignedScanRow:
    row_id: str
    stock_id: str
    role: str
    scene_id: str
    film_frame_id: str
    roll_id: str
    source_rgb: np.ndarray
    scan_rgb: np.ndarray
    homography_source_to_scan: np.ndarray

    def __post_init__(self) -> None:
        source = _rgb(self.source_rgb)
        scan = _rgb(self.scan_rgb)
        homography = np.asarray(self.homography_source_to_scan, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            raise ThreeStockPairedSamplingError("homography must be finite 3x3")
        if abs(float(np.linalg.det(homography))) <= 1e-12:
            raise ThreeStockPairedSamplingError("homography must be invertible")
        if self.role not in {"development", "confirmation"}:
            raise ThreeStockPairedSamplingError("unsupported row role")
        if not all(
            (
                self.row_id,
                self.stock_id,
                self.scene_id,
                self.film_frame_id,
                self.roll_id,
            )
        ):
            raise ThreeStockPairedSamplingError("row identities must be non-empty")
        object.__setattr__(self, "source_rgb", source)
        object.__setattr__(self, "scan_rgb", scan)
        object.__setattr__(self, "homography_source_to_scan", homography)


def _rgb(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 2:
        raise ThreeStockPairedSamplingError("expected non-empty RGB image")
    if array.dtype not in (np.uint8, np.uint16):
        raise ThreeStockPairedSamplingError("expected uint8 or uint16 image")
    return np.ascontiguousarray(array)


def _normalized(value: np.ndarray) -> np.ndarray:
    return value.astype(np.float64) / float(np.iinfo(value.dtype).max)


def _source_grid(height: int, width: int, config: Mapping[str, Any]) -> np.ndarray:
    rows = int(config["source_grid_rows"])
    columns = int(config["source_grid_columns"])
    margin = float(config["source_inner_margin_fraction"])
    if rows < 2 or columns < 2 or not 0.0 <= margin < 0.5:
        raise ThreeStockPairedSamplingError("invalid source grid contract")
    x = np.linspace(margin * (width - 1), (1.0 - margin) * (width - 1), columns)
    y = np.linspace(margin * (height - 1), (1.0 - margin) * (height - 1), rows)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float64)


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(
        points.astype(np.float32).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    if not np.isfinite(projected).all():
        raise ThreeStockPairedSamplingError("homography projection is non-finite")
    return projected.astype(np.float64)


def _bilinear(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    normalized = _normalized(image).astype(np.float32)
    mapped = cv2.remap(
        normalized,
        points[:, 0].astype(np.float32).reshape(1, -1),
        points[:, 1].astype(np.float32).reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    result = np.asarray(mapped, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(result).all() or np.min(result) < 0.0 or np.max(result) > 1.0:
        raise ThreeStockPairedSamplingError("sampled RGB values are invalid")
    return result


def extract_common_paired_samples(
    rows: Sequence[AlignedScanRow],
    sampling: Mapping[str, Any],
) -> tuple[
    dict[str, list[StockFrameSamples]],
    dict[str, list[StockFrameSamples]],
    dict[str, Any],
]:
    """Sample every scan on one shared source-coordinate mask per role-scene."""

    if not rows:
        raise ThreeStockPairedSamplingError("aligned scan rows are empty")
    if sampling.get("interpolation") != "bilinear":
        raise ThreeStockPairedSamplingError("unsupported interpolation")
    grouped: dict[tuple[str, str], list[AlignedScanRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.role, row.scene_id)].append(row)
    development: dict[str, list[StockFrameSamples]] = defaultdict(list)
    confirmation: dict[str, list[StockFrameSamples]] = defaultdict(list)
    scene_facts: list[dict[str, Any]] = []
    edge = float(sampling["scan_edge_margin_pixels"])
    minimum_fraction = float(sampling["minimum_shared_valid_fraction_per_scene"])
    if edge < 0.0 or not 0.0 < minimum_fraction <= 1.0:
        raise ThreeStockPairedSamplingError("invalid shared-validity contract")

    for (role, scene_id), selected in sorted(grouped.items()):
        canonical_source = selected[0].source_rgb
        if any(
            row.source_rgb.shape != canonical_source.shape
            or row.source_rgb.dtype != canonical_source.dtype
            or not np.array_equal(row.source_rgb, canonical_source)
            for row in selected[1:]
        ):
            raise ThreeStockPairedSamplingError(
                f"source pixels differ within {role}/{scene_id}"
            )
        height, width = canonical_source.shape[:2]
        source_points = _source_grid(height, width, sampling)
        projected_by_row: dict[str, np.ndarray] = {}
        shared = np.ones(len(source_points), dtype=bool)
        for row in selected:
            projected = _project(source_points, row.homography_source_to_scan)
            projected_by_row[row.row_id] = projected
            scan_height, scan_width = row.scan_rgb.shape[:2]
            shared &= (
                (projected[:, 0] >= edge)
                & (projected[:, 0] <= scan_width - 1 - edge)
                & (projected[:, 1] >= edge)
                & (projected[:, 1] <= scan_height - 1 - edge)
            )
        shared_count = int(np.sum(shared))
        shared_fraction = shared_count / len(source_points)
        if shared_fraction < minimum_fraction:
            raise ThreeStockPairedSamplingError(
                f"shared valid support {shared_fraction:.6f} below gate for {role}/{scene_id}"
            )
        common_points = source_points[shared]
        source_samples = _bilinear(canonical_source, common_points)
        destination = development if role == "development" else confirmation
        for row in sorted(selected, key=lambda value: value.row_id):
            target_samples = _bilinear(
                row.scan_rgb, projected_by_row[row.row_id][shared]
            )
            destination[row.stock_id].append(
                StockFrameSamples(
                    scene_id=scene_id,
                    frame_id=row.row_id,
                    roll_id=row.roll_id,
                    source=source_samples.copy(),
                    target=target_samples,
                )
            )
        scene_facts.append(
            {
                "role": role,
                "scene_id": scene_id,
                "row_count": len(selected),
                "grid_points": len(source_points),
                "shared_valid_points": shared_count,
                "shared_valid_fraction": shared_fraction,
            }
        )
    facts = {
        "scene_count": len(scene_facts),
        "row_count": len(rows),
        "minimum_observed_shared_valid_fraction": min(
            row["shared_valid_fraction"] for row in scene_facts
        ),
        "scenes": scene_facts,
    }
    return dict(development), dict(confirmation), facts


__all__ = [
    "AlignedScanRow",
    "ThreeStockPairedSamplingError",
    "extract_common_paired_samples",
]
