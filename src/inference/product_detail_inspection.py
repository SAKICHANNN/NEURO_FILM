"""Exact-export spatial detail inspection for the private product desktop."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image

from .product_desktop import (
    DesktopExportReceipt,
    DesktopFinishingEffects,
    ProductDesktopError,
    ProductDesktopWorkflow,
    product_output_format,
    sha256_file,
)


@dataclass(frozen=True)
class DesktopDetailPreview:
    """One owned viewer crop derived from an exact temporary product export."""

    style_id: str
    look_amount: float
    input_path: Path
    input_sha256: str
    output_format_id: str
    full_width: int
    full_height: int
    crop_box: tuple[int, int, int, int]
    crop_png: bytes
    crop_sha256: str
    full_output_sha256: str
    strict_recipe_sha256: str


class _ExportWorkflow(Protocol):
    scratch_root: Path

    def export(
        self,
        style_id: str,
        output_path: Path,
        *,
        output_format_id: str,
        effects: DesktopFinishingEffects | None = None,
    ) -> DesktopExportReceipt: ...


def _unit_coordinate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductDesktopError(f"{name} must be a finite number in [0, 1]")
    coordinate = float(value)
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
        raise ProductDesktopError(f"{name} must be a finite number in [0, 1]")
    return coordinate


def normalized_card_point(
    x: object,
    y: object,
    *,
    widget_width: object,
    widget_height: object,
    image_width: object,
    image_height: object,
) -> tuple[float, float]:
    """Map one click inside a centered card image to a normalized point."""

    values = (x, y, widget_width, widget_height, image_width, image_height)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ProductDesktopError("detail card geometry must use integers")
    if widget_width < 1 or widget_height < 1 or image_width < 1 or image_height < 1:
        raise ProductDesktopError("detail card geometry must be positive")
    if image_width > widget_width or image_height > widget_height:
        raise ProductDesktopError("detail image exceeds its visible card")
    left = (widget_width - image_width) // 2
    top = (widget_height - image_height) // 2
    local_x = x - left
    local_y = y - top
    if not 0 <= local_x < image_width or not 0 <= local_y < image_height:
        raise ProductDesktopError("choose a point inside the rendered preview")
    normalized_x = 0.5 if image_width == 1 else local_x / (image_width - 1)
    normalized_y = 0.5 if image_height == 1 else local_y / (image_height - 1)
    return float(normalized_x), float(normalized_y)


def detail_crop_box(
    width: object,
    height: object,
    point: tuple[object, object],
    *,
    crop_limit: object = 512,
) -> tuple[int, int, int, int]:
    """Resolve one clamped native-pixel crop around a normalized point."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (width, height, crop_limit)
    ):
        raise ProductDesktopError("detail geometry must use positive integers")
    if not isinstance(point, tuple) or len(point) != 2:
        raise ProductDesktopError("detail point must contain x and y")
    normalized_x = _unit_coordinate(point[0], "detail x")
    normalized_y = _unit_coordinate(point[1], "detail y")
    crop_width = min(width, crop_limit)
    crop_height = min(height, crop_limit)
    center_x = math.floor(normalized_x * (width - 1) + 0.5)
    center_y = math.floor(normalized_y * (height - 1) + 0.5)
    left = min(max(center_x - crop_width // 2, 0), width - crop_width)
    top = min(max(center_y - crop_height // 2, 0), height - crop_height)
    return left, top, left + crop_width, top + crop_height


def _crop_png(
    output_path: Path, point: tuple[float, float], crop_limit: int
) -> tuple[
    int,
    int,
    tuple[int, int, int, int],
    bytes,
]:
    with Image.open(output_path) as opened:
        if int(getattr(opened, "n_frames", 1)) != 1:
            raise ProductDesktopError("detail output must contain one image")
        width, height = opened.size
        crop_box = detail_crop_box(width, height, point, crop_limit=crop_limit)
        crop = opened.convert("RGB").crop(crop_box)
    buffer = BytesIO()
    crop.save(buffer, format="PNG", compress_level=6)
    return width, height, crop_box, buffer.getvalue()


def _cleanup_exact_receipt(
    root: Path,
    receipt: DesktopExportReceipt,
) -> None:
    expected = {
        receipt.output_path: receipt.output_sha256,
        receipt.recipe_path: receipt.recipe_sha256,
    }
    if receipt.output_path.parent != root or receipt.recipe_path.parent != root:
        raise ProductDesktopError("detail export escaped its owned workspace")
    members = {path for path in root.iterdir()}
    if members != set(expected):
        raise ProductDesktopError("detail workspace member set changed")
    if any(
        not path.is_file() or sha256_file(path) != digest
        for path, digest in expected.items()
    ):
        raise ProductDesktopError("detail workspace identity changed")
    for path in sorted(expected, key=lambda candidate: candidate.name, reverse=True):
        path.unlink()
    root.rmdir()


def render_exact_export_detail(
    workflow: ProductDesktopWorkflow | _ExportWorkflow,
    style_id: str,
    *,
    output_format_id: str,
    point: tuple[object, object] = (0.5, 0.5),
    crop_limit: object = 512,
    effects: DesktopFinishingEffects | None = None,
) -> DesktopDetailPreview:
    """Render, inspect, and remove one exact temporary full-resolution export."""

    if not isinstance(point, tuple) or len(point) != 2:
        raise ProductDesktopError("detail point must contain x and y")
    normalized_point = (
        _unit_coordinate(point[0], "detail x"),
        _unit_coordinate(point[1], "detail y"),
    )
    if (
        isinstance(crop_limit, bool)
        or not isinstance(crop_limit, int)
        or crop_limit < 1
    ):
        raise ProductDesktopError("detail crop limit must be a positive integer")
    output = product_output_format(output_format_id)
    scratch_root = Path(workflow.scratch_root).resolve(strict=True)
    owned_root = scratch_root / f"u7-16a-detail-{os.getpid()}-{uuid.uuid4().hex}"
    owned_root.mkdir(exist_ok=False)
    destination = owned_root / f"detail{output.canonical_extension}"
    receipt: DesktopExportReceipt | None = None
    try:
        export_kwargs: dict[str, object] = {"output_format_id": output_format_id}
        if effects is not None:
            export_kwargs["effects"] = effects
        receipt = workflow.export(style_id, destination, **export_kwargs)
        if (
            receipt.style_id != style_id
            or receipt.output_format_id != output_format_id
            or receipt.output_path != destination
            or receipt.recipe_path != destination.with_suffix(".recipe.json")
            or sha256_file(receipt.output_path) != receipt.output_sha256
            or sha256_file(receipt.recipe_path) != receipt.recipe_sha256
        ):
            raise ProductDesktopError("detail export receipt drifted")
        full_width, full_height, crop_box, crop_png = _crop_png(
            receipt.output_path,
            normalized_point,
            crop_limit,
        )
        result = DesktopDetailPreview(
            style_id=receipt.style_id,
            look_amount=receipt.look_amount,
            input_path=receipt.input_path,
            input_sha256=receipt.input_sha256,
            output_format_id=receipt.output_format_id,
            full_width=full_width,
            full_height=full_height,
            crop_box=crop_box,
            crop_png=crop_png,
            crop_sha256=hashlib.sha256(crop_png).hexdigest(),
            full_output_sha256=receipt.output_sha256,
            strict_recipe_sha256=receipt.recipe_sha256,
        )
        _cleanup_exact_receipt(owned_root, receipt)
        return result
    except BaseException:
        if receipt is None and owned_root.exists() and not any(owned_root.iterdir()):
            owned_root.rmdir()
        raise


__all__ = [
    "DesktopDetailPreview",
    "detail_crop_box",
    "normalized_card_point",
    "render_exact_export_detail",
]
