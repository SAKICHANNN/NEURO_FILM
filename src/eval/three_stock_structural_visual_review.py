"""Deterministic review sheets for frozen U4.3A structural queue leaders."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter, sobel, uniform_filter

from src.color_engine.srgb_transfer import encoded_srgb_to_linear


class ThreeStockStructuralVisualReviewError(ValueError):
    """Raised when the frozen review inputs or selection do not match."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_exact(path: Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    if _sha256(data) != expected_sha256:
        raise ThreeStockStructuralVisualReviewError(f"input hash mismatch: {path}")
    return data


def _read_rgb8(path: Path, expected_sha256: str) -> np.ndarray:
    data = _read_exact(path, expected_sha256)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB":
            raise ThreeStockStructuralVisualReviewError(f"expected RGB8 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def _selected_pairs(report: dict[str, Any], depth: int) -> list[tuple[str, str]]:
    queues = report["scientific_payload"]["review_queues"]
    selected = {
        (row["source_id"], row["arm_id"])
        for rows in queues.values()
        for row in rows[:depth]
    }
    return sorted(selected)


def _luminance(rgb8: np.ndarray) -> np.ndarray:
    encoded = rgb8.astype(np.float32) / np.float32(255.0)
    linear = encoded_srgb_to_linear(encoded)
    return np.asarray(
        linear @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
        dtype=np.float32,
    )


def _crop_box(score: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
    height, width = score.shape
    if crop_size > min(height, width):
        raise ThreeStockStructuralVisualReviewError("crop exceeds image geometry")
    smoothed = uniform_filter(score.astype(np.float32), size=33, mode="reflect")
    half = crop_size // 2
    valid = np.full(smoothed.shape, -np.inf, dtype=np.float32)
    y_min, y_max = half, height - (crop_size - half)
    x_min, x_max = half, width - (crop_size - half)
    valid[y_min : y_max + 1, x_min : x_max + 1] = smoothed[
        y_min : y_max + 1, x_min : x_max + 1
    ]
    y, x = np.unravel_index(int(np.argmax(valid)), valid.shape)
    left = int(x - half)
    top = int(y - half)
    return left, top, left + crop_size, top + crop_size


def _review_boxes(
    source: np.ndarray, output: np.ndarray, crop_size: int
) -> dict[str, list[int]]:
    before_luma = _luminance(source)
    after_luma = _luminance(output)
    before_gx = sobel(before_luma, axis=1, mode="reflect")
    before_gy = sobel(before_luma, axis=0, mode="reflect")
    after_gx = sobel(after_luma, axis=1, mode="reflect")
    after_gy = sobel(after_luma, axis=0, mode="reflect")
    before_gradient = np.hypot(before_gx, before_gy)
    after_gradient = np.hypot(after_gx, after_gy)
    edge_floor = max(float(np.percentile(before_gradient, 75.0)), 1e-6)
    edge_mask = before_gradient >= edge_floor
    denominator = np.maximum(before_gradient * after_gradient, 1e-6)
    cosine = np.clip(
        (before_gx * after_gx + before_gy * after_gy) / denominator, -1.0, 1.0
    )
    edge_disagreement = np.where(edge_mask, 1.0 - cosine, 0.0)
    edge_gain = np.where(
        edge_mask,
        np.clip(after_gradient / np.maximum(before_gradient, 1e-6) - 1.0, 0.0, 10.0),
        0.0,
    )
    before_high = np.abs(
        before_luma - gaussian_filter(before_luma, sigma=1.0, mode="reflect")
    )
    after_high = np.abs(
        after_luma - gaussian_filter(after_luma, sigma=1.0, mode="reflect")
    )
    flat_mask = before_high <= float(np.percentile(before_high, 25.0))
    flat_added = np.where(flat_mask, np.maximum(after_high - before_high, 0.0), 0.0)
    return {
        "flat_new_high_frequency": list(_crop_box(flat_added, crop_size)),
        "edge_direction_disagreement": list(_crop_box(edge_disagreement, crop_size)),
        "edge_gain_tail": list(_crop_box(edge_gain, crop_size)),
    }


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def _sheet_bytes(
    source: np.ndarray,
    output: np.ndarray,
    boxes: dict[str, list[int]],
    source_id: str,
    arm_id: str,
) -> bytes:
    source_image = Image.fromarray(source, mode="RGB")
    output_image = Image.fromarray(output, mode="RGB")
    canvas = Image.new("RGB", (1536, 1048), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{source_id} | {arm_id}", fill="black")
    canvas.paste(_fit_panel(source_image, (768, 420)), (0, 34))
    canvas.paste(_fit_panel(output_image, (768, 420)), (768, 34))
    draw.text((12, 458), "SOURCE OVERVIEW", fill="black")
    draw.text((780, 458), "OUTPUT OVERVIEW", fill="black")
    for index, (label, box_values) in enumerate(boxes.items()):
        left, top, right, bottom = box_values
        x = index * 512 + 128
        draw.text((index * 512 + 8, 486), f"{label} crop={box_values}", fill="black")
        canvas.paste(source_image.crop((left, top, right, bottom)), (x, 510))
        canvas.paste(output_image.crop((left, top, right, bottom)), (x, 778))
        draw.text((index * 512 + 8, 752), "SOURCE 1:1", fill="black")
        draw.text((index * 512 + 8, 1020), "OUTPUT 1:1", fill="black")
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def _population_sheet_bytes(
    source: np.ndarray,
    outputs: list[tuple[str, np.ndarray, dict[str, list[int]]]],
    source_id: str,
) -> bytes:
    """Build one overview plus exact-pixel crop sheet for all three K=1 arms."""
    if len(outputs) != 3:
        raise ThreeStockStructuralVisualReviewError(
            "population review requires exactly three K=1 outputs"
        )
    source_image = Image.fromarray(source, mode="RGB")
    canvas = Image.new("RGB", (1536, 2456), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{source_id} | three fixed K=1 Look Approximation arms", fill="black")

    overview_images = [("SOURCE", source_image)] + [
        (arm_id, Image.fromarray(output, mode="RGB"))
        for arm_id, output, _boxes in outputs
    ]
    for index, (label, image) in enumerate(overview_images):
        x = (index % 2) * 768
        y = 34 + (index // 2) * 390
        canvas.paste(_fit_panel(image, (768, 350)), (x, y))
        draw.text((x + 8, y + 354), label, fill="black")

    row_top = 820
    for arm_index, (arm_id, output, boxes) in enumerate(outputs):
        output_image = Image.fromarray(output, mode="RGB")
        y = row_top + arm_index * 540
        draw.text((8, y), arm_id, fill="black")
        for crop_index, (label, box_values) in enumerate(boxes.items()):
            left, top, right, bottom = box_values
            x = crop_index * 512
            draw.text((x + 8, y + 22), f"{label} {box_values}", fill="black")
            canvas.paste(source_image.crop((left, top, right, bottom)), (x, y + 46))
            canvas.paste(output_image.crop((left, top, right, bottom)), (x + 256, y + 46))
            draw.text((x + 8, y + 306), "SOURCE 1:1", fill="black")
            draw.text((x + 264, y + 306), "OUTPUT 1:1", fill="black")

    stream = io.BytesIO()
    canvas.save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def build_review_material(
    config: dict[str, Any], root: Path, output_dir: Path, *, reverse: bool = False
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("U4.3B output directory is create-only")
    report_spec = config["input_report"]
    report_bytes = _read_exact(root / report_spec["path"], report_spec["sha256"])
    if len(report_bytes) != int(report_spec["bytes"]):
        raise ThreeStockStructuralVisualReviewError("report byte count mismatch")
    report = json.loads(report_bytes)
    if report["scientific_identity"] != report_spec["scientific_identity"]:
        raise ThreeStockStructuralVisualReviewError("scientific identity mismatch")
    selection = config["selection"]
    selected = _selected_pairs(report, int(selection["queue_depth"]))
    expected = sorted(tuple(pair) for pair in selection["pairs"])
    if selected != expected or len(selected) != int(selection["required_pair_count"]):
        raise ThreeStockStructuralVisualReviewError("frozen selection mismatch")
    queues = report["scientific_payload"]["review_queues"]
    if len(queues) != int(selection["required_queue_count"]):
        raise ThreeStockStructuralVisualReviewError("queue count mismatch")

    manifest_spec = report["scientific_payload"]["inputs"][2]
    manifest = json.loads(
        _read_exact(root / manifest_spec["path"], manifest_spec["sha256"])
    )
    source_rows = {row["id"]: row for row in manifest}
    diagnostic_rows = {
        (row["source_id"], row["arm_id"]): row
        for row in report["scientific_payload"]["rows"]
    }
    ordered = list(reversed(selected)) if reverse else selected
    output_dir.mkdir(parents=True)
    render_root = root / "outputs/eval/rf3_three_stock_proxy_baseline_d0_v1/run_a"
    sheets: list[dict[str, Any]] = []
    crop_size = int(config["review_material"]["crop_size"])
    for source_id, arm_id in ordered:
        row = diagnostic_rows[(source_id, arm_id)]
        source_record = source_rows[source_id]
        source = _read_rgb8(
            root / source_record["decoded_path"], source_record["decoded_sha256"]
        )
        output = _read_rgb8(
            render_root / row["output"]["relative_path"], row["output"]["sha256"]
        )
        boxes = _review_boxes(source, output, crop_size)
        sheet = _sheet_bytes(source, output, boxes, source_id, arm_id)
        name = f"{source_id}__{arm_id}.png"
        (output_dir / name).write_bytes(sheet)
        memberships = sorted(
            queue_name
            for queue_name, queue_rows in queues.items()
            if any(
                item["source_id"] == source_id and item["arm_id"] == arm_id
                for item in queue_rows[: int(selection["queue_depth"])]
            )
        )
        sheets.append(
            {
                "source_id": source_id,
                "arm_id": arm_id,
                "queue_memberships": memberships,
                "source_sha256": row["source"]["sha256"],
                "output_sha256": row["output"]["sha256"],
                "review_boxes": boxes,
                "sheet": {
                    "relative_path": name,
                    "bytes": len(sheet),
                    "sha256": _sha256(sheet),
                },
            }
        )
    sheets.sort(key=lambda item: (item["source_id"], item["arm_id"]))
    payload = {
        "input_report": {
            "path": report_spec["path"],
            "bytes": len(report_bytes),
            "sha256": _sha256(report_bytes),
            "scientific_identity": report["scientific_identity"],
        },
        "selection_rule": selection["rule"],
        "sheets": sheets,
        "render_calls": 0,
        "network_reads": 0,
        "adjudications_present": False,
    }
    return {
        "schema": "neuro-film.u4-3b-three-stock-structural-review-material.v1",
        "experiment_id": config["experiment_id"],
        "status": "REVIEW_MATERIAL_READY",
        "scientific_payload": payload,
        "scientific_identity": _sha256(_canonical_bytes(payload)),
        "claim_ceiling": config["claim_ceiling"],
    }


def build_population_review_material(
    config: dict[str, Any], root: Path, output_dir: Path, *, reverse: bool = False
) -> dict[str, Any]:
    """Build grouped review material for every source and every fixed K=1 arm."""
    if output_dir.exists():
        raise FileExistsError("U4.3D output directory is create-only")
    report_spec = config["input_report"]
    report_bytes = _read_exact(root / report_spec["path"], report_spec["sha256"])
    if len(report_bytes) != int(report_spec["bytes"]):
        raise ThreeStockStructuralVisualReviewError("report byte count mismatch")
    report = json.loads(report_bytes)
    if report["scientific_identity"] != report_spec["scientific_identity"]:
        raise ThreeStockStructuralVisualReviewError("scientific identity mismatch")

    manifest_spec = report["scientific_payload"]["inputs"][2]
    manifest = json.loads(
        _read_exact(root / manifest_spec["path"], manifest_spec["sha256"])
    )
    source_rows = {row["id"]: row for row in manifest}
    diagnostic_rows = {
        (row["source_id"], row["arm_id"]): row
        for row in report["scientific_payload"]["rows"]
    }
    selection = config["selection"]
    source_ids = list(selection["source_ids"])
    arm_ids = list(selection["arm_ids"])
    if len(source_ids) != int(selection["required_source_count"]):
        raise ThreeStockStructuralVisualReviewError("source count mismatch")
    if len(set(source_ids)) != len(source_ids) or len(set(arm_ids)) != 3:
        raise ThreeStockStructuralVisualReviewError("selection identities are not unique")
    expected_pairs = {(source_id, arm_id) for source_id in source_ids for arm_id in arm_ids}
    if not expected_pairs.issubset(diagnostic_rows):
        raise ThreeStockStructuralVisualReviewError("frozen population selection mismatch")

    ordered_sources = list(reversed(source_ids)) if reverse else source_ids
    output_dir.mkdir(parents=True)
    render_root = root / config["review_material"]["render_root"]
    crop_size = int(config["review_material"]["crop_size"])
    sheets: list[dict[str, Any]] = []
    for source_id in ordered_sources:
        source_record = source_rows[source_id]
        source = _read_rgb8(
            root / source_record["decoded_path"], source_record["decoded_sha256"]
        )
        outputs: list[tuple[str, np.ndarray, dict[str, list[int]]]] = []
        rows: list[dict[str, Any]] = []
        for arm_id in arm_ids:
            row = diagnostic_rows[(source_id, arm_id)]
            output = _read_rgb8(
                render_root / row["output"]["relative_path"],
                row["output"]["sha256"],
            )
            if output.shape != source.shape:
                raise ThreeStockStructuralVisualReviewError("output geometry mismatch")
            boxes = _review_boxes(source, output, crop_size)
            outputs.append((arm_id, output, boxes))
            rows.append(
                {
                    "arm_id": arm_id,
                    "output_sha256": row["output"]["sha256"],
                    "review_boxes": boxes,
                }
            )
        sheet = _population_sheet_bytes(source, outputs, source_id)
        name = f"{source_id}__three_k1_population.png"
        (output_dir / name).write_bytes(sheet)
        sheets.append(
            {
                "source_id": source_id,
                "source_sha256": source_record["decoded_sha256"],
                "rows": rows,
                "sheet": {
                    "relative_path": name,
                    "bytes": len(sheet),
                    "sha256": _sha256(sheet),
                },
            }
        )
    sheets.sort(key=lambda item: source_ids.index(item["source_id"]))
    payload = {
        "input_report": {
            "path": report_spec["path"],
            "bytes": len(report_bytes),
            "sha256": _sha256(report_bytes),
            "scientific_identity": report["scientific_identity"],
        },
        "selection_rule": selection["rule"],
        "source_count": len(source_ids),
        "output_count": len(source_ids) * len(arm_ids),
        "sheets": sheets,
        "render_calls": 0,
        "network_reads": 0,
        "adjudications_present": False,
    }
    return {
        "schema": "neuro-film.u4-3d-three-stock-population-review-material.v1",
        "experiment_id": config["experiment_id"],
        "status": "REVIEW_MATERIAL_READY",
        "scientific_payload": payload,
        "scientific_identity": _sha256(_canonical_bytes(payload)),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "ThreeStockStructuralVisualReviewError",
    "build_population_review_material",
    "build_review_material",
]
