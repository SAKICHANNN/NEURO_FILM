"""Deterministic helpers for the U1.4C2 Rec.2020 visual/OOD audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_filmr_selection(config: dict, *, root: Path) -> list[Path]:
    """Validate the frozen metadata/hash selection without inspecting output pixels."""

    manifest_path = root / config["source_manifest"]
    if sha256_path(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("FILM-R source manifest hash mismatch")
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_pair = {row["pair_id"]: row for row in manifest_rows}
    selected: list[Path] = []
    for sample in config["selection"]["samples"]:
        row = by_pair.get(sample["pair_id"])
        if row is None or row["expert_restoration"]["sha256"] != sample["sha256"]:
            raise ValueError(f"frozen selection does not match manifest: {sample['pair_id']}")
        path = root / "data" / "real_film" / "filmr_v2" / "files" / sample["filename"]
        if not path.is_file():
            raise ValueError(f"selected source is missing: {sample['pair_id']}")
        if path.stat().st_size != sample["bytes"]:
            raise ValueError(f"selected source byte count mismatch: {sample['pair_id']}")
        if sha256_path(path) != sample["sha256"]:
            raise ValueError(f"selected source hash mismatch: {sample['pair_id']}")
        selected.append(path)
    if len(selected) != config["automatic_gates"]["expected_sources"]:
        raise ValueError("selected source count does not match the frozen gate")
    return selected


def encoded_srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3 or not np.isfinite(value).all():
        raise ValueError("encoded sRGB must be finite HxWx3")
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_srgb_to_encoded(linear: np.ndarray) -> np.ndarray:
    value = np.asarray(linear, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3 or not np.isfinite(value).all():
        raise ValueError("linear sRGB must be finite HxWx3")
    value = np.clip(value, 0.0, 1.0)
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def save_srgb_preview(linear_srgb: np.ndarray, path: Path) -> str:
    encoded = linear_srgb_to_encoded(linear_srgb)
    pixels = np.rint(encoded * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", compress_level=6)
    return sha256_path(path)


def residual_gradient_quantile(source: np.ndarray, output: np.ndarray, quantile: float = 0.999) -> float:
    residual = np.asarray(output - source, dtype=np.float32)
    horizontal = np.max(np.abs(np.diff(residual, axis=1)), axis=2).reshape(-1)
    vertical = np.max(np.abs(np.diff(residual, axis=0)), axis=2).reshape(-1)
    return float(max(np.quantile(horizontal, quantile), np.quantile(vertical, quantile)))


def anonymous_candidate_order(sample_id: str, round_id: int, candidates: list[str]) -> list[str]:
    return sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"u1.4c2:{round_id}:{sample_id}:{candidate}".encode("utf-8")
        ).hexdigest(),
    )


def risk_rank(records: list[dict], limit: int) -> list[dict]:
    """Rank by the sum of deterministic descending ranks across frozen risks."""

    keys = [
        "new_rec2020_boundary_fraction",
        "srgb_preview_clip_fraction",
        "residual_gradient_q999",
    ]
    rank_sums = {record["logical_id"]: 0 for record in records}
    for key in keys:
        ordered = sorted(records, key=lambda row: (-row[key], row["logical_id"]))
        for rank, record in enumerate(ordered, start=1):
            rank_sums[record["logical_id"]] += rank
    ranked = sorted(
        records,
        key=lambda row: (rank_sums[row["logical_id"]], row["logical_id"]),
    )[:limit]
    return [
        {
            "risk_rank": index,
            "rank_sum": rank_sums[record["logical_id"]],
            **record,
        }
        for index, record in enumerate(ranked, start=1)
    ]


def build_anonymous_sheet(
    *,
    source_preview: Path,
    candidate_previews: dict[str, Path],
    sample_id: str,
    round_id: int,
    output_path: Path,
) -> dict[str, str]:
    """Build one two-row anonymous severe-artifact sheet and return its private map."""

    order = anonymous_candidate_order(sample_id, round_id, sorted(candidate_previews))
    labels = [chr(ord("A") + index) for index in range(len(order))]
    private_map = dict(zip(labels, order, strict=True))
    thumb_width, thumb_height = 384, 288
    label_height = 28
    columns = 7
    rows = 2
    canvas = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#202020")

    with Image.open(source_preview) as source_image:
        source = source_image.convert("RGB")
        source.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        for row in range(rows):
            x = (thumb_width - source.width) // 2
            y = row * (thumb_height + label_height) + (thumb_height - source.height) // 2
            canvas.paste(source, (x, y))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for row in range(rows):
        draw.text((8, row * (thumb_height + label_height) + thumb_height + 5), "SOURCE", fill="white")
        for col in range(1, columns):
            index = row * 6 + (col - 1)
            label = labels[index]
            with Image.open(candidate_previews[private_map[label]]) as candidate_image:
                candidate = candidate_image.convert("RGB")
                candidate.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                x = col * thumb_width + (thumb_width - candidate.width) // 2
                y = row * (thumb_height + label_height) + (thumb_height - candidate.height) // 2
                canvas.paste(candidate, (x, y))
            draw.text(
                (col * thumb_width + 8, row * (thumb_height + label_height) + thumb_height + 5),
                label,
                fill="white",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=6)
    return private_map


__all__ = [
    "anonymous_candidate_order",
    "build_anonymous_sheet",
    "encoded_srgb_to_linear",
    "linear_srgb_to_encoded",
    "residual_gradient_quantile",
    "risk_rank",
    "save_srgb_preview",
    "sha256_path",
    "validate_filmr_selection",
]
