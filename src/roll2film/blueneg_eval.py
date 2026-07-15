"""BlueNeg development-only alignment, sampling, and access enforcement."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.pipeline import load_working_image


class BlueNegEvaluationError(ValueError):
    """Raised when alignment, role access, or fixed budgets are violated."""


@dataclass(frozen=True)
class BlueNegRollSamples:
    roll_id: str
    support_content_ids: tuple[str, ...]
    query_content_ids: tuple[str, ...]
    support_source: np.ndarray
    support_target: np.ndarray
    query_source: np.ndarray
    query_target: np.ndarray


def _seed(base: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{base}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _ordered(rows: list[dict[str, Any]], seed: int, namespace: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{namespace}:{row['filename']}".encode()
        ).hexdigest(),
    )


def align_blueneg_pair(
    preview: np.ndarray,
    pseudogt: np.ndarray,
    bbox: np.ndarray | list[int] | tuple[int, ...],
    *,
    minimum_side: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect an official bbox with preview and apply its offset to pseudo-GT."""
    source = np.asarray(preview)
    target = np.asarray(pseudogt)
    bounds = np.asarray(bbox, dtype=np.int64)
    if source.ndim != 3 or target.ndim != 3 or source.shape[-1] != 3 or target.shape[-1] != 3:
        raise BlueNegEvaluationError("BlueNeg alignment requires HxWx3 images")
    if bounds.shape != (4,):
        raise BlueNegEvaluationError("BlueNeg bbox must contain four integers")
    x0, y0, x1, y1 = (int(value) for value in bounds)
    if x1 <= x0 or y1 <= y0:
        raise BlueNegEvaluationError("BlueNeg bbox is empty")
    expected_width = x1 - x0
    expected_height = y1 - y0
    if target.shape[:2] != (expected_height, expected_width):
        raise BlueNegEvaluationError(
            "pseudo-GT shape does not match the official bbox extent"
        )
    source_height, source_width = source.shape[:2]
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, source_width), min(y1, source_height)
    if sx1 <= sx0 or sy1 <= sy0:
        raise BlueNegEvaluationError("official bbox has no preview intersection")
    tx0, ty0 = sx0 - x0, sy0 - y0
    tx1, ty1 = tx0 + (sx1 - sx0), ty0 + (sy1 - sy0)
    if tx0 < 0 or ty0 < 0 or tx1 > target.shape[1] or ty1 > target.shape[0]:
        raise BlueNegEvaluationError("aligned pseudo-GT crop escapes target bounds")
    aligned_source = source[sy0:sy1, sx0:sx1]
    aligned_target = target[ty0:ty1, tx0:tx1]
    if aligned_source.shape != aligned_target.shape:
        raise BlueNegEvaluationError("aligned source/target shapes differ")
    if min(aligned_source.shape[:2]) < minimum_side:
        raise BlueNegEvaluationError("aligned BlueNeg crop is below minimum side")
    return aligned_source, aligned_target


def load_aligned_blueneg_pair(
    root: Path,
    row: dict[str, Any],
    transformation: dict[str, np.ndarray],
    *,
    minimum_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    preview_path = (root / Path(*str(row["preview_path"]).split("/"))).resolve()
    pseudogt_value = row.get("pseudogt_path")
    if not pseudogt_value:
        raise BlueNegEvaluationError(f"frame lacks pseudo-GT: {row['filename']}")
    pseudogt_path = (root / Path(*str(pseudogt_value).split("/"))).resolve()
    try:
        preview_path.relative_to(root.resolve())
        pseudogt_path.relative_to(root.resolve())
    except ValueError as exc:
        raise BlueNegEvaluationError("BlueNeg payload path escapes dataset root") from exc
    preview = load_working_image(preview_path).pixels
    pseudogt = load_working_image(pseudogt_path).pixels
    return align_blueneg_pair(
        preview,
        pseudogt,
        transformation["bbox"],
        minimum_side=minimum_side,
    )


def _sample_flat(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    pixels = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    if len(pixels) < count:
        raise BlueNegEvaluationError(
            f"fixed pixel budget {count} exceeds aligned crop {len(pixels)}"
        )
    indices = np.random.default_rng(seed).choice(len(pixels), size=count, replace=False)
    return pixels[indices]


def _sample_aligned(
    source: np.ndarray,
    target: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_pixels = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target_pixels = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source_pixels.shape != target_pixels.shape or len(source_pixels) < count:
        raise BlueNegEvaluationError("aligned query does not satisfy fixed budget")
    indices = np.random.default_rng(seed).choice(
        len(source_pixels), size=count, replace=False
    )
    return source_pixels[indices], target_pixels[indices]


def load_development_roll_samples(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    transformations: dict[str, dict[str, np.ndarray]],
    roll_id: str,
    support_frames: int,
    support_pixels: int,
    query_frames: int,
    query_pixels: int,
    support_seed: int,
    query_seed: int,
    minimum_side: int,
) -> BlueNegRollSamples:
    """Decode only a development roll under the frozen support/query roles."""
    return _load_roll_samples(
        root=root,
        rows=rows,
        transformations=transformations,
        roll_id=roll_id,
        expected_pool="development_roll",
        support_frames=support_frames,
        support_pixels=support_pixels,
        query_frames=query_frames,
        query_pixels=query_pixels,
        support_seed=support_seed,
        query_seed=query_seed,
        minimum_side=minimum_side,
    )


def load_confirmatory_roll_samples(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    transformations: dict[str, dict[str, np.ndarray]],
    roll_id: str,
    support_frames: int,
    support_pixels: int,
    query_frames: int,
    query_pixels: int,
    support_seed: int,
    query_seed: int,
    minimum_side: int,
) -> BlueNegRollSamples:
    """Decode only a confirmatory roll after the family decision is frozen."""
    return _load_roll_samples(
        root=root,
        rows=rows,
        transformations=transformations,
        roll_id=roll_id,
        expected_pool="confirmatory_roll",
        support_frames=support_frames,
        support_pixels=support_pixels,
        query_frames=query_frames,
        query_pixels=query_pixels,
        support_seed=support_seed,
        query_seed=query_seed,
        minimum_side=minimum_side,
    )


def _load_roll_samples(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    transformations: dict[str, dict[str, np.ndarray]],
    roll_id: str,
    expected_pool: str,
    support_frames: int,
    support_pixels: int,
    query_frames: int,
    query_pixels: int,
    support_seed: int,
    query_seed: int,
    minimum_side: int,
) -> BlueNegRollSamples:
    roll_rows = [row for row in rows if str(row["roll_id"]) == roll_id]
    if not roll_rows:
        raise BlueNegEvaluationError(f"roll not found in frame manifest: {roll_id}")
    pools = {str(row["research_pool"]) for row in roll_rows}
    if pools != {expected_pool}:
        raise BlueNegEvaluationError(
            f"{expected_pool} loader rejects roll {roll_id}: {sorted(pools)}"
        )
    support_rows = _ordered(
        [row for row in roll_rows if row["frame_role"] == "unpaired_support"],
        support_seed,
        f"support:{roll_id}",
    )[:support_frames]
    query_rows = _ordered(
        [row for row in roll_rows if row["frame_role"] == "hidden_aligned_query"],
        query_seed,
        f"query:{roll_id}",
    )[:query_frames]
    if len(support_rows) != support_frames or len(query_rows) != query_frames:
        raise BlueNegEvaluationError(f"roll {roll_id} cannot satisfy fixed frame budget")
    support_source: list[np.ndarray] = []
    support_target: list[np.ndarray] = []
    for row in support_rows:
        filename = str(row["filename"])
        source, target = load_aligned_blueneg_pair(
            root,
            row,
            transformations[filename],
            minimum_side=minimum_side,
        )
        support_source.append(
            _sample_flat(source, support_pixels, _seed(support_seed, f"source:{filename}"))
        )
        support_target.append(
            _sample_flat(target, support_pixels, _seed(support_seed, f"target:{filename}"))
        )
    query_source: list[np.ndarray] = []
    query_target: list[np.ndarray] = []
    for row in query_rows:
        filename = str(row["filename"])
        source, target = load_aligned_blueneg_pair(
            root,
            row,
            transformations[filename],
            minimum_side=minimum_side,
        )
        sampled_source, sampled_target = _sample_aligned(
            source,
            target,
            query_pixels,
            _seed(query_seed, f"aligned:{filename}"),
        )
        query_source.append(sampled_source)
        query_target.append(sampled_target)
    return BlueNegRollSamples(
        roll_id=roll_id,
        support_content_ids=tuple(str(row["filename"]) for row in support_rows),
        query_content_ids=tuple(str(row["filename"]) for row in query_rows),
        support_source=np.concatenate(support_source, axis=0),
        support_target=np.concatenate(support_target, axis=0),
        query_source=np.asarray(query_source),
        query_target=np.asarray(query_target),
    )


def shuffled_target_support_groups(
    rolls: tuple[BlueNegRollSamples, BlueNegRollSamples],
    *,
    frames_per_roll: int,
    pixels_per_frame: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Reassign complete target-support frames between two rolls deterministically."""
    chunks = []
    ownership = []
    for roll in rolls:
        expected = frames_per_roll * pixels_per_frame
        if roll.support_target.shape != (expected, 3):
            raise BlueNegEvaluationError("support target does not match frozen frame budget")
        chunks.extend(np.split(roll.support_target, frames_per_roll))
        ownership.extend([roll.roll_id] * frames_per_roll)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chunks))
    first_owners = {ownership[int(index)] for index in order[:frames_per_roll]}
    if first_owners == {rolls[0].roll_id}:
        order = np.roll(order, 1)
    return {
        rolls[0].roll_id: np.concatenate([chunks[int(index)] for index in order[:frames_per_roll]]),
        rolls[1].roll_id: np.concatenate([chunks[int(index)] for index in order[frames_per_roll:]]),
    }
