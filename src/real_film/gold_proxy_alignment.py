"""Safe BlueNeg transformation loading and Gold display-proxy alignment gates."""

from __future__ import annotations

import io
import pickle
import pickletools
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class GoldProxyAlignmentError(ValueError):
    """Raised when RF1.4B0 evidence or alignment fails closed."""


_ALLOWED_GLOBAL_NAMES = {
    "numpy.core.multiarray _reconstruct",
    "numpy._core.multiarray _reconstruct",
    "numpy ndarray",
    "numpy dtype",
}
_REQUIRED_GLOBAL_SETS = {
    frozenset({"numpy.core.multiarray _reconstruct", "numpy ndarray", "numpy dtype"}),
    frozenset({"numpy._core.multiarray _reconstruct", "numpy ndarray", "numpy dtype"}),
}
_FORBIDDEN_OPCODES = {
    "STACK_GLOBAL",
    "INST",
    "OBJ",
    "NEWOBJ",
    "NEWOBJ_EX",
    "EXT1",
    "EXT2",
    "EXT4",
    "PERSID",
    "BINPERSID",
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in {
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
        }:
            return np._core.multiarray._reconstruct  # type: ignore[attr-defined]
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")


def safe_load_transformations(path: Path) -> dict[str, dict[str, Any]]:
    """Load the pinned NumPy-only pickle after opcode and structure validation."""
    payload = path.read_bytes()
    globals_seen: set[str] = set()
    for opcode, argument, _ in pickletools.genops(payload):
        if opcode.name in _FORBIDDEN_OPCODES:
            raise GoldProxyAlignmentError(f"forbidden pickle opcode: {opcode.name}")
        if opcode.name == "GLOBAL":
            value = str(argument)
            globals_seen.add(value)
            if value not in _ALLOWED_GLOBAL_NAMES:
                raise GoldProxyAlignmentError(f"forbidden pickle global: {value}")
    if frozenset(globals_seen) not in _REQUIRED_GLOBAL_SETS:
        raise GoldProxyAlignmentError(
            f"unexpected NumPy global set: {sorted(globals_seen)!r}"
        )
    try:
        loaded = _RestrictedNumpyUnpickler(io.BytesIO(payload)).load()
    except (pickle.UnpicklingError, ValueError, TypeError) as exc:
        raise GoldProxyAlignmentError(f"restricted pickle load failed: {exc}") from exc
    if not isinstance(loaded, dict) or not loaded:
        raise GoldProxyAlignmentError("transformations root must be a non-empty dict")
    validated: dict[str, dict[str, Any]] = {}
    for key, value in loaded.items():
        if not isinstance(key, str) or not isinstance(value, dict) or set(value) != {"matrix", "bbox"}:
            raise GoldProxyAlignmentError("invalid transformation record structure")
        matrix = np.asarray(value["matrix"], dtype=np.float64)
        bbox = value["bbox"]
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise GoldProxyAlignmentError(f"invalid matrix for {key}")
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise GoldProxyAlignmentError(f"invalid bbox for {key}")
        coordinates = tuple(int(item) for item in bbox)
        if any(float(item) != float(converted) for item, converted in zip(bbox, coordinates, strict=True)):
            raise GoldProxyAlignmentError(f"non-integer bbox for {key}")
        validated[key] = {"matrix": matrix, "bbox": coordinates}
    return validated


def audit_gold_proxy_alignment(
    *,
    download_root: Path,
    file_records: Sequence[Mapping[str, Any]],
    transformations: Mapping[str, Mapping[str, Any]],
    film_stock_id: str,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate official bboxes against every frozen preview/proxy sibling."""
    by_frame: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in file_records:
        if str(row.get("film_stock_id")) != film_stock_id:
            continue
        lane = str(row.get("lane"))
        if lane in by_frame[str(row["frame_id"])]:
            raise GoldProxyAlignmentError(f"duplicate frame/lane: {row['frame_id']} {lane}")
        by_frame[str(row["frame_id"])][lane] = row
    paired = {
        frame_id: lanes
        for frame_id, lanes in by_frame.items()
        if "display_proxy" in lanes
    }
    if len(paired) != int(gates["expected_pairs"]):
        raise GoldProxyAlignmentError(
            f"pair count mismatch: {len(paired)} != {gates['expected_pairs']}"
        )
    pair_records: list[dict[str, Any]] = []
    roll_counts: Counter[str] = Counter()
    for frame_id, lanes in sorted(paired.items()):
        if set(lanes) != {"negative_preview", "display_proxy"}:
            raise GoldProxyAlignmentError(f"incomplete preview/proxy sibling: {frame_id}")
        if frame_id not in transformations:
            raise GoldProxyAlignmentError(f"missing official transformation: {frame_id}")
        preview_row = lanes["negative_preview"]
        proxy_row = lanes["display_proxy"]
        if preview_row["roll_id"] != proxy_row["roll_id"]:
            raise GoldProxyAlignmentError(f"sibling roll mismatch: {frame_id}")
        preview_path = download_root / Path(*Path(str(preview_row["path"])).parts)
        proxy_path = download_root / Path(*Path(str(proxy_row["path"])).parts)
        with Image.open(preview_path) as preview_image:
            preview_image.verify()
            preview_size = preview_image.size
        with Image.open(proxy_path) as proxy_image:
            proxy_image.verify()
            proxy_size = proxy_image.size
        transformation = transformations[frame_id]
        matrix = np.asarray(transformation["matrix"], dtype=np.float64)
        determinant = float(np.linalg.det(matrix))
        if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
            raise GoldProxyAlignmentError(f"singular transformation: {frame_id}")
        x0, y0, x1, y1 = (int(item) for item in transformation["bbox"])
        bbox_in_bounds = 0 <= x0 < x1 <= preview_size[0] and 0 <= y0 < y1 <= preview_size[1]
        if not bbox_in_bounds:
            raise GoldProxyAlignmentError(f"bbox outside preview: {frame_id}")
        bbox_size = (x1 - x0, y1 - y0)
        if bbox_size != proxy_size:
            raise GoldProxyAlignmentError(
                f"bbox/proxy dimensions differ for {frame_id}: {bbox_size} != {proxy_size}"
            )
        roll_id = str(preview_row["roll_id"])
        roll_counts[roll_id] += 1
        pair_records.append(
            {
                "frame_id": frame_id,
                "roll_id": roll_id,
                "preview_path": str(preview_row["path"]),
                "proxy_path": str(proxy_row["path"]),
                "preview_size": list(preview_size),
                "proxy_size": list(proxy_size),
                "bbox": [x0, y0, x1, y1],
                "matrix_determinant": determinant,
            }
        )
    if len(roll_counts) < int(gates["minimum_rolls"]):
        raise GoldProxyAlignmentError("too few paired physical rolls")
    minimum_each = int(gates["minimum_pairs_each_roll"])
    if any(count < minimum_each for count in roll_counts.values()):
        raise GoldProxyAlignmentError(
            f"under-supported paired roll: {dict(sorted(roll_counts.items()))}"
        )
    return {
        "passed": True,
        "paired_frames": len(pair_records),
        "paired_rolls": len(roll_counts),
        "pairs_by_roll": dict(sorted(roll_counts.items())),
        "all_official_transform_keys_present": True,
        "all_matrices_finite_nonsingular": True,
        "all_bboxes_in_bounds": True,
        "all_bbox_proxy_dimensions_equal": True,
        "colour_fit_performed": False,
        "pair_records": pair_records,
    }
