from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from src.real_film.gold_proxy_alignment import (
    GoldProxyAlignmentError,
    audit_gold_proxy_alignment,
    safe_load_transformations,
)


def test_restricted_loader_accepts_numpy_transformations(tmp_path: Path) -> None:
    path = tmp_path / "transformations.pkl"
    path.write_bytes(pickle.dumps({"frame": {"matrix": np.eye(3), "bbox": (1, 2, 5, 6)}}, protocol=3))
    loaded = safe_load_transformations(path)
    assert loaded["frame"]["bbox"] == (1, 2, 5, 6)
    assert np.array_equal(loaded["frame"]["matrix"], np.eye(3))


def test_restricted_loader_rejects_non_numpy_global(tmp_path: Path) -> None:
    path = tmp_path / "malicious.pkl"
    path.write_bytes(pickle.dumps(eval))
    try:
        safe_load_transformations(path)
        raise AssertionError("expected forbidden pickle rejection")
    except GoldProxyAlignmentError as exc:
        assert "forbidden pickle" in str(exc)


def test_alignment_requires_exact_bbox_proxy_dimensions(tmp_path: Path) -> None:
    root = tmp_path / "pixels"
    preview_path = "negative-preview-8bit/blue/roll1-01.preview.png"
    proxy_path = "pseudogt-8bit/blue/roll1-01.pseudogt.png"
    for relative, size in ((preview_path, (10, 8)), (proxy_path, (6, 4))):
        path = root / Path(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size).save(path)
    rows = [
        {"film_stock_id": "gold", "frame_id": "roll1-01", "roll_id": "roll1", "lane": "negative_preview", "path": preview_path},
        {"film_stock_id": "gold", "frame_id": "roll1-01", "roll_id": "roll1", "lane": "display_proxy", "path": proxy_path},
    ]
    report = audit_gold_proxy_alignment(
        download_root=root,
        file_records=rows,
        transformations={"roll1-01": {"matrix": np.eye(3), "bbox": (2, 2, 8, 6)}},
        film_stock_id="gold",
        gates={"expected_pairs": 1, "minimum_rolls": 1, "minimum_pairs_each_roll": 1},
    )
    assert report["passed"] is True
    broken = {"roll1-01": {"matrix": np.eye(3), "bbox": (2, 2, 7, 6)}}
    try:
        audit_gold_proxy_alignment(
            download_root=root,
            file_records=rows,
            transformations=broken,
            film_stock_id="gold",
            gates={"expected_pairs": 1, "minimum_rolls": 1, "minimum_pairs_each_roll": 1},
        )
        raise AssertionError("expected bbox/proxy mismatch")
    except GoldProxyAlignmentError as exc:
        assert "dimensions differ" in str(exc)
