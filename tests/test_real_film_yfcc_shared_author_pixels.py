from __future__ import annotations

import pytest

from src.real_film.yfcc_shared_author_pixels import (
    YfccSharedAuthorPixelError,
    evaluate_shared_author_pixel_download,
)


def _config() -> dict:
    return {
        "pilot_id": "test",
        "stock_ids": ["left", "right"],
        "selection": {"maximum_retained_files": 4},
        "download_limits": {"maximum_bytes_total": 100, "maximum_bytes_per_file": 50},
        "pixel_gate": {"minimum_retained_files_per_stock": 2, "minimum_usable_shared_authors": 2},
        "claim_ceiling": "test",
    }


def _row(photoid: int, uid: str, stock: str) -> dict:
    return {
        "photoid": photoid,
        "author_uid": uid,
        "film_stock_id": stock,
        "bytes": 10,
    }


def test_combined_download_passes_only_bilateral_author_gate() -> None:
    results = {
        "left": {"rows": [_row(1, "u1", "left"), _row(2, "u2", "left")], "attempts": []},
        "right": {"rows": [_row(3, "u1", "right"), _row(4, "u2", "right")], "attempts": []},
    }
    report = evaluate_shared_author_pixel_download(results, _config())
    assert report["acquisition_gate_passed"] is True
    assert report["usable_shared_author_uids"] == ["u1", "u2"]
    assert report["operator_fitting_allowed"] is False
    assert report["training_allowed"] is False


def test_combined_download_rejects_duplicate_photoid() -> None:
    results = {
        "left": {"rows": [_row(1, "u1", "left")], "attempts": []},
        "right": {"rows": [_row(1, "u1", "right")], "attempts": []},
    }
    with pytest.raises(YfccSharedAuthorPixelError, match="duplicate photoid"):
        evaluate_shared_author_pixel_download(results, _config())
