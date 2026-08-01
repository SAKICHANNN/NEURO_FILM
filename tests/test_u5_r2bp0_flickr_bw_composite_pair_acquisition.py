from __future__ import annotations

import numpy as np
import pytest

from src.eval.flickr_bw_composite_pair_acquisition import (
    FlickrBwCompositeAcquisitionError,
    split_composite,
)


CONTRACT = {
    "white_threshold": 248,
    "minimum_white_row_fraction": 0.95,
    "central_search_start": 450,
    "central_search_end": 575,
    "minimum_separator_rows": 10,
    "minimum_content_row_fraction": 0.02,
    "minimum_content_column_fraction": 0.02,
    "minimum_crop_width": 700,
    "minimum_crop_height": 450,
}


def _composite() -> np.ndarray:
    image = np.full((1024, 765, 3), 255, dtype=np.uint8)
    y, x = np.mgrid[:496, :749]
    film = np.stack(((x + y) % 240, (2 * x + y) % 240, (x + 2 * y) % 240), axis=2).astype(
        np.uint8
    )
    digital = np.flip(film, axis=1).copy()
    image[8:504, 8:757] = film
    image[522:1018, 8:757] = digital
    return image


def test_split_composite_finds_exact_roles() -> None:
    film, digital, diagnostics = split_composite(_composite(), CONTRACT)
    assert film.shape == (496, 749, 3)
    assert digital.shape == (496, 749, 3)
    assert diagnostics["separator_rows_inclusive"] == [504, 521]
    assert np.array_equal(film[:, ::-1], digital)


def test_split_composite_rejects_missing_separator() -> None:
    image = _composite()
    image[504:522] = 0
    with pytest.raises(FlickrBwCompositeAcquisitionError, match="separator"):
        split_composite(image, CONTRACT)
