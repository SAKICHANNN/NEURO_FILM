from __future__ import annotations

import numpy as np
from PIL import Image

from src.eval import scorpion_cross_scanner_coherence_d0 as subject


def test_coherence_prefers_shared_structure() -> None:
    rng = np.random.default_rng(7)
    shared = rng.normal(size=(768, 768))
    other = rng.normal(size=(768, 768))
    analysis = {
        "crop_size": 768,
        "window_size": 128,
        "window_stride": 128,
        "frequency_min_cycles_per_pixel": 0.03125,
        "frequency_max_cycles_per_pixel": 0.25,
    }
    a = subject._spectra(shared + 0.05 * rng.normal(size=shared.shape), analysis)
    b = subject._spectra(shared + 0.05 * rng.normal(size=shared.shape), analysis)
    wrong = subject._spectra(other, analysis)
    assert subject._coherence(a, b, analysis) > 10.0 * subject._coherence(
        a, wrong, analysis
    )


def test_central_directory_parser_rejects_trailing_bytes() -> None:
    try:
        subject._central_rows(b"not-a-central-directory")
    except ValueError as exc:
        assert "parse drift" in str(exc)
    else:
        raise AssertionError("invalid central directory accepted")


def test_rank_luminance_uses_true_midranks(tmp_path) -> None:
    pixels = np.zeros((1024, 1024, 3), dtype=np.uint8)
    pixels[:, 512:] = 255
    path = tmp_path / "two_values.png"
    Image.fromarray(pixels).save(path)
    ranked = subject._rank_luminance(path)
    assert np.unique(ranked).tolist() == [0.25, 0.75]
