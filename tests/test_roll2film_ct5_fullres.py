from __future__ import annotations

import numpy as np

from src.roll2film.ct5_fullres import full_resolution_diagnostics, linear_to_u8


def test_full_resolution_diagnostics_flag_raw_excursions_without_nonfinite_values() -> None:
    source = np.full((32, 48, 3), 0.4, dtype=np.float64)
    target = source * np.array([1.05, 0.98, 1.02])
    output = target.copy()
    output[0, 0] = np.array([1.2, -0.1, 0.5])
    metrics = full_resolution_diagnostics(source, target, output)

    assert metrics["raw_out_of_range_fraction"] > 0.0
    assert metrics["raw_out_of_range_pixel_fraction"] > 0.0
    assert metrics["mean_delta_e00_to_target"] > 0.0
    assert linear_to_u8(output).dtype == np.uint8
