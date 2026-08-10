"""Lifetime-bounded downstream composition for the experimental glare chain."""

from __future__ import annotations

import numpy as np

from src.eval.scanner_glare_typed_chain import glare_profile
from src.film_physics.scanner import (
    ScannerProfile,
    apply_scanner_profile,
    apply_scanner_profile_row_tiled,
    compile_scanner_context,
)
from src.film_physics.scanner_glare import compile_scanner_glare_kernel
from src.film_physics.scanner_glare_channel_serial_fft import (
    apply_scanner_glare_channel_serial_block_fft,
)


def apply_typed_scanner_glare_chain_tiled_downstream(
    transmittance: np.ndarray,
    profile: ScannerProfile,
    *,
    pixel_pitch_um: float,
    glare_row_chunk: int,
    downstream_tile_rows: int,
) -> np.ndarray:
    """Use retained glare then the existing halo-aware tiled scanner downstream."""
    values = np.asarray(transmittance, dtype=np.float64)
    spectral = apply_scanner_profile(
        values, profile, pixel_pitch_um=pixel_pitch_um, stages=("spectral",)
    )
    glare = glare_profile()
    kernel = compile_scanner_glare_kernel(glare, kernel_size=177)
    spread = apply_scanner_glare_channel_serial_block_fft(
        spectral,
        kernel,
        flare_fraction=glare.flare_fraction,
        row_chunk=glare_row_chunk,
    )
    del spectral
    downstream_stages = ("dmax", "mtf", "noise")
    context = compile_scanner_context(spread, profile, stages=downstream_stages)
    return apply_scanner_profile_row_tiled(
        spread,
        profile,
        pixel_pitch_um=pixel_pitch_um,
        context=context,
        tile_rows=downstream_tile_rows,
        stages=downstream_stages,
    )
