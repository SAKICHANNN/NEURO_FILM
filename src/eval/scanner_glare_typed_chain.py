"""Experimental typed scanner chain with the retained multiscale glare operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.scanner import (
    ScannerProfile,
    apply_scanner_profile,
    apply_scanner_profile_row_tiled,
    compile_scanner_context,
)
from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    apply_scanner_glare,
    compile_scanner_glare_kernel,
)
from src.film_physics.scanner_glare_channel_serial_fft import (
    apply_scanner_glare_channel_serial_block_fft,
)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p6zg_scanner_glare_typed_chain_contract.v1"
    ):
        raise ValueError("unsupported U6.P6ZG contract")
    if payload["chain"]["stage_order"] != [
        "spectral",
        "p6za-multiscale-glare",
        "dmax",
        "mtf",
        "noise",
    ]:
        raise ValueError("typed scanner stage order drift")
    if (
        payload["chain"]["replace_existing_generic_flare"] is not True
        or payload["chain"]["combine_with_existing_generic_flare"] is not False
    ):
        raise ValueError("scanner glare replacement semantics drift")
    return payload


def scanner_profile(root: Path, contract: dict[str, Any]) -> ScannerProfile:
    source = json.loads(
        (root / contract["parents"]["scanner_profile_contract_path"]).read_text(
            encoding="utf-8"
        )
    )["profiles"][contract["chain"]["scanner_profile"]]
    dmax = source["dmax_density_rgb"]
    return ScannerProfile(
        profile_id=str(source["profile_id"]),
        illuminant_rgb=tuple(float(value) for value in source["illuminant_rgb"]),
        spectral_matrix=tuple(
            tuple(float(value) for value in row) for row in source["spectral_matrix"]
        ),
        local_flare_fraction=float(source["local_flare_fraction"]),
        global_flare_fraction=float(source["global_flare_fraction"]),
        flare_sigma_um=float(source["flare_sigma_um"]),
        dmax_density_rgb=None
        if dmax is None
        else tuple(float(value) for value in dmax),
        mtf_sigma_um_rgb=tuple(float(value) for value in source["mtf_sigma_um_rgb"]),
        shot_noise_variance_scale=float(source["shot_noise_variance_scale"]),
        read_noise_variance=float(source["read_noise_variance"]),
        seed=int(source["seed"]),
    )


def glare_profile() -> MultiscaleScannerGlareProfile:
    return MultiscaleScannerGlareProfile(
        components=(
            ScannerGlareComponent(weight=0.8, sigma_pixels=2.0),
            ScannerGlareComponent(weight=0.2, sigma_pixels=12.0),
        ),
        flare_fraction=0.08,
        truncate_sigma=6.0,
    )


def apply_typed_scanner_glare_chain(
    transmittance: np.ndarray,
    profile: ScannerProfile,
    *,
    pixel_pitch_um: float,
    glare_algorithm: str,
    glare_row_chunk: int,
) -> np.ndarray:
    """Apply spectral -> replacement glare -> Dmax -> MTF -> noise."""
    values = np.asarray(transmittance, dtype=np.float64)
    spectral = apply_scanner_profile(
        values, profile, pixel_pitch_um=pixel_pitch_um, stages=("spectral",)
    )
    glare = glare_profile()
    kernel = compile_scanner_glare_kernel(glare, kernel_size=177)
    if glare_algorithm == "full-fft":
        spread = apply_scanner_glare(
            spectral, kernel, flare_fraction=glare.flare_fraction
        )
    elif glare_algorithm == "channel-serial-block-fft":
        spread = apply_scanner_glare_channel_serial_block_fft(
            spectral,
            kernel,
            flare_fraction=glare.flare_fraction,
            row_chunk=glare_row_chunk,
        )
    else:
        raise ValueError("unsupported typed scanner glare algorithm")
    return apply_scanner_profile(
        spread,
        profile,
        pixel_pitch_um=pixel_pitch_um,
        stages=("dmax", "mtf", "noise"),
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
