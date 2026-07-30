"""Shared deterministic procedural FilmFX staging primitives."""

from __future__ import annotations

from pathlib import Path

from src.filmfx import (
    composite_layers,
    dust_scratch_layer,
    grain_residual_layer,
    halation_layer,
)
from src.preprocess import load_working_image, working_image_to_srgb_float

from .composition import FilmEffectBinding
from .contracts import ReferenceMatchContractError
from .files import _encode_srgb


def signed_seed(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < -(2**31)
        or value > 2**31 - 1
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a signed 32-bit integer"
        )
    return value


def validate_procedural_filmfx(effects: FilmEffectBinding) -> None:
    if effects.halation_model != "simple":
        raise ReferenceMatchContractError(
            "physical FilmFX requires separately bound resolved controls"
        )
    if effects.grain <= 0.0 and effects.halation <= 0.0 and effects.dust <= 0.0:
        raise ReferenceMatchContractError(
            "FilmFX execution requires at least one active effect"
        )


def protect_staging_inputs(
    outputs: tuple[Path, ...],
    report: Path,
    *,
    input_paths: tuple[Path, ...],
    input_report: Path,
    protected_label: str,
) -> None:
    protected = {
        str(path.resolve(strict=True)).casefold() for path in input_paths
    }
    protected.add(str(input_report.resolve(strict=True)).casefold())
    destinations = {
        str(path.resolve(strict=False)).casefold() for path in outputs
    }
    destinations.add(str(report.resolve(strict=False)).casefold())
    if protected & destinations:
        raise ReferenceMatchContractError(
            f"FilmFX destinations must not overwrite {protected_label} "
            "staging artifacts"
        )


def render_procedural_filmfx(
    *,
    source_path: Path,
    destination: Path,
    effects: FilmEffectBinding,
    base_seed: int,
    source_index: int,
    output_bit_depth: int,
) -> tuple[str, int, int]:
    working = load_working_image(source_path)
    if (
        working.working_space != "linear_srgb"
        or working.transfer_state != "display_linear"
        or working.alpha_policy != "absent"
    ):
        raise ReferenceMatchContractError(
            "FilmFX execution requires alpha-free display-linear "
            "linear-sRGB staging input"
        )
    base = working_image_to_srgb_float(working)
    row_seed = signed_seed(base_seed + source_index * 1009, "row_seed")
    dust_seed = signed_seed(row_seed + 17, "dust_seed")
    layers = []
    if effects.grain > 0.0:
        layers.append(
            grain_residual_layer(
                base,
                strength=effects.grain,
                seed=row_seed,
                color=effects.interpretation != "bw_developer_scan",
            )
        )
    if effects.halation > 0.0:
        layers.append(halation_layer(base, strength=effects.halation))
    if effects.dust > 0.0:
        layers.append(
            dust_scratch_layer(
                base.shape,
                strength=effects.dust,
                seed=dust_seed,
            )
        )
    rendered = composite_layers(base, layers, output_margin=4)
    output_format, _clipped_fraction = _encode_srgb(
        rendered,
        destination,
        output_bit_depth=output_bit_depth,
    )
    return output_format, row_seed, dust_seed


__all__ = [
    "protect_staging_inputs",
    "render_procedural_filmfx",
    "signed_seed",
    "validate_procedural_filmfx",
]
