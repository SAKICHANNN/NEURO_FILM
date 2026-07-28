"""Offline marked-Poisson developed-image structure reference.

Colour material accumulates dye-cloud optical density. B&W material forms a
Boolean metallic-silver opacity field. Neither branch adds noise to display
RGB; both stop in an explicit physical domain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)


@dataclass(frozen=True)
class DevelopedStructureContext:
    material: str
    input_shape: tuple[int, int]
    output_zoom: int
    output_pixel_pitch_um: float
    centers_by_layer: tuple[np.ndarray, ...]
    radius_input_pixels: tuple[float, ...]
    mark_optical_density: tuple[float, ...]
    monte_carlo_offsets_yx: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        if self.material not in {"colour-dye-cloud", "bw-metallic-silver"}:
            raise ValueError("unsupported developed structure material")
        if (
            len(self.input_shape) != 2
            or any(not isinstance(value, int) or value <= 0 for value in self.input_shape)
        ):
            raise ValueError("input_shape must contain two positive integers")
        if not isinstance(self.output_zoom, int) or self.output_zoom <= 0:
            raise ValueError("output_zoom must be a positive integer")
        PhysicalScale(self.output_pixel_pitch_um)
        centers = tuple(np.asarray(value, dtype=np.float64) for value in self.centers_by_layer)
        expected_layers = 3 if self.material == "colour-dye-cloud" else 1
        if len(centers) != expected_layers:
            raise ValueError("material has the wrong layer count")
        owned_centers = []
        for layer in centers:
            if (
                layer.ndim != 2
                or layer.shape[1:] != (2,)
                or not np.all(np.isfinite(layer))
                or np.any(layer < 0.0)
                or np.any(layer[:, 0] >= self.input_shape[0])
                or np.any(layer[:, 1] >= self.input_shape[1])
            ):
                raise ValueError("structure centers must be finite in-bounds y/x rows")
            owned = np.array(layer, copy=True)
            owned.setflags(write=False)
            owned_centers.append(owned)
        radii = tuple(float(value) for value in self.radius_input_pixels)
        marks = tuple(float(value) for value in self.mark_optical_density)
        if len(radii) != expected_layers or len(marks) != expected_layers:
            raise ValueError("material radius/mark counts must match layers")
        if any(not math.isfinite(value) or value <= 0.0 for value in radii + marks):
            raise ValueError("material radii and marks must be finite and positive")
        offsets = np.asarray(self.monte_carlo_offsets_yx, dtype=np.float64)
        if (
            offsets.ndim != 2
            or offsets.shape[1:] != (2,)
            or len(offsets) == 0
            or not np.all(np.isfinite(offsets))
            or np.any(offsets < -0.5)
            or np.any(offsets > 0.5)
        ):
            raise ValueError("Monte Carlo offsets must be finite within half a pixel")
        owned_offsets = np.array(offsets, copy=True)
        owned_offsets.setflags(write=False)
        object.__setattr__(self, "centers_by_layer", tuple(owned_centers))
        object.__setattr__(self, "radius_input_pixels", radii)
        object.__setattr__(self, "mark_optical_density", marks)
        object.__setattr__(self, "monte_carlo_offsets_yx", owned_offsets)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (
            self.input_shape[0] * self.output_zoom,
            self.input_shape[1] * self.output_zoom,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.material.encode("ascii"))
        digest.update(np.asarray(self.input_shape, dtype="<u8").tobytes())
        digest.update(np.asarray([self.output_zoom], dtype="<u8").tobytes())
        digest.update(np.asarray([self.output_pixel_pitch_um], dtype="<f8").tobytes())
        for values in self.centers_by_layer:
            digest.update(np.asarray(values, dtype="<f8").tobytes())
        digest.update(np.asarray(self.radius_input_pixels, dtype="<f8").tobytes())
        digest.update(np.asarray(self.mark_optical_density, dtype="<f8").tobytes())
        digest.update(np.asarray(self.monte_carlo_offsets_yx, dtype="<f8").tobytes())
        digest.update(np.asarray([self.seed], dtype="<u8").tobytes())
        return digest.hexdigest()


def _sample_centers(
    counts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    total = int(np.sum(counts, dtype=np.int64))
    centers = np.empty((total, 2), dtype=np.float64)
    cursor = 0
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            count = int(counts[row, column])
            if count:
                centers[cursor : cursor + count, 0] = row + rng.random(count)
                centers[cursor : cursor + count, 1] = column + rng.random(count)
                cursor += count
    return centers


def build_colour_dye_cloud_context(
    target_density: np.ndarray,
    *,
    radius_um_cmy: tuple[float, float, float],
    mark_optical_density_cmy: tuple[float, float, float],
    output_zoom: int,
    output_pixel_pitch_um: float,
    monte_carlo_samples: int,
    seed: int,
) -> DevelopedStructureContext:
    density = np.asarray(target_density, dtype=np.float64)
    if (
        density.ndim != 3
        or density.shape[-1] != 3
        or not density.size
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
    ):
        raise ValueError("target dye density must be finite nonnegative HxWx3")
    if (
        isinstance(monte_carlo_samples, bool)
        or not isinstance(monte_carlo_samples, int)
        or monte_carlo_samples <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("samples and seed must be nonnegative")
    PhysicalScale(output_pixel_pitch_um)
    if not isinstance(output_zoom, int) or isinstance(output_zoom, bool) or output_zoom <= 0:
        raise ValueError("output_zoom must be a positive integer")
    radii_um = tuple(float(value) for value in radius_um_cmy)
    marks = tuple(float(value) for value in mark_optical_density_cmy)
    if len(radii_um) != 3 or len(marks) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in radii_um + marks
    ):
        raise ValueError("colour radii and marks must be three finite positive values")
    input_pitch_um = output_zoom * float(output_pixel_pitch_um)
    radii_input = tuple(value / input_pitch_um for value in radii_um)
    sequence = np.random.SeedSequence(seed)
    children = sequence.spawn(4)
    centers = []
    for layer in range(3):
        rng = np.random.default_rng(children[layer])
        area = math.pi * radii_input[layer] ** 2
        intensity = density[..., layer] / (area * marks[layer])
        centers.append(_sample_centers(rng.poisson(intensity), rng))
    offset_rng = np.random.default_rng(children[3])
    offsets = offset_rng.uniform(-0.5, 0.5, size=(monte_carlo_samples, 2))
    return DevelopedStructureContext(
        "colour-dye-cloud",
        density.shape[:2],
        int(output_zoom),
        float(output_pixel_pitch_um),
        tuple(centers),
        radii_input,
        marks,
        offsets,
        int(seed),
    )


def build_bw_silver_context(
    target_density: np.ndarray,
    *,
    radius_um: float,
    output_zoom: int,
    output_pixel_pitch_um: float,
    monte_carlo_samples: int,
    seed: int,
) -> DevelopedStructureContext:
    density = np.asarray(target_density, dtype=np.float64)
    if (
        density.ndim != 2
        or not density.size
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
    ):
        raise ValueError("target silver density must be finite nonnegative 2D")
    if (
        isinstance(monte_carlo_samples, bool)
        or not isinstance(monte_carlo_samples, int)
        or monte_carlo_samples <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("samples and seed must be nonnegative")
    PhysicalScale(output_pixel_pitch_um)
    if not isinstance(output_zoom, int) or isinstance(output_zoom, bool) or output_zoom <= 0:
        raise ValueError("output_zoom must be a positive integer")
    if not math.isfinite(radius_um) or radius_um <= 0.0:
        raise ValueError("silver radius_um must be finite and positive")
    input_pitch_um = output_zoom * float(output_pixel_pitch_um)
    radius_input = float(radius_um) / input_pitch_um
    area = math.pi * radius_input**2
    intensity = density * math.log(10.0) / area
    sequence = np.random.SeedSequence(seed)
    center_seed, offset_seed = sequence.spawn(2)
    rng = np.random.default_rng(center_seed)
    centers = _sample_centers(rng.poisson(intensity), rng)
    offsets = np.random.default_rng(offset_seed).uniform(
        -0.5, 0.5, size=(monte_carlo_samples, 2)
    )
    return DevelopedStructureContext(
        "bw-metallic-silver",
        density.shape,
        int(output_zoom),
        float(output_pixel_pitch_um),
        (centers,),
        (radius_input,),
        (1.0,),
        offsets,
        int(seed),
    )


def _render_layer_region(
    context: DevelopedStructureContext,
    *,
    layer: int,
    output_origin_yx: tuple[int, int],
    output_shape: tuple[int, int],
    boolean: bool,
) -> np.ndarray:
    origin_y, origin_x = output_origin_yx
    height, width = output_shape
    full_height, full_width = context.output_shape
    if not (
        0 <= origin_y < origin_y + height <= full_height
        and 0 <= origin_x < origin_x + width <= full_width
    ):
        raise ValueError("developed structure region is outside the output")
    accumulated = np.zeros((height, width), dtype=np.float32)
    centers = context.centers_by_layer[layer]
    radius = context.radius_input_pixels[layer] * context.output_zoom
    mark = np.float32(context.mark_optical_density[layer])
    for offset_y, offset_x in context.monte_carlo_offsets_yx:
        sample = (
            np.zeros((height, width), dtype=bool)
            if boolean
            else np.zeros((height, width), dtype=np.float32)
        )
        for center_y, center_x in centers:
            cy = context.output_zoom * center_y + offset_y
            cx = context.output_zoom * center_x + offset_x
            y0 = max(0, int(math.ceil(cy - radius - origin_y)))
            y1 = min(height - 1, int(math.floor(cy + radius - origin_y)))
            x0 = max(0, int(math.ceil(cx - radius - origin_x)))
            x1 = min(width - 1, int(math.floor(cx + radius - origin_x)))
            if y0 > y1 or x0 > x1:
                continue
            ys = np.arange(y0, y1 + 1, dtype=np.float64) + origin_y
            xs = np.arange(x0, x1 + 1, dtype=np.float64) + origin_x
            mask = (ys[:, None] - cy) ** 2 + (xs[None, :] - cx) ** 2 < radius**2
            target = sample[y0 : y1 + 1, x0 : x1 + 1]
            if boolean:
                target |= mask
            else:
                target += mask.astype(np.float32) * mark
        accumulated += sample
    return accumulated / np.float32(len(context.monte_carlo_offsets_yx))


def render_developed_structure_region(
    context: DevelopedStructureContext,
    *,
    output_origin_yx: tuple[int, int],
    output_shape: tuple[int, int],
) -> PhysicalDomainArray:
    if context.material == "colour-dye-cloud":
        layers = [
            _render_layer_region(
                context,
                layer=layer,
                output_origin_yx=output_origin_yx,
                output_shape=output_shape,
                boolean=False,
            )
            for layer in range(3)
        ]
        values = np.stack(layers, axis=-1)
        return PhysicalDomainArray(
            values,
            PhysicalDomain.DEVELOPED_DENSITY,
            PhysicalUnit.OPTICAL_DENSITY,
            ("cyan-dye", "magenta-dye", "yellow-dye"),
            PhysicalScale(context.output_pixel_pitch_um),
        )
    covered = _render_layer_region(
        context,
        layer=0,
        output_origin_yx=output_origin_yx,
        output_shape=output_shape,
        boolean=True,
    )
    transmission = np.maximum(1.0 - covered, np.finfo(np.float32).tiny)
    values = np.repeat(transmission[..., None], 3, axis=2)
    return PhysicalDomainArray(
        values,
        PhysicalDomain.TRANSMITTANCE,
        PhysicalUnit.TRANSMITTANCE,
        ("neutral-r", "neutral-g", "neutral-b"),
        PhysicalScale(context.output_pixel_pitch_um),
    )


def render_developed_structure(
    context: DevelopedStructureContext,
) -> PhysicalDomainArray:
    return render_developed_structure_region(
        context, output_origin_yx=(0, 0), output_shape=context.output_shape
    )
