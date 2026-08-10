"""Counter-addressed temporally independent grain innovations."""

from __future__ import annotations

import numpy as np

from src.film_physics.contracts import coordinate_counter_u64
from src.film_physics.structure_compiler import counter_normal_region


def temporal_grain_innovation_region(
    *,
    profile_sha256: str,
    seed: int,
    frame: int,
    layer: int,
    full_shape: tuple[int, int],
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Return one frame/layer innovation region with exact tile semantics."""
    stream_seed = coordinate_counter_u64(
        profile_sha256=profile_sha256,
        seed=seed,
        frame=frame,
        x=0,
        y=0,
        layer=layer,
    )
    with np.errstate(over="ignore"):
        return counter_normal_region(
            full_shape,
            origin_yx=origin_yx,
            shape=shape,
            seed=stream_seed,
        )


def temporal_grain_innovation_frame(
    *,
    profile_sha256: str,
    seed: int,
    frame: int,
    full_shape: tuple[int, int],
    layer_count: int,
) -> np.ndarray:
    """Return an HWC stack of independent frame/layer innovations."""
    if not isinstance(layer_count, int) or layer_count < 1:
        raise ValueError("layer count must be positive")
    return np.stack(
        [
            temporal_grain_innovation_region(
                profile_sha256=profile_sha256,
                seed=seed,
                frame=frame,
                layer=layer,
                full_shape=full_shape,
                origin_yx=(0, 0),
                shape=full_shape,
            )
            for layer in range(layer_count)
        ],
        axis=-1,
    )
