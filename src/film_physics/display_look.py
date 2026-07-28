"""Artifact-backed AO6 display-look reference implementation.

This is a deterministic colour component, not a calibrated film response.  It
exists so the physical profile consumer does not need to reopen the experiment
configuration tree after a profile has been compiled.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import (
    _style_transfer_rgb_with_context,
    apply_output_margin,
    build_safe_lab_source_context,
)
from src.color_engine.safe_lab_rgb_context import (
    style_transfer_rgb_with_source_context,
)
from src.color_engine.safe_lab import (
    SafeLabSourceContext,
    safe_lab_context_from_lab,
    validate_safe_lab_source_context,
)
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.roll2film.density_domain import (
    DensityDomainNegativePrintOperator,
)
from src.roll2film.factorized_boundary_guard import (
    apply_factorized_boundary_guard,
)
from src.roll2film.positive_film import PositiveFilmResponseOperator


DISPLAY_LOOK_SCHEMA = (
    "neuro_film.ao6_source_context_display_look_component.v2"
)


def make_display_look_payload(
    *,
    density_operator: DensityDomainNegativePrintOperator,
    anchor_stats: dict[str, Any],
    anchor: dict[str, Any],
    base: dict[str, Any],
    residual_operator: PositiveFilmResponseOperator,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Freeze only the parameters required by the canonical reference kernel."""

    guard = candidate["factorization"]
    payload = {
        "schema": DISPLAY_LOOK_SCHEMA,
        "base": {
            "order": base["order"],
            "density_strength": float(base["density_strength"]),
            "final_output_margin": int(base["final_output_margin"]),
            "density_operator": density_operator.to_dict(),
        },
        "anchor": {
            "stats": anchor_stats,
            "style": str(anchor["style"]),
            "strength": float(anchor["strength"]),
            "luma_strength": float(anchor["luma_strength"]),
            "grain": float(anchor["grain"]),
            "seed": int(anchor["seed"]),
            "gamut_mode": str(anchor["gamut_mode"]),
        },
        "residual": {
            "operator": residual_operator.to_dict(),
            "tone_strength": float(candidate["tone_strength"]),
            "chroma_strength": float(candidate["chroma_strength"]),
            "luma_weights": [
                float(value) for value in guard["luma_weights"]
            ],
            "hard_boundary_epsilon_encoded_srgb": float(
                guard["hard_boundary_epsilon_encoded_srgb"]
            ),
            "guard_boundary_epsilon_encoded_srgb": float(
                guard["guard_boundary_epsilon_encoded_srgb"]
            ),
            "hard_clip_allowed": bool(guard["hard_clip_allowed"]),
        },
        "source_context_scope": "one-full-frame",
        "claim_ceiling": (
            "AO6-like deterministic display-look approximation; not stock, "
            "exposure, emulsion, process or scanner truth"
        ),
    }
    validate_display_look_payload(payload)
    return payload


def validate_display_look_payload(payload: dict[str, Any]) -> None:
    expected = {
        "schema",
        "base",
        "anchor",
        "residual",
        "source_context_scope",
        "claim_ceiling",
    }
    if set(payload) != expected or payload.get("schema") != DISPLAY_LOOK_SCHEMA:
        raise ValueError("display-look component fields drift")
    base = payload["base"]
    anchor = payload["anchor"]
    residual = payload["residual"]
    if (
        set(base)
        != {
            "order",
            "density_strength",
            "final_output_margin",
            "density_operator",
        }
        or base["order"] != "density_then_anchor"
        or float(base["density_strength"]) != 0.5
        or int(base["final_output_margin"]) != 4
        or set(anchor)
        != {
            "stats",
            "style",
            "strength",
            "luma_strength",
            "grain",
            "seed",
            "gamut_mode",
        }
        or set(residual)
        != {
            "operator",
            "tone_strength",
            "chroma_strength",
            "luma_weights",
            "hard_boundary_epsilon_encoded_srgb",
            "guard_boundary_epsilon_encoded_srgb",
            "hard_clip_allowed",
        }
        or float(residual["tone_strength"]) != 0.15
        or float(residual["chroma_strength"]) != 0.35
        or residual["hard_clip_allowed"]
        or payload["source_context_scope"] != "one-full-frame"
    ):
        raise ValueError("unsupported display-look component")
    DensityDomainNegativePrintOperator.from_dict(base["density_operator"])
    PositiveFilmResponseOperator.from_dict(residual["operator"])
    weights = np.asarray(residual["luma_weights"], dtype=np.float64)
    if (
        weights.shape != (3,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or abs(float(np.sum(weights)) - 1.0) > 1e-12
        or not isinstance(anchor["stats"], dict)
        or not anchor["stats"]
    ):
        raise ValueError("invalid display-look parameters")


def build_source_context_display_look(
    payload: dict[str, Any], source: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Build one full-frame AO6 display-look callable from artifact bytes."""

    apply_base, apply_residual = build_source_context_display_look_stages(
        payload, source
    )

    def apply(encoded: np.ndarray) -> np.ndarray:
        return apply_residual(apply_base(encoded))

    return apply


def build_source_context_display_look_stages(
    payload: dict[str, Any], source: np.ndarray
) -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """Expose the exact base/residual split for profiling and native ports."""

    validate_display_look_payload(payload)
    base = payload["base"]
    anchor = payload["anchor"]
    residual = payload["residual"]
    density_operator = DensityDomainNegativePrintOperator.from_dict(
        base["density_operator"]
    )
    residual_operator = PositiveFilmResponseOperator.from_dict(
        residual["operator"]
    )

    def apply_density(encoded: np.ndarray) -> np.ndarray:
        linear = encoded_srgb_to_linear(
            np.asarray(encoded, dtype=np.float64)
        )
        return linear_srgb_to_encoded(
            density_operator.apply(
                linear, strength=float(base["density_strength"])
            )
        )

    source_value = np.asarray(source, dtype=np.float32)
    source_shape = source_value.shape
    source_context = build_safe_lab_source_context(
        apply_density(source_value)
    )
    del source_value

    def apply_base(encoded: np.ndarray) -> np.ndarray:
        encoded_value = np.asarray(encoded, dtype=np.float32)
        base_output = style_transfer_rgb_with_source_context(
            np.asarray(apply_density(encoded_value), dtype=np.float32),
            anchor["stats"],
            str(anchor["style"]),
            float(anchor["strength"]),
            float(anchor["luma_strength"]),
            float(anchor["grain"]),
            int(anchor["seed"]),
            True,
            gamut_mode=str(anchor["gamut_mode"]),
            output_margin=0,
            source_context=source_context,
        )
        base_output = apply_output_margin(
            np.asarray(base_output, dtype=np.float64),
            int(base["final_output_margin"]),
        )
        base_output = np.asarray(base_output, dtype=np.float32)
        return base_output

    def apply_residual(base_output: np.ndarray) -> np.ndarray:
        result = apply_factorized_boundary_guard(
            residual_operator,
            encoded_srgb_to_linear(base_output),
            tone_strength=float(residual["tone_strength"]),
            chroma_strength=float(residual["chroma_strength"]),
            luma_weights=np.asarray(
                residual["luma_weights"], dtype=np.float64
            ),
            hard_boundary_epsilon_encoded_srgb=float(
                residual["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                residual["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        output = linear_srgb_to_encoded(result.output)
        if (
            output.shape != source_shape
            or not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError("display-look component left encoded RGB")
        return output

    return apply_base, apply_residual


def build_source_context_display_look_row_streamed(
    payload: dict[str, Any],
    source: np.ndarray,
    *,
    tile_rows: int,
    reuse_input_buffer: bool = False,
    source_context: SafeLabSourceContext | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the same display look with row-bounded base and residual arrays."""

    validate_display_look_payload(payload)
    if (
        isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
    ):
        raise ValueError("tile_rows must be a positive integer")
    base = payload["base"]
    anchor = payload["anchor"]
    residual = payload["residual"]
    density_operator = DensityDomainNegativePrintOperator.from_dict(
        base["density_operator"]
    )
    residual_operator = PositiveFilmResponseOperator.from_dict(
        residual["operator"]
    )

    def apply_density(encoded: np.ndarray) -> np.ndarray:
        linear = encoded_srgb_to_linear(
            np.asarray(encoded, dtype=np.float64)
        )
        return linear_srgb_to_encoded(
            density_operator.apply(
                linear, strength=float(base["density_strength"])
            )
        )

    source_shape = np.asarray(source).shape
    if source_context is None:
        source_context = build_density_source_context_row_staged(
            payload,
            source,
            tile_rows=tile_rows,
        )
    else:
        validate_safe_lab_source_context(source_context)
        if source_context.source_shape != source_shape:
            raise ValueError(
                "precomputed source context must match source shape"
            )

    def apply(encoded: np.ndarray) -> np.ndarray:
        encoded_value = np.asarray(encoded)
        if (
            encoded_value.shape != source_shape
            or not np.all(np.isfinite(encoded_value))
            or np.any(encoded_value < 0.0)
            or np.any(encoded_value > 1.0)
        ):
            raise ValueError(
                "row-streamed display input must be finite encoded RGB "
                "matching the source context shape"
            )
        if reuse_input_buffer:
            if (
                encoded_value.dtype != np.float64
                or not encoded_value.flags.c_contiguous
                or not encoded_value.flags.writeable
            ):
                raise ValueError(
                    "reused display input must be writable C-contiguous float64"
                )
            output = encoded_value
        else:
            output = np.empty(source_shape, dtype=np.float64)
        for y0 in range(0, source_shape[0], tile_rows):
            y1 = min(source_shape[0], y0 + tile_rows)
            base_output = _style_transfer_rgb_with_context(
                np.asarray(
                    apply_density(
                        np.asarray(
                            encoded_value[y0:y1], dtype=np.float32
                        )
                    ),
                    dtype=np.float32,
                ),
                anchor["stats"],
                str(anchor["style"]),
                float(anchor["strength"]),
                float(anchor["luma_strength"]),
                float(anchor["grain"]),
                int(anchor["seed"]),
                True,
                gamut_mode=str(anchor["gamut_mode"]),
                output_margin=0,
                source_context=source_context,
            )
            base_output = np.asarray(
                apply_output_margin(
                    np.asarray(base_output, dtype=np.float64),
                    int(base["final_output_margin"]),
                ),
                dtype=np.float32,
            )
            result = apply_factorized_boundary_guard(
                residual_operator,
                encoded_srgb_to_linear(base_output),
                tone_strength=float(residual["tone_strength"]),
                chroma_strength=float(residual["chroma_strength"]),
                luma_weights=np.asarray(
                    residual["luma_weights"], dtype=np.float64
                ),
                hard_boundary_epsilon_encoded_srgb=float(
                    residual[
                        "hard_boundary_epsilon_encoded_srgb"
                    ]
                ),
                guard_boundary_epsilon_encoded_srgb=float(
                    residual[
                        "guard_boundary_epsilon_encoded_srgb"
                    ]
                ),
            )
            output[y0:y1] = linear_srgb_to_encoded(result.output)
        if (
            not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError(
                "row-streamed display-look component left encoded RGB"
            )
        return output

    return apply


def build_density_source_context_row_staged(
    payload: dict[str, Any],
    source: np.ndarray,
    *,
    tile_rows: int,
) -> SafeLabSourceContext:
    """Build the exact legacy context without full-frame density temporaries."""

    validate_display_look_payload(payload)
    if (
        isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
    ):
        raise ValueError("tile_rows must be a positive integer")
    source_value = np.asarray(source)
    if (
        source_value.ndim != 3
        or source_value.shape[-1] != 3
        or source_value.shape[0] == 0
        or source_value.shape[1] == 0
        or not np.all(np.isfinite(source_value))
        or np.any(source_value < 0.0)
        or np.any(source_value > 1.0)
    ):
        raise ValueError("source context input must be finite encoded HxWx3")
    base = payload["base"]
    density_operator = DensityDomainNegativePrintOperator.from_dict(
        base["density_operator"]
    )
    lab = np.empty(source_value.shape, dtype=np.float32)
    for y0 in range(0, source_value.shape[0], tile_rows):
        y1 = min(source_value.shape[0], y0 + tile_rows)
        source_rows = np.asarray(
            source_value[y0:y1], dtype=np.float32
        )
        linear = encoded_srgb_to_linear(source_rows.astype(np.float64))
        density = density_operator.apply(
            linear, strength=float(base["density_strength"])
        )
        encoded = linear_srgb_to_encoded(density)
        lab[y0:y1] = rgb2lab(
            np.asarray(encoded, dtype=np.float32)
        )
    return safe_lab_context_from_lab(lab, source_value.shape)


def build_density_source_context_inplace_packed_lab(
    payload: dict[str, Any],
    source: np.ndarray,
    *,
    tile_rows: int,
) -> SafeLabSourceContext:
    """Replace consumed float64 encoded storage with exact packed float32 Lab."""

    validate_display_look_payload(payload)
    source_value = np.asarray(source)
    if (
        source_value.dtype != np.float64
        or source_value.ndim != 3
        or source_value.shape[-1] != 3
        or source_value.shape[0] == 0
        or source_value.shape[1] == 0
        or not source_value.flags.c_contiguous
        or not source_value.flags.writeable
        or isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
        or not np.all(np.isfinite(source_value))
        or np.any(source_value < 0.0)
        or np.any(source_value > 1.0)
    ):
        raise ValueError(
            "packed Lab context requires writable C-contiguous float64 HxWx3"
        )
    base = payload["base"]
    density_operator = DensityDomainNegativePrintOperator.from_dict(
        base["density_operator"]
    )
    packed_lab = source_value.view(np.float32).reshape(-1)[
        : source_value.size
    ].reshape(source_value.shape)
    for y0 in range(0, source_value.shape[0], tile_rows):
        y1 = min(source_value.shape[0], y0 + tile_rows)
        source_rows = np.asarray(
            source_value[y0:y1], dtype=np.float32
        )
        linear = encoded_srgb_to_linear(source_rows.astype(np.float64))
        density = density_operator.apply(
            linear, strength=float(base["density_strength"])
        )
        encoded = linear_srgb_to_encoded(density)
        packed_lab[y0:y1] = rgb2lab(
            np.asarray(encoded, dtype=np.float32)
        )
    return safe_lab_context_from_lab(
        packed_lab, tuple(int(size) for size in source_value.shape)
    )


__all__ = [
    "DISPLAY_LOOK_SCHEMA",
    "build_source_context_display_look",
    "build_source_context_display_look_stages",
    "build_source_context_display_look_row_streamed",
    "build_density_source_context_row_staged",
    "build_density_source_context_inplace_packed_lab",
    "make_display_look_payload",
    "validate_display_look_payload",
]
