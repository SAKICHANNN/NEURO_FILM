"""Artifact-backed AO6 display-look reference implementation.

This is a deterministic colour component, not a calibrated film response.  It
exists so the physical profile consumer does not need to reopen the experiment
configuration tree after a profile has been compiled.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from scripts.pipeline_color_baseline import (
    apply_output_margin,
    build_safe_lab_source_context,
)
from src.color_engine.safe_lab_rgb_context import (
    style_transfer_rgb_with_source_context,
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
    source_context = build_safe_lab_source_context(
        apply_density(source_value)
    )

    def apply(encoded: np.ndarray) -> np.ndarray:
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
            output.shape != source_value.shape
            or not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError("display-look component left encoded RGB")
        return output

    return apply


__all__ = [
    "DISPLAY_LOOK_SCHEMA",
    "build_source_context_display_look",
    "make_display_look_payload",
    "validate_display_look_payload",
]
