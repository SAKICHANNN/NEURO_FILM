"""Output-exact row-bounded analytic renderer candidate for CB66."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.eval.analytic_y_chromaticity_streaming import (
    select_analytic_y_chromaticity_candidate_streamed,
)
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fujifilm_characteristic_luma_chroma import (
    FujifilmCharacteristicLumaChromaError,
    apply_characteristic_luma_chroma,
)
from src.eval.nonexpansive_fraction_transport_external_sort import (
    nonexpansive_fraction_transport_target_external_sorted,
)
from src.inference.analytic_y_chromaticity_profile import (
    AnalyticYChromaticityProfileError,
    AnalyticYChromaticityRuntime,
)
from src.preprocess.types import WorkingImage


def apply_characteristic_luma_chroma_output_row_materialized(
    source_linear: np.ndarray,
    curve: PchipInterpolator,
    *,
    weights: np.ndarray,
    strength: float,
    boundary_epsilon: float,
    row_chunk: int = 128,
) -> np.ndarray:
    """Return the exact CB11 output while bounding all diagnostic temporaries by rows."""
    source = np.asarray(source_linear)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
        or isinstance(row_chunk, bool)
        or not isinstance(row_chunk, int)
        or row_chunk <= 0
    ):
        raise FujifilmCharacteristicLumaChromaError("CB66 input drift")
    output = np.empty_like(source)
    for y0 in range(0, source.shape[0], row_chunk):
        y1 = min(source.shape[0], y0 + row_chunk)
        output[y0:y1], _, _ = apply_characteristic_luma_chroma(
            source[y0:y1],
            curve,
            weights=weights,
            strength=strength,
            boundary_epsilon=boundary_epsilon,
        )
    return output


def render_analytic_y_chromaticity_memory_candidate(
    working: WorkingImage,
    runtime: AnalyticYChromaticityRuntime,
    *,
    scratch_root: Path,
    row_chunk: int = 128,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render CB61 semantics using row-bounded safe-base and external target sort."""
    if (
        working.working_space != "linear_srgb"
        or working.transfer_state != "display_linear"
        or not scratch_root.is_dir()
    ):
        raise AnalyticYChromaticityProfileError("CB66 runtime input drift")
    source = np.asarray(working.pixels, dtype=np.float32)
    operator = runtime.cb11["operator"]
    weights = np.asarray(operator["luminance_weights"], dtype=np.float64)
    epsilon = float(operator["boundary_epsilon"])
    safe_base = apply_characteristic_luma_chroma_output_row_materialized(
        source,
        runtime.curve,
        weights=weights,
        strength=float(operator["nominal_strength"]),
        boundary_epsilon=epsilon,
        row_chunk=row_chunk,
    )
    ao6 = render_fixed_pair(source, runtime.artifact, runtime.ao6_config["component"])[
        runtime.ao6_config["arm_id"]
    ]
    config = runtime.cb52
    target = nonexpansive_fraction_transport_target_external_sorted(
        safe_base,
        ao6,
        weights=weights,
        minimum_valid_fraction=float(config["operator"]["minimum_valid_fraction"]),
        fraction_knots=int(config["operator"]["fraction_knots"]),
        maximum_fraction_slope=float(config["operator"]["maximum_fraction_slope"]),
        row_chunk=row_chunk,
        scratch_root=scratch_root,
    )
    del safe_base, ao6
    candidate, _, _, facts = select_analytic_y_chromaticity_candidate_streamed(
        source,
        target,
        curve=runtime.curve,
        strength=float(operator["nominal_strength"]),
        boundary_epsilon=epsilon,
        dose_grid=config["operator"]["dose_grid"],
        maximum_gradient_ratio=float(
            config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
        ),
        maximum_lstar_inversion_fraction=float(
            config["automatic_gates"][
                "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
            ]
        ),
        lstar_order_epsilon=float(config["operator"]["lstar_order_epsilon"]),
        row_chunk=row_chunk,
        scratch_root=scratch_root,
    )
    return candidate, facts


__all__ = [
    "apply_characteristic_luma_chroma_output_row_materialized",
    "render_analytic_y_chromaticity_memory_candidate",
]
