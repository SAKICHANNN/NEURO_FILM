"""Exact analytic renderer combining parallel target rows and compact selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.analytic_y_chromaticity_compact_selector import (
    select_analytic_y_chromaticity_candidate_compact,
)
from src.eval.analytic_y_chromaticity_memory_optimized import (
    apply_characteristic_luma_chroma_output_row_materialized,
)
from src.eval.fixed_ao6_single_target import render_fixed_ao6_single_target
from src.eval.nonexpansive_fraction_transport_parallel import (
    nonexpansive_fraction_transport_target_parallel,
)
from src.inference.analytic_y_chromaticity_profile import (
    AnalyticYChromaticityProfileError,
    AnalyticYChromaticityRuntime,
)
from src.preprocess.types import WorkingImage


def render_analytic_y_chromaticity_compact_parallel_candidate(
    working: WorkingImage,
    runtime: AnalyticYChromaticityRuntime,
    *,
    scratch_root: Path,
    row_chunk: int = 128,
    workers: int = 4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render CB69 semantics with parallel target rows and compact selector state."""
    if (
        working.working_space != "linear_srgb"
        or working.transfer_state != "display_linear"
        or not scratch_root.is_dir()
        or workers != 4
    ):
        raise AnalyticYChromaticityProfileError("CB71 runtime input drift")
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
    ao6 = render_fixed_ao6_single_target(
        source,
        runtime.artifact,
        runtime.ao6_config["component"],
        row_chunk=row_chunk,
        workers=workers,
    )
    config = runtime.cb52
    target = nonexpansive_fraction_transport_target_parallel(
        safe_base,
        ao6,
        weights=weights,
        minimum_valid_fraction=float(config["operator"]["minimum_valid_fraction"]),
        fraction_knots=int(config["operator"]["fraction_knots"]),
        maximum_fraction_slope=float(
            config["operator"]["maximum_fraction_slope"]
        ),
        row_chunk=row_chunk,
        scratch_root=scratch_root,
        workers=workers,
    )
    del safe_base, ao6
    return select_analytic_y_chromaticity_candidate_compact(
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


__all__ = ["render_analytic_y_chromaticity_compact_parallel_candidate"]
