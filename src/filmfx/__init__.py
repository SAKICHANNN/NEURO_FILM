"""Film effect layer compositing."""

from .compositor import composite_layers
from .effects import density_halation_layer, dust_scratch_layer, grain_residual_layer, halation_layer, physical_halation_layer
from .halation_controls import (
    HALATION_PRESETS,
    HalationPreset,
    PhysicalHalationControls,
    build_physical_halation_layer,
    describe_physical_halation_controls,
    get_halation_preset,
    halation_gui_schema,
    list_halation_presets,
    resolve_physical_halation_controls,
    validate_physical_halation_controls,
)
from .layers import FilmLayer, layer_metrics
from .tiled_grain import (
    GRAIN_BLUR_HALO,
    StagedGrainMetadata,
    staged_grain_residual_layer,
)
from .tiled_effects import (
    DustScratchContext,
    DustScratchExecutionMetadata,
    build_dust_scratch_context,
    composite_dust_scratch_tiled,
    composite_simple_halation_tiled,
    dust_scratch_alpha_window,
    simple_halation_required_halo,
)

__all__ = [
    "FilmLayer",
    "GRAIN_BLUR_HALO",
    "DustScratchContext",
    "DustScratchExecutionMetadata",
    "HALATION_PRESETS",
    "HalationPreset",
    "PhysicalHalationControls",
    "StagedGrainMetadata",
    "build_physical_halation_layer",
    "build_dust_scratch_context",
    "composite_layers",
    "composite_dust_scratch_tiled",
    "composite_simple_halation_tiled",
    "density_halation_layer",
    "describe_physical_halation_controls",
    "dust_scratch_layer",
    "dust_scratch_alpha_window",
    "grain_residual_layer",
    "halation_layer",
    "get_halation_preset",
    "halation_gui_schema",
    "list_halation_presets",
    "physical_halation_layer",
    "resolve_physical_halation_controls",
    "simple_halation_required_halo",
    "staged_grain_residual_layer",
    "validate_physical_halation_controls",
    "layer_metrics",
]
