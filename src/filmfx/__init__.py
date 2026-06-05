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

__all__ = [
    "FilmLayer",
    "HALATION_PRESETS",
    "HalationPreset",
    "PhysicalHalationControls",
    "build_physical_halation_layer",
    "composite_layers",
    "density_halation_layer",
    "describe_physical_halation_controls",
    "dust_scratch_layer",
    "grain_residual_layer",
    "halation_layer",
    "get_halation_preset",
    "halation_gui_schema",
    "list_halation_presets",
    "physical_halation_layer",
    "resolve_physical_halation_controls",
    "validate_physical_halation_controls",
    "layer_metrics",
]
