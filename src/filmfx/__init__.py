"""Film effect layer compositing."""

from .compositor import composite_layers
from .effects import density_halation_layer, dust_scratch_layer, grain_residual_layer, halation_layer, physical_halation_layer
from .halation_controls import (
    PhysicalHalationControls,
    build_physical_halation_layer,
    describe_physical_halation_controls,
    resolve_physical_halation_controls,
)
from .layers import FilmLayer, layer_metrics

__all__ = [
    "FilmLayer",
    "PhysicalHalationControls",
    "build_physical_halation_layer",
    "composite_layers",
    "density_halation_layer",
    "describe_physical_halation_controls",
    "dust_scratch_layer",
    "grain_residual_layer",
    "halation_layer",
    "physical_halation_layer",
    "resolve_physical_halation_controls",
    "layer_metrics",
]
