"""Film effect layer compositing."""

from .compositor import composite_layers
from .effects import dust_scratch_layer, grain_residual_layer, halation_layer, physical_halation_layer
from .halation_controls import PhysicalHalationControls, resolve_physical_halation_controls
from .layers import FilmLayer, layer_metrics

__all__ = [
    "FilmLayer",
    "PhysicalHalationControls",
    "composite_layers",
    "dust_scratch_layer",
    "grain_residual_layer",
    "halation_layer",
    "physical_halation_layer",
    "resolve_physical_halation_controls",
    "layer_metrics",
]
