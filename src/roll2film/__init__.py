"""Data-independent Roll2Film operator-identification research primitives."""

from .identification import estimate_gaussian_transport_operator
from .operators import AffineColorOperator
from .simulator import PseudoRoll, PseudoRollConfig, simulate_pseudo_roll

__all__ = [
    "AffineColorOperator",
    "PseudoRoll",
    "PseudoRollConfig",
    "estimate_gaussian_transport_operator",
    "simulate_pseudo_roll",
]
