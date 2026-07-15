"""Data-independent Roll2Film operator-identification research primitives."""

from .identification import estimate_gaussian_transport_operator
from .lut import DenseLUT3D, bake_dense_lut
from .operators import AffineColorOperator
from .simulator import PseudoRoll, PseudoRollConfig, simulate_pseudo_roll
from .splines import AffineMonotoneSplineOperator, RationalQuadraticSpline

__all__ = [
    "AffineColorOperator",
    "AffineMonotoneSplineOperator",
    "DenseLUT3D",
    "PseudoRoll",
    "PseudoRollConfig",
    "RationalQuadraticSpline",
    "bake_dense_lut",
    "estimate_gaussian_transport_operator",
    "simulate_pseudo_roll",
]
