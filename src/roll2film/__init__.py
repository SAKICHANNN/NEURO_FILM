"""Data-independent Roll2Film operator-identification research primitives."""

from .identification import (
    estimate_affine_spline_transport_operator,
    estimate_gaussian_transport_operator,
)
from .constrained import (
    ConstrainedGlobalColorOperator,
    LUTConstraintReport,
    LUTConstraintSpec,
    audit_lut_constraints,
    identity_lut,
)
from .lut import DenseLUT3D, bake_dense_lut
from .lab_statistics import LabStatisticsDescriptor, extract_lab_statistics
from .operators import AffineColorOperator
from .simulator import PseudoRoll, PseudoRollConfig, simulate_pseudo_roll
from .splines import AffineMonotoneSplineOperator, RationalQuadraticSpline

__all__ = [
    "AffineColorOperator",
    "AffineMonotoneSplineOperator",
    "ConstrainedGlobalColorOperator",
    "DenseLUT3D",
    "LUTConstraintReport",
    "LUTConstraintSpec",
    "LabStatisticsDescriptor",
    "PseudoRoll",
    "PseudoRollConfig",
    "RationalQuadraticSpline",
    "audit_lut_constraints",
    "bake_dense_lut",
    "estimate_affine_spline_transport_operator",
    "estimate_gaussian_transport_operator",
    "extract_lab_statistics",
    "identity_lut",
    "simulate_pseudo_roll",
]
