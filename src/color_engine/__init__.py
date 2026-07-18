"""Working-space-aware deterministic colour-engine primitives."""

from .gamut import (
    compress_chroma_to_working_gamut,
    compress_source_to_working_gamut,
    in_working_gamut,
)
from .lab import lab_to_linear_rgb, linear_rgb_to_lab
from .rec2020_safe_lab import apply_rec2020_safe_lab
from .safe_lab import SafeLabSourceContext, apply_safe_lab_transform, safe_lab_context_from_lab

__all__ = [
    "SafeLabSourceContext",
    "apply_rec2020_safe_lab",
    "apply_safe_lab_transform",
    "compress_chroma_to_working_gamut",
    "compress_source_to_working_gamut",
    "in_working_gamut",
    "lab_to_linear_rgb",
    "linear_rgb_to_lab",
    "safe_lab_context_from_lab",
]
