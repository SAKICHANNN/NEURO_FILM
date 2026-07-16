"""Input preprocessing primitives for film rendering."""

from .color_state import resolve_look_approximation_claim
from .pipeline import inspect_input, load_working_image
from .output_encode import (
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
    srgb_icc_profile,
    srgb_icc_profile_sha256,
)
from .raster_decode import working_image_to_legacy_srgb8
from .types import (
    DecodeWarning,
    InputInspection,
    SourceProfile,
    WorkingImage,
)

__all__ = [
    "DecodeWarning",
    "InputInspection",
    "SourceProfile",
    "WorkingImage",
    "inspect_input",
    "load_working_image",
    "resolve_look_approximation_claim",
    "save_srgb8",
    "save_srgb16_tiff",
    "save_srgb16_png",
    "srgb_icc_profile",
    "srgb_icc_profile_sha256",
    "working_image_to_legacy_srgb8",
]
