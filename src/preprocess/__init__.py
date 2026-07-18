"""Input preprocessing primitives for film rendering."""

from .color_state import resolve_look_approximation_claim
from .color_management import (
    LINEAR_RGB_TRANSFORM_VERSION,
    convert_linear_rgb,
    convert_working_image_space,
    linear_rgb_matrix,
)
from .pipeline import inspect_input, load_working_image
from .output_encode import (
    normalized_icc_profile_sha256,
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
    srgb_icc_profile,
    srgb_icc_profile_fingerprint_sha256,
    srgb_icc_profile_sha256,
)
from .raster_decode import working_image_to_legacy_srgb8, working_image_to_srgb_float
from .types import (
    DecodeWarning,
    InputInspection,
    SourceProfile,
    WorkingImage,
)

__all__ = [
    "DecodeWarning",
    "InputInspection",
    "LINEAR_RGB_TRANSFORM_VERSION",
    "SourceProfile",
    "WorkingImage",
    "convert_linear_rgb",
    "convert_working_image_space",
    "inspect_input",
    "load_working_image",
    "linear_rgb_matrix",
    "normalized_icc_profile_sha256",
    "resolve_look_approximation_claim",
    "save_srgb8",
    "save_srgb16_tiff",
    "save_srgb16_png",
    "srgb_icc_profile",
    "srgb_icc_profile_fingerprint_sha256",
    "srgb_icc_profile_sha256",
    "working_image_to_legacy_srgb8",
    "working_image_to_srgb_float",
]
