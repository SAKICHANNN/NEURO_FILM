"""Input preprocessing primitives for film rendering."""

from .pipeline import inspect_input, load_working_image
from .output_encode import save_srgb8, srgb_icc_profile, srgb_icc_profile_sha256
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
    "save_srgb8",
    "srgb_icc_profile",
    "srgb_icc_profile_sha256",
    "working_image_to_legacy_srgb8",
]
