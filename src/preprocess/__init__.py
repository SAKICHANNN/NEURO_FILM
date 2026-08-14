"""Input preprocessing primitives for film rendering."""

from .color_management import (
    LINEAR_RGB_TRANSFORM_VERSION,
    REC2020_SDR_CICP,
    REC2020_TRANSFER_VERSION,
    convert_linear_rgb,
    convert_working_image_space,
    linear_rec2020_to_rec2020,
    linear_rgb_matrix,
    rec2020_to_linear_rec2020,
)
from .color_state import resolve_look_approximation_claim
from .output_encode import (
    normalized_icc_profile_sha256,
    save_rec2020_16_png,
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
    srgb_icc_profile,
    srgb_icc_profile_fingerprint_sha256,
    srgb_icc_profile_sha256,
)
from .pipeline import inspect_input, load_working_image
from .png_stream import StreamingSrgbPngWriter
from .raster_decode import working_image_to_legacy_srgb8, working_image_to_srgb_float
from .romm_rec2020_product import (
    OFFICIAL_ROMM_ICC_SHA256,
    ROMM_REC2020_CAPABILITY_ID,
    ROMM_REC2020_QUALIFICATION_EVIDENCE_SHA256,
    ROMM_REC2020_RECEIPT_SCHEMA,
    ROMMRec2020ConversionError,
    convert_official_romm_rgb16_to_rec2020_png,
)
from .types import (
    DecodeWarning,
    InputInspection,
    SourceProfile,
    WorkingImage,
)

__all__ = [
    "LINEAR_RGB_TRANSFORM_VERSION",
    "OFFICIAL_ROMM_ICC_SHA256",
    "REC2020_SDR_CICP",
    "REC2020_TRANSFER_VERSION",
    "ROMM_REC2020_CAPABILITY_ID",
    "ROMM_REC2020_QUALIFICATION_EVIDENCE_SHA256",
    "ROMM_REC2020_RECEIPT_SCHEMA",
    "DecodeWarning",
    "InputInspection",
    "ROMMRec2020ConversionError",
    "SourceProfile",
    "StreamingSrgbPngWriter",
    "WorkingImage",
    "convert_linear_rgb",
    "convert_official_romm_rgb16_to_rec2020_png",
    "convert_working_image_space",
    "inspect_input",
    "linear_rec2020_to_rec2020",
    "linear_rgb_matrix",
    "load_working_image",
    "normalized_icc_profile_sha256",
    "rec2020_to_linear_rec2020",
    "resolve_look_approximation_claim",
    "save_rec2020_16_png",
    "save_srgb8",
    "save_srgb16_png",
    "save_srgb16_tiff",
    "srgb_icc_profile",
    "srgb_icc_profile_fingerprint_sha256",
    "srgb_icc_profile_sha256",
    "working_image_to_legacy_srgb8",
    "working_image_to_srgb_float",
]
