"""Execute the experimental scanner chain from canonical profile bytes."""

from __future__ import annotations

import re

import numpy as np

from src.eval.scanner_glare_tiled_downstream import (
    apply_typed_scanner_glare_chain_tiled_downstream,
)
from src.film_physics.scanner_chain_profile import ScannerChainProfile

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def apply_scanner_chain_from_bundle(
    transmittance: np.ndarray,
    bundle_bytes: bytes,
    *,
    expected_profile_sha256: str,
) -> np.ndarray:
    """Validate exact bundle identity and input domain before pixel execution."""
    if (
        not isinstance(expected_profile_sha256, str)
        or _SHA256_RE.fullmatch(expected_profile_sha256) is None
    ):
        raise ValueError("expected_profile_sha256 must be lowercase hexadecimal")
    profile = ScannerChainProfile.from_json_bytes(bundle_bytes)
    if profile.canonical_bytes() != bundle_bytes:
        raise ValueError("scanner chain bundle bytes must be canonical")
    if profile.profile_sha256 != expected_profile_sha256:
        raise ValueError("scanner chain profile identity mismatch")

    values = np.asarray(transmittance)
    if values.dtype != np.float64:
        raise TypeError("scanner chain bundle input must be float64")
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("scanner chain bundle input must be HxWx3")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("scanner chain bundle input must be finite in (0, 1]")

    return apply_typed_scanner_glare_chain_tiled_downstream(
        values,
        profile.scanner_profile,
        pixel_pitch_um=profile.pixel_pitch_um,
        glare_row_chunk=profile.glare_row_chunk,
        downstream_tile_rows=profile.downstream_tile_rows,
    )


__all__ = ["apply_scanner_chain_from_bundle"]
