"""Execute an interpretation medium through a separate scanner profile."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from .contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
)
from .exposure_development import InterpretationRoute
from .interpretation_medium import (
    InterpretationMedium,
    PostScanPolarity,
)
from .scanner import SCANNER_STAGES, ScannerProfile, apply_scanner_profile


SCANNED_INTERPRETATION_SCHEMA = "neuro_film.scanned_interpretation.v1"
SCANNED_INTERPRETATION_ORDER = (
    "developed_density",
    "interpretation_medium",
    "scanner_profile",
    "post_scan_polarity",
)


def scanner_profile_identity(profile: ScannerProfile) -> str:
    if not isinstance(profile, ScannerProfile):
        raise TypeError("profile must be ScannerProfile")
    payload = {
        "profile_id": profile.profile_id,
        "illuminant_rgb": list(profile.illuminant_rgb),
        "spectral_matrix": [list(row) for row in profile.spectral_matrix],
        "local_flare_fraction": profile.local_flare_fraction,
        "global_flare_fraction": profile.global_flare_fraction,
        "flare_sigma_um": profile.flare_sigma_um,
        "dmax_density_rgb": (
            None
            if profile.dmax_density_rgb is None
            else list(profile.dmax_density_rgb)
        ),
        "mtf_sigma_um_rgb": list(profile.mtf_sigma_um_rgb),
        "shot_noise_variance_scale": profile.shot_noise_variance_scale,
        "read_noise_variance": profile.read_noise_variance,
        "seed": profile.seed,
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def apply_post_scan_polarity(
    scan_linear: np.ndarray, polarity: PostScanPolarity
) -> np.ndarray:
    values = np.asarray(scan_linear)
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("scan-linear values must be float32 or float64")
    if values.ndim != 3 or values.shape[-1] != 3 or values.size == 0:
        raise ValueError("scan-linear values must have non-empty HxWx3 shape")
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("scan-linear values must be bounded in [0, 1]")
    selected = PostScanPolarity(polarity)
    if selected is PostScanPolarity.IDENTITY:
        return values.copy()
    return (values.dtype.type(1.0) - values).astype(values.dtype)


@dataclass(frozen=True)
class ScannedInterpretation:
    scan_linear: PhysicalDomainArray
    interpretation_route: InterpretationRoute
    medium_descriptor: dict[str, Any]
    scanner_profile_id: str
    scanner_profile_sha256: str
    stages: tuple[str, ...]
    order: tuple[str, ...] = SCANNED_INTERPRETATION_ORDER

    def __post_init__(self) -> None:
        self.scan_linear.require(PhysicalDomain.SCAN_LINEAR)
        route = InterpretationRoute(self.interpretation_route)
        stages = tuple(self.stages)
        if not isinstance(self.medium_descriptor, dict):
            raise TypeError("medium_descriptor must be a dictionary")
        if not isinstance(self.scanner_profile_id, str) or not self.scanner_profile_id:
            raise ValueError("scanner_profile_id must be non-empty")
        if (
            not isinstance(self.scanner_profile_sha256, str)
            or len(self.scanner_profile_sha256) != 64
        ):
            raise ValueError("scanner profile identity must be SHA-256")
        if tuple(self.order) != SCANNED_INTERPRETATION_ORDER:
            raise ValueError("scanned interpretation order mismatch")
        object.__setattr__(self, "interpretation_route", route)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "order", SCANNED_INTERPRETATION_ORDER)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": SCANNED_INTERPRETATION_SCHEMA,
            "interpretation_route": self.interpretation_route.value,
            "medium": self.medium_descriptor,
            "scanner_profile_id": self.scanner_profile_id,
            "scanner_profile_sha256": self.scanner_profile_sha256,
            "stages": list(self.stages),
            "order": list(self.order),
            "output": self.scan_linear.descriptor(),
        }


def scan_interpretation_medium(
    medium: InterpretationMedium,
    profile: ScannerProfile,
    *,
    pixel_pitch_um: float,
    stages: Iterable[str] = SCANNER_STAGES,
    expected_route: InterpretationRoute | None = None,
) -> ScannedInterpretation:
    """Scan first and apply the route polarity only to scan-linear output."""
    if not isinstance(medium, InterpretationMedium):
        raise TypeError("medium must be InterpretationMedium")
    if not isinstance(profile, ScannerProfile):
        raise TypeError("profile must be ScannerProfile")
    if (
        expected_route is not None
        and medium.interpretation_route is not InterpretationRoute(expected_route)
    ):
        raise ValueError("interpretation medium route binding mismatch")
    selected = tuple(stages)
    before = medium.values.tobytes()
    scanned = apply_scanner_profile(
        medium.values,
        profile,
        pixel_pitch_um=pixel_pitch_um,
        stages=selected,
    )
    output = apply_post_scan_polarity(
        scanned, medium.post_scan_polarity
    )
    if medium.values.tobytes() != before:
        raise RuntimeError("scanner mutated interpretation medium")
    state = PhysicalDomainArray(
        output,
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        medium.channels,
        medium.scale,
    )
    return ScannedInterpretation(
        state,
        medium.interpretation_route,
        medium.descriptor(),
        profile.profile_id,
        scanner_profile_identity(profile),
        selected,
    )
