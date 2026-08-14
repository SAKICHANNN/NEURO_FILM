"""Historical density-dependent Wiener-granularity amplitude prior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BWWienerDensityPrior:
    film_identity: str
    development: str
    reported_gamma: float
    densities: tuple[float, ...]
    wiener_granularity_spectrum_cm2: tuple[float, ...]
    source_sha256: str
    current_400tx_claim: bool = False
    spatial_spectrum_shape_available: bool = False
    render_allowed: bool = False

    def __post_init__(self) -> None:
        density = np.asarray(self.densities, dtype=np.float64)
        wiener = np.asarray(self.wiener_granularity_spectrum_cm2, dtype=np.float64)
        if density.shape != (4,) or wiener.shape != density.shape:
            raise ValueError(
                "historical Wiener prior requires four aligned observations"
            )
        if not np.all(np.isfinite(density)) or not np.all(np.diff(density) > 0.0):
            raise ValueError("historical densities must be finite and increasing")
        if not np.all(np.isfinite(wiener)) or not np.all(wiener > 0.0):
            raise ValueError("historical Wiener amplitudes must be finite and positive")
        if self.reported_gamma <= 0.0 or len(self.source_sha256) != 64:
            raise ValueError("historical source metadata is invalid")
        if (
            self.current_400tx_claim
            or self.spatial_spectrum_shape_available
            or self.render_allowed
        ):
            raise ValueError("historical amplitude-only prior cannot expand its claim")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "neuro-film.bw-wiener-density-prior.v1",
            "film_identity": self.film_identity,
            "development": self.development,
            "reported_gamma": self.reported_gamma,
            "densities": list(self.densities),
            "wiener_granularity_spectrum_cm2": list(
                self.wiener_granularity_spectrum_cm2
            ),
            "source_sha256": self.source_sha256,
            "current_400tx_claim": self.current_400tx_claim,
            "spatial_spectrum_shape_available": self.spatial_spectrum_shape_available,
            "render_allowed": self.render_allowed,
        }

    def identity(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def render(self, *_args: Any, **_kwargs: Any) -> None:
        raise ValueError("historical Wiener amplitude prior cannot render")


__all__ = ["BWWienerDensityPrior"]
