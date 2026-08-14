"""Version-limited B&W modulation-transfer source prior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BWMTFSourcePrior:
    film_identity: str
    process: str
    densitometry: str
    frequencies_cycles_per_mm: tuple[float, ...]
    response_fractions: tuple[float, ...]
    source_pdf_sha256: str
    current_stock_measurement_claim: bool = False
    scanner_response_included: bool = False
    render_allowed: bool = False

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_cycles_per_mm, dtype=np.float64)
        responses = np.asarray(self.response_fractions, dtype=np.float64)
        if (
            frequencies.ndim != 1
            or frequencies.size < 2
            or responses.shape != frequencies.shape
        ):
            raise ValueError("MTF prior requires aligned one-dimensional samples")
        if not np.all(np.isfinite(frequencies)) or not np.all(np.diff(frequencies) > 0):
            raise ValueError("MTF frequencies must be finite and strictly increasing")
        if not np.all(np.isfinite(responses)) or not np.all(responses > 0):
            raise ValueError("MTF responses must be finite and positive")
        if len(self.source_pdf_sha256) != 64:
            raise ValueError("source PDF identity must be SHA-256")
        if (
            self.current_stock_measurement_claim
            or self.scanner_response_included
            or self.render_allowed
        ):
            raise ValueError(
                "source-only MTF prior cannot authorize expanded claims or rendering"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "neuro-film.bw-mtf-source-prior.v1",
            "film_identity": self.film_identity,
            "process": self.process,
            "densitometry": self.densitometry,
            "frequencies_cycles_per_mm": list(self.frequencies_cycles_per_mm),
            "response_fractions": list(self.response_fractions),
            "source_pdf_sha256": self.source_pdf_sha256,
            "current_stock_measurement_claim": self.current_stock_measurement_claim,
            "scanner_response_included": self.scanner_response_included,
            "render_allowed": self.render_allowed,
        }

    def identity(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def render(self, *_args: Any, **_kwargs: Any) -> None:
        raise ValueError("version-limited MTF source prior cannot render")


__all__ = ["BWMTFSourcePrior"]
