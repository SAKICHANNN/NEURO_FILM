"""Typed exposure-time reciprocity primitives.

The primitive operates on exposure time in seconds.  It does not infer colour,
process, scanner, or a particular-roll response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ReciprocityDomainError(ValueError):
    """Raised when reciprocity arithmetic is requested outside its domain."""


def _positive_finite(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ReciprocityDomainError(f"{name} must be positive and finite")
    return array


@dataclass(frozen=True)
class PowerReciprocityProfile:
    """One manufacturer-specified low-intensity reciprocity exponent."""

    profile_id: str
    exponent: float
    reference_time_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ReciprocityDomainError("profile_id must be non-empty")
        if not np.isfinite(self.exponent) or self.exponent < 1.0:
            raise ReciprocityDomainError("exponent must be finite and at least one")
        if (
            not np.isfinite(self.reference_time_seconds)
            or self.reference_time_seconds <= 0.0
        ):
            raise ReciprocityDomainError(
                "reference_time_seconds must be positive and finite"
            )

    def corrected_time_seconds(
        self, metered_time_seconds: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the physical exposure time needed for the metered response."""

        metered = _positive_finite(metered_time_seconds, name="metered_time_seconds")
        reference = self.reference_time_seconds
        return np.where(
            metered <= reference,
            metered,
            reference * np.power(metered / reference, self.exponent),
        )

    def effective_time_seconds(
        self, physical_time_seconds: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the metered-time equivalent of a physical exposure duration."""

        physical = _positive_finite(
            physical_time_seconds, name="physical_time_seconds"
        )
        reference = self.reference_time_seconds
        return np.where(
            physical <= reference,
            physical,
            reference * np.power(physical / reference, 1.0 / self.exponent),
        )

    def effective_exposure(
        self,
        incident_rate: ArrayLike,
        physical_time_seconds: ArrayLike,
    ) -> NDArray[np.float64]:
        """Apply the time efficiency to a nonnegative constant incident rate."""

        rate = np.asarray(incident_rate, dtype=np.float64)
        if not np.all(np.isfinite(rate)) or np.any(rate < 0.0):
            raise ReciprocityDomainError("incident_rate must be nonnegative and finite")
        effective_time = self.effective_time_seconds(physical_time_seconds)
        return np.multiply(rate, effective_time, dtype=np.float64)


@dataclass(frozen=True)
class DocumentedIdentityInterval:
    """A source-bounded interval where no reciprocity correction is reported."""

    profile_id: str
    minimum_seconds: float
    maximum_seconds: float

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ReciprocityDomainError("profile_id must be non-empty")
        if (
            not np.isfinite(self.minimum_seconds)
            or not np.isfinite(self.maximum_seconds)
            or self.minimum_seconds <= 0.0
            or self.maximum_seconds < self.minimum_seconds
        ):
            raise ReciprocityDomainError("invalid documented identity interval")

    def corrected_time_seconds(
        self, metered_time_seconds: ArrayLike
    ) -> NDArray[np.float64]:
        metered = _positive_finite(metered_time_seconds, name="metered_time_seconds")
        if np.any(metered < self.minimum_seconds) or np.any(
            metered > self.maximum_seconds
        ):
            raise ReciprocityDomainError(
                "time lies outside the documented identity interval"
            )
        return metered.copy()


__all__ = [
    "DocumentedIdentityInterval",
    "PowerReciprocityProfile",
    "ReciprocityDomainError",
]
