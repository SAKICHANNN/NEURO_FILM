"""Typed contracts for product-grade input preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


TransferState = Literal["scene_linear", "display_linear", "display_referred", "unknown"]
AlphaPolicy = Literal["absent", "preserved", "composited"]
SourceKind = Literal["raster", "raw", "unknown"]


@dataclass(frozen=True)
class DecodeWarning:
    """A non-fatal assumption or limitation in the decode path."""

    code: str
    message: str


@dataclass(frozen=True)
class SourceProfile:
    """Color/profile metadata discovered before or during decode."""

    kind: Literal["icc", "cicp", "nclx", "raw_metadata", "assumed_srgb", "unknown"]
    description: str
    bytes_length: int = 0


@dataclass
class InputInspection:
    """Read-only metadata inspection for an input file."""

    path: Path
    exists: bool
    source_kind: SourceKind
    format_name: str = "unknown"
    mode: str = "unknown"
    width: int = 0
    height: int = 0
    bit_depth: int | None = None
    has_alpha: bool = False
    orientation: int | None = None
    orientation_applied: bool = False
    frame_count: int = 1
    source_profile: SourceProfile = field(default_factory=lambda: SourceProfile("unknown", "unknown"))
    transfer_state: TransferState = "unknown"
    hdr_metadata: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[DecodeWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "source_kind": self.source_kind,
            "format_name": self.format_name,
            "mode": self.mode,
            "width": self.width,
            "height": self.height,
            "bit_depth": self.bit_depth,
            "has_alpha": self.has_alpha,
            "orientation": self.orientation,
            "orientation_applied": self.orientation_applied,
            "frame_count": self.frame_count,
            "source_profile": {
                "kind": self.source_profile.kind,
                "description": self.source_profile.description,
                "bytes_length": self.source_profile.bytes_length,
            },
            "transfer_state": self.transfer_state,
            "hdr_metadata": self.hdr_metadata,
            "raw_metadata": self.raw_metadata,
            "warnings": [warning.__dict__ for warning in self.warnings],
        }


@dataclass
class WorkingImage:
    """High-precision image handed to auto-base and film rendering stages."""

    pixels: np.ndarray
    working_space: str
    transfer_state: TransferState
    source_transfer_state: TransferState
    source_profile: SourceProfile
    hdr_metadata: dict[str, Any]
    orientation_applied: bool
    alpha_policy: AlphaPolicy
    bit_depth_in: int | None
    source_path: Path
    warnings: list[DecodeWarning] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError("WorkingImage.pixels must be HxWx3.")
        if self.pixels.dtype != np.float32:
            raise TypeError("WorkingImage.pixels must be float32.")
