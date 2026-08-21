"""Private P87 Ultra HDR MatchView to Rec.2100 PQ PNG bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.preprocess.rec2100_pq_transfer import (
    save_absolute_rec2020_cdm2_to_pq_rgb16_png,
)

from .core_contracts import MATCH_PROFILE_ABSOLUTE_REC2020
from .ultra_hdr_ingress import (
    PreparedUltraHDRMatchViewV1,
    validate_prepared_ultrahdr_match_view_v1,
)


def publish_ultrahdr_match_view_pq_png_v1(
    value: PreparedUltraHDRMatchViewV1,
    path: Path,
    *,
    row_count: int = 64,
) -> tuple[str, np.ndarray]:
    """Publish a validated P87 absolute Rec.2020 view through U1.4G."""

    validate_prepared_ultrahdr_match_view_v1(value)
    if value.descriptor.profile_id != MATCH_PROFILE_ABSOLUTE_REC2020:
        raise ValueError("PQ publication requires the absolute Rec.2020 profile")
    return save_absolute_rec2020_cdm2_to_pq_rgb16_png(
        value.pixels,
        path,
        row_count=row_count,
    )


__all__ = ["publish_ultrahdr_match_view_pq_png_v1"]
