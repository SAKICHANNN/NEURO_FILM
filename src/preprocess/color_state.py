"""Fail-closed output-claim policy for the current uncalibrated renderer."""

from __future__ import annotations

from typing import Any

from .types import WorkingImage


def resolve_look_approximation_claim(working: WorkingImage) -> dict[str, Any]:
    """Return the strongest truthful claim allowed for the current renderer."""
    unknown = working.source_transfer_state == "unknown"
    return {
        "render_mode": "Style-safe",
        "output_label": "film-inspired",
        "evidence_grade": "look-approximation",
        "input_color_state": working.source_transfer_state,
        "color_state_policy": (
            "look_approximation_fail_closed"
            if unknown
            else "look_approximation_only"
        ),
        "calibrated_reference_allowed": False,
    }
