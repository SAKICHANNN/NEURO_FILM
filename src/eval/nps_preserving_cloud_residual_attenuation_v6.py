"""Frozen U6.P4DW target-resolution cloud residual attenuation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.nps_preserving_cloud_residual_lod_v5 import (
    _apply_residual_gain,
    load_contract,
)
from src.eval.nps_preserving_cloud_residual_lod_v5 import evaluate as _evaluate_v5


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    report = _evaluate_v5(root, contract_path)
    report["schema"] = (
        "neuro_film.u6_p4dw_nps_preserving_cloud_residual_attenuation_report.v6"
    )
    return report


__all__ = ["_apply_residual_gain", "evaluate", "load_contract"]
