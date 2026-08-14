"""CB74 third source-disjoint analytic colour confirmation adjudication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.analytic_y_chromaticity_adjudication import (
    AnalyticYChromaticityAdjudicationError,
    adjudicate,
)

SCHEMA = "neuro_film.u5_r2cb74_analytic_y_chromaticity_adjudication_contract.v1"
EXPERIMENT_ID = "U5.R2CB74A"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise AnalyticYChromaticityAdjudicationError("CB74A contract drift")
    return payload


__all__ = ["adjudicate", "load_contract"]
