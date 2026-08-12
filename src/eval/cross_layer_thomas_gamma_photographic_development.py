"""P4HD photographic test of the fixed cross-layer Thomas/Gamma copula."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.layer_gamma_photographic_development import _evaluate_photographic
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.independent_density_nps import (
    apply_cross_layer_thomas_gamma_copula,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = (
    "neuro-film.u6-p4hd-cross-layer-thomas-gamma-photographic-development-contract.v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HD contract")
    return payload


def evaluate(
    contract: dict[str, Any], *, root: Path, contact_sheet_path: Path
) -> dict[str, Any]:
    candidate = contract["candidate"]
    correlation = np.asarray(candidate["correlation_matrix"], dtype=np.float64)
    row_block_height = int(candidate["canonical_receipt_row_block_height"])

    def apply_physical(
        source: np.ndarray,
        profile: DensityConditionedThomasProfile,
        prior: ManufacturerCharacteristicPrior,
        seeds: tuple[int, int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return apply_cross_layer_thomas_gamma_copula(
            source,
            profile=profile,
            prior=prior,
            layer_seeds=seeds,
            correlation_matrix=correlation,
            canonical_receipt_row_block_height=row_block_height,
        )

    return _evaluate_photographic(
        contract,
        root=root,
        contact_sheet_path=contact_sheet_path,
        apply_physical=apply_physical,
    )


__all__ = ["evaluate", "load_contract"]
