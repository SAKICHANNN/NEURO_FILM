"""P4HE paired downstream scanner-MTF test of P4HD structure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.layer_gamma_photographic_development import (
    _evaluate_photographic,
    sha256_file,
)
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.independent_density_nps import (
    apply_cross_layer_thomas_gamma_copula,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.spatial_response import (
    SpatialResponseProfile,
    apply_scanner_mtf,
)

SCHEMA = (
    "neuro-film.u6-p4he-paired-scanner-mtf-photographic-development-contract.v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HE contract")
    return payload


def evaluate(
    contract: dict[str, Any], *, root: Path, contact_sheet_path: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    p4fa_result = parents["p4fa_result"]
    p4fa_contract_binding = parents["p4fa_contract"]
    for binding in (p4fa_result, p4fa_contract_binding):
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"P4HE parent drift: {path}")
    p4fa_evidence = json.loads((root / p4fa_result["path"]).read_text("utf-8"))
    p4fa_contract = json.loads(
        (root / p4fa_contract_binding["path"]).read_text("utf-8")
    )
    if p4fa_evidence.get("decision") != p4fa_result["required_decision"]:
        raise ValueError("P4HE P4FA decision drift")

    candidate = contract["candidate"]
    frozen_spatial = p4fa_contract["spatial_profiles"]
    if (
        candidate["scanner_mtf_sigma_pixels_rgb"]
        != frozen_spatial["scanner_mtf_sigma_pixels_rgb"]
        or candidate["gaussian_truncate"] != frozen_spatial["gaussian_truncate"]
        or candidate["baseline_and_candidate_share_downstream_stage"] is not True
        or candidate["amplitude_multiplier"] != 1.0
        or candidate["cohort_fitting_allowed"] is not False
        or candidate["hard_clipping_allowed"] is not False
        or candidate["posthoc_limiting_allowed"] is not False
    ):
        raise ValueError("P4HE paired scanner policy drift")

    correlation = np.asarray(candidate["correlation_matrix"], dtype=np.float64)
    row_block_height = int(candidate["canonical_receipt_row_block_height"])
    scanner_sigmas = tuple(
        float(value) for value in candidate["scanner_mtf_sigma_pixels_rgb"]
    )
    scanner_profile = SpatialResponseProfile(
        pixel_pitch_um=1.0,
        forward_scatter_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_gain_rgb=(0.0, 0.0, 0.0),
        dye_diffusion_sigma_um_rgb=(0.0, 0.0, 0.0),
        scanner_mtf_sigma_um_rgb=scanner_sigmas,
        gaussian_truncate=float(candidate["gaussian_truncate"]),
    )

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

    def finalize_pair(
        source: np.ndarray, physical: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        baseline = np.ascontiguousarray(
            apply_scanner_mtf(source.astype(np.float64), scanner_profile),
            dtype=np.float32,
        )
        candidate_output = np.ascontiguousarray(
            apply_scanner_mtf(physical.astype(np.float64), scanner_profile),
            dtype=np.float32,
        )
        return baseline, candidate_output, {
            "paired_scanner_baseline_sha256": hashlib.sha256(
                memoryview(baseline).cast("B")
            ).hexdigest(),
            "paired_scanner_candidate_sha256": hashlib.sha256(
                memoryview(candidate_output).cast("B")
            ).hexdigest(),
            "scanner_mtf_sigma_pixels_rgb": list(scanner_sigmas),
        }

    return _evaluate_photographic(
        contract,
        root=root,
        contact_sheet_path=contact_sheet_path,
        apply_physical=apply_physical,
        result_parent_name="p4hd_result",
        finalize_pair=finalize_pair,
    )


__all__ = ["evaluate", "load_contract"]
