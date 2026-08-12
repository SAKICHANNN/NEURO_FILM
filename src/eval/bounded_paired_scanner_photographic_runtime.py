"""P4HI bounded copula execution inside the fixed P4HE photographic chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.neutral_base_photographic_ablation import sha256_file
from src.eval.paired_scanner_mtf_photographic_development import _evaluate_paired
from src.film_physics.bounded_histogram_copula import (
    apply_bounded_multipass_histogram_copula,
)
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = "neuro-film.u6-p4hi-bounded-paired-scanner-photographic-runtime-contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HI contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"P4HI parent drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(
    contract: dict[str, Any], *, root: Path, contact_sheet_path: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    p4he_result = _load_bound(root, parents["p4he_result"])
    p4he_contract = _load_bound(root, parents["p4he_contract"])
    p4hh_result = _load_bound(root, parents["p4hh_result"])
    p4hh_contract = _load_bound(root, parents["p4hh_contract"])
    if (
        p4he_result.get("decision")
        != parents["p4he_result"]["required_decision"]
        or p4hh_result.get("decision")
        != parents["p4hh_result"]["required_decision"]
    ):
        raise ValueError("P4HI parent decision drift")
    candidate = contract["candidate"]
    if (
        candidate["rank_bins"] != p4hh_contract["candidate"]["rank_bins"]
        or candidate["canonical_row_block_height"]
        != p4hh_contract["candidate"]["canonical_row_block_height"]
        or candidate["compact_contact_previews"] is not True
        or candidate["full_frame_visual_rows_retained"] is not False
    ):
        raise ValueError("P4HI execution policy drift")
    correlation = np.asarray(
        p4he_contract["candidate"]["correlation_matrix"], dtype=np.float64
    )

    def apply_physical(
        source: np.ndarray,
        profile: DensityConditionedThomasProfile,
        prior: ManufacturerCharacteristicPrior,
        seeds: tuple[int, int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return apply_bounded_multipass_histogram_copula(
            source,
            profile=profile,
            prior=prior,
            layer_seeds=seeds,
            correlation_matrix=correlation,
            canonical_receipt_row_block_height=candidate[
                "canonical_row_block_height"
            ],
            rank_bins=candidate["rank_bins"],
        )

    result = _evaluate_paired(
        p4he_contract,
        root=root,
        contact_sheet_path=contact_sheet_path,
        result_parent_name="p4hd_result",
        apply_physical_override=apply_physical,
        compact_visual_rows=True,
    )
    stable = result["stable"]
    stable["contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    stable["bounded_execution"] = {
        "rank_bins": candidate["rank_bins"],
        "canonical_row_block_height": candidate["canonical_row_block_height"],
        "maximum_peak_live_temporary_bytes": max(
            row["peak_live_temporary_bytes"] for row in stable["rows"]
        ),
        "full_frame_intermediate_count_excluding_input_output": max(
            row["full_frame_intermediate_count_excluding_input_output"]
            for row in stable["rows"]
        ),
        "full_frame_visual_rows_retained": False,
    }
    stable["claim_ceiling"] = contract["claim_ceiling"]
    result["schema"] = contract["schema"].replace("contract", "result")
    result["stable_evidence_id"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return result


__all__ = ["evaluate", "load_contract"]
