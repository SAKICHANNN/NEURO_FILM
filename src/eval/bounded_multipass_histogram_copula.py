"""P4HH parity and memory-model test for bounded multipass copula execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cross_layer_thomas_gamma_copula import _load_parent
from src.eval.neutral_base_photographic_ablation import sha256_file
from src.film_physics.bounded_histogram_copula import (
    apply_bounded_multipass_histogram_copula,
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

SCHEMA = "neuro-film.u6-p4hh-bounded-multipass-histogram-copula-contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HH contract")
    return payload


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    bound: dict[str, dict[str, Any]] = {}
    for name, binding in parents.items():
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"P4HH parent drift: {path}")
        bound[name] = json.loads(path.read_text(encoding="utf-8"))
    if (
        bound["p4hg_result"].get("decision")
        != parents["p4hg_result"]["required_decision"]
    ):
        raise ValueError("P4HH parent decision drift")
    hg = bound["p4hg_contract"]
    hc_binding = hg["parents"]["p4hc_contract"]
    hc_path = root / hc_binding["path"]
    if not hc_path.is_file() or sha256_file(hc_path) != hc_binding["sha256"]:
        raise ValueError("P4HH P4HC contract drift")
    hc = json.loads(hc_path.read_text(encoding="utf-8"))
    profile = DensityConditionedThomasProfile.from_dict(
        _load_parent(root, hc["parents"]["p4bw_bundle"])
    )
    prior = ManufacturerCharacteristicPrior.from_dict(
        _load_parent(root, hc["parents"]["p2q_bundle"])["prior"]
    )
    base_candidate = hc["candidate"]
    candidate = contract["candidate"]
    if (
        candidate["rank_bins"] != hg["candidate"]["rank_bins"]
        or candidate["canonical_row_block_height"]
        != base_candidate["canonical_receipt_row_block_height"]
        or candidate["full_frame_intermediates_allowed"] != ["input", "output"]
    ):
        raise ValueError("P4HH execution policy drift")
    shape = tuple(int(value) for value in base_candidate["field_shape"])
    seeds = tuple(int(value) for value in base_candidate["layer_field_seeds"])
    correlation = np.asarray(base_candidate["correlation_matrix"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for level in base_candidate["channel_levels"]:
        for ratio_values in base_candidate["rgb_ratios"]:
            pixel = float(level) * np.asarray(ratio_values, dtype=np.float64)
            base = np.broadcast_to(pixel, (*shape, 3)).copy()
            reference, _ = apply_cross_layer_thomas_gamma_copula(
                base,
                profile=profile,
                prior=prior,
                layer_seeds=seeds,
                correlation_matrix=correlation,
                canonical_receipt_row_block_height=candidate[
                    "canonical_row_block_height"
                ],
                rank_bins=candidate["rank_bins"],
            )
            streamed, diagnostics = apply_bounded_multipass_histogram_copula(
                base,
                profile=profile,
                prior=prior,
                layer_seeds=seeds,
                correlation_matrix=correlation,
                canonical_receipt_row_block_height=candidate[
                    "canonical_row_block_height"
                ],
                rank_bins=candidate["rank_bins"],
            )
            repeated, repeated_diagnostics = apply_bounded_multipass_histogram_copula(
                base,
                profile=profile,
                prior=prior,
                layer_seeds=seeds,
                correlation_matrix=correlation,
                canonical_receipt_row_block_height=candidate[
                    "canonical_row_block_height"
                ],
                rank_bins=candidate["rank_bins"],
            )
            error = np.abs(streamed.astype(np.float64) - reference.astype(np.float64))
            rows.append(
                {
                    "level": level,
                    "rgb_ratio": ratio_values,
                    "output_sha256": hashlib.sha256(
                        memoryview(streamed).cast("B")
                    ).hexdigest(),
                    "rmse_vs_p4hg": float(np.sqrt(np.mean(error * error))),
                    "p999_absolute_vs_p4hg": float(np.quantile(error, 0.999)),
                    "maximum_absolute_vs_p4hg": float(np.max(error)),
                    "repeat_exact": bool(np.array_equal(streamed, repeated)),
                    "diagnostics_exact": diagnostics == repeated_diagnostics,
                    "peak_live_temporary_bytes": diagnostics[
                        "peak_live_temporary_bytes"
                    ],
                    "full_frame_intermediate_count_excluding_input_output": diagnostics[
                        "full_frame_intermediate_count_excluding_input_output"
                    ],
                    "finite_unit_output": bool(
                        np.all(np.isfinite(streamed))
                        and np.all(streamed >= 0.0)
                        and np.all(streamed <= 1.0)
                    ),
                }
            )
    gates = contract["automatic_gates"]
    checks = {
        "rmse": max(row["rmse_vs_p4hg"] for row in rows)
        <= gates["maximum_output_rmse_vs_p4hg"],
        "p999": max(row["p999_absolute_vs_p4hg"] for row in rows)
        <= gates["maximum_output_p999_absolute_vs_p4hg"],
        "maximum": max(row["maximum_absolute_vs_p4hg"] for row in rows)
        <= gates["maximum_output_absolute_vs_p4hg"],
        "temporary_budget": max(row["peak_live_temporary_bytes"] for row in rows)
        <= gates["maximum_peak_live_temporary_bytes"],
        "repeat_identity": all(
            row["repeat_exact"] and row["diagnostics_exact"] for row in rows
        ),
        "finite_unit_output": all(row["finite_unit_output"] for row in rows),
        "no_full_frame_intermediate": all(
            row["full_frame_intermediate_count_excluding_input_output"] == 0
            for row in rows
        ),
    }
    stable = {
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
        "maximum_output_rmse_vs_p4hg": max(row["rmse_vs_p4hg"] for row in rows),
        "maximum_output_p999_absolute_vs_p4hg": max(
            row["p999_absolute_vs_p4hg"] for row in rows
        ),
        "maximum_output_absolute_vs_p4hg": max(
            row["maximum_absolute_vs_p4hg"] for row in rows
        ),
        "maximum_peak_live_temporary_bytes": max(
            row["peak_live_temporary_bytes"] for row in rows
        ),
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "decision": contract["decision_if_pass"]
        if all(checks.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("contract", "result"),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
