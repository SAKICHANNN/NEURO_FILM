"""P4HA photographic development of per-layer support-matched density."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.native_cloud_residual_display import _apply_residual
from src.eval.neutral_base_photographic_ablation import (
    _contact_sheet,
    _load_source,
    _new_boundary_fraction,
    sha256_file,
)
from src.eval.physical_spatial_photographic_stress import (
    _flat_region_p99,
    _isolated_excursions,
)
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.independent_density_nps import (
    apply_layer_support_matched_gamma_density,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.filmfx.fast_blur import gaussian_filter_safe
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime

SCHEMA = "neuro-film.u6-p4ha-layer-gamma-photographic-development-contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HA contract")
    return payload


def _load_bound_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"P4HA parent drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _high_frequency_chroma_p999(residual: np.ndarray) -> float:
    chroma = residual - residual.mean(axis=2, keepdims=True)
    high_frequency = chroma - gaussian_filter_safe(chroma, sigma=(1.2, 1.2, 0.0))
    return float(np.quantile(np.max(np.abs(high_frequency), axis=2), 0.999))


def _evaluate_photographic(
    contract: dict[str, Any],
    *,
    root: Path,
    contact_sheet_path: Path,
    apply_physical: Callable[
        [
            np.ndarray,
            DensityConditionedThomasProfile,
            ManufacturerCharacteristicPrior,
            tuple[int, int, int],
        ],
        tuple[np.ndarray, dict[str, Any]],
    ],
) -> dict[str, Any]:
    parents = contract["parents"]
    result_names = [name for name in parents if name.endswith("_result")]
    if len(result_names) != 1:
        raise ValueError("P4HA/P4HB requires one result parent")
    result_binding = parents[result_names[0]]
    result = _load_bound_json(root, result_binding)
    profile_payload = _load_bound_json(root, parents["p4bw_bundle"])
    prior_payload = _load_bound_json(root, parents["p2q_bundle"])
    profile = DensityConditionedThomasProfile.from_dict(profile_payload)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if (
        result.get("decision") != result_binding["required_decision"]
        or profile.identity() != parents["p4bw_bundle"]["required_profile_id"]
        or prior.identity() != parents["p2q_bundle"]["required_prior_id"]
    ):
        raise ValueError("P4HA parent identity drift")

    source_contract = contract["source"]
    manifest_path = root / source_contract["manifest"]
    if sha256_file(manifest_path) != source_contract["manifest_sha256"]:
        raise ValueError("P4HA source manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        len(manifest) != source_contract["expected_rows"]
        or len({row["make"] for row in manifest})
        != source_contract["expected_camera_makes"]
        or any(
            row.get("allowed_use") != source_contract["required_allowed_use"]
            or row.get("rights_scope") != source_contract["required_rights_scope"]
            or row.get("decoded_color_state") != source_contract["required_color_state"]
            for row in manifest
        )
    ):
        raise ValueError("P4HA source population drift")

    candidate = contract["candidate"]
    if (
        candidate["amplitude_multiplier"] != 1.0
        or candidate["cohort_fitting_allowed"] is not False
        or candidate["hard_clipping_allowed"] is not False
        or candidate["posthoc_limiting_allowed"] is not False
    ):
        raise ValueError("P4HA candidate policy drift")
    base_seeds = tuple(int(value) for value in candidate["layer_field_seeds"])
    stride = int(candidate["field_seed_stride_per_source"])
    gates = contract["automatic_gates"]
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    with __import__("tempfile").TemporaryDirectory(
        ignore_cleanup_errors=True
    ) as directory:
        standard = _runtime(Path(directory))
        for index, source_row in enumerate(manifest):
            source_encoded, source = _load_source(source_row, root)
            seeds = tuple(value + index * stride for value in base_seeds)
            physical, diagnostics = apply_physical(source, profile, prior, seeds)
            ao6_linear = _apply_residual(standard, source)
            combined_linear = _apply_residual(standard, physical)
            physical_encoded = np.ascontiguousarray(
                linear_srgb_to_encoded(physical.astype(np.float64)), np.float32
            )
            ao6_encoded = np.ascontiguousarray(
                linear_srgb_to_encoded(ao6_linear.astype(np.float64)), np.float32
            )
            combined_encoded = np.ascontiguousarray(
                linear_srgb_to_encoded(combined_linear.astype(np.float64)), np.float32
            )
            difference = combined_encoded - ao6_encoded
            absolute = np.abs(difference)
            row = {
                "id": source_row["id"],
                "make": source_row["make"],
                "decoded_sha256": source_row["decoded_sha256"],
                "shape": list(source.shape),
                "layer_field_seeds": list(seeds),
                "physical_output_sha256": hashlib.sha256(
                    memoryview(physical).cast("B")
                ).hexdigest(),
                "combined_output_sha256": hashlib.sha256(
                    memoryview(combined_encoded).cast("B")
                ).hexdigest(),
                "finite": bool(
                    np.all(np.isfinite(physical))
                    and np.all(np.isfinite(combined_encoded))
                ),
                "physical_residual_rms": diagnostics["bounded_residual_rms"],
                "minimum_target_sigma_d": diagnostics["minimum_target_sigma_d"],
                "maximum_target_sigma_d": diagnostics["maximum_target_sigma_d"],
                "support_degenerate_fraction": diagnostics[
                    "support_degenerate_fraction"
                ],
                "minimum_developed_density": diagnostics["minimum_developed_density"],
                "limited_fraction": diagnostics["limited_fraction"],
                "hard_clipping_used": diagnostics["hard_clipping_used"] != 0.0,
                "combined_vs_ao6_p95_abs": float(np.quantile(absolute, 0.95)),
                "combined_vs_ao6_p99_abs": float(np.quantile(absolute, 0.99)),
                "flat_region_p99_abs": _flat_region_p99(source, difference),
                "high_frequency_chroma_p999": _high_frequency_chroma_p999(difference),
                "isolated_excursion_count": _isolated_excursions(
                    difference,
                    threshold=float(gates["isolated_excursion_threshold"]),
                    radius=int(gates["isolated_support_radius_pixels"]),
                    minimum_support=int(gates["minimum_isolated_support_count"]),
                ),
                "new_boundary_fraction_vs_ao6": _new_boundary_fraction(
                    ao6_encoded, combined_encoded
                ),
            }
            if "receipt_ids" in diagnostics:
                row["receipt_ids"] = diagnostics["receipt_ids"]
            rows.append(row)
            visual_rows.append(
                {
                    "id": row["id"],
                    "make": row["make"],
                    "source": source_encoded,
                    "ao6": ao6_encoded,
                    "physical": physical_encoded,
                    "combined": combined_encoded,
                }
            )

    population_p95 = float(
        np.quantile([row["combined_vs_ao6_p95_abs"] for row in rows], 0.95)
    )
    population_p99 = float(
        np.quantile([row["combined_vs_ao6_p99_abs"] for row in rows], 0.99)
    )
    checks = {
        "physical_residual": min(row["physical_residual_rms"] for row in rows)
        >= gates["minimum_per_row_physical_residual_rms"],
        "nontrivial_population": population_p95
        >= gates["minimum_population_p95_combined_vs_ao6_abs"],
        "population_tail": population_p99
        <= gates["maximum_population_p99_combined_vs_ao6_abs"],
        "flat_regions": max(row["flat_region_p99_abs"] for row in rows)
        <= gates["maximum_flat_region_p99_combined_vs_ao6_abs"],
        "high_frequency_chroma": max(row["high_frequency_chroma_p999"] for row in rows)
        <= gates["maximum_high_frequency_chroma_p999"],
        "isolated_excursions": sum(row["isolated_excursion_count"] for row in rows)
        <= gates["maximum_isolated_excursion_count"],
        "new_boundaries": max(row["new_boundary_fraction_vs_ao6"] for row in rows)
        <= gates["maximum_new_boundary_fraction_vs_ao6"],
        "finite": all(row["finite"] for row in rows),
        "no_clipping_or_limiting": not any(
            row["hard_clipping_used"] or row["limited_fraction"] != 0.0 for row in rows
        ),
    }
    contact_sha = _contact_sheet(visual_rows, contact_sheet_path)
    stable = {
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "manifest_sha256": source_contract["manifest_sha256"],
        "profile_id": profile.identity(),
        "prior_id": prior.identity(),
        "rows": rows,
        "population_p95_combined_vs_ao6_abs": population_p95,
        "population_p99_combined_vs_ao6_abs": population_p99,
        "worst_high_frequency_chroma_p999": max(
            row["high_frequency_chroma_p999"] for row in rows
        ),
        "contact_sheet_sha256": contact_sha,
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
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


def evaluate(
    contract: dict[str, Any], *, root: Path, contact_sheet_path: Path
) -> dict[str, Any]:
    def apply_physical(
        source: np.ndarray,
        profile: DensityConditionedThomasProfile,
        prior: ManufacturerCharacteristicPrior,
        seeds: tuple[int, int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return apply_layer_support_matched_gamma_density(
            source, profile=profile, prior=prior, layer_seeds=seeds
        )

    return _evaluate_photographic(
        contract,
        root=root,
        contact_sheet_path=contact_sheet_path,
        apply_physical=apply_physical,
    )


__all__ = ["_evaluate_photographic", "evaluate", "load_contract"]
