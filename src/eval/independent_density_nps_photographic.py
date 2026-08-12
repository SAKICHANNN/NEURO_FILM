"""P4GW photographic development of an independent density NPS field."""

from __future__ import annotations

import hashlib
import json
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
from src.film_physics.independent_density_nps import apply_independent_density_nps
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime

SCHEMA = (
    "neuro-film.u6-p4gw-independent-density-nps-photographic-development-contract.v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4GW contract")
    return payload


def evaluate(
    contract: dict[str, Any], *, root: Path, contact_sheet_path: Path
) -> dict[str, Any]:
    cloud_parent = contract["parents"]["cloud_family"]
    cloud_path = root / cloud_parent["path"]
    cloud = json.loads(cloud_path.read_text(encoding="utf-8"))
    density_parent = contract["parents"]["safe_density_primitive"]
    density_path = root / density_parent["path"]
    density = json.loads(density_path.read_text(encoding="utf-8"))
    if (
        sha256_file(cloud_path) != cloud_parent["sha256"]
        or cloud["decision"] != cloud_parent["required_decision"]
    ):
        raise ValueError("P4GW cloud parent drift")
    if (
        sha256_file(density_path) != density_parent["sha256"]
        or density["automatic"]["automatic_pass"]
        is not density_parent["required_automatic_pass"]
        or density["visual"]["severe_artifact_pass"]
        is not density_parent["required_severe_artifact_pass"]
        or bool(density["visual"]["winning_arms"])
        is not density_parent["required_visual_value_pass"]
    ):
        raise ValueError("P4GW safe density parent drift")
    source_contract = contract["source"]
    manifest_path = root / source_contract["manifest"]
    if sha256_file(manifest_path) != source_contract["manifest_sha256"]:
        raise ValueError("P4GW source manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        len(manifest) != source_contract["expected_rows"]
        or len({row["make"] for row in manifest})
        != source_contract["expected_camera_makes"]
    ):
        raise ValueError("P4GW source population drift")

    gates = contract["automatic_gates"]
    candidate = contract["candidate"]
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    with __import__("tempfile").TemporaryDirectory(
        ignore_cleanup_errors=True
    ) as directory:
        standard = _runtime(Path(directory))
        for index, source_row in enumerate(manifest):
            source_encoded, source = _load_source(source_row, root)
            physical, diagnostics = apply_independent_density_nps(
                source,
                density_sigma=float(candidate["density_sigma"]),
                seed=int(candidate["field_seed"])
                + index * int(candidate["field_seed_stride"]),
            )
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
                "physical_residual_rms": float(
                    np.sqrt(np.mean((physical.astype(np.float64) - source) ** 2))
                ),
                "maximum_limited_fraction": diagnostics["limited_fraction"],
                "field_mean": diagnostics["field_mean"],
                "field_std": diagnostics["field_std"],
                "hard_clipping_used": diagnostics["hard_clipping_used"] != 0.0,
                "combined_vs_ao6_p95_abs": float(np.quantile(absolute, 0.95)),
                "combined_vs_ao6_p99_abs": float(np.quantile(absolute, 0.99)),
                "flat_region_p99_abs": _flat_region_p99(source, difference),
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
    p95 = float(np.quantile([row["combined_vs_ao6_p95_abs"] for row in rows], 0.95))
    p99 = float(np.quantile([row["combined_vs_ao6_p99_abs"] for row in rows], 0.99))
    checks = {
        "physical_residual": min(row["physical_residual_rms"] for row in rows)
        >= gates["minimum_per_row_physical_residual_rms"],
        "limited_fraction": max(row["maximum_limited_fraction"] for row in rows)
        <= gates["maximum_per_row_limited_fraction"],
        "nontrivial_population": p95
        >= gates["minimum_population_p95_combined_vs_ao6_abs"],
        "population_tail": p99 <= gates["maximum_population_p99_combined_vs_ao6_abs"],
        "flat_regions": max(row["flat_region_p99_abs"] for row in rows)
        <= gates["maximum_flat_region_p99_combined_vs_ao6_abs"],
        "isolated_excursions": sum(row["isolated_excursion_count"] for row in rows)
        <= gates["maximum_isolated_excursion_count"],
        "new_boundaries": max(row["new_boundary_fraction_vs_ao6"] for row in rows)
        <= gates["maximum_new_boundary_fraction_vs_ao6"],
        "field_zero_dc": max(abs(row["field_mean"]) for row in rows)
        <= gates["maximum_field_mean_absolute"],
        "field_unit_std": max(abs(row["field_std"] - 1.0) for row in rows)
        <= gates["maximum_field_std_absolute_error"],
        "finite": all(row["finite"] for row in rows),
        "no_hard_clipping": not any(row["hard_clipping_used"] for row in rows),
    }
    sheet_sha = _contact_sheet(visual_rows, contact_sheet_path)
    stable = {
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "manifest_sha256": source_contract["manifest_sha256"],
        "rows": rows,
        "population_p95_combined_vs_ao6_abs": p95,
        "population_p99_combined_vs_ao6_abs": p99,
        "contact_sheet_sha256": sheet_sha,
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


__all__ = ["evaluate", "load_contract"]
