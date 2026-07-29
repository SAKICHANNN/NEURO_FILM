"""U6.P4G crop-boundary audit for density-conditioned material structure."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_density_conditioned_lod import (
    evaluate_density_conditioned_lod_with_compiler,
)
from src.eval.physical_density_conditioned_structure import (
    profiles_from_contract,
)
from src.film_physics.density_conditioned_structure import (
    DensityConditionedLayerProfile,
    compile_effective_mark_loss_profiles,
    render_density_conditioned_structure,
)


SCHEMA = "neuro_film.u6_p4g_film_structure_boundary_semantics_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4g_film_structure_boundary_semantics_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4G parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4G contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["compiler"]["per_image_fit_or_normalization_allowed"]
        or contract["compiler"]["empirical_parameter_fit_allowed"]
        or contract["compiler"]["output_clipping_allowed"]
    ):
        raise ValueError("unsupported U6.P4G contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    _load_exact(
        root,
        parents["p4f_contract"],
        parents["p4f_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4f_decision"],
        parents["p4f_decision_sha256"],
    )
    if (
        decision["decision"]
        != "close_effective_mark_loss_lod_retain_p4d_base_pitch_only"
        or not decision["next_leaf"].startswith("U6.P4G")
    ):
        raise ValueError("U6.P4F did not open the boundary audit")
    return contract, parent


def _normalized_profiles(
    profiles: tuple[DensityConditionedLayerProfile, ...],
) -> tuple[DensityConditionedLayerProfile, ...]:
    return tuple(
        replace(profile, boundary_mode="normalized-support-v1")
        for profile in profiles
    )


def _maximum_flat_boundary_mean_error(
    parent: dict[str, Any], *, seed_offset: int
) -> float:
    profiles = _normalized_profiles(
        profiles_from_contract(parent, seed_offset=seed_offset)
    )
    level = 1.2
    shape = (384, 512, len(profiles))
    target = np.full(shape, level, dtype=np.float64)
    density = render_density_conditioned_structure(
        target, profiles
    ).density.astype(np.float64)
    mask = np.zeros(shape[:2], dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return max(
        abs(float(np.mean(density[..., channel][mask])) - level)
        for channel in range(len(profiles))
    )


def evaluate_structure_boundary_semantics(
    contract: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    incumbent = evaluate_density_conditioned_lod_with_compiler(
        contract,
        parent,
        profile_compiler=compile_effective_mark_loss_profiles,
        report_schema=REPORT_SCHEMA,
    )
    candidate = evaluate_density_conditioned_lod_with_compiler(
        contract,
        parent,
        profile_compiler=compile_effective_mark_loss_profiles,
        report_schema=REPORT_SCHEMA,
        base_profile_transform=_normalized_profiles,
    )
    zero_fill_factor_2 = max(
        row["maximum_direct_to_reference_local_mean_difference"]
        for split in ("development", "confirmation")
        for row in incumbent[split]["factors"]
        if row["factor"] == 2
    )
    flat_errors = {
        "development": _maximum_flat_boundary_mean_error(
            parent,
            seed_offset=int(
                contract["synthetic_scene"]["development_seed_offset"]
            ),
        ),
        "confirmation": _maximum_flat_boundary_mean_error(
            parent,
            seed_offset=int(
                contract["synthetic_scene"]["confirmation_seed_offset"]
            ),
        ),
    }
    gates = contract["automatic_gates"]
    extra_checks = {
        "zero_fill_negative_control": zero_fill_factor_2
        >= float(
            gates[
                "minimum_zero_fill_factor_2_reference_local_mean_failure"
            ]
        ),
        "normalized_boundary_flat_mean": max(flat_errors.values())
        <= float(gates["maximum_normalized_boundary_flat_mean_error"]),
    }
    checks = {**candidate["checks"], **extra_checks}
    passed = all(checks.values())
    candidate.update(
        {
            "checks": checks,
            "automatic_pass": passed,
            "decision": (
                contract["branch_rule"]["pass"]
                if passed
                else contract["branch_rule"]["fail"]
            ),
            "incumbent_zero_fill_factor_2_maximum_reference_local_mean_difference": (
                zero_fill_factor_2
            ),
            "normalized_boundary_flat_mean_errors": flat_errors,
        }
    )
    return candidate


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_structure_boundary_semantics",
    "load_contract",
]
