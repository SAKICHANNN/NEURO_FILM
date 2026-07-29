"""U6.P4F analytic effective-mark-loss physical-LOD audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.physical_density_conditioned_lod import (
    evaluate_density_conditioned_lod_with_compiler,
)
from src.film_physics.density_conditioned_structure import (
    compile_effective_mark_loss_profiles,
)


SCHEMA = "neuro_film.u6_p4f_effective_mark_loss_lod_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4f_effective_mark_loss_lod_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4F parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4F contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["compiler"]["per_image_fit_or_normalization_allowed"]
        or contract["compiler"]["empirical_parameter_fit_allowed"]
    ):
        raise ValueError("unsupported U6.P4F contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    _load_exact(
        root,
        parents["p4e_contract"],
        parents["p4e_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4e_decision"],
        parents["p4e_decision_sha256"],
    )
    if (
        decision["decision"]
        != "close_density_conditioned_lod_integration_retain_p4d_base_pitch_only"
        or not decision["next_leaf"].startswith("U6.P4F")
    ):
        raise ValueError("U6.P4E did not open the analytic successor")
    return contract, parent


def evaluate_effective_mark_loss_lod(
    contract: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    return evaluate_density_conditioned_lod_with_compiler(
        contract,
        parent,
        profile_compiler=compile_effective_mark_loss_profiles,
        report_schema=REPORT_SCHEMA,
    )


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_effective_mark_loss_lod",
    "load_contract",
]
