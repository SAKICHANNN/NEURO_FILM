"""Frozen BN7 head-to-head evaluation of triangular transport versus AO6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.fivek_projection_curve_visual_product_value import _load_exact_json
from src.eval.fivek_triangular_logit_transport_visual_product_value import (
    ARMS,
    FiveKTriangularVisualError,
    run_visual_product_value,
    validate_contract as validate_base_contract,
)


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen challenger, model, source, and head-to-head policy."""

    validated = validate_base_contract(root, config)
    challenger_spec = config.get("challenger_evidence", {})
    challenger = _load_exact_json(
        root,
        challenger_spec["decision"],
        challenger_spec["decision_sha256"],
    )
    gate = config.get("automatic_gate", {})
    blind = config.get("blind_protocol", {})
    if (
        challenger.get("status") != challenger_spec.get("required_status")
        or challenger.get("next_branch")
        != challenger_spec.get("required_next_branch")
        or challenger.get("pass") is not True
        or config.get("comparison_reference_arm") != ARMS[0]
        or tuple(blind.get("primary_pair", ())) != (ARMS[0], ARMS[2])
        or int(blind.get("sources_per_round", -1)) != 10
        or int(blind.get("aggregate_choice_denominator", -1)) != 30
        or "minimum_median_adaptive_vs_direct_ao6_delta_e76" not in gate
        or "minimum_sources_with_adaptive_vs_direct_ao6_delta_e76_ge_0p5"
        not in gate
        or "minimum_median_adaptive_vs_global_delta_e76" in gate
    ):
        raise FiveKTriangularVisualError("BN7 incumbent boundary drift")
    return {**validated, "challenger": challenger}


def run_incumbent_comparison(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    """Render the unchanged operator and AO6 incumbent on the frozen source set."""

    validate_contract(root, config)
    return run_visual_product_value(
        root=root,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        software_commit=software_commit,
    )


def load_contract(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["load_contract", "run_incumbent_comparison", "validate_contract"]
