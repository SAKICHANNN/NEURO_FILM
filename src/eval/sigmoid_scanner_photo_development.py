"""U6.P4IK fixed-photo development of the retained sigmoid scanner chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.opponent_diffusion_photographic_confirmation import (
    CONTRACT_SCHEMA as P4HZ_SCHEMA,
)
from src.eval.opponent_diffusion_photographic_confirmation import (
    OpponentDiffusionRuntime,
    SigmoidCharacteristicScannerRuntime,
    _load_bound_json,
    evaluate_with_runtime,
)
from src.film_physics.compact_log_scanner_compiler import CompactLogScannerCompiler

SCHEMA = "neuro-film.u6-p4ik-sigmoid-scanner-photo-development-contract.v1"
RESULT_SCHEMA = "neuro-film.u6-p4ik-sigmoid-scanner-photo-development-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IK contract")
    return payload


def evaluate(contract: dict[str, Any], *, root: Path, output_dir: Path) -> dict[str, Any]:
    binding = contract["parents"]["p4ij_evidence"]
    evidence = _load_bound_json(root, binding, "P4IJ evidence")
    if (
        evidence.get("decision") != binding["required_decision"]
        or evidence.get("formal_runs", {}).get("stable_evidence_id")
        != binding["required_stable_evidence_id"]
    ):
        raise ValueError("P4IJ evidence drift")
    sigmoid = contract["sigmoid"]
    if max(evidence["metrics"]["curve_fit_rmse_red_green_blue"]) > sigmoid["maximum_curve_fit_rmse"]:
        raise ValueError("P4IJ curve-fit gate drift")
    row = contract["compiler"]
    compiler = CompactLogScannerCompiler(
        row["compiler_id"],
        tuple(tuple(values) for values in row["matrix_density_to_log10_rgb"]),
        tuple(row["bias_log10_rgb"]),
    )

    def build(base: OpponentDiffusionRuntime) -> OpponentDiffusionRuntime:
        return SigmoidCharacteristicScannerRuntime(
            components=base.components,
            correlation=base.correlation,
            candidate=base.candidate,
            native_toolchains={"backend": "python-p4ij-sigmoid-scanner-v1"},
            compiler=compiler,
            fit_samples=int(sigmoid["fit_samples"]),
            initial_slope=float(sigmoid["initial_slope"]),
            maximum_iterations=int(sigmoid["maximum_iterations"]),
        )

    adapted = dict(contract)
    adapted["schema"] = P4HZ_SCHEMA
    result = evaluate_with_runtime(
        adapted,
        root=root,
        output_dir=output_dir,
        runtime_builder=build,
        result_schema=RESULT_SCHEMA,
    )
    result["p4ik_config_sha256"] = hashlib.sha256(
        (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    return result


__all__ = ["evaluate", "load_contract"]
