"""Fresh photographic automatic confirmation of the fixed P4HX mechanism."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.opponent_dye_cloud_diffusion_d0 import (
    _diffuse_opponent_density,
    _diffused_opponent_density_residual,
)
from src.eval.p4hu_ao6_value import (
    _load_bound_json,
    _load_manifest,
    _scanner_profile,
    aggregate_arm_rows,
    automatic_arm_checks,
    evaluate_source_arms,
)
from src.film_physics.analytical_density_direction import (
    apply_analytical_density_direction,
)
from src.film_physics.bounded_photographic_profile import (
    reconstruct_bounded_photographic_profile,
)
from src.film_physics.compact_log_scanner_compiler import CompactLogScannerCompiler
from src.film_physics.independent_density_nps import (
    apply_cross_layer_thomas_gamma_copula,
)
from src.film_physics.relative_display_characteristic_ingress import (
    relative_display_to_finite_density_transmittance,
)
from src.film_physics.sigmoid_characteristic import (
    fit_sigmoid_characteristic,
    render_sigmoid_scanner_positive,
)

CONTRACT_SCHEMA = (
    "neuro-film.u6-p4hz-opponent-diffusion-photographic-confirmation-contract.v1"
)
RESULT_SCHEMA = (
    "neuro-film.u6-p4hz-opponent-diffusion-photographic-confirmation-result.v1"
)


def _canonical_sha256(value: Any) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported U6.P4HZ contract")
    return payload


def _load_ao6(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    document = _load_bound_json(root, binding, "AO6 artifact")
    artifact = document.get("artifact", document)
    if artifact.get("bundle_sha256") != binding["required_bundle_sha256"]:
        raise ValueError("AO6 artifact bundle drift")
    return artifact["component_payloads"]["ao6-source-context-display-look"]


@dataclass
class OpponentDiffusionRuntime:
    """Fixed Python reference realization of the P4HX developed-density stage."""

    components: Any
    correlation: np.ndarray
    candidate: dict[str, Any]
    native_toolchains: dict[str, Any]

    def apply_source(
        self, source: np.ndarray, *, seeds: tuple[int, int, int]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        control, diagnostics = apply_cross_layer_thomas_gamma_copula(
            source,
            profile=self.components.profile,
            prior=self.components.prior,
            layer_seeds=seeds,
            correlation_matrix=self.correlation,
            canonical_receipt_row_block_height=int(
                self.candidate["canonical_row_block_height"]
            ),
            rank_bins=int(self.candidate["rank_bins"]),
        )
        output, common_error = _diffuse_opponent_density(
            source,
            control,
            sigmas=self.candidate["dye_diffusion_sigma_pixels_cmy"],
            truncate=float(self.candidate["dye_diffusion_truncate"]),
        )
        residual = output.astype(np.float64) - source.astype(np.float64)
        diagnostics.update(
            {
                "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
                "minimum_developed_density": float(
                    np.min(-np.log10(np.maximum(output, np.finfo(np.float32).tiny)))
                ),
                "limited_fraction": 0.0,
                "hard_clipping_used": 0.0,
                "rank_bins": int(self.candidate["rank_bins"]),
                "pass_count": 4,
                "canonical_row_block_height": int(
                    self.candidate["canonical_row_block_height"]
                ),
                "peak_live_temporary_bytes": int(source.nbytes * 6),
                "full_frame_intermediate_count_excluding_input_output": 4,
                "opponent_common_density_error": common_error,
            }
        )
        return output, diagnostics


@dataclass
class AnalyticalOpponentDiffusionRuntime(OpponentDiffusionRuntime):
    """P4IA direction-preserving envelope around fixed P4HX structure."""

    def apply_source(
        self, source: np.ndarray, *, seeds: tuple[int, int, int]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        control, diagnostics = apply_cross_layer_thomas_gamma_copula(
            source,
            profile=self.components.profile,
            prior=self.components.prior,
            layer_seeds=seeds,
            correlation_matrix=self.correlation,
            canonical_receipt_row_block_height=int(
                self.candidate["canonical_row_block_height"]
            ),
            rank_bins=int(self.candidate["rank_bins"]),
        )
        density, common_error = _diffused_opponent_density_residual(
            source,
            control,
            sigmas=self.candidate["dye_diffusion_sigma_pixels_cmy"],
            truncate=float(self.candidate["dye_diffusion_truncate"]),
        )
        output, envelope = apply_analytical_density_direction(source, density)
        residual = output.astype(np.float64) - source.astype(np.float64)
        diagnostics.update(
            {
                "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
                "minimum_developed_density": float(
                    np.min(-np.log10(np.maximum(output, np.finfo(np.float32).tiny)))
                ),
                "limited_fraction": envelope["limited_fraction"],
                "hard_clipping_used": 0.0,
                "rank_bins": int(self.candidate["rank_bins"]),
                "pass_count": 4,
                "canonical_row_block_height": int(
                    self.candidate["canonical_row_block_height"]
                ),
                "peak_live_temporary_bytes": int(source.nbytes * 6),
                "full_frame_intermediate_count_excluding_input_output": 4,
                "opponent_common_density_error": common_error,
                "minimum_density_direction_scale": envelope[
                    "minimum_density_direction_scale"
                ],
                "maximum_density_direction_scale_error": envelope[
                    "maximum_density_direction_scale_error"
                ],
            }
        )
        return output, diagnostics


@dataclass
class CharacteristicIngressAnalyticalRuntime(AnalyticalOpponentDiffusionRuntime):
    """P4IC finite characteristic ingress followed by P4IA structure."""

    def apply_source(
        self, source: np.ndarray, *, seeds: tuple[int, int, int]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, transmittance, ingress = relative_display_to_finite_density_transmittance(
            source, self.components.prior
        )
        output, diagnostics = super().apply_source(transmittance, seeds=seeds)
        diagnostics["ingress_zero_input_transmittance_minimum"] = ingress[
            "zero_input_transmittance_minimum"
        ]
        diagnostics["ingress_one_input_transmittance_minimum"] = ingress[
            "one_input_transmittance_minimum"
        ]
        return output, diagnostics


@dataclass
class SigmoidCharacteristicScannerRuntime(AnalyticalOpponentDiffusionRuntime):
    """Sigmoid characteristic ingress, P4IA structure and compact scanner."""

    compiler: CompactLogScannerCompiler
    fit_samples: int = 4097
    initial_slope: float = 8.0
    maximum_iterations: int = 20000

    def apply_source(
        self, source: np.ndarray, *, seeds: tuple[int, int, int]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        curves, fit_rmse = fit_sigmoid_characteristic(
            self.components.prior,
            samples=self.fit_samples,
            initial_slope=self.initial_slope,
            maximum_iterations=self.maximum_iterations,
        )
        density = np.stack(
            [curve.apply_normalized(source[..., channel]) for channel, curve in enumerate(curves)],
            axis=-1,
        )
        base_transmittance = np.ascontiguousarray(
            np.power(10.0, -density), dtype=np.float32
        )
        structured_transmittance, diagnostics = super().apply_source(
            base_transmittance, seeds=seeds
        )
        output, scanner = render_sigmoid_scanner_positive(
            source,
            structured_transmittance,
            curves=curves,
            compiler=self.compiler,
        )
        residual = output.astype(np.float64) - source.astype(np.float64)
        diagnostics.update(
            {
                "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
                "sigmoid_curve_fit_rmse_maximum": max(fit_rmse),
                "scanner_minimum_shared_scale": scanner["minimum_shared_scale"],
                "scanner_limited_fraction": scanner["limited_fraction"],
                "scanner_maximum_shared_direction_error": scanner[
                    "maximum_shared_direction_error"
                ],
            }
        )
        return output, diagnostics


def _runtime(
    profile: dict[str, Any], candidate: dict[str, Any]
) -> OpponentDiffusionRuntime:
    execution = profile["execution"]
    if (
        execution["layer_field_seeds"] != candidate["layer_field_seeds"]
        or execution["field_seed_stride_per_source"]
        != candidate["field_seed_stride_per_source"]
        or execution["rank_bins"] != candidate["rank_bins"]
        or execution["canonical_row_block_height"]
        != candidate["canonical_row_block_height"]
    ):
        raise ValueError("P4HZ execution identity drift")
    return OpponentDiffusionRuntime(
        components=reconstruct_bounded_photographic_profile(profile),
        correlation=np.asarray(execution["correlation_matrix"], dtype=np.float64),
        candidate=candidate,
        native_toolchains={"backend": "python-reference-p4hx-opponent-density-v1"},
    )


def evaluate_with_runtime(
    contract: dict[str, Any],
    *,
    root: Path,
    output_dir: Path,
    runtime_class: type[OpponentDiffusionRuntime] = OpponentDiffusionRuntime,
    runtime_builder: Callable[[OpponentDiffusionRuntime], OpponentDiffusionRuntime]
    | None = None,
    result_schema: str = RESULT_SCHEMA,
    source_arm_evaluator: Callable[..., dict[str, Any]] = evaluate_source_arms,
) -> dict[str, Any]:
    parents = contract["parents"]
    evidence = _load_bound_json(root, parents["p4hx_evidence"], "P4HX evidence")
    if evidence.get("decision") != parents["p4hx_evidence"]["required_decision"]:
        raise ValueError("P4HX decision drift")
    preflight = _load_bound_json(root, parents["p4hy_report"], "P4HY report")
    if (
        preflight.get("automatic_pass")
        is not parents["p4hy_report"]["required_automatic_pass"]
    ):
        raise ValueError("P4HY preflight decision drift")
    profile = _load_bound_json(root, parents["profile"], "P4HK profile")
    if profile.get("bundle_sha256") != parents["profile"]["bundle_sha256"]:
        raise ValueError("P4HK profile bundle drift")
    rows, manifest_sha = _load_manifest(root, contract["source"])
    candidate = contract["candidate"]
    arms = tuple(contract["comparison"]["arms"])
    if len(arms) != 4 or len(set(arms)) != 4:
        raise ValueError("P4HZ requires four unique ordered arms")
    base_runtime = _runtime(profile, candidate)
    runtime = (
        runtime_builder(base_runtime)
        if runtime_builder is not None
        else runtime_class(
            components=base_runtime.components,
            correlation=base_runtime.correlation,
            candidate=base_runtime.candidate,
            native_toolchains={"backend": runtime_class.__name__},
        )
    )
    scanner = _scanner_profile(candidate)
    ao6 = _load_ao6(root, parents["ao6_artifact"])
    output_dir.mkdir(parents=True, exist_ok=False)
    evaluated: list[dict[str, Any]] = []
    execution_failures: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        seeds = tuple(
            int(seed) + index * int(candidate["field_seed_stride_per_source"])
            for seed in candidate["layer_field_seeds"]
        )
        try:
            result = source_arm_evaluator(
                root=root,
                output_dir=output_dir,
                row=row,
                transform=contract["source"].get("transforms", {}).get(row["id"]),
                seeds=seeds,
                runtime=runtime,
                ao6_payload=ao6,
                scanner=scanner,
                gates=contract["automatic_gates"],
                arm_ids=arms,
            )
        except RuntimeError as error:
            execution_failures.append({"source_id": row["id"], "error": str(error)})
            break
        evaluated.append(result)
    if execution_failures:
        core = {
            "schema": result_schema,
            "experiment_id": contract["experiment_id"],
            "config_sha256": _canonical_sha256(contract),
            "manifest_sha256": manifest_sha,
            "source_count": len(rows),
            "evaluated_source_count": len(evaluated),
            "arm_ids": list(arms),
            "profile_bundle_sha256": profile["bundle_sha256"],
            "runtime_identity": runtime.native_toolchains,
            "rows": evaluated,
            "execution_failures": execution_failures,
            "gates": {
                "complete_inventory": False,
                "finite_and_bounded": False,
            },
            "automatic_pass": False,
            "severe_visual_review_allowed": False,
            "blind_review_allowed": False,
            "decision": contract["decision_if_fail"],
            "production_default_changed": False,
            "claim_ceiling": contract["claim_ceiling"],
        }
        return {**core, "stable_evidence_id": _canonical_sha256(core)}
    aggregates = aggregate_arm_rows(evaluated)
    checks = automatic_arm_checks(
        aggregates,
        contract["automatic_gates"],
        expected_rows=len(rows),
        expected_arm_count=len(arms),
    )
    passed = all(checks.values())
    core = {
        "schema": result_schema,
        "experiment_id": contract["experiment_id"],
        "config_sha256": _canonical_sha256(contract),
        "manifest_sha256": manifest_sha,
        "source_count": len(rows),
        "arm_ids": list(arms),
        "profile_bundle_sha256": profile["bundle_sha256"],
        "runtime_identity": runtime.native_toolchains,
        "rows": evaluated,
        "aggregates": aggregates,
        "gates": checks,
        "automatic_pass": passed,
        "severe_visual_review_allowed": passed,
        "blind_review_allowed": False,
        "decision": (
            contract["decision_if_pass"] if passed else contract["decision_if_fail"]
        ),
        "production_default_changed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _canonical_sha256(core)}


def evaluate(
    contract: dict[str, Any], *, root: Path, output_dir: Path
) -> dict[str, Any]:
    return evaluate_with_runtime(contract, root=root, output_dir=output_dir)


__all__ = ["CONTRACT_SCHEMA", "RESULT_SCHEMA", "evaluate", "load_contract"]
