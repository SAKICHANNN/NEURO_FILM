"""U1.4C11 confirmation of analytical OKLab ingress on ICC-semantic inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.oklab_analytical_interior import (
    analytical_oklab_interior_rec2020,
)
from src.color_engine.oklch_local_minde import local_minde_rec2020
from src.color_engine.rec2020_safe_lab import apply_rec2020_safe_lab
from src.eval import rec2020_native_prophoto_confirmation as c4
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.gamutmlp_oklch_local_minde_baseline import _mapping_metrics
from src.eval.rec2020_source_anchored_interior import (
    _array_sha256,
    _median_lab_delta_e76,
    _new_boundary_fraction,
    _save_and_verify,
    _style_kwargs,
    source_anchored_interior_residual,
)
from src.preprocess import load_working_image

SCHEMA = "neuro-film.u1-4c11-oklab-analytical-interior-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c11-oklab-analytical-interior-report.v1"
EXPERIMENT_ID = "U1.4C11"
CONTRACT_SHA256 = "7ddb5ebc94b50c3f657e02340a11d24bef1c837c46eb39c81ec7dfbfa94a2e2c"


class AnalyticalInteriorConfirmationError(RuntimeError):
    """Raised when a frozen C11 identity or execution invariant fails."""


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise AnalyticalInteriorConfirmationError("C11 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise AnalyticalInteriorConfirmationError("C11 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise AnalyticalInteriorConfirmationError("C11 contract structure drift")
    mapper = payload["mapper"]
    if (
        mapper.get("softness") != 1.0 / 64.0
        or mapper.get("rgb16_margin") != 2.0 / 65535.0
        or mapper.get("bisection_iterations") != 24
        or mapper.get("map_only_out_of_gamut_pixels") is not True
        or mapper.get("preserve_oklab_ab_direction") is not True
        or mapper.get("cohort_fitting_allowed") is not False
        or mapper.get("hard_component_clipping_allowed") is not False
        or mapper.get("posthoc_limiting_allowed") is not False
    ):
        raise AnalyticalInteriorConfirmationError("C11 mapper policy drift")
    for parent in payload["parents"].values():
        _relative(parent["path"])
    return payload


def _validate_inputs(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    for name, parent in contract["parents"].items():
        path = root / _relative(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise AnalyticalInteriorConfirmationError(f"C11 parent drift: {name}")
        if path.suffix == ".json":
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    if payloads["c10_evidence"].get("decision") != contract["parents"][
        "c10_evidence"
    ]["required_decision"]:
        raise AnalyticalInteriorConfirmationError("C11 requires the C10 failure branch")
    if payloads["c10_report"].get("metrics", {}).get(
        "luminance_domain_failure_count"
    ) != int(contract["parents"]["c10_report"]["required_failure_count"]):
        raise AnalyticalInteriorConfirmationError("C11 C10 domain evidence drift")
    c4_config = c4.load_contract(root / _relative(contract["parents"]["c4_contract"]["path"]))
    rows = c4._validate_inputs(c4_config, root)
    if (
        len(rows) != int(contract["source"]["expected_rows"])
        or c4_config["source"]["expected_embedded_icc_sha256"]
        != contract["source"]["expected_embedded_icc_sha256"]
        or c4_config["source"]["maximum_evaluation_side"]
        != int(contract["source"]["maximum_evaluation_side"])
    ):
        raise AnalyticalInteriorConfirmationError("C11 source inventory drift")
    return c4_config, rows


def _semantic_source(
    row: Mapping[str, Any], c4_config: Mapping[str, Any], root: Path
) -> tuple[np.ndarray, Path]:
    path = root / c4._relative_path(row["path"])
    working = load_working_image(path)
    if (
        working.pixels.dtype != np.float32
        or working.working_space != "linear_rec2020"
        or working.transfer_state != "display_linear"
    ):
        raise AnalyticalInteriorConfirmationError("C11 product semantic ingress drift")
    return (
        c4.resize_float(
            working.pixels, int(c4_config["source"]["maximum_evaluation_side"])
        ),
        path,
    )


def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def _checks(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, bool]:
    numeric_values = [
        value
        for value in metrics.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    finite = bool(np.all(np.isfinite(np.asarray(numeric_values, dtype=np.float64))))
    return {
        "finite": finite,
        "in_gamut_exact": metrics["in_gamut_input_change_count"] == 0,
        "mapped_in_gamut": metrics["mapped_output_minimum"] >= 0.0
        and metrics["mapped_output_maximum"] <= 1.0,
        "new_mapped_boundary": metrics[
            "maximum_new_rgb16_boundary_fraction_vs_semantic_source"
        ]
        <= float(gates["maximum_new_rgb16_boundary_fraction_vs_semantic_source"]),
        "hue_direction": metrics["maximum_per_source_p99_oklab_hue_error_degrees"]
        <= float(gates["maximum_per_source_p99_oklab_hue_error_degrees"]),
        "chroma_retention": metrics["minimum_per_source_median_chroma_scale"]
        >= float(gates["minimum_per_source_median_chroma_scale"]),
        "chroma_collapse": metrics[
            "maximum_per_source_fraction_chroma_scale_below_1e_6"
        ]
        <= float(gates["maximum_per_source_fraction_chroma_scale_below_1e_6"]),
        "population_style": metrics["population_median_style_retention_ratio"]
        >= float(gates["minimum_population_median_style_retention_ratio"]),
        "worst_style": metrics["worst_render_style_retention_ratio"]
        >= float(gates["minimum_worst_render_style_retention_ratio"]),
        "residual_scale": metrics["population_median_residual_scale"]
        >= float(gates["minimum_population_median_residual_scale"]),
        "low_scale_fraction": metrics["population_median_fraction_scale_below_0p5"]
        <= float(gates["maximum_population_median_fraction_scale_below_0p5"]),
        "new_render_boundary": metrics[
            "maximum_new_rgb16_boundary_fraction_vs_mapped_source"
        ]
        <= float(gates["maximum_new_rgb16_boundary_fraction_vs_mapped_source"]),
        "gradient": metrics["maximum_p999_gradient_ratio_vs_mapped_source"]
        <= float(gates["maximum_p999_gradient_ratio_vs_mapped_source"]),
        "lstar_inversion": metrics["maximum_adjacent_lstar_sign_inversion_fraction"]
        <= float(gates["maximum_adjacent_lstar_sign_inversion_fraction"]),
    }


def evaluate(
    contract: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise AnalyticalInteriorConfirmationError("C11 output directory must be create-only")
    c4_config, source_rows = _validate_inputs(contract, root)
    output_dir.mkdir(parents=True)
    mapper = contract["mapper"]
    source_facts: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []

    for row in source_rows:
        semantic, source_path = _semantic_source(row, c4_config, root)
        mapped, chroma_ratio = analytical_oklab_interior_rec2020(
            semantic,
            softness=float(mapper["softness"]),
            margin=float(mapper["rgb16_margin"]),
            iterations=int(mapper["bisection_iterations"]),
        )
        local_minde, _ = local_minde_rec2020(semantic)
        out_of_gamut = np.any((semantic < 0.0) | (semantic > 1.0), axis=-1)
        if not bool(np.any(out_of_gamut)):
            raise AnalyticalInteriorConfirmationError("C11 source lacks OOG pixels")
        in_gamut = ~out_of_gamut
        in_gamut_changes = int(np.count_nonzero(mapped[in_gamut] != semantic[in_gamut]))
        candidate_metrics = _mapping_metrics(semantic, mapped, out_of_gamut)
        baseline_metrics = _mapping_metrics(semantic, local_minde, out_of_gamut)
        source_facts.append(
            {
                "id": row["id"],
                "source_sha256": row["sha256"],
                "shape": list(mapped.shape),
                "semantic_array_sha256": _array_sha256(semantic),
                "mapped_array_sha256": _array_sha256(mapped),
                "local_minde_array_sha256": _array_sha256(local_minde),
                "out_of_gamut_fraction": float(np.mean(out_of_gamut)),
                "in_gamut_input_change_count": in_gamut_changes,
                "new_rgb16_boundary_fraction_vs_semantic_source": _new_boundary_fraction(
                    semantic, mapped
                ),
                "median_chroma_scale": float(np.median(chroma_ratio[out_of_gamut])),
                "fraction_chroma_scale_below_1e_6": float(
                    np.mean(chroma_ratio[out_of_gamut] < 1e-6)
                ),
                "p99_oklab_hue_error_degrees": float(
                    np.quantile(candidate_metrics["hue_error"], 0.99)
                ),
                "candidate_mapping": _public_metrics(candidate_metrics),
                "local_minde_mapping": _public_metrics(baseline_metrics),
                "mapped_minimum": float(np.min(mapped)),
                "mapped_maximum": float(np.max(mapped)),
            }
        )

        working = c4._working(mapped, source_path)
        source_lab = linear_rgb_to_lab(mapped, working_space="linear_rec2020")
        for style in contract["render"]["styles"]:
            for mode in contract["render"]["gamut_modes"]:
                candidate = apply_rec2020_safe_lab(
                    working, **_style_kwargs(style, mode, root)
                ).pixels
                guarded, scale = source_anchored_interior_residual(
                    mapped, candidate, margin=float(mapper["rgb16_margin"])
                )
                candidate_lab = linear_rgb_to_lab(
                    candidate, working_space="linear_rec2020"
                )
                guarded_lab = linear_rgb_to_lab(
                    guarded, working_space="linear_rec2020"
                )
                baseline_style = _median_lab_delta_e76(source_lab, candidate_lab)
                guarded_style = _median_lab_delta_e76(source_lab, guarded_lab)
                retention = guarded_style / baseline_style if baseline_style > 1e-12 else 1.0
                relative = Path("renders") / style / mode / f"{row['id']}.png"
                render_rows.append(
                    {
                        "id": row["id"],
                        "style": style,
                        "gamut_mode": mode,
                        "source_sha256": row["sha256"],
                        "mapped_array_sha256": _array_sha256(mapped),
                        "candidate_array_sha256": _array_sha256(candidate),
                        "guarded_array_sha256": _array_sha256(guarded),
                        "output_path": relative.as_posix(),
                        "output_sha256": _save_and_verify(
                            c4._working(guarded, source_path), output_dir / relative
                        ),
                        "style_retention_ratio": retention,
                        "residual_scale_median": float(np.median(scale)),
                        "fraction_residual_scale_below_0p5": float(np.mean(scale < 0.5)),
                        "new_rgb16_boundary_fraction_vs_mapped_source": _new_boundary_fraction(
                            mapped, guarded
                        ),
                        "p999_gradient_ratio_vs_mapped_source": _gradient_p999_ratio(
                            mapped, guarded
                        ),
                        "adjacent_lstar_sign_inversion_fraction": _gradient_inversion_fraction(
                            source_lab[..., 0], guarded_lab[..., 0], epsilon=0.02
                        ),
                    }
                )

    metrics = {
        "source_count": len(source_facts),
        "render_count": len(render_rows),
        "in_gamut_input_change_count": sum(
            row["in_gamut_input_change_count"] for row in source_facts
        ),
        "mapped_output_minimum": min(row["mapped_minimum"] for row in source_facts),
        "mapped_output_maximum": max(row["mapped_maximum"] for row in source_facts),
        "maximum_new_rgb16_boundary_fraction_vs_semantic_source": max(
            row["new_rgb16_boundary_fraction_vs_semantic_source"] for row in source_facts
        ),
        "maximum_per_source_p99_oklab_hue_error_degrees": max(
            row["p99_oklab_hue_error_degrees"] for row in source_facts
        ),
        "minimum_per_source_median_chroma_scale": min(
            row["median_chroma_scale"] for row in source_facts
        ),
        "maximum_per_source_fraction_chroma_scale_below_1e_6": max(
            row["fraction_chroma_scale_below_1e_6"] for row in source_facts
        ),
        "population_median_style_retention_ratio": float(
            np.median([row["style_retention_ratio"] for row in render_rows])
        ),
        "worst_render_style_retention_ratio": min(
            row["style_retention_ratio"] for row in render_rows
        ),
        "population_median_residual_scale": float(
            np.median([row["residual_scale_median"] for row in render_rows])
        ),
        "population_median_fraction_scale_below_0p5": float(
            np.median([row["fraction_residual_scale_below_0p5"] for row in render_rows])
        ),
        "maximum_new_rgb16_boundary_fraction_vs_mapped_source": max(
            row["new_rgb16_boundary_fraction_vs_mapped_source"] for row in render_rows
        ),
        "maximum_p999_gradient_ratio_vs_mapped_source": max(
            row["p999_gradient_ratio_vs_mapped_source"] for row in render_rows
        ),
        "maximum_adjacent_lstar_sign_inversion_fraction": max(
            row["adjacent_lstar_sign_inversion_fraction"] for row in render_rows
        ),
        "population_median_mapping_delta_e_ratio_vs_local_minde": float(
            np.median(
                [row["candidate_mapping"]["median_delta_e_ok"] for row in source_facts]
            )
            / max(
                float(
                    np.median(
                        [row["local_minde_mapping"]["median_delta_e_ok"] for row in source_facts]
                    )
                ),
                1e-12,
            )
        ),
        "population_median_hue_error_ratio_vs_local_minde": float(
            np.median(
                [row["candidate_mapping"]["median_hue_error_degrees"] for row in source_facts]
            )
            / max(
                float(
                    np.median(
                        [row["local_minde_mapping"]["median_hue_error_degrees"] for row in source_facts]
                    )
                ),
                1e-12,
            )
        ),
    }
    checks = _checks(metrics, contract["automatic_gates"])
    checks["inventory"] = metrics["source_count"] == int(contract["source"]["expected_rows"])
    checks["render_inventory"] = metrics["render_count"] == int(
        contract["source"]["expected_rows"]
    ) * len(contract["render"]["styles"]) * len(contract["render"]["gamut_modes"])
    automatic_pass = all(checks.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "mapper_id": mapper["id"],
        "sources": source_facts,
        "rows": render_rows,
        "metrics": metrics,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "automatic_pass": automatic_pass,
        "decision": contract["decisions"]["pass" if automatic_pass else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
        "production_default_changed": False,
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "CONTRACT_SHA256",
    "AnalyticalInteriorConfirmationError",
    "evaluate",
    "load_contract",
]
