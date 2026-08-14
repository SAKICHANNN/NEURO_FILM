"""U1.4C8 W3C OKLCh local-MINDE baseline on the frozen C7 cohort."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.oklch_local_minde import (
    linear_rec2020_to_oklab,
    local_minde_rec2020,
)
from src.color_engine.rec2020_safe_lab import apply_rec2020_safe_lab
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.gamutmlp_prophoto_rec2020_confirmation import (
    _load_declared_prophoto,
)
from src.eval.gamutmlp_prophoto_rec2020_confirmation import (
    _relative as _c7_relative,
)
from src.eval.gamutmlp_rec2020_stress_confirmation import (
    _validate_inputs as _validate_c7_inputs,
)
from src.eval.gamutmlp_rec2020_stress_confirmation import (
    load_contract as load_c7_contract,
)
from src.eval.rec2020_native_prophoto_confirmation import _working
from src.eval.rec2020_source_anchored_interior import (
    _array_sha256,
    _median_lab_delta_e76,
    _new_boundary_fraction,
    _save_and_verify,
    _style_kwargs,
    source_anchored_interior_residual,
)

SCHEMA = "neuro-film.u1-4c8-w3c-oklch-local-minde-baseline-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c8-w3c-oklch-local-minde-baseline-report.v1"
EXPERIMENT_ID = "U1.4C8"
CONTRACT_SHA256 = "59ab5b3bbfec534b4f34bdbe04fd8c4c29ed659e2c2131f9dafc3ba07c9eef2a"


class LocalMindeBaselineError(RuntimeError):
    """Raised when a frozen C8 identity or comparison invariant fails."""


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise LocalMindeBaselineError("C8 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise LocalMindeBaselineError("C8 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise LocalMindeBaselineError("C8 contract structure drift")
    for parent in payload["parents"].values():
        _relative(parent["path"])
    challenger = payload["challenger"]
    if (
        challenger.get("jnd_delta_e_ok") != 0.02
        or challenger.get("chroma_epsilon") != 0.0001
        or challenger.get("map_only_out_of_gamut_pixels") is not True
        or challenger.get("comparison_population")
        != "pixels outside linear Rec.2020 before ingress mapping only"
        or payload["baseline"].get("comparison_population")
        != "pixels outside linear Rec.2020 before ingress mapping only"
        or challenger.get("source_or_operator_fitting_allowed") is not False
        or challenger.get("hard_clipping_as_final_mapper_allowed") is not False
        or challenger.get("posthoc_limiting_allowed") is not False
    ):
        raise LocalMindeBaselineError("C8 local-MINDE policy drift")
    return payload


def _validate_inputs(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for name, parent in config["parents"].items():
        path = root / _relative(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise LocalMindeBaselineError(f"C8 parent identity drift: {name}")
        if path.suffix == ".json":
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
        if parent.get("required_automatic_pass") is True and payloads[name].get(
            "automatic_pass"
        ) is not True:
            raise LocalMindeBaselineError(f"C8 requires passing parent: {name}")
        required_stable = parent.get("required_stable_evidence_id")
        if required_stable is not None:
            actual = payloads[name].get("formal_replay", {}).get(
                "stable_evidence_id"
            )
            if actual != required_stable:
                raise LocalMindeBaselineError("C8 C7 stable identity drift")

    c7_config = load_c7_contract(root / _relative(config["parents"]["c7_contract"]["path"]))
    rows = _validate_c7_inputs(c7_config, root)
    c7_report = payloads["c7_report"]
    source = config["source"]
    if (
        len(rows) != int(source["expected_rows"])
        or len({row["camera"] for row in rows})
        != int(source["expected_camera_models"])
        or c7_config["source"]["required_allowed_use"]
        != source["required_allowed_use"]
        or c7_config["source"]["required_rights_scope"]
        != source["required_rights_scope"]
        or c7_report.get("source_manifest_sha256")
        != config["parents"]["source_manifest"]["sha256"]
        or c7_report.get("automatic_pass") is not True
        or c7_report.get("stable_evidence_id")
        != config["parents"]["c7_evidence"]["required_stable_evidence_id"]
    ):
        raise LocalMindeBaselineError("C8 C7 inventory or report drift")
    return c7_config, rows, c7_report


def _hue_error_degrees(origin: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    origin_chroma = np.hypot(origin[..., 1], origin[..., 2])
    mapped_chroma = np.hypot(mapped[..., 1], mapped[..., 2])
    valid = (origin_chroma > 4e-6) & (mapped_chroma > 4e-6)
    delta = np.zeros(origin.shape[:-1], dtype=np.float64)
    if bool(np.any(valid)):
        one = np.arctan2(origin[..., 2][valid], origin[..., 1][valid])
        two = np.arctan2(mapped[..., 2][valid], mapped[..., 1][valid])
        wrapped = np.abs((two - one + np.pi) % (2.0 * np.pi) - np.pi)
        delta[valid] = np.degrees(wrapped)
    return delta


def _mapping_metrics(
    origin: np.ndarray, mapped: np.ndarray, population: np.ndarray
) -> dict[str, Any]:
    if population.shape != origin.shape[:2] or not bool(np.any(population)):
        raise LocalMindeBaselineError("C8 mapping population must contain OOG pixels")
    origin_oklab = linear_rec2020_to_oklab(origin)
    mapped_oklab = linear_rec2020_to_oklab(mapped)
    delta = np.linalg.norm(mapped_oklab - origin_oklab, axis=-1)[population]
    hue = _hue_error_degrees(origin_oklab, mapped_oklab)[population]
    lightness = np.abs(mapped_oklab[..., 0] - origin_oklab[..., 0])[population]
    return {
        "median_delta_e_ok": float(np.median(delta)),
        "p95_delta_e_ok": float(np.quantile(delta, 0.95)),
        "median_hue_error_degrees": float(np.median(hue)),
        "p95_hue_error_degrees": float(np.quantile(hue, 0.95)),
        "median_lightness_error": float(np.median(lightness)),
        "p95_lightness_error": float(np.quantile(lightness, 0.95)),
        "delta_e_ok": delta,
        "hue_error": hue,
        "lightness_error": lightness,
    }


def _finite_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def evaluate(
    config: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise LocalMindeBaselineError("C8 output directory must be create-only")
    output_dir.mkdir(parents=True)
    c7_config, source_rows, c7_report = _validate_inputs(config, root)
    c7_render_index = {
        (row["id"], row["style"], row["gamut_mode"]): row
        for row in c7_report["rows"]
    }
    if len(c7_render_index) != len(c7_report["rows"]):
        raise LocalMindeBaselineError("C8 C7 render keys are not unique")

    source_facts: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    challenger_lightness_errors: list[np.ndarray] = []
    challenger = config["challenger"]
    for source_row in source_rows:
        c4_mapped, extended, inherited = _load_declared_prophoto(
            source_row, c7_config, root
        )
        mapped, chroma_ratio = local_minde_rec2020(
            extended,
            jnd=float(challenger["jnd_delta_e_ok"]),
            epsilon=float(challenger["chroma_epsilon"]),
        )
        out_of_gamut = np.any((extended < 0.0) | (extended > 1.0), axis=-1)
        c4_metrics = _mapping_metrics(extended, c4_mapped, out_of_gamut)
        challenger_metrics = _mapping_metrics(extended, mapped, out_of_gamut)
        challenger_lightness_errors.append(challenger_metrics["lightness_error"])
        source_id = (
            f"{source_row['camera']}-{source_row['source_id']}-{source_row['style']}"
        )
        boundary_fraction = float(
            np.mean(
                np.any(
                    (mapped <= 1.0 / 65536.0)
                    | (mapped >= 1.0 - 1.0 / 65536.0),
                    axis=-1,
                )
            )
        )
        source_facts.append(
            {
                "id": source_id,
                "camera": source_row["camera"],
                "source_sha256": source_row["member_sha256"],
                "shape": list(mapped.shape),
                "extended_rec2020_array_sha256": _array_sha256(extended),
                "c4_mapped_source_array_sha256": _array_sha256(c4_mapped),
                "challenger_mapped_source_array_sha256": _array_sha256(mapped),
                "precompression_rec2020_out_of_gamut_fraction": inherited[
                    "precompression_rec2020_out_of_gamut_fraction"
                ],
                "challenger_mapped_source_boundary_fraction": boundary_fraction,
                "challenger_chroma_ratio_median": float(np.median(chroma_ratio)),
                "challenger_fraction_chroma_ratio_below_0p5": float(
                    np.mean(chroma_ratio < 0.5)
                ),
                "c4_mapping": {
                    key: value
                    for key, value in c4_metrics.items()
                    if not isinstance(value, np.ndarray)
                },
                "challenger_mapping": {
                    key: value
                    for key, value in challenger_metrics.items()
                    if not isinstance(value, np.ndarray)
                },
            }
        )

        source_path = root / _c7_relative(source_row["path"])
        working = _working(mapped, source_path)
        source_lab = linear_rgb_to_lab(mapped, working_space="linear_rec2020")
        for style in config["render"]["styles"]:
            for mode in config["render"]["gamut_modes"]:
                key = (source_id, style, mode)
                if key not in c7_render_index:
                    raise LocalMindeBaselineError("C8 C7 render alignment drift")
                c4_row = c7_render_index[key]
                candidate = apply_rec2020_safe_lab(
                    working, **_style_kwargs(style, mode, root)
                ).pixels
                guarded, scale = source_anchored_interior_residual(
                    mapped, candidate, margin=2.0 / 65535.0
                )
                candidate_lab = linear_rgb_to_lab(
                    candidate, working_space="linear_rec2020"
                )
                guarded_lab = linear_rgb_to_lab(
                    guarded, working_space="linear_rec2020"
                )
                baseline_style = _median_lab_delta_e76(source_lab, candidate_lab)
                guarded_style = _median_lab_delta_e76(source_lab, guarded_lab)
                retention = (
                    guarded_style / baseline_style if baseline_style > 1e-12 else 1.0
                )
                relative_path = Path("renders") / style / mode / f"{source_id}.png"
                output_sha = _save_and_verify(
                    _working(guarded, source_path), output_dir / relative_path
                )
                render_rows.append(
                    {
                        "id": source_id,
                        "camera": source_row["camera"],
                        "style": style,
                        "gamut_mode": mode,
                        "source_sha256": source_row["member_sha256"],
                        "shape": list(mapped.shape),
                        "challenger_mapped_source_array_sha256": _array_sha256(mapped),
                        "candidate_array_sha256": _array_sha256(candidate),
                        "guarded_array_sha256": _array_sha256(guarded),
                        "output_path": relative_path.as_posix(),
                        "output_sha256": output_sha,
                        "baseline_style_delta_e76": baseline_style,
                        "guarded_style_delta_e76": guarded_style,
                        "style_retention_ratio": retention,
                        "c4_style_retention_ratio": c4_row["style_retention_ratio"],
                        "style_retention_margin_vs_c4": retention
                        - c4_row["style_retention_ratio"],
                        "residual_scale_median": float(np.median(scale)),
                        "fraction_residual_scale_below_0p5": float(
                            np.mean(scale < 0.5)
                        ),
                        "new_rgb16_boundary_fraction_vs_mapped_source": _new_boundary_fraction(
                            mapped, guarded
                        ),
                        "p999_gradient_ratio_vs_mapped_source": _gradient_p999_ratio(
                            mapped, guarded
                        ),
                        "adjacent_lstar_sign_inversion_fraction": _gradient_inversion_fraction(
                            source_lab[..., 0], guarded_lab[..., 0], epsilon=0.02
                        ),
                        "output_minimum": float(np.min(guarded)),
                        "output_maximum": float(np.max(guarded)),
                    }
                )

    c4_delta = np.asarray(
        [row["c4_mapping"]["median_delta_e_ok"] for row in source_facts]
    )
    challenger_delta = np.asarray(
        [row["challenger_mapping"]["median_delta_e_ok"] for row in source_facts]
    )
    c4_hue = np.asarray(
        [row["c4_mapping"]["median_hue_error_degrees"] for row in source_facts]
    )
    challenger_hue = np.asarray(
        [
            row["challenger_mapping"]["median_hue_error_degrees"]
            for row in source_facts
        ]
    )
    challenger_retentions = np.asarray(
        [row["style_retention_ratio"] for row in render_rows]
    )
    c4_retentions = np.asarray(
        [row["c4_style_retention_ratio"] for row in render_rows]
    )
    metrics = {
        "source_count": len(source_facts),
        "camera_model_count": len({row["camera"] for row in source_facts}),
        "render_count": len(render_rows),
        "maximum_challenger_mapped_source_boundary_fraction": max(
            row["challenger_mapped_source_boundary_fraction"] for row in source_facts
        ),
        "population_median_oklab_mapping_delta_e_ratio_vs_c4": _finite_ratio(
            float(np.median(challenger_delta)), float(np.median(c4_delta))
        ),
        "sources_with_no_worse_median_oklab_mapping_delta_e": int(
            np.count_nonzero(challenger_delta <= c4_delta)
        ),
        "population_median_oklab_hue_error_ratio_vs_c4": _finite_ratio(
            float(np.median(challenger_hue)), float(np.median(c4_hue))
        ),
        "population_p95_oklab_lightness_error": float(
            np.quantile(np.concatenate(challenger_lightness_errors), 0.95)
        ),
        "challenger_population_median_style_retention_ratio": float(
            np.median(challenger_retentions)
        ),
        "challenger_worst_render_style_retention_ratio": float(
            np.min(challenger_retentions)
        ),
        "c4_population_median_style_retention_ratio": float(
            np.median(c4_retentions)
        ),
        "c4_worst_render_style_retention_ratio": float(np.min(c4_retentions)),
        "style_retention_median_margin_vs_c4": float(
            np.median(challenger_retentions) - np.median(c4_retentions)
        ),
        "style_retention_worst_margin_vs_c4": float(
            np.min(challenger_retentions) - np.min(c4_retentions)
        ),
        "maximum_new_rgb16_boundary_fraction_vs_mapped_source": max(
            row["new_rgb16_boundary_fraction_vs_mapped_source"] for row in render_rows
        ),
        "population_median_residual_scale": float(
            np.median([row["residual_scale_median"] for row in render_rows])
        ),
        "population_median_fraction_scale_below_0p5": float(
            np.median(
                [row["fraction_residual_scale_below_0p5"] for row in render_rows]
            )
        ),
        "maximum_p999_gradient_ratio_vs_mapped_source": max(
            row["p999_gradient_ratio_vs_mapped_source"] for row in render_rows
        ),
        "maximum_adjacent_lstar_sign_inversion_fraction": max(
            row["adjacent_lstar_sign_inversion_fraction"] for row in render_rows
        ),
        "output_minimum": min(row["output_minimum"] for row in render_rows),
        "output_maximum": max(row["output_maximum"] for row in render_rows),
    }
    gates = config["automatic_gates"]
    checks = {
        "inventory": metrics["source_count"] == int(config["source"]["expected_rows"])
        and metrics["camera_model_count"]
        == int(config["source"]["expected_camera_models"])
        and metrics["render_count"]
        == int(config["source"]["expected_rows"])
        * len(config["render"]["styles"])
        * len(config["render"]["gamut_modes"]),
        "finite_in_gamut_outputs": bool(
            np.isfinite(metrics["output_minimum"])
            and np.isfinite(metrics["output_maximum"])
            and metrics["output_minimum"] >= 0.0
            and metrics["output_maximum"] <= 1.0
        ),
        "mapped_source_boundary": metrics[
            "maximum_challenger_mapped_source_boundary_fraction"
        ]
        <= float(gates["maximum_mapped_source_boundary_fraction"]),
        "new_rgb16_boundary": metrics[
            "maximum_new_rgb16_boundary_fraction_vs_mapped_source"
        ]
        <= float(gates["maximum_new_rgb16_boundary_fraction_vs_mapped_source"]),
        "population_style_retention": metrics[
            "challenger_population_median_style_retention_ratio"
        ]
        >= float(gates["minimum_population_median_style_retention_ratio"]),
        "worst_style_retention": metrics[
            "challenger_worst_render_style_retention_ratio"
        ]
        >= float(gates["minimum_worst_render_style_retention_ratio"]),
        "residual_scale": metrics["population_median_residual_scale"]
        >= float(gates["minimum_population_median_residual_scale"]),
        "low_scale_fraction": metrics["population_median_fraction_scale_below_0p5"]
        <= float(gates["maximum_population_median_fraction_scale_below_0p5"]),
        "gradient": metrics["maximum_p999_gradient_ratio_vs_mapped_source"]
        <= float(gates["maximum_p999_gradient_ratio_vs_mapped_source"]),
        "gradient_order": metrics[
            "maximum_adjacent_lstar_sign_inversion_fraction"
        ]
        <= float(gates["maximum_adjacent_lstar_sign_inversion_fraction"]),
        "mapping_delta": metrics[
            "population_median_oklab_mapping_delta_e_ratio_vs_c4"
        ]
        <= float(gates["maximum_population_median_oklab_mapping_delta_e_ratio_vs_c4"]),
        "mapping_delta_rows": metrics[
            "sources_with_no_worse_median_oklab_mapping_delta_e"
        ]
        >= int(gates["minimum_sources_with_no_worse_median_oklab_mapping_delta_e"]),
        "mapping_hue": metrics["population_median_oklab_hue_error_ratio_vs_c4"]
        <= float(gates["maximum_population_median_oklab_hue_error_ratio_vs_c4"]),
        "mapping_lightness": metrics["population_p95_oklab_lightness_error"]
        <= float(gates["maximum_population_p95_oklab_lightness_error"]),
        "style_retention_median_vs_c4": metrics[
            "style_retention_median_margin_vs_c4"
        ]
        >= float(gates["minimum_style_retention_median_margin_vs_c4"]),
        "style_retention_worst_vs_c4": metrics["style_retention_worst_margin_vs_c4"]
        >= float(gates["minimum_style_retention_worst_margin_vs_c4"]),
    }
    automatic_pass = all(checks.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "source_manifest_sha256": config["parents"]["source_manifest"]["sha256"],
        "baseline_mapping_id": config["baseline"]["id"],
        "challenger_mapping_id": config["challenger"]["id"],
        "c3_mechanism_id": config["render"]["c3_mechanism_id"],
        "sources": source_facts,
        "rows": render_rows,
        "metrics": metrics,
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "automatic_pass": automatic_pass,
        "visual_or_independent_adjudication_opened": automatic_pass,
        "product_ingress_opened": False,
        "decision": config["branches"]["pass" if automatic_pass else "fail"],
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONTRACT_SHA256",
    "LocalMindeBaselineError",
    "evaluate",
    "load_contract",
    "write_report",
]
