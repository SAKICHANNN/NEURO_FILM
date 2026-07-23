#!/usr/bin/env python
"""Run the frozen U5.R2D2 canonicalizer and hard-retrieval sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.canonicalizer_sensitivity import (  # noqa: E402
    build_canonical_observations,
    build_neutral_bank,
    canonical_rows_for_split,
    neutral_bank_manifest,
    neutral_bank_manifest_sha256,
    retrieval_diagnostics,
    signature_matrix,
    target_matrix,
)
from src.roll2film.synthetic_benchmark import (  # noqa: E402
    PCAPredictor,
    bootstrap_median_interval,
    evaluate_predictions,
)
from src.roll2film.synthetic_recovery import (  # noqa: E402
    SyntheticBoundedOperator,
    project_operator_parameters,
    uniform_probe_grid,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _load_hashed(path: Path, expected: str) -> tuple[dict[str, Any] | list[Any], bytes]:
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: expected {expected}, observed {actual}")
    return json.loads(payload), payload


def _compact(
    evaluation: dict[str, Any],
    config: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    summary = evaluation["summary"]
    return {
        "overall": {
            **summary["overall"],
            "operator_group_bootstrap_95_interval_median_uniform_rgb_rmse": (
                bootstrap_median_interval(
                    summary["operator_metrics"],
                    key="uniform_rgb_rmse",
                    replicates=int(config["evaluation"]["bootstrap_replicates"]),
                    seed=int(config["evaluation"]["bootstrap_seed"]) + seed_offset,
                )
            ),
        },
        "by_family": summary["by_family"],
    }


def _fit_policy_set(
    canonicalizer: str,
    fit_rows: list[Any],
    validation_rows: list[Any],
    config: dict[str, Any],
    parent_config: dict[str, Any],
) -> tuple[dict[str, PCAPredictor], dict[str, Any], str]:
    x_fit = signature_matrix(fit_rows, canonicalizer)
    x_validation = signature_matrix(validation_rows, canonicalizer)
    y_fit = target_matrix(fit_rows)
    specification = config["policies"]
    models: dict[str, PCAPredictor] = {}
    validation: dict[str, Any] = {}
    ridge_candidates = []
    for alpha in specification["ridge_alphas"]:
        model = PCAPredictor(
            kind="ridge",
            hyperparameter=alpha,
            components=int(specification["pca_components"]),
            seed=int(config["neutral_bank"]["seed"]),
        ).fit(x_fit, y_fit)
        result = evaluate_predictions(
            validation_rows, model.predict(x_validation), parent_config
        )
        score = float(result["summary"]["overall"]["median_uniform_rgb_rmse"])
        ridge_candidates.append((score, float(alpha), model, result))
    ridge_candidates.sort(key=lambda item: (item[0], item[1]))
    ridge_score, ridge_alpha, ridge_model, ridge_result = ridge_candidates[0]
    models["ridge"] = ridge_model
    validation["ridge"] = {
        "hyperparameter": ridge_alpha,
        "median_uniform_rgb_rmse": ridge_score,
        "all_alpha_scores": [
            {"alpha": alpha, "median_uniform_rgb_rmse": score}
            for score, alpha, _, _ in ridge_candidates
        ],
        "summary": {
            "overall": ridge_result["summary"]["overall"],
            "by_family": ridge_result["summary"]["by_family"],
        },
    }
    for policy, neighbors in (
        ("hard_case_top1", int(specification["hard_case_neighbors"])),
        ("sparse_case_top3", int(specification["sparse_case_neighbors"])),
    ):
        model = PCAPredictor(
            kind="knn",
            hyperparameter=neighbors,
            components=int(specification["pca_components"]),
            seed=int(config["neutral_bank"]["seed"]),
        ).fit(x_fit, y_fit)
        result = evaluate_predictions(
            validation_rows, model.predict(x_validation), parent_config
        )
        models[policy] = model
        validation[policy] = {
            "neighbors": neighbors,
            "median_uniform_rgb_rmse": float(
                result["summary"]["overall"]["median_uniform_rgb_rmse"]
            ),
            "summary": {
                "overall": result["summary"]["overall"],
                "by_family": result["summary"]["by_family"],
            },
        }
    selected = min(
        validation,
        key=lambda policy: (
            validation[policy]["median_uniform_rgb_rmse"],
            policy,
        ),
    )
    return models, validation, selected


def _improvement(reference: float, challenger: float) -> float:
    return float((reference - challenger) / max(reference, 1e-15))


def _family_degradation(
    reference: dict[str, Any],
    challenger: dict[str, Any],
) -> float:
    values = []
    for family in reference:
        baseline = float(reference[family]["median_uniform_rgb_rmse"])
        candidate = float(challenger[family]["median_uniform_rgb_rmse"])
        values.append((candidate - baseline) / max(baseline, 1e-15))
    return float(max(values))


def _pairwise_rendered_disagreement(
    predictions: dict[str, np.ndarray],
    config: dict[str, Any],
    parent_config: dict[str, Any],
) -> dict[str, Any]:
    names = sorted(predictions)
    probes = uniform_probe_grid(int(config["evaluation"]["uniform_probe_grid_size"]))
    projected = {
        name: [
            project_operator_parameters(
                row,
                maximum_off_diagonal_sum=float(
                    parent_config["operator"][
                        "matrix_maximum_total_off_diagonal_per_row"
                    ]
                ),
                minimum_tone_increment=float(
                    parent_config["operator"]["tone_minimum_knot_increment"]
                ),
            )
            for row in values
        ]
        for name, values in predictions.items()
    }
    pairs = {}
    all_values = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            errors = []
            for observation_index, (left_parameters, right_parameters) in enumerate(
                zip(projected[left], projected[right])
            ):
                left_operator = SyntheticBoundedOperator(
                    left_parameters,
                    "combined",
                    f"{left}-{observation_index}",
                )
                right_operator = SyntheticBoundedOperator(
                    right_parameters,
                    "combined",
                    f"{right}-{observation_index}",
                )
                errors.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (
                                    left_operator.apply(probes)
                                    - right_operator.apply(probes)
                                )
                                ** 2
                            )
                        )
                    )
                )
            value = float(np.median(errors))
            pairs[f"{left}__{right}"] = value
            all_values.append(value)
    return {
        "pairwise_median_rendered_rgb_rmse": pairs,
        "median_pairwise_rendered_rgb_rmse": float(np.median(all_values)),
        "maximum_pairwise_rendered_rgb_rmse": float(np.max(all_values)),
    }


def _make_contact_sheet(
    path: Path,
    rows: list[Any],
    predictions: np.ndarray,
    evaluation: dict[str, Any],
    config: dict[str, Any],
    parent_config: dict[str, Any],
    label: str,
) -> None:
    records = evaluation["per_observation"]
    worst = sorted(
        range(len(records)),
        key=lambda index: records[index]["uniform_rgb_rmse"],
        reverse=True,
    )[:3]
    size = int(config["evaluation"]["uniform_probe_grid_size"])
    probes = uniform_probe_grid(size)
    chart = probes.reshape(size, size * size, 3)
    scale = 2
    label_width = 120
    height = size * scale
    canvas = Image.new("RGB", (label_width + chart.shape[1] * scale, height * 9), "white")
    draw = ImageDraw.Draw(canvas)
    output_row = 0
    for index in worst:
        row = rows[index]
        parameters = project_operator_parameters(
            predictions[index],
            maximum_off_diagonal_sum=float(
                parent_config["operator"]["matrix_maximum_total_off_diagonal_per_row"]
            ),
            minimum_tone_increment=float(
                parent_config["operator"]["tone_minimum_knot_increment"]
            ),
        )
        true = SyntheticBoundedOperator(row.parameters, row.family, row.operator_id)
        predicted = SyntheticBoundedOperator(
            parameters, "combined", f"predicted-{row.operator_id}"
        )
        for role, values in (
            ("input", chart),
            ("true", true.apply(chart)),
            ("predicted", predicted.apply(chart)),
        ):
            image = Image.fromarray(
                np.round(np.clip(values, 0, 1) * 255).astype(np.uint8),
                mode="RGB",
            ).resize(
                (chart.shape[1] * scale, height),
                resample=Image.Resampling.NEAREST,
            )
            y = output_row * height
            canvas.paste(image, (label_width, y))
            draw.text((3, y + 2), f"{label}\n{row.operator_id}\n{role}", fill="black")
            output_row += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)


def run(config: dict[str, Any], config_sha256: str, output_directory: Path) -> dict[str, Any]:
    parent_spec = config["parent"]
    parent_config_payload, _ = _load_hashed(
        ROOT / parent_spec["config"], parent_spec["config_sha256"]
    )
    parent_report_payload, _ = _load_hashed(
        ROOT / parent_spec["report"], parent_spec["report_sha256"]
    )
    parent_manifest_payload, _ = _load_hashed(
        ROOT / parent_spec["manifest"], parent_spec["manifest_sha256"]
    )
    parent_config = parent_config_payload
    parent_report = parent_report_payload
    parent_manifest = parent_manifest_payload
    if not isinstance(parent_manifest, list):
        raise ValueError("parent manifest must be a JSON list")

    bank = build_neutral_bank(config)
    bank_manifest = neutral_bank_manifest(bank)
    bank_bytes = (json.dumps(bank_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_directory.mkdir(parents=True, exist_ok=True)
    bank_path = output_directory / "neutral_bank_manifest.json"
    bank_path.write_bytes(bank_bytes)
    observations = build_canonical_observations(
        parent_config, parent_manifest, config, bank
    )
    split_rows = {
        split: canonical_rows_for_split(observations, split)
        for split in ("fit", "validation", "confirmation", "stress")
    }

    canonicalizers = list(config["canonicalizers"]) + [
        "nearest_raw_lab_leave_true_palette_out",
        "nearest_basic_normalized_quantiles_leave_true_palette_out",
    ]
    methods: dict[str, Any] = {}
    selected_confirmation_predictions: dict[str, np.ndarray] = {}
    evaluation_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for canonicalizer_index, canonicalizer in enumerate(canonicalizers):
        models, validation, selected = _fit_policy_set(
            canonicalizer,
            split_rows["fit"],
            split_rows["validation"],
            config,
            parent_config,
        )
        policies: dict[str, Any] = {}
        evaluation_cache[canonicalizer] = {}
        for policy_index, (policy, model) in enumerate(models.items()):
            confirmation_predictions = model.predict(
                signature_matrix(split_rows["confirmation"], canonicalizer)
            )
            stress_predictions = model.predict(
                signature_matrix(split_rows["stress"], canonicalizer)
            )
            confirmation_evaluation = evaluate_predictions(
                split_rows["confirmation"],
                confirmation_predictions,
                parent_config,
            )
            stress_evaluation = evaluate_predictions(
                split_rows["stress"],
                stress_predictions,
                parent_config,
            )
            evaluation_cache[canonicalizer][policy] = confirmation_evaluation
            policies[policy] = {
                "confirmation": _compact(
                    confirmation_evaluation,
                    config,
                    canonicalizer_index * 20 + policy_index * 2,
                ),
                "stress": _compact(
                    stress_evaluation,
                    config,
                    canonicalizer_index * 20 + policy_index * 2 + 1,
                ),
            }
            if policy == selected:
                selected_confirmation_predictions[canonicalizer] = (
                    confirmation_predictions
                )
        methods[canonicalizer] = {
            "selected_policy_from_validation": selected,
            "validation": validation,
            "policies": policies,
            "selected_confirmation": policies[selected]["confirmation"],
            "selected_stress": policies[selected]["stress"],
        }

    practical = list(config["practical_canonicalizers"])
    global_confirmation = parent_report["methods"]["global_mean"]["confirmation"]
    global_rmse = float(global_confirmation["overall"]["median_uniform_rgb_rmse"])
    gates_spec = config["gates"]
    useful: dict[str, bool] = {}
    useful_diagnostics: dict[str, Any] = {}
    for canonicalizer in practical:
        selected = methods[canonicalizer]["selected_confirmation"]
        rmse = float(selected["overall"]["median_uniform_rgb_rmse"])
        capture = float(selected["overall"]["median_captured_style_fraction"])
        degradation = _family_degradation(
            global_confirmation["by_family"],
            selected["by_family"],
        )
        useful[canonicalizer] = bool(
            _improvement(global_rmse, rmse)
            >= gates_spec["minimum_practical_improvement_over_global_fraction"]
            and capture >= gates_spec["minimum_practical_captured_style_fraction"]
            and degradation
            <= gates_spec["maximum_practical_family_degradation_over_global_fraction"]
        )
        useful_diagnostics[canonicalizer] = {
            "selected_policy": methods[canonicalizer][
                "selected_policy_from_validation"
            ],
            "improvement_over_global_fraction": _improvement(global_rmse, rmse),
            "captured_style_fraction": capture,
            "maximum_family_degradation_over_global_fraction": degradation,
            "useful": useful[canonicalizer],
        }

    disagreement = _pairwise_rendered_disagreement(
        {
            name: selected_confirmation_predictions[name]
            for name in practical
        },
        config,
        parent_config,
    )
    exact_rmse = float(
        methods["exact_raw_reference_oracle"]["selected_confirmation"]["overall"][
            "median_uniform_rgb_rmse"
        ]
    )
    exact_relative_error = abs(
        exact_rmse - float(parent_spec["canonical_oracle_confirmation_rgb_rmse"])
    ) / float(parent_spec["canonical_oracle_confirmation_rgb_rmse"])

    hard_wins: dict[str, Any] = {}
    sparse_wins: dict[str, Any] = {}
    for canonicalizer in practical:
        ridge = methods[canonicalizer]["policies"]["ridge"]["confirmation"]
        top1 = methods[canonicalizer]["policies"]["hard_case_top1"]["confirmation"]
        top3 = methods[canonicalizer]["policies"]["sparse_case_top3"]["confirmation"]
        ridge_rmse = float(ridge["overall"]["median_uniform_rgb_rmse"])
        top1_rmse = float(top1["overall"]["median_uniform_rgb_rmse"])
        top3_rmse = float(top3["overall"]["median_uniform_rgb_rmse"])
        hard_wins[canonicalizer] = {
            "improvement_over_ridge_fraction": _improvement(ridge_rmse, top1_rmse),
            "maximum_family_degradation_over_ridge_fraction": _family_degradation(
                ridge["by_family"], top1["by_family"]
            ),
        }
        sparse_wins[canonicalizer] = {
            "improvement_over_top1_fraction": _improvement(top1_rmse, top3_rmse),
            "maximum_family_degradation_over_top1_fraction": _family_degradation(
                top1["by_family"], top3["by_family"]
            ),
        }

    hard_case_value = any(
        useful[name]
        and values["improvement_over_ridge_fraction"]
        >= gates_spec["minimum_hard_top1_improvement_over_ridge_fraction"]
        and values["maximum_family_degradation_over_ridge_fraction"]
        <= gates_spec["maximum_hard_top1_family_degradation_fraction"]
        for name, values in hard_wins.items()
    )
    sparse_case_value = any(
        useful[name]
        and values["improvement_over_top1_fraction"]
        >= gates_spec["minimum_sparse_top3_improvement_over_top1_fraction"]
        for name, values in sparse_wins.items()
    )
    all_valid = all(
        float(policy["confirmation"]["overall"]["valid_fraction"])
        == gates_spec["projected_policy_valid_fraction"]
        for method in methods.values()
        for policy in method["policies"].values()
    )
    gates = {
        "parent_hashes_and_group_splits_reproduced": True,
        "all_projected_policies_valid": all_valid,
        "exact_oracle_reproduced": (
            exact_relative_error
            <= gates_spec["maximum_parent_oracle_relative_reproduction_error"]
        ),
        "minimum_three_practical_canonicalizers_useful": (
            sum(useful.values())
            >= gates_spec["minimum_independently_useful_practical_canonicalizers"]
        ),
        "practical_canonicalizers_agree": (
            disagreement["median_pairwise_rendered_rgb_rmse"]
            <= gates_spec["maximum_practical_pairwise_rendered_rgb_rmse"]
        ),
        "hard_case_top1_value": hard_case_value,
        "sparse_case_top3_value": sparse_case_value,
    }

    best_practical = min(
        practical,
        key=lambda name: float(
            methods[name]["selected_confirmation"]["overall"][
                "median_uniform_rgb_rmse"
            ]
        ),
    )
    best_policy = methods[best_practical]["selected_policy_from_validation"]
    sheet_path = output_directory / "best_practical_worst_probe_grid.png"
    _make_contact_sheet(
        sheet_path,
        split_rows["confirmation"],
        selected_confirmation_predictions[best_practical],
        evaluation_cache[best_practical][best_policy],
        config,
        parent_config,
        f"{best_practical}/{best_policy}",
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "parent_hashes_verified": {
            "config_sha256": parent_spec["config_sha256"],
            "report_sha256": parent_spec["report_sha256"],
            "manifest_sha256": parent_spec["manifest_sha256"],
        },
        "neutral_bank_manifest_sha256": _sha256_bytes(bank_bytes),
        "neutral_bank_semantic_sha256": neutral_bank_manifest_sha256(bank),
        "neutral_bank_items": len(bank),
        "methods": methods,
        "retrieval_diagnostics_confirmation": retrieval_diagnostics(
            split_rows["confirmation"], bank
        ),
        "retrieval_diagnostics_stress": retrieval_diagnostics(
            split_rows["stress"], bank
        ),
        "practical_usefulness": useful_diagnostics,
        "useful_practical_count": int(sum(useful.values())),
        "canonicalizer_disagreement": disagreement,
        "exact_oracle_relative_reproduction_error": exact_relative_error,
        "hard_case_diagnostics": hard_wins,
        "sparse_case_diagnostics": sparse_wins,
        "gates": gates,
        "all_primary_canonicalization_gates_passed": all(
            gates[name]
            for name in (
                "parent_hashes_and_group_splits_reproduced",
                "all_projected_policies_valid",
                "exact_oracle_reproduced",
                "minimum_three_practical_canonicalizers_useful",
                "practical_canonicalizers_agree",
            )
        ),
        "best_practical_by_validation_policy": {
            "canonicalizer": best_practical,
            "policy": best_policy,
        },
        "visual_contact_sheet": {
            "path": sheet_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_path(sheet_path),
            "binding": False,
        },
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2d2_canonicalizer_retrieval_sensitivity_v1.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "outputs/u5_r2d2_canonicalizer_retrieval_sensitivity_v1",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    report = run(
        config,
        _sha256_bytes(config_bytes),
        args.output_directory,
    )
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    report_path = args.output_directory / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "sha256": _sha256_bytes(encoded),
                "primary_gates_passed": report[
                    "all_primary_canonicalization_gates_passed"
                ],
                "useful_practical_count": report["useful_practical_count"],
                "gates": report["gates"],
                "best_practical": report["best_practical_by_validation_policy"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
