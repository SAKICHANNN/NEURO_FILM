#!/usr/bin/env python
"""Run the frozen U5.R2D1 generated known-operator benchmark."""

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

from src.roll2film.synthetic_benchmark import (  # noqa: E402
    bootstrap_median_interval,
    build_observations,
    evaluate_predictions,
    feature_matrix,
    fit_predictor_candidates,
    rows_for_split,
    target_matrix,
)
from src.roll2film.synthetic_recovery import (  # noqa: E402
    SyntheticBoundedOperator,
    generate_operator_manifest,
    manifest_sha256,
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


def _compact_evaluation(
    evaluation: dict[str, Any],
    config: dict[str, Any],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    summary = evaluation["summary"]
    operator_metrics = summary["operator_metrics"]
    interval = bootstrap_median_interval(
        operator_metrics,
        key="uniform_rgb_rmse",
        replicates=int(config["evaluation"]["operator_group_bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + seed_offset,
    )
    return {
        "overall": {
            **summary["overall"],
            "operator_group_bootstrap_95_interval_median_uniform_rgb_rmse": interval,
        },
        "by_family": summary["by_family"],
    }


def _split_overlap(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {
        split: {row["operator_id"] for row in manifest if row["split"] == split}
        for split in ("fit", "validation", "confirmation", "stress")
    }
    pairs: dict[str, int] = {}
    keys = list(ids)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            pairs[f"{left}__{right}"] = len(ids[left] & ids[right])
    return {"pairwise_overlap": pairs, "maximum_overlap": max(pairs.values())}


def _identity_parameters() -> np.ndarray:
    return np.array([0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75], dtype=np.float64)


def _improvement(reference: float, challenger: float) -> float:
    return float((reference - challenger) / max(reference, 1e-15))


def _make_contact_sheet(
    path: Path,
    confirmation_rows: list[Any],
    raw_predictions: np.ndarray,
    interaction_evaluation: dict[str, Any],
    config: dict[str, Any],
) -> None:
    records = interaction_evaluation["per_observation"]
    worst_indices = sorted(
        range(len(records)),
        key=lambda index: records[index]["uniform_rgb_rmse"],
        reverse=True,
    )[:3]
    grid_size = int(config["evaluation"]["uniform_probe_grid_size"])
    grid = uniform_probe_grid(grid_size)
    chart = grid.reshape(grid_size, grid_size * grid_size, 3)
    scale = 2
    label_width = 92
    row_height = grid_size * scale
    canvas = Image.new(
        "RGB",
        (label_width + chart.shape[1] * scale, row_height * 9),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    output_row = 0
    for observation_index in worst_indices:
        row = confirmation_rows[observation_index]
        predicted_parameters = project_operator_parameters(
            raw_predictions[observation_index],
            maximum_off_diagonal_sum=float(
                config["operator"]["matrix_maximum_total_off_diagonal_per_row"]
            ),
            minimum_tone_increment=float(
                config["operator"]["tone_minimum_knot_increment"]
            ),
        )
        true_operator = SyntheticBoundedOperator(
            row.parameters, row.family, row.operator_id
        )
        predicted_operator = SyntheticBoundedOperator(
            predicted_parameters, "combined", f"predicted-{row.operator_id}"
        )
        images = (
            ("input", chart),
            ("true", true_operator.apply(chart)),
            ("predicted", predicted_operator.apply(chart)),
        )
        for label, values in images:
            rendered = Image.fromarray(
                np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8),
                mode="RGB",
            ).resize(
                (chart.shape[1] * scale, row_height),
                resample=Image.Resampling.NEAREST,
            )
            y = output_row * row_height
            canvas.paste(rendered, (label_width, y))
            draw.text((4, y + 3), f"{row.operator_id}\n{label}", fill="black")
            output_row += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)


def run_benchmark(
    config: dict[str, Any],
    *,
    config_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    manifest = generate_operator_manifest(config)
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path = output_directory / "operator_manifest.json"
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    observations = build_observations(config, manifest)
    fit_rows = rows_for_split(observations, "fit")
    validation_rows = rows_for_split(observations, "validation")
    confirmation_rows = rows_for_split(observations, "confirmation")
    stress_rows = rows_for_split(observations, "stress")

    methods: dict[str, dict[str, Any]] = {}
    prediction_cache: dict[str, dict[str, np.ndarray]] = {}
    representations = (
        "source_content_only_negative",
        "style_target_only",
        "interaction_source_target",
        "canonical_reference_delta_oracle",
        "matched_query_delta_oracle",
        "shuffled_operator_negative",
    )
    for representation_index, representation in enumerate(representations):
        predictor, selection = fit_predictor_candidates(
            representation,
            fit_rows,
            validation_rows,
            config,
            shuffled=representation == "shuffled_operator_negative",
        )
        confirmation_predictions = predictor.predict(
            feature_matrix(confirmation_rows, representation)
        )
        stress_predictions = predictor.predict(feature_matrix(stress_rows, representation))
        confirmation_evaluation = evaluate_predictions(
            confirmation_rows, confirmation_predictions, config
        )
        stress_evaluation = evaluate_predictions(stress_rows, stress_predictions, config)
        methods[representation] = {
            "selection": {
                key: value
                for key, value in selection.items()
                if key != "validation_summary"
            },
            "validation": {
                "overall": selection["validation_summary"]["overall"],
                "by_family": selection["validation_summary"]["by_family"],
            },
            "confirmation": _compact_evaluation(
                confirmation_evaluation,
                config,
                seed_offset=representation_index * 2,
            ),
            "stress": _compact_evaluation(
                stress_evaluation,
                config,
                seed_offset=representation_index * 2 + 1,
            ),
        }
        prediction_cache[representation] = {
            "confirmation": confirmation_predictions,
            "stress": stress_predictions,
        }
        if representation == "interaction_source_target":
            interaction_evaluation = confirmation_evaluation

    fit_targets = target_matrix(fit_rows)
    mean_parameters = project_operator_parameters(
        np.mean(fit_targets, axis=0),
        maximum_off_diagonal_sum=float(
            config["operator"]["matrix_maximum_total_off_diagonal_per_row"]
        ),
        minimum_tone_increment=float(config["operator"]["tone_minimum_knot_increment"]),
    )
    global_confirmation_predictions = np.tile(
        mean_parameters, (len(confirmation_rows), 1)
    )
    global_stress_predictions = np.tile(mean_parameters, (len(stress_rows), 1))
    methods["global_mean"] = {
        "confirmation": _compact_evaluation(
            evaluate_predictions(
                confirmation_rows, global_confirmation_predictions, config
            ),
            config,
            seed_offset=100,
        ),
        "stress": _compact_evaluation(
            evaluate_predictions(stress_rows, global_stress_predictions, config),
            config,
            seed_offset=101,
        ),
        "parameters": mean_parameters.tolist(),
    }

    identity = _identity_parameters()
    identity_evaluation = evaluate_predictions(
        confirmation_rows,
        np.tile(identity, (len(confirmation_rows), 1)),
        config,
    )
    truth_identity_by_family = {
        family: values["median_uniform_rgb_rmse"]
        for family, values in identity_evaluation["summary"]["by_family"].items()
    }

    def confirmation_rmse(method: str) -> float:
        return float(methods[method]["confirmation"]["overall"]["median_uniform_rgb_rmse"])

    global_rmse = confirmation_rmse("global_mean")
    source_rmse = confirmation_rmse("source_content_only_negative")
    target_rmse = confirmation_rmse("style_target_only")
    interaction_rmse = confirmation_rmse("interaction_source_target")
    canonical_rmse = confirmation_rmse("canonical_reference_delta_oracle")
    shuffled_rmse = confirmation_rmse("shuffled_operator_negative")
    best_non_oracle_capture = max(
        float(methods[name]["confirmation"]["overall"]["median_captured_style_fraction"])
        for name in ("style_target_only", "interaction_source_target")
    )
    overlap = _split_overlap(manifest)
    gates_spec = config["gates"]
    all_valid = all(
        float(method["confirmation"]["overall"]["valid_fraction"])
        == float(gates_spec["projected_policy_valid_fraction"])
        for method in methods.values()
    )
    gates = {
        "truth_non_trivial_each_family": all(
            value >= gates_spec["minimum_truth_identity_rgb_rmse_each_family"]
            for value in truth_identity_by_family.values()
        ),
        "zero_operator_group_leakage": (
            overlap["maximum_overlap"] == gates_spec["cross_split_operator_overlap"]
        ),
        "all_projected_policies_valid": all_valid,
        "source_content_negative_control": (
            _improvement(global_rmse, source_rmse)
            <= gates_spec["maximum_source_content_improvement_over_global_fraction"]
        ),
        "shuffled_operator_negative_control": (
            _improvement(global_rmse, shuffled_rmse)
            <= gates_spec["maximum_shuffle_improvement_over_global_fraction"]
        ),
        "canonical_information": (
            _improvement(target_rmse, canonical_rmse)
            >= gates_spec["minimum_canonical_delta_improvement_over_target_only_fraction"]
        ),
        "deployable_interaction_over_global": (
            _improvement(global_rmse, interaction_rmse)
            >= gates_spec["minimum_interaction_improvement_over_global_fraction"]
        ),
        "deployable_interaction_over_target_only": (
            _improvement(target_rmse, interaction_rmse)
            >= gates_spec["minimum_interaction_improvement_over_target_only_fraction"]
        ),
        "best_non_oracle_useful_recovery": (
            best_non_oracle_capture
            >= gates_spec["minimum_best_non_oracle_captured_style_fraction"]
        ),
    }

    contact_sheet_path = output_directory / "interaction_worst_probe_grid.png"
    _make_contact_sheet(
        contact_sheet_path,
        confirmation_rows,
        prediction_cache["interaction_source_target"]["confirmation"],
        interaction_evaluation,
        config,
    )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "manifest_semantic_sha256": manifest_sha256(manifest),
        "manifest_rows": len(manifest),
        "observation_rows": len(observations),
        "split_overlap": overlap,
        "truth_identity_median_uniform_rgb_rmse_by_family": truth_identity_by_family,
        "methods": methods,
        "gate_diagnostics": {
            "source_improvement_over_global_fraction": _improvement(
                global_rmse, source_rmse
            ),
            "shuffle_improvement_over_global_fraction": _improvement(
                global_rmse, shuffled_rmse
            ),
            "canonical_improvement_over_target_only_fraction": _improvement(
                target_rmse, canonical_rmse
            ),
            "interaction_improvement_over_global_fraction": _improvement(
                global_rmse, interaction_rmse
            ),
            "interaction_improvement_over_target_only_fraction": _improvement(
                target_rmse, interaction_rmse
            ),
            "best_non_oracle_captured_style_fraction": best_non_oracle_capture,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "visual_contact_sheet": {
            "path": contact_sheet_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_path(contact_sheet_path),
            "binding": False,
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2d1_synthetic_operator_recovery_v1.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "outputs/u5_r2d1_synthetic_operator_recovery_v1",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    report = run_benchmark(
        config,
        config_sha256=_sha256_bytes(config_bytes),
        output_directory=args.output_directory,
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
                "all_gates_passed": report["all_gates_passed"],
                "gates": report["gates"],
                "gate_diagnostics": report["gate_diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
