"""U5.R2AF1 manifest evaluator for the isolated AceTone LUT tokenizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.spectral_film_lut_bank import (
    array_sha256,
    canonical_sha256,
    local_jacobian_metrics,
    median_delta_e76,
    sha256_file,
    synthetic_cube,
)
from src.eval.velvia_datasheet_witness import (
    _RGB_TO_XYZ,
    encoded_srgb_to_linear,
    xyz_to_lab,
)


MANIFEST_SCHEMA = "u5-r2af1-acetone-tokenizer-manifest-v1"


def analytic_lut(
    cube: np.ndarray, curve_a: list[float], matrix: list[list[float]]
) -> np.ndarray:
    values = np.asarray(cube, dtype=np.float64)
    coefficients = np.asarray(curve_a, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if coefficients.shape != (3,) or transform.shape != (3, 3):
        raise ValueError("analytic LUT requires three curve terms and a 3x3 matrix")
    curved = values + coefficients * values * (1.0 - values)
    output = curved @ transform.T
    return np.asarray(output, dtype=np.float32)


def build_population(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    cube = synthetic_cube(int(config["population"]["cube_size"]))
    return {
        row["id"]: analytic_lut(cube, row["curve_a"], row["matrix"])
        for row in config["population"]["records"]
    }


def _lab(encoded: np.ndarray) -> np.ndarray:
    linear = encoded_srgb_to_linear(np.clip(np.asarray(encoded), 0.0, 1.0))
    return xyz_to_lab(linear @ _RGB_TO_XYZ.T)


def delta_e_summary(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    differences = np.linalg.norm(_lab(source) - _lab(target), axis=-1)
    return {
        "median_delta_e76": float(np.median(differences)),
        "p95_delta_e76": float(np.percentile(differences, 95)),
    }


def metric_negative_control(size: int) -> float:
    swapped = synthetic_cube(size)[..., [1, 0, 2]]
    return local_jacobian_metrics(swapped)["negative_jacobian_fraction"]


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected AF1 manifest schema")
    return value


def _validate_manifest(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    source = config["external_source"]
    runtime = config["runtime"]
    if manifest["experiment_id"] != config["experiment_id"]:
        raise ValueError("experiment identity mismatch")
    if manifest["config_sha256"] != canonical_sha256(config):
        raise ValueError("config hash mismatch")
    if manifest["external_revision"] != source["revision"]:
        raise ValueError("external revision mismatch")
    if manifest["model_source_sha256"] != source["model_source_sha256"]:
        raise ValueError("external model-source hash mismatch")
    if manifest["checkpoint_sha256"] != source["checkpoint_sha256"]:
        raise ValueError("external checkpoint hash mismatch")
    if manifest["checkpoint_bytes"] != source["checkpoint_bytes"]:
        raise ValueError("external checkpoint size mismatch")
    if manifest["runtime"] != runtime:
        raise ValueError("runtime mismatch")
    expected = [row["id"] for row in config["population"]["records"]]
    observed = [row["id"] for row in manifest["records"]]
    if observed != expected:
        raise ValueError("population or order mismatch")


def _resolve_artifact(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("artifact path escapes project root") from exc
    return path


def _load_artifact(
    root: Path,
    record: Mapping[str, Any],
    prefix: str,
) -> np.ndarray:
    path = _resolve_artifact(root, record[f"{prefix}_path"])
    if sha256_file(path) != record[f"{prefix}_file_sha256"]:
        raise ValueError(f"{prefix} file hash mismatch: {record['id']}")
    value = np.load(path, allow_pickle=False)
    if array_sha256(value) != record[f"{prefix}_array_sha256"]:
        raise ValueError(f"{prefix} array hash mismatch: {record['id']}")
    return value


def _load_run(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    _validate_manifest(manifest, config)
    sources: dict[str, np.ndarray] = {}
    reconstructions: dict[str, np.ndarray] = {}
    codes: dict[str, np.ndarray] = {}
    for row in manifest["records"]:
        name = row["id"]
        sources[name] = _load_artifact(root, row, "source")
        reconstructions[name] = _load_artifact(root, row, "reconstruction")
        codes[name] = _load_artifact(root, row, "codes")
    duplicate = manifest["duplicate_control"]
    reconstructions["__duplicate__"] = _load_artifact(
        root, duplicate, "reconstruction"
    )
    codes["__duplicate__"] = _load_artifact(root, duplicate, "codes")
    return sources, reconstructions, codes


def evaluate_manifests(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    expected = build_population(config)
    sources_a, recon_a, codes_a = _load_run(first, config, root=root)
    sources_b, recon_b, codes_b = _load_run(second, config, root=root)
    gates = config["automatic_gates"]
    size = int(config["population"]["cube_size"])
    duplicate_id = config["population"]["duplicate_control_id"]

    expected_shape = (size, size, size, 3)
    source_controls_pass = True
    exact_repeat = True
    records: list[dict[str, Any]] = []
    for name in expected:
        source = np.asarray(sources_a[name])
        reconstruction = np.asarray(recon_a[name])
        if source.shape != expected_shape or reconstruction.shape != expected_shape:
            raise ValueError(f"unexpected LUT shape: {name}")
        if not np.array_equal(source, expected[name]):
            raise ValueError(f"source generation mismatch: {name}")
        source_metrics = local_jacobian_metrics(source)
        source_finite = bool(np.all(np.isfinite(source)))
        source_range = bool(
            np.min(source) >= gates["source_range_min"]
            and np.max(source) <= gates["source_range_max"]
        )
        source_record_pass = bool(
            source_finite
            and source_range
            and source_metrics["negative_jacobian_fraction"]
            <= gates["source_maximum_negative_jacobian_fraction"]
            and source_metrics["minimum_jacobian_determinant"]
            >= gates["source_minimum_jacobian_determinant"]
        )
        source_controls_pass = source_controls_pass and source_record_pass

        repeat = bool(
            np.array_equal(source, sources_b[name])
            and np.array_equal(reconstruction, recon_b[name])
            and np.array_equal(codes_a[name], codes_b[name])
        )
        exact_repeat = exact_repeat and repeat
        reconstruction_metrics = local_jacobian_metrics(reconstruction)
        delta_e = delta_e_summary(source, reconstruction)
        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(reconstruction, dtype=np.float64)
                        - np.asarray(source, dtype=np.float64)
                    )
                    ** 2
                )
            )
        )
        source_interior = np.all(
            (source > gates["interior_clip_margin"])
            & (source < 1.0 - gates["interior_clip_margin"]),
            axis=-1,
        )
        interior_values = reconstruction[source_interior]
        new_clipping = (
            0.0
            if interior_values.size == 0
            else float(
                np.mean(
                    (interior_values <= 1e-6) | (interior_values >= 1.0 - 1e-6)
                )
            )
        )
        metric_values = {
            "source_all_finite": source_finite,
            "source_minimum_jacobian_determinant": source_metrics[
                "minimum_jacobian_determinant"
            ],
            "source_negative_jacobian_fraction": source_metrics[
                "negative_jacobian_fraction"
            ],
            "reconstruction_all_finite": bool(
                np.all(np.isfinite(reconstruction))
            ),
            "reconstruction_minimum": float(np.min(reconstruction)),
            "reconstruction_maximum": float(np.max(reconstruction)),
            "reconstruction_minimum_jacobian_determinant": reconstruction_metrics[
                "minimum_jacobian_determinant"
            ],
            "reconstruction_negative_jacobian_fraction": reconstruction_metrics[
                "negative_jacobian_fraction"
            ],
            "reconstruction_maximum_jacobian_spectral_norm": reconstruction_metrics[
                "maximum_jacobian_spectral_norm"
            ],
            "rgb_rmse": rmse,
            **delta_e,
            "new_interior_clipping_fraction": new_clipping,
            "repeat_exact": repeat,
        }
        gate_values = {
            "finite_and_range": bool(
                metric_values["reconstruction_all_finite"]
                and metric_values["reconstruction_minimum"]
                >= gates["reconstruction_range_min"]
                and metric_values["reconstruction_maximum"]
                <= gates["reconstruction_range_max"]
            ),
            "jacobian_orientation": metric_values[
                "reconstruction_negative_jacobian_fraction"
            ]
            <= gates["maximum_negative_jacobian_fraction"],
            "jacobian_amplification": metric_values[
                "reconstruction_maximum_jacobian_spectral_norm"
            ]
            <= gates["maximum_jacobian_spectral_norm"],
            "rgb_fidelity": rmse <= gates["maximum_rgb_rmse"],
            "delta_e_fidelity": bool(
                delta_e["median_delta_e76"] <= gates["maximum_median_delta_e76"]
                and delta_e["p95_delta_e76"] <= gates["maximum_p95_delta_e76"]
            ),
            "interior_clipping": new_clipping
            <= gates["maximum_new_interior_clipping_fraction"],
        }
        records.append(
            {
                "id": name,
                "source_controls_pass": source_record_pass,
                "metrics": metric_values,
                "gates": gate_values,
                "all_reconstruction_gates_pass": all(gate_values.values()),
            }
        )

    duplicate_exact = bool(
        np.array_equal(recon_a[duplicate_id], recon_a["__duplicate__"])
        and np.array_equal(codes_a[duplicate_id], codes_a["__duplicate__"])
        and np.array_equal(recon_b[duplicate_id], recon_b["__duplicate__"])
        and np.array_equal(codes_b[duplicate_id], codes_b["__duplicate__"])
    )
    negative_control_fraction = metric_negative_control(size)
    negative_control_pass = (
        negative_control_fraction
        >= gates["negative_metric_control_minimum_fraction"]
    )
    topology_pass = all(
        row["gates"]["jacobian_orientation"]
        and row["gates"]["jacobian_amplification"]
        for row in records
    )
    fidelity_pass = all(
        row["gates"]["rgb_fidelity"] and row["gates"]["delta_e_fidelity"]
        for row in records
    )
    other_reconstruction_pass = all(
        row["gates"]["finite_and_range"] and row["gates"]["interior_clipping"]
        for row in records
    )
    controls_pass = bool(
        source_controls_pass
        and negative_control_pass
        and exact_repeat
        and duplicate_exact
    )
    if not controls_pass:
        decision = "invalid_control_or_replay_failure"
    elif not topology_pass:
        decision = "close_topology_failure"
    elif not fidelity_pass:
        decision = "close_fidelity_failure"
    elif not other_reconstruction_pass:
        decision = "close_range_or_clipping_failure"
    else:
        decision = "retain_synthetic_representation_feasibility_only"

    return {
        "schema_version": "u5-r2af1-acetone-tokenizer-topology-report-v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": canonical_sha256(config),
        "run_ids": [first["run_id"], second["run_id"]],
        "controls": {
            "source_controls_pass": source_controls_pass,
            "exact_repeat": exact_repeat,
            "duplicate_exact": duplicate_exact,
            "negative_metric_control_fraction": negative_control_fraction,
            "negative_metric_control_pass": negative_control_pass,
        },
        "records": records,
        "summary": {
            "topology_pass": topology_pass,
            "fidelity_pass": fidelity_pass,
            "other_reconstruction_pass": other_reconstruction_pass,
            "all_controls_pass": controls_pass,
            "survivor_count": sum(
                row["all_reconstruction_gates_pass"] for row in records
            ),
            "population_count": len(records),
        },
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }


def report_sha256(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            report, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
