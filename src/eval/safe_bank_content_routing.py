"""A0 hard content-routing feasibility for two immutable safe operators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from src.eval.global_frontier import sha256_file


class SafeBankRoutingError(ValueError):
    """Raised when H1 evidence or routing configuration is invalid."""


def content_descriptor(path: Path, spec: Mapping[str, Any]) -> np.ndarray:
    with Image.open(path) as image:
        rgb = (
            np.asarray(
                ImageOps.exif_transpose(image)
                .convert("RGB")
                .resize(tuple(spec["resize"]), Image.Resampling.BILINEAR),
                dtype=np.float64,
            )
            / 255.0
        )
    luma = (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    )
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0.0,
    )
    chroma = maximum - minimum
    features = [
        np.histogram(
            luma,
            bins=int(spec["luma_histogram_bins"]),
            range=(0.0, 1.0),
            density=False,
        )[0].astype(np.float64)
        / luma.size,
        np.quantile(luma, np.asarray(spec["luma_quantiles"], dtype=np.float64)),
        np.histogram(
            saturation,
            bins=int(spec["saturation_histogram_bins"]),
            range=(0.0, 1.0),
            density=False,
        )[0].astype(np.float64)
        / saturation.size,
    ]
    rows, columns = (int(v) for v in spec["spatial_grid"])
    height, width = luma.shape
    if height % rows or width % columns:
        raise SafeBankRoutingError("resize must divide exactly into spatial cells")
    if spec["include_spatial_luma"]:
        features.append(
            luma.reshape(rows, height // rows, columns, width // columns)
            .mean(axis=(1, 3))
            .reshape(-1)
        )
    if spec["include_spatial_chroma_magnitude"]:
        features.append(
            chroma.reshape(rows, height // rows, columns, width // columns)
            .mean(axis=(1, 3))
            .reshape(-1)
        )
    if spec["include_global_rgb_mean_std"]:
        flat = rgb.reshape(-1, 3)
        features.extend((flat.mean(axis=0), flat.std(axis=0)))
    result = np.concatenate(features).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise SafeBankRoutingError("descriptor contains non-finite values")
    return result


def _validate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    for path_key, hash_key in (
        ("input_manifest", "input_manifest_sha256"),
        ("anchor_manifest", "anchor_manifest_sha256"),
        ("anchor_report", "anchor_report_sha256"),
        ("density_manifest", "density_manifest_sha256"),
        ("density_report", "density_report_sha256"),
    ):
        if sha256_file(root / str(config[path_key])) != str(config[hash_key]):
            raise SafeBankRoutingError(f"{path_key} hash mismatch")
    source_payload = json.loads(
        (root / str(config["input_manifest"])).read_text(encoding="utf-8")
    )
    samples = source_payload.get("frozen_set", source_payload)["samples"]
    if len(samples) != 41:
        raise SafeBankRoutingError("H1 requires the exact 41-row A0 bank")
    by_id = {str(row["id"]): dict(row) for row in samples}
    for sample_id, row in by_id.items():
        if sha256_file(root / row["source_path"]) != row["source_sha256"]:
            raise SafeBankRoutingError(f"source hash mismatch: {sample_id}")

    reports = {}
    manifests = {}
    for prefix in ("anchor", "density"):
        report = json.loads(
            (root / str(config[f"{prefix}_report"])).read_text(encoding="utf-8")
        )
        candidate_id = str(config[f"{prefix}_candidate_id"])
        candidate = report["candidates"][candidate_id]
        metrics = {str(row["sample_id"]): row for row in candidate["per_image"]}
        manifest_path = root / str(config[f"{prefix}_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = {
            str(row["sample_id"]): row
            for row in manifest["records"]
            if row["candidate_id"] == candidate_id
        }
        if metrics.keys() != by_id.keys() or records.keys() != by_id.keys():
            raise SafeBankRoutingError(f"{prefix} bank coverage mismatch")
        for sample_id, row in records.items():
            output_path = Path(str(row["output"]))
            if not output_path.is_absolute():
                if prefix == "density":
                    output_path = manifest_path.parent / output_path
                else:
                    output_path = root / output_path
            if sha256_file(output_path) != row["output_sha256"]:
                raise SafeBankRoutingError(
                    f"{prefix} output hash mismatch: {sample_id}"
                )
            if row["source_sha256"] != by_id[sample_id]["source_sha256"]:
                raise SafeBankRoutingError(f"{prefix} source lineage mismatch")
        reports[prefix] = metrics
        manifests[prefix] = records
    return {"samples": by_id, "reports": reports, "manifests": manifests}


def oracle_arrays(
    validated: Mapping[str, Any],
    view: str,
    margin: float,
) -> dict[str, Any]:
    ids = sorted(validated["samples"])
    anchor = validated["reports"]["anchor"]
    density = validated["reports"]["density"]

    def utility(row: Mapping[str, Any]) -> float:
        if view == "non_basic_residual":
            return float(row["median_non_basic_residual_delta_e76"])
        if view == "style_delta_e76":
            return float(row["median_style_delta_e76"])
        if view == "style_plus_non_basic":
            return float(row["median_non_basic_residual_delta_e76"]) + float(
                row["median_style_delta_e76"]
            )
        raise SafeBankRoutingError(f"unsupported Oracle view: {view}")

    anchor_values = np.asarray([utility(anchor[sample_id]) for sample_id in ids])
    density_values = np.asarray([utility(density[sample_id]) for sample_id in ids])
    delta = density_values - anchor_values
    assigned = np.abs(delta) >= margin
    labels = (delta > 0.0).astype(np.int64)
    return {
        "ids": ids,
        "anchor": anchor_values,
        "density": density_values,
        "assigned": assigned,
        "labels": labels,
    }


def _fold_standardize(
    descriptors: np.ndarray,
    training: np.ndarray,
    test_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = descriptors[training].mean(axis=0)
    scale = descriptors[training].std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (
        (descriptors[training] - mean) / scale,
        (descriptors[test_index] - mean) / scale,
    )


def loo_predict(
    descriptors: np.ndarray,
    labels: np.ndarray,
    assigned: np.ndarray,
    router: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    test_indices = np.flatnonzero(assigned)
    predictions = []
    traces = []
    for test_index in test_indices:
        training = np.flatnonzero(assigned & (np.arange(len(labels)) != test_index))
        x_train, x_test = _fold_standardize(descriptors, training, test_index)
        y_train = labels[training]
        kind = str(router["kind"])
        selected: list[int] = []
        if kind == "majority" or len(np.unique(y_train)) < 2:
            counts = np.bincount(y_train, minlength=2)
            prediction = int(1 if counts[1] > counts[0] else 0)
        elif kind == "knn":
            distances = np.linalg.norm(x_train - x_test, axis=1)
            order = np.lexsort((training, distances))
            chosen = order[: int(router["k"])]
            selected = training[chosen].tolist()
            if router["distance_weighted"]:
                weights = 1.0 / np.maximum(distances[chosen], 1e-12)
                votes = np.asarray(
                    [
                        np.sum(weights[y_train[chosen] == label])
                        for label in (0, 1)
                    ]
                )
                prediction = int(1 if votes[1] > votes[0] else 0)
            else:
                counts = np.bincount(y_train[chosen], minlength=2)
                prediction = int(1 if counts[1] > counts[0] else 0)
        elif kind == "logistic":
            model = LogisticRegression(
                C=float(router["c"]),
                solver="liblinear",
                random_state=int(router["seed"]),
                max_iter=1000,
            )
            model.fit(x_train, y_train)
            prediction = int(model.predict(x_test[None, :])[0])
        else:
            raise SafeBankRoutingError(f"unsupported router: {kind}")
        predictions.append(prediction)
        traces.append(
            {
                "test_index": int(test_index),
                "selected_training_indices": selected,
            }
        )
    return np.asarray(predictions, dtype=np.int64), traces


def _metrics(
    oracle: Mapping[str, Any],
    predictions: np.ndarray,
) -> dict[str, float]:
    assigned = np.asarray(oracle["assigned"], dtype=bool)
    labels = np.asarray(oracle["labels"], dtype=np.int64)
    truth = labels[assigned]
    anchor = np.asarray(oracle["anchor"], dtype=np.float64)[assigned]
    density = np.asarray(oracle["density"], dtype=np.float64)[assigned]
    achieved = np.where(predictions == 1, density, anchor)
    utility = np.stack((anchor, density), axis=1)
    oracle_utility = np.max(utility, axis=1)
    global_means = utility.mean(axis=0)
    global_class = int(np.argmax(global_means))
    global_utility = utility[:, global_class]
    denominator = float(np.mean(oracle_utility - global_utility))
    regret_closed = (
        float(np.mean(achieved - global_utility)) / denominator
        if denominator > 1e-12
        else 0.0
    )
    return {
        "accuracy": float(np.mean(predictions == truth)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "class_0_recall": float(np.mean(predictions[truth == 0] == 0)),
        "class_1_recall": float(np.mean(predictions[truth == 1] == 1)),
        "mean_achieved_utility": float(np.mean(achieved)),
        "median_achieved_utility": float(np.median(achieved)),
        "mean_global_utility": float(np.mean(global_utility)),
        "mean_oracle_utility": float(np.mean(oracle_utility)),
        "median_global_utility": float(np.median(global_utility)),
        "median_oracle_utility": float(np.median(oracle_utility)),
        "fraction_mean_regret_closed": regret_closed,
        "global_class": global_class,
    }


def run_audit(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validated = _validate(root, config)
    ids = sorted(validated["samples"])
    descriptors = np.stack(
        [
            content_descriptor(
                root / validated["samples"][sample_id]["source_path"],
                config["content_descriptor"],
            )
            for sample_id in ids
        ]
    )
    views = {}
    rng = np.random.default_rng(int(config["seed"]))
    for view in config["oracle_views"]:
        oracle = oracle_arrays(
            validated,
            str(view),
            float(config["minimum_assignment_margin"]),
        )
        assigned = np.asarray(oracle["assigned"], dtype=bool)
        labels = np.asarray(oracle["labels"], dtype=np.int64)
        class_counts = np.bincount(labels[assigned], minlength=2)
        anchor = np.asarray(oracle["anchor"])[assigned]
        density = np.asarray(oracle["density"])[assigned]
        oracle_utility = np.maximum(anchor, density)
        best_global_median = max(float(np.median(anchor)), float(np.median(density)))
        oracle_gain = float(np.median(oracle_utility) / best_global_median - 1.0)
        router_rows = {}
        null_labels = [
            labels.copy()
            for _ in range(int(config["permutations"]))
        ]
        assigned_values = labels[assigned]
        for item in null_labels:
            item[assigned] = rng.permutation(assigned_values)
        for router in config["routers"]:
            prediction, traces = loo_predict(
                descriptors, labels, assigned, router
            )
            metrics = _metrics(oracle, prediction)
            null_scores = []
            for permuted in null_labels:
                null_prediction, _ = loo_predict(
                    descriptors, permuted, assigned, router
                )
                null_scores.append(
                    float(
                        balanced_accuracy_score(
                            permuted[assigned], null_prediction
                        )
                    )
                )
            observed = metrics["balanced_accuracy"]
            metrics["permutation_p_value"] = float(
                (1 + np.sum(np.asarray(null_scores) >= observed))
                / (1 + len(null_scores))
            )
            metrics["predictions"] = prediction.tolist()
            metrics["test_ids"] = [
                ids[index] for index in np.flatnonzero(assigned)
            ]
            metrics["traces"] = [
                {
                    "test_id": ids[row["test_index"]],
                    "selected_training_ids": [
                        ids[index]
                        for index in row["selected_training_indices"]
                    ],
                }
                for row in traces
            ]
            router_rows[str(router["router_id"])] = metrics
        views[str(view)] = {
            "assigned_count": int(np.sum(assigned)),
            "coverage": float(np.mean(assigned)),
            "class_counts": class_counts.tolist(),
            "oracle_median_gain_fraction": oracle_gain,
            "routers": router_rows,
        }

    primary = views[str(config["primary_view"])]
    gates = config["gates"]
    viable = []
    for router_id, row in primary["routers"].items():
        if router_id == "majority":
            continue
        if (
            row["balanced_accuracy"]
            >= float(gates["minimum_router_balanced_accuracy"])
            and row["permutation_p_value"]
            <= float(gates["maximum_permutation_p_value"])
            and row["fraction_mean_regret_closed"]
            >= float(gates["minimum_fraction_mean_regret_closed"])
        ):
            viable.append(router_id)
    component_best = []
    for view in ("non_basic_residual", "style_delta_e76"):
        rows = [
            row
            for key, row in views[view]["routers"].items()
            if key != "majority"
        ]
        component_best.append(
            max(rows, key=lambda row: row["balanced_accuracy"])
        )
    contradicted = all(
        row["balanced_accuracy"] < 0.5
        or row["fraction_mean_regret_closed"] < 0.0
        for row in component_best
    )
    gate_report = {
        "oracle_gain": primary["oracle_median_gain_fraction"]
        >= float(gates["minimum_composite_oracle_median_gain_fraction"]),
        "class_support": min(primary["class_counts"])
        >= int(gates["minimum_assigned_rows_per_class"]),
        "router": bool(viable),
        "component_views_not_both_contradictory": not contradicted,
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "sample_count": len(ids),
        "descriptor_dimension": int(descriptors.shape[1]),
        "descriptor_sha256": __import__("hashlib").sha256(
            np.asarray(descriptors, dtype="<f8").tobytes()
        ).hexdigest(),
        "views": views,
        "viable_primary_routers": viable,
        "gates": gate_report,
        "decision": (
            "retain_simple_hard_router_for_visual_diagnostic"
            if all(gate_report.values())
            else "close_content_routing_for_two_operator_a0_bank"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "SafeBankRoutingError",
    "content_descriptor",
    "loo_predict",
    "oracle_arrays",
    "run_audit",
]
