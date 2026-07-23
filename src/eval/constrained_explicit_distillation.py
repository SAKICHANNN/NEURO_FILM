"""Synthetic-grid distillation into bounded explicit colour operators."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from src.eval.canoncgt_reference_condition import out_of_range_fraction
from src.eval.global_frontier import new_hard_clipping_fraction, sha256_file
from src.eval.projected_lut_frontier import (
    convert_public_lut,
    lut_diagnostics,
    validate_contract as validate_f1_contract,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)
from src.roll2film.lut import DenseLUT3D


OPERATOR_SCHEMA = "neuro_film.bounded_curve_matrix.v1"


class ExplicitDistillationError(ValueError):
    """Raised when frozen G0 evidence or parameters are invalid."""


@dataclass(frozen=True)
class BoundedCurveMatrixOperator:
    """Strictly monotone per-channel curves followed by a convex RGB matrix."""

    curve_values: np.ndarray
    matrix: np.ndarray

    def __post_init__(self) -> None:
        curves = np.asarray(self.curve_values, dtype=np.float64)
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if curves.ndim != 2 or curves.shape[0] != 3 or curves.shape[1] < 2:
            raise ExplicitDistillationError("curve_values must have shape (3,K)")
        if matrix.shape != (3, 3):
            raise ExplicitDistillationError("matrix must have shape (3,3)")
        if not np.all(np.isfinite(curves)) or not np.all(np.isfinite(matrix)):
            raise ExplicitDistillationError("operator parameters must be finite")
        if not np.array_equal(curves[:, 0], np.zeros(3)):
            raise ExplicitDistillationError("curve lower endpoints must be zero")
        if not np.array_equal(curves[:, -1], np.ones(3)):
            raise ExplicitDistillationError("curve upper endpoints must be one")
        if np.any(np.diff(curves, axis=1) <= 0.0):
            raise ExplicitDistillationError("curves must be strictly increasing")
        if np.any(matrix < 0.0) or np.max(np.abs(matrix.sum(axis=1) - 1.0)) > 1e-12:
            raise ExplicitDistillationError("matrix must be non-negative row-stochastic")
        if float(np.linalg.det(matrix)) <= 0.0:
            raise ExplicitDistillationError("matrix must preserve orientation")
        curves = curves.copy()
        matrix = matrix.copy()
        curves.setflags(write=False)
        matrix.setflags(write=False)
        object.__setattr__(self, "curve_values", curves)
        object.__setattr__(self, "matrix", matrix)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(rgb, dtype=np.float64)
        if (
            values.ndim < 2
            or values.shape[-1] != 3
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ExplicitDistillationError(
                "operator input must be finite [0,1] RGB"
            )
        axis = np.linspace(0.0, 1.0, self.curve_values.shape[1])
        curved = np.stack(
            [
                np.interp(values[..., channel], axis, self.curve_values[channel])
                for channel in range(3)
            ],
            axis=-1,
        )
        return curved @ self.matrix.T

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OPERATOR_SCHEMA,
            "curve_values": self.curve_values.tolist(),
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BoundedCurveMatrixOperator":
        if payload.get("schema") != OPERATOR_SCHEMA:
            raise ExplicitDistillationError("unsupported operator schema")
        return cls(
            curve_values=np.asarray(payload["curve_values"], dtype=np.float64),
            matrix=np.asarray(payload["matrix"], dtype=np.float64),
        )


def synthetic_grid(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def compose_f1_target(
    canonical_public: np.ndarray,
    restyle_public: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    grid = synthetic_grid(grid_size)
    canonical = DenseLUT3D(
        convert_public_lut(canonical_public),
        np.zeros(3),
        np.ones(3),
        "trilinear",
    ).apply(grid)
    # The external implementation uses border padding for the second LUT.
    final = DenseLUT3D(
        convert_public_lut(restyle_public),
        np.zeros(3),
        np.ones(3),
        "trilinear",
    ).apply(np.clip(canonical, 0.0, 1.0))
    return np.clip(final, 0.0, 1.0), {
        "raw_target_minimum": float(np.min(final)),
        "raw_target_maximum": float(np.max(final)),
        "raw_target_out_of_range_fraction": out_of_range_fraction(final),
    }


def fit_operator(
    inputs: np.ndarray,
    targets: np.ndarray,
    fit: Mapping[str, Any],
) -> tuple[BoundedCurveMatrixOperator, dict[str, Any]]:
    import torch

    if str(fit["dtype"]) != "float64" or str(fit["device"]) != "cpu":
        raise ExplicitDistillationError("G0 requires float64 CPU fitting")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    x = torch.as_tensor(
        np.asarray(inputs, dtype=np.float64).reshape(-1, 3),
        dtype=torch.float64,
    )
    target = torch.as_tensor(
        np.asarray(targets, dtype=np.float64).reshape(-1, 3),
        dtype=torch.float64,
    )
    knot_count = int(fit["curve_knot_count"])
    interval_count = knot_count - 1
    minimum_interval = float(fit["minimum_curve_interval"])
    remaining = 1.0 - interval_count * minimum_interval
    if remaining <= 0.0:
        raise ExplicitDistillationError("minimum curve interval is infeasible")
    maximum_mix = float(fit["maximum_matrix_mix"])
    identity = torch.eye(3, dtype=torch.float64)
    axis = torch.linspace(0.0, 1.0, knot_count, dtype=torch.float64)
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    histories = []

    for restart in range(int(fit["restarts"])):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(fit["seed"]) + restart)
        curve_logits = torch.nn.Parameter(
            torch.randn((3, interval_count), generator=generator, dtype=torch.float64)
            * (0.0 if restart == 0 else 0.05)
        )
        matrix_logits = torch.nn.Parameter(
            torch.randn((3, 3), generator=generator, dtype=torch.float64)
            * (0.0 if restart == 0 else 0.05)
        )
        alpha_logit = torch.nn.Parameter(
            torch.tensor(-3.0 + 0.2 * restart, dtype=torch.float64)
        )
        optimizer = torch.optim.Adam(
            [curve_logits, matrix_logits, alpha_logit],
            lr=float(fit["learning_rate"]),
        )
        loss_value = float("inf")
        for _ in range(int(fit["steps"])):
            optimizer.zero_grad(set_to_none=True)
            increments = minimum_interval + remaining * torch.softmax(
                curve_logits, dim=1
            )
            curves = torch.cat(
                (
                    torch.zeros((3, 1), dtype=torch.float64),
                    torch.cumsum(increments, dim=1),
                ),
                dim=1,
            )
            alpha = maximum_mix * torch.sigmoid(alpha_logit)
            matrix = (1.0 - alpha) * identity + alpha * torch.softmax(
                matrix_logits, dim=1
            )
            scaled = x * interval_count
            lower = torch.clamp(
                torch.floor(scaled).to(torch.int64),
                min=0,
                max=interval_count - 1,
            )
            fraction = scaled - lower.to(torch.float64)
            channels = []
            for channel in range(3):
                low = curves[channel].gather(0, lower[:, channel])
                high = curves[channel].gather(0, lower[:, channel] + 1)
                channels.append(
                    low + fraction[:, channel] * (high - low)
                )
            curved = torch.stack(channels, dim=1)
            predicted = curved @ matrix.T
            regularization = torch.mean((matrix - identity) ** 2) + torch.mean(
                (curves - axis[None, :]) ** 2
            )
            loss = torch.mean((predicted - target) ** 2) + float(
                fit["identity_regularization"]
            ) * regularization
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
        curve_array = curves.detach().cpu().numpy()
        curve_array[:, 0] = 0.0
        curve_array[:, -1] = 1.0
        matrix_array = matrix.detach().cpu().numpy()
        histories.append({"restart": restart, "final_objective": loss_value})
        key = (loss_value, restart)
        if best is None or key < (best[0], best[1]):
            best = (loss_value, restart, curve_array, matrix_array)

    assert best is not None
    operator = BoundedCurveMatrixOperator(best[2], best[3])
    prediction = operator.apply(np.asarray(inputs, dtype=np.float64))
    return operator, {
        "selected_restart": best[1],
        "objective": best[0],
        "target_rmse": float(
            np.sqrt(np.mean((prediction - np.asarray(targets)) ** 2))
        ),
        "restarts": histories,
    }


def operator_diagnostics(
    operator: BoundedCurveMatrixOperator,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["structure_gates"]
    grid_size = int(config["fit"]["synthetic_grid_size"])
    baked = operator.apply(synthetic_grid(grid_size))
    lut_audit = lut_diagnostics(baked)
    replay = BoundedCurveMatrixOperator.from_dict(operator.to_dict()).apply(
        synthetic_grid(grid_size)
    )
    report = {
        "minimum_curve_interval": float(
            np.min(np.diff(operator.curve_values, axis=1))
        ),
        "matrix_minimum_entry": float(np.min(operator.matrix)),
        "matrix_maximum_row_sum_error": float(
            np.max(np.abs(operator.matrix.sum(axis=1) - 1.0))
        ),
        "matrix_determinant": float(np.linalg.det(operator.matrix)),
        "replay_maximum_absolute_error": float(np.max(np.abs(replay - baked))),
        **lut_audit,
    }
    report["structure_safe"] = bool(
        report["minimum_curve_interval"]
        >= float(gates["minimum_curve_interval"])
        and report["matrix_minimum_entry"]
        >= float(gates["matrix_minimum_entry"])
        and report["matrix_maximum_row_sum_error"]
        <= float(gates["matrix_maximum_row_sum_error"])
        and report["matrix_determinant"]
        >= float(gates["matrix_minimum_determinant"])
        and report["minimum_node"] >= float(gates["node_minimum"])
        and report["maximum_node"] <= float(gates["node_maximum"])
        and report["minimum_corresponding_channel_grid_step"]
        >= float(gates["minimum_corresponding_channel_grid_step"])
        and report["minimum_tetrahedron_jacobian_determinant"]
        >= float(gates["minimum_tetrahedron_jacobian_determinant"])
        and report["replay_maximum_absolute_error"]
        <= float(gates["maximum_replay_absolute_error"])
    )
    return report


def _validated(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    for path_key, hash_key in (
        ("source_f2_config", "source_f2_config_sha256"),
        ("source_f2_report", "source_f2_report_sha256"),
    ):
        if sha256_file(root / str(config[path_key])) != str(config[hash_key]):
            raise ExplicitDistillationError(f"{path_key} hash mismatch")
    validated = validate_f1_contract(root, config)
    if set(validated["samples"]) != set(str(v) for v in config["sample_ids"]):
        raise ExplicitDistillationError("sample bank drift")
    return validated


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = _validated(root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    grid = synthetic_grid(int(config["fit"]["synthetic_grid_size"]))
    for reference_id in config["reference_ids"]:
        candidate_dir = output_dir / reference_id
        operator_dir = candidate_dir / "operators"
        operator_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in config["sample_ids"]:
            source_row = validated["records"][(reference_id, sample_id)]
            canonical_path = (
                validated["manifest_path"].parent / source_row["canonical_lut"]
            )
            restyle_path = (
                validated["manifest_path"].parent / source_row["restyle_lut"]
            )
            target, target_audit = compose_f1_target(
                np.load(canonical_path, allow_pickle=False),
                np.load(restyle_path, allow_pickle=False),
                grid_size=int(config["fit"]["synthetic_grid_size"]),
            )
            operator, fit_audit = fit_operator(grid, target, config["fit"])
            structure = operator_diagnostics(operator, config)
            operator_path = operator_dir / f"{sample_id}.json"
            operator_bytes = (
                json.dumps(operator.to_dict(), indent=2, sort_keys=True) + "\n"
            ).encode()
            operator_path.write_bytes(operator_bytes)
            sample = validated["samples"][sample_id]
            with Image.open(root / str(sample["source_path"])) as image:
                source = (
                    np.asarray(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        dtype=np.float64,
                    )
                    / 255.0
                )
            output_raw = operator.apply(source)
            output_path = candidate_dir / f"{sample_id}.png"
            Image.fromarray(
                np.rint(output_raw * 255.0).astype(np.uint8), mode="RGB"
            ).save(output_path, format="PNG", compress_level=6)
            records.append(
                {
                    "reference_id": reference_id,
                    "provenance_bucket": validated["references"][reference_id][
                        "provenance_bucket"
                    ],
                    "sample_id": sample_id,
                    "source_sha256": sample["source_sha256"],
                    "source_canonical_lut_sha256": sha256_file(canonical_path),
                    "source_restyle_lut_sha256": sha256_file(restyle_path),
                    "operator": f"{reference_id}/operators/{sample_id}.json",
                    "operator_sha256": hashlib.sha256(operator_bytes).hexdigest(),
                    "fit": fit_audit,
                    "target": target_audit,
                    "structure": structure,
                    "raw_final_minimum": float(np.min(output_raw)),
                    "raw_final_maximum": float(np.max(output_raw)),
                    "raw_final_out_of_range_fraction": out_of_range_fraction(
                        output_raw
                    ),
                    "output": f"{reference_id}/{sample_id}.png",
                    "output_sha256": sha256_file(output_path),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "candidate_count": len(config["reference_ids"]),
        "sample_count": len(config["sample_ids"]),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = output_dir / "manifest.json"
    path.write_bytes(encoded)
    return {
        "manifest_path": path,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest": manifest,
    }


def evaluate_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    validated = _validated(root, config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        (str(row["reference_id"]), str(row["sample_id"])): row
        for row in manifest["records"]
    }
    expected = {
        (str(reference), str(sample))
        for reference in config["reference_ids"]
        for sample in config["sample_ids"]
    }
    if records.keys() != expected:
        raise ExplicitDistillationError("manifest bank is incomplete")
    budget = 250000
    epsilon = 1.0 / 255.0
    gates = config["automatic_gates"]
    summaries: dict[str, Any] = {}
    sampled_outputs: dict[tuple[str, str], np.ndarray] = {}
    for reference_id in config["reference_ids"]:
        per_image = []
        for sample_id in config["sample_ids"]:
            row = records[(reference_id, sample_id)]
            output_path = manifest_path.parent / row["output"]
            if sha256_file(output_path) != row["output_sha256"]:
                raise ExplicitDistillationError("output hash mismatch")
            operator_path = manifest_path.parent / row["operator"]
            if sha256_file(operator_path) != row["operator_sha256"]:
                raise ExplicitDistillationError("operator hash mismatch")
            source = sample_rgb_image(
                root / str(validated["samples"][sample_id]["source_path"]),
                budget,
            )
            output = sample_rgb_image(output_path, budget)
            sampled_outputs[(reference_id, sample_id)] = output
            style, residual = style_and_basic_residual(source, output)
            per_image.append(
                {
                    "sample_id": sample_id,
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source, output, epsilon
                    ),
                    "raw_final_out_of_range_fraction": row[
                        "raw_final_out_of_range_fraction"
                    ],
                    "target_rmse": row["fit"]["target_rmse"],
                }
            )
        summary = {
            "reference_id": reference_id,
            "provenance_bucket": records[
                (reference_id, str(config["sample_ids"][0]))
            ]["provenance_bucket"],
            "all_operators_structure_safe": all(
                records[(reference_id, sample)]["structure"]["structure_safe"]
                for sample in config["sample_ids"]
            ),
            "gold_median_style_delta_e76": float(
                np.median([row["median_style_delta_e76"] for row in per_image])
            ),
            "gold_median_non_basic_residual_delta_e76": float(
                np.median(
                    [
                        row["median_non_basic_residual_delta_e76"]
                        for row in per_image
                    ]
                )
            ),
            "worst_gold_new_hard_clipping_fraction": max(
                row["new_hard_clipping_fraction"] for row in per_image
            ),
            "worst_gold_raw_final_out_of_range_fraction": max(
                row["raw_final_out_of_range_fraction"] for row in per_image
            ),
            "median_synthetic_target_rmse": float(
                np.median([row["target_rmse"] for row in per_image])
            ),
            "per_image": per_image,
        }
        automatic = {
            "structure": summary["all_operators_structure_safe"],
            "style": summary["gold_median_style_delta_e76"]
            >= float(gates["gold_median_style_delta_e76_minimum"]),
            "non_basic": summary["gold_median_non_basic_residual_delta_e76"]
            >= float(
                gates["gold_median_non_basic_residual_delta_e76_minimum"]
            ),
            "clipping": summary["worst_gold_new_hard_clipping_fraction"]
            <= float(gates["worst_gold_new_hard_clipping_fraction_maximum"]),
            "raw_range": summary["worst_gold_raw_final_out_of_range_fraction"]
            <= float(
                gates["worst_gold_raw_final_out_of_range_fraction_maximum"]
            ),
        }
        summary["automatic_gates"] = automatic
        summary["automatic_survivor"] = all(automatic.values())
        summaries[reference_id] = summary

    sensitivity_values = []
    reference_ids = [str(value) for value in config["reference_ids"]]
    for sample_id in config["sample_ids"]:
        labs = {
            reference: rgb2lab(
                sampled_outputs[(reference, sample_id)].reshape(-1, 1, 3)
            ).reshape(-1, 3)
            for reference in reference_ids
        }
        for index, first in enumerate(reference_ids):
            for second in reference_ids[index + 1 :]:
                sensitivity_values.append(
                    float(
                        np.median(
                            np.linalg.norm(labs[first] - labs[second], axis=1)
                        )
                    )
                )
    sensitivity = float(np.median(sensitivity_values))
    sensitivity_pass = sensitivity >= float(
        gates["reference_sensitivity_median_pairwise_delta_e76_minimum"]
    )
    f2 = json.loads(
        (root / str(config["source_f2_report"])).read_text(encoding="utf-8")
    )
    f2_styles = [
        float(
            f2["candidates"][
                f"{reference}__safe_contract_cap100"
            ]["gold_median_style_delta_e76"]
        )
        for reference in reference_ids
    ]
    style_advantage = float(
        np.median(
            [summaries[reference]["gold_median_style_delta_e76"] for reference in reference_ids]
        )
        - np.median(f2_styles)
    )
    advantage_pass = style_advantage >= float(
        gates["branch_median_style_advantage_over_f2_cap100_minimum"]
    )
    survivors = sorted(
        reference
        for reference, row in summaries.items()
        if row["automatic_survivor"]
    )
    ranked = sorted(
        survivors,
        key=lambda reference: (
            -summaries[reference][
                "gold_median_non_basic_residual_delta_e76"
            ],
            -summaries[reference]["gold_median_style_delta_e76"],
            reference,
        ),
    )
    shortlist = []
    seen = set()
    if sensitivity_pass and advantage_pass:
        for reference in ranked:
            bucket = summaries[reference]["provenance_bucket"]
            if bucket in seen:
                continue
            seen.add(bucket)
            shortlist.append(reference)
            if len(shortlist) == 3:
                break
    return {
        "candidate_count": len(reference_ids),
        "sample_count": len(config["sample_ids"]),
        "candidates": summaries,
        "automatic_survivors": survivors,
        "reference_sensitivity_median_pairwise_output_delta_e76": sensitivity,
        "reference_sensitivity_gate_passed": sensitivity_pass,
        "median_style_advantage_over_f2_cap100": style_advantage,
        "branch_style_advantage_gate_passed": advantage_pass,
        "shortlist": shortlist,
        "automatic_decision": (
            "visual_gate_required"
            if shortlist
            else (
                "no_automatic_survivor"
                if not survivors
                else "branch_level_gate_failed"
            )
        ),
    }


__all__ = [
    "BoundedCurveMatrixOperator",
    "ExplicitDistillationError",
    "compose_f1_target",
    "evaluate_bank",
    "fit_operator",
    "operator_diagnostics",
    "render_bank",
    "synthetic_grid",
]
