"""Fixed RGB8 quantization headroom for immutable explicit operators."""

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
from src.eval.constrained_explicit_distillation import (
    BoundedCurveMatrixOperator,
    synthetic_grid,
)
from src.eval.global_frontier import new_hard_clipping_fraction, sha256_file
from src.eval.projected_lut_frontier import lut_diagnostics
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)


HEADROOM_SCHEMA = "neuro_film.quantized_headroom_operator.v1"


class QuantizationHeadroomError(ValueError):
    """Raised when G1 lineage or operator evidence is invalid."""


@dataclass(frozen=True)
class QuantizedHeadroomOperator:
    base: BoundedCurveMatrixOperator
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.lower)
            or not np.isfinite(self.upper)
            or not 0.0 < self.lower < self.upper < 1.0
        ):
            raise QuantizationHeadroomError("invalid interior headroom")

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return self.lower + (self.upper - self.lower) * self.base.apply(rgb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HEADROOM_SCHEMA,
            "lower": self.lower,
            "upper": self.upper,
            "base": self.base.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuantizedHeadroomOperator":
        if payload.get("schema") != HEADROOM_SCHEMA:
            raise QuantizationHeadroomError("unsupported headroom schema")
        return cls(
            base=BoundedCurveMatrixOperator.from_dict(payload["base"]),
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
        )


def operator_diagnostics(
    operator: QuantizedHeadroomOperator,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    grid = synthetic_grid(17)
    baked = operator.apply(grid)
    replay = QuantizedHeadroomOperator.from_dict(operator.to_dict()).apply(grid)
    report = {
        **lut_diagnostics(baked),
        "replay_maximum_absolute_error": float(np.max(np.abs(replay - baked))),
    }
    gates = config["structure_gates"]
    tolerance = 1e-15
    report["structure_safe"] = bool(
        report["minimum_node"]
        >= float(gates["minimum_raw_output"]) - tolerance
        and report["maximum_node"]
        <= float(gates["maximum_raw_output"]) + tolerance
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
        ("source_g0_config", "source_g0_config_sha256"),
        ("source_g0_manifest", "source_g0_manifest_sha256"),
        ("source_g0_report", "source_g0_report_sha256"),
    ):
        if sha256_file(root / str(config[path_key])) != str(config[hash_key]):
            raise QuantizationHeadroomError(f"{path_key} hash mismatch")
    g0_config = json.loads(
        (root / str(config["source_g0_config"])).read_text(encoding="utf-8")
    )
    manifest_path = root / str(config["source_g0_manifest"])
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
        raise QuantizationHeadroomError("G0 operator bank drift")
    f1_config = json.loads(
        (root / str(g0_config["source_f1_config"])).read_text(encoding="utf-8")
    )
    input_manifest = json.loads(
        (root / str(f1_config["input_set"]["manifest"])).read_text(
            encoding="utf-8"
        )
    )
    samples = {
        str(row["id"]): row
        for row in input_manifest.get("frozen_set", input_manifest)["samples"]
    }
    for key, row in records.items():
        operator_path = manifest_path.parent / row["operator"]
        if sha256_file(operator_path) != row["operator_sha256"]:
            raise QuantizationHeadroomError(f"G0 operator hash mismatch: {key}")
        source = samples[key[1]]
        if sha256_file(root / source["source_path"]) != source["source_sha256"]:
            raise QuantizationHeadroomError(f"source hash mismatch: {key[1]}")
    return {
        "manifest_path": manifest_path,
        "records": records,
        "samples": samples,
    }


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = _validated(root, config)
    lower = float(config["headroom"]["lower"])
    upper = float(config["headroom"]["upper"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for reference_id in config["reference_ids"]:
        candidate_dir = output_dir / reference_id
        operator_dir = candidate_dir / "operators"
        operator_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in config["sample_ids"]:
            g0_row = validated["records"][(reference_id, sample_id)]
            g0_operator_path = (
                validated["manifest_path"].parent / g0_row["operator"]
            )
            base = BoundedCurveMatrixOperator.from_dict(
                json.loads(g0_operator_path.read_text(encoding="utf-8"))
            )
            operator = QuantizedHeadroomOperator(base, lower, upper)
            audit = operator_diagnostics(operator, config)
            operator_bytes = (
                json.dumps(operator.to_dict(), indent=2, sort_keys=True) + "\n"
            ).encode()
            operator_path = operator_dir / f"{sample_id}.json"
            operator_path.write_bytes(operator_bytes)
            sample = validated["samples"][sample_id]
            with Image.open(root / sample["source_path"]) as image:
                source = (
                    np.asarray(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        dtype=np.float64,
                    )
                    / 255.0
                )
            raw = operator.apply(source)
            pixels = np.rint(raw * 255.0).astype(np.uint8)
            output_path = candidate_dir / f"{sample_id}.png"
            Image.fromarray(pixels, mode="RGB").save(
                output_path, format="PNG", compress_level=6
            )
            records.append(
                {
                    "reference_id": reference_id,
                    "sample_id": sample_id,
                    "source_sha256": sample["source_sha256"],
                    "source_g0_operator_sha256": g0_row["operator_sha256"],
                    "operator": f"{reference_id}/operators/{sample_id}.json",
                    "operator_sha256": hashlib.sha256(operator_bytes).hexdigest(),
                    "structure": audit,
                    "raw_final_minimum": float(np.min(raw)),
                    "raw_final_maximum": float(np.max(raw)),
                    "raw_final_out_of_range_fraction": out_of_range_fraction(raw),
                    "minimum_rgb8_code": int(np.min(pixels)),
                    "maximum_rgb8_code": int(np.max(pixels)),
                    "output": f"{reference_id}/{sample_id}.png",
                    "output_sha256": sha256_file(output_path),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
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
        raise QuantizationHeadroomError("G1 manifest incomplete")
    g0 = json.loads(
        (root / str(config["source_g0_report"])).read_text(encoding="utf-8")
    )
    gates = config["automatic_gates"]
    budget = 250000
    epsilon = 1.0 / 255.0
    summaries: dict[str, Any] = {}
    sampled: dict[tuple[str, str], np.ndarray] = {}
    for reference_id in config["reference_ids"]:
        per_image = []
        for sample_id in config["sample_ids"]:
            row = records[(reference_id, sample_id)]
            output_path = manifest_path.parent / row["output"]
            if sha256_file(output_path) != row["output_sha256"]:
                raise QuantizationHeadroomError("output hash mismatch")
            if row["minimum_rgb8_code"] < 1 or row["maximum_rgb8_code"] > 254:
                raise QuantizationHeadroomError("RGB8 headroom contract failed")
            source = sample_rgb_image(
                root / validated["samples"][sample_id]["source_path"], budget
            )
            output = sample_rgb_image(output_path, budget)
            sampled[(reference_id, sample_id)] = output
            style, residual = style_and_basic_residual(source, output)
            per_image.append(
                {
                    "sample_id": sample_id,
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source, output, epsilon
                    ),
                }
            )
        summary = {
            "reference_id": reference_id,
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
                records[(reference_id, sample)][
                    "raw_final_out_of_range_fraction"
                ]
                for sample in config["sample_ids"]
            ),
            "style_delta_vs_g0": 0.0,
            "non_basic_delta_vs_g0": 0.0,
            "per_image": per_image,
        }
        g0_row = g0["candidates"][reference_id]
        summary["style_delta_vs_g0"] = float(
            summary["gold_median_style_delta_e76"]
            - g0_row["gold_median_style_delta_e76"]
        )
        summary["non_basic_delta_vs_g0"] = float(
            summary["gold_median_non_basic_residual_delta_e76"]
            - g0_row["gold_median_non_basic_residual_delta_e76"]
        )
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

    reference_ids = [str(value) for value in config["reference_ids"]]
    pairwise = []
    for sample_id in config["sample_ids"]:
        labs = {
            reference: rgb2lab(
                sampled[(reference, sample_id)].reshape(-1, 1, 3)
            ).reshape(-1, 3)
            for reference in reference_ids
        }
        for index, first in enumerate(reference_ids):
            for second in reference_ids[index + 1 :]:
                pairwise.append(
                    float(
                        np.median(
                            np.linalg.norm(labs[first] - labs[second], axis=1)
                        )
                    )
                )
    sensitivity = float(np.median(pairwise))
    sensitivity_pass = sensitivity >= float(
        gates["reference_sensitivity_median_pairwise_delta_e76_minimum"]
    )
    style_loss = float(
        -np.median(
            [min(0.0, summaries[ref]["style_delta_vs_g0"]) for ref in reference_ids]
        )
    )
    non_basic_loss = float(
        -np.median(
            [
                min(0.0, summaries[ref]["non_basic_delta_vs_g0"])
                for ref in reference_ids
            ]
        )
    )
    retention_pass = bool(
        style_loss <= float(gates["maximum_median_style_loss_vs_g0"])
        and non_basic_loss
        <= float(gates["maximum_median_non_basic_loss_vs_g0"])
    )
    survivors = sorted(
        ref for ref in reference_ids if summaries[ref]["automatic_survivor"]
    )
    ranked = sorted(
        survivors,
        key=lambda ref: (
            -summaries[ref]["gold_median_non_basic_residual_delta_e76"],
            -summaries[ref]["gold_median_style_delta_e76"],
            ref,
        ),
    )
    # Provenance buckets are fixed in groups of three by the frozen reference bank.
    bucket = {
        ref: ("commons_ektar_category" if int(ref[-2:]) <= 3 else
              "commons_ultramax_category" if int(ref[-2:]) <= 6 else
              "yfcc_velvia_category")
        for ref in reference_ids
    }
    shortlist = []
    seen = set()
    if sensitivity_pass and retention_pass:
        for ref in ranked:
            if bucket[ref] in seen:
                continue
            seen.add(bucket[ref])
            shortlist.append(ref)
            if len(shortlist) == 3:
                break
    return {
        "candidate_count": len(reference_ids),
        "sample_count": len(config["sample_ids"]),
        "candidates": summaries,
        "automatic_survivors": survivors,
        "reference_sensitivity_median_pairwise_output_delta_e76": sensitivity,
        "reference_sensitivity_gate_passed": sensitivity_pass,
        "median_style_loss_vs_g0": style_loss,
        "median_non_basic_loss_vs_g0": non_basic_loss,
        "retention_gate_passed": retention_pass,
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
    "QuantizationHeadroomError",
    "QuantizedHeadroomOperator",
    "evaluate_bank",
    "operator_diagnostics",
    "render_bank",
]
