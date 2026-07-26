#!/usr/bin/env python
"""Run the frozen U5.R2N1 clean-room ModFlows B0 synthetic audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.modflows_b0_synthetic import (  # noqa: E402
    encode_modflows_images,
    load_modflows_b0_encoder,
    transfer_modflows,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _palette_image(
    palette: np.ndarray,
    *,
    size: int,
    layout: str,
    seed: int,
) -> np.ndarray:
    colors = np.asarray(palette, dtype=np.float32)
    if colors.shape != (8, 3) or size % 8:
        raise ValueError("palette controls require eight colors and divisible size")
    if layout == "vertical":
        labels = np.broadcast_to(
            np.arange(8, dtype=np.int64)[None, :].repeat(size // 8, axis=1),
            (size, size),
        )
    elif layout == "horizontal":
        labels = np.broadcast_to(
            np.arange(8, dtype=np.int64)[:, None].repeat(size // 8, axis=0),
            (size, size),
        )
    elif layout == "permutation":
        labels = np.repeat(np.arange(8, dtype=np.int64), size * size // 8)
        np.random.default_rng(seed).shuffle(labels)
        labels = labels.reshape(size, size)
    else:
        raise ValueError("unsupported palette layout")
    return colors[labels]


def _synthetic_images(size: int, seed: int) -> tuple[list[str], np.ndarray]:
    warm = np.asarray(
        [
            [0.04, 0.02, 0.015],
            [0.16, 0.055, 0.025],
            [0.35, 0.12, 0.045],
            [0.58, 0.24, 0.08],
            [0.78, 0.42, 0.16],
            [0.92, 0.64, 0.32],
            [0.98, 0.82, 0.58],
            [0.995, 0.95, 0.84],
        ],
        dtype=np.float32,
    )
    cool = np.asarray(
        [
            [0.01, 0.025, 0.05],
            [0.025, 0.08, 0.18],
            [0.04, 0.18, 0.34],
            [0.07, 0.32, 0.52],
            [0.14, 0.50, 0.68],
            [0.32, 0.68, 0.78],
            [0.58, 0.84, 0.88],
            [0.84, 0.96, 0.96],
        ],
        dtype=np.float32,
    )
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gray = np.repeat(ramp[None, :, None], size, axis=0)
    gray = np.repeat(gray, 3, axis=2)
    cube = np.asarray(
        [[r, g, b] for r in (0.0, 1.0) for g in (0.0, 1.0) for b in (0.0, 1.0)],
        dtype=np.float32,
    )
    names = [
        "gray_ramp",
        "rgb_cube_tiles",
        "warm_palette_geometry_a",
        "warm_palette_geometry_b",
        "warm_palette_pixel_permutation",
        "cool_palette_geometry_a",
        "cool_palette_geometry_b",
        "cool_palette_pixel_permutation",
    ]
    images = [
        gray,
        _palette_image(cube, size=size, layout="vertical", seed=seed),
        _palette_image(warm, size=size, layout="vertical", seed=seed),
        _palette_image(warm, size=size, layout="horizontal", seed=seed),
        _palette_image(warm, size=size, layout="permutation", seed=seed + 1),
        _palette_image(cool, size=size, layout="vertical", seed=seed),
        _palette_image(cool, size=size, layout="horizontal", seed=seed),
        _palette_image(cool, size=size, layout="permutation", seed=seed + 2),
    ]
    return names, np.stack(images, axis=0)


def _normalized(parameters: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(parameters, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise RuntimeError("embedding contains a zero or invalid norm")
    return parameters / norms


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _rgb_grid(size: int, *, margin: float = 0.0) -> np.ndarray:
    axis = np.linspace(margin, 1.0 - margin, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1,
        3,
    )


def _jacobian_metrics(
    points: np.ndarray,
    *,
    content: np.ndarray,
    style: np.ndarray,
    steps: int,
    epsilon: float,
) -> tuple[float, float]:
    columns = []
    for channel in range(3):
        plus = points.copy()
        minus = points.copy()
        plus[:, channel] += epsilon
        minus[:, channel] -= epsilon
        columns.append(
            (
                transfer_modflows(
                    plus,
                    content_parameters=content,
                    style_parameters=style,
                    steps_per_leg=steps,
                )
                - transfer_modflows(
                    minus,
                    content_parameters=content,
                    style_parameters=style,
                    steps_per_leg=steps,
                )
            )
            / (2.0 * epsilon)
        )
    jacobians = np.stack(columns, axis=2)
    determinants = np.linalg.det(jacobians)
    norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
    return float(np.min(determinants)), float(np.max(norms))


def run_audit(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    checkpoint_config = config["checkpoint"]
    checkpoint_path = ROOT / checkpoint_config["destination"]
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if (
        checkpoint_bytes != int(checkpoint_config["expected_bytes"])
        or checkpoint_sha256 != checkpoint_config["expected_sha256"]
    ):
        raise ValueError("checkpoint does not match frozen size/hash")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, architecture = load_modflows_b0_encoder(checkpoint_path)
    encoder_config = config["clean_room_encoder"]
    names, images = _synthetic_images(
        int(encoder_config["input_size"][0]),
        int(config["synthetic_images"]["seed"]),
    )
    if names != config["synthetic_images"]["families"]:
        raise RuntimeError("synthetic family order differs from frozen config")
    kwargs = {
        "normalization_mean": tuple(
            float(value) for value in encoder_config["normalization_mean"]
        ),
        "normalization_std": tuple(
            float(value) for value in encoder_config["normalization_std"]
        ),
    }
    first = encode_modflows_images(model, images, **kwargs)
    second = encode_modflows_images(model, images, **kwargs)
    repeat_error = float(np.max(np.abs(first - second)))
    embeddings = dict(zip(names, first, strict=True))
    normalized = dict(zip(names, _normalized(first), strict=True))

    geometry_cosines = [
        _cosine(
            normalized["warm_palette_geometry_a"],
            normalized["warm_palette_geometry_b"],
        ),
        _cosine(
            normalized["cool_palette_geometry_a"],
            normalized["cool_palette_geometry_b"],
        ),
    ]
    permutation_cosines = [
        _cosine(
            normalized["warm_palette_geometry_a"],
            normalized["warm_palette_pixel_permutation"],
        ),
        _cosine(
            normalized["cool_palette_geometry_a"],
            normalized["cool_palette_pixel_permutation"],
        ),
    ]
    geometry_distances = [
        float(
            np.linalg.norm(
                normalized["warm_palette_geometry_a"]
                - normalized["warm_palette_geometry_b"]
            )
        ),
        float(
            np.linalg.norm(
                normalized["cool_palette_geometry_a"]
                - normalized["cool_palette_geometry_b"]
            )
        ),
    ]
    palette_distances = [
        float(
            np.linalg.norm(
                normalized["warm_palette_geometry_a"]
                - normalized["cool_palette_geometry_a"]
            )
        ),
        float(
            np.linalg.norm(
                normalized["warm_palette_geometry_b"]
                - normalized["cool_palette_geometry_b"]
            )
        ),
    ]
    palette_geometry_ratio = float(
        np.mean(palette_distances) / max(np.mean(geometry_distances), 1e-12)
    )

    velocity_config = config["clean_room_velocity"]
    steps = int(velocity_config["steps_per_leg"])
    cube = _rgb_grid(int(config["synthetic_images"]["cube_grid_size"]))
    identity_errors = {}
    for name, parameters in embeddings.items():
        replay = transfer_modflows(
            cube,
            content_parameters=parameters,
            style_parameters=parameters,
            steps_per_leg=steps,
        )
        identity_errors[name] = float(np.max(np.abs(replay - cube)))

    transfer_pairs = (
        ("warm_to_cool", embeddings["warm_palette_geometry_a"], embeddings["cool_palette_geometry_a"]),
        ("cool_to_warm", embeddings["cool_palette_geometry_a"], embeddings["warm_palette_geometry_a"]),
    )
    transfer_reports = {}
    overall_minimum = float("inf")
    overall_maximum = float("-inf")
    minimum_determinant = float("inf")
    maximum_norm = 0.0
    transfer_repeat_error = 0.0
    jacobian_points = _rgb_grid(
        int(config["synthetic_images"]["jacobian_grid_size"]),
        margin=2.0 * float(config["synthetic_images"]["finite_difference_epsilon"]),
    )
    for pair_name, content, style in transfer_pairs:
        output = transfer_modflows(
            cube,
            content_parameters=content,
            style_parameters=style,
            steps_per_leg=steps,
        )
        repeated = transfer_modflows(
            cube,
            content_parameters=content,
            style_parameters=style,
            steps_per_leg=steps,
        )
        pair_repeat = float(np.max(np.abs(output - repeated)))
        pair_determinant, pair_norm = _jacobian_metrics(
            jacobian_points,
            content=content,
            style=style,
            steps=steps,
            epsilon=float(config["synthetic_images"]["finite_difference_epsilon"]),
        )
        pair_minimum = float(np.min(output))
        pair_maximum = float(np.max(output))
        transfer_reports[pair_name] = {
            "raw_minimum": pair_minimum,
            "raw_maximum": pair_maximum,
            "minimum_sampled_jacobian_determinant": pair_determinant,
            "maximum_sampled_jacobian_norm": pair_norm,
            "repeat_maximum_absolute_error": pair_repeat,
        }
        overall_minimum = min(overall_minimum, pair_minimum)
        overall_maximum = max(overall_maximum, pair_maximum)
        minimum_determinant = min(minimum_determinant, pair_determinant)
        maximum_norm = max(maximum_norm, pair_norm)
        transfer_repeat_error = max(transfer_repeat_error, pair_repeat)

    gates = config["gates"]
    checks = {
        "checkpoint_hash_and_size": True,
        "weights_only_and_architecture": architecture["classifier_weight_shape"]
        == (515, 1280)
        and architecture["classifier_bias_shape"] == (515,),
        "embedding_finite": bool(np.all(np.isfinite(first))),
        "embedding_repeat_exact": repeat_error
        <= gates["embedding_repeat_maximum_absolute_error"],
        "same_palette_geometry": min(geometry_cosines)
        >= gates["minimum_same_palette_geometry_cosine_similarity"],
        "same_palette_permutation": min(permutation_cosines)
        >= gates["minimum_same_palette_permutation_cosine_similarity"],
        "palette_over_geometry": palette_geometry_ratio
        >= gates["minimum_palette_distance_over_geometry_distance_ratio"],
        "same_embedding_identity": max(identity_errors.values())
        <= gates["maximum_same_embedding_identity_error"],
        "raw_range": overall_minimum >= gates["raw_transfer_minimum"]
        and overall_maximum <= gates["raw_transfer_maximum"],
        "positive_orientation": minimum_determinant
        > gates["minimum_sampled_jacobian_determinant"],
        "jacobian_norm": maximum_norm
        <= gates["maximum_sampled_jacobian_norm"],
        "transfer_repeat_exact": transfer_repeat_error
        <= gates["transfer_repeat_maximum_absolute_error"],
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "checkpoint": {
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
        },
        "architecture": architecture,
        "embedding": {
            "output_dimension": int(first.shape[1]),
            "maximum_repeat_error": repeat_error,
            "minimum_same_palette_geometry_cosine_similarity": min(
                geometry_cosines
            ),
            "minimum_same_palette_permutation_cosine_similarity": min(
                permutation_cosines
            ),
            "mean_same_palette_geometry_distance": float(
                np.mean(geometry_distances)
            ),
            "mean_different_palette_distance": float(np.mean(palette_distances)),
            "palette_distance_over_geometry_distance_ratio": palette_geometry_ratio,
            "parameter_minimum": float(np.min(first)),
            "parameter_maximum": float(np.max(first)),
            "parameter_rms": float(
                np.sqrt(np.mean(first * first, dtype=np.float64))
            ),
        },
        "maximum_same_embedding_identity_error": max(identity_errors.values()),
        "identity_errors": identity_errors,
        "transfer_reports": transfer_reports,
        "raw_transfer_minimum": overall_minimum,
        "raw_transfer_maximum": overall_maximum,
        "minimum_sampled_jacobian_determinant": minimum_determinant,
        "maximum_sampled_jacobian_norm": maximum_norm,
        "transfer_repeat_maximum_absolute_error": transfer_repeat_error,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2n1_modflows_b0_synthetic_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2n1_modflows_b0_synthetic_audit_v1/report.json",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    report = run_audit(json.loads(config_bytes), _sha256(config_bytes))
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(encoded),
                "all_checks_passed": report["all_checks_passed"],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
