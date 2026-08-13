"""StatLUT-inspired spatially agnostic explicit-operator D0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro-film.u5-r2statlut0-statistical-lut-identifiability-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2statlut0-statistical-lut-identifiability-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported statistical-LUT D0 contract")
    return payload


def _scene(index: int, height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 104729 * index)
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    phases = rng.uniform(-np.pi, np.pi, size=6)
    weights = rng.uniform(0.15, 0.85, size=(3, 4))
    fields = np.stack(
        (
            x,
            y,
            0.5 + 0.5 * np.sin((2.0 + index % 3) * np.pi * x + phases[0]),
            0.5 + 0.5 * np.cos((2.0 + index % 4) * np.pi * y + phases[1]),
        ),
        axis=-1,
    )
    image = np.einsum("hwk,ck->hwc", fields, weights)
    blobs = np.zeros_like(image)
    for blob in range(5):
        cx, cy = rng.uniform(0.05, 0.95, size=2)
        sigma = rng.uniform(0.035, 0.18)
        colour = rng.uniform(-0.35, 0.35, size=3)
        mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
        blobs += mask[..., None] * colour
    image = image / np.maximum(np.max(image, axis=(0, 1), keepdims=True), 1e-9)
    return np.ascontiguousarray(np.clip(0.04 + 0.90 * image + blobs, 0.02, 0.98), dtype=np.float64)


def _parameters(effect: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * effect / 12.0
    gain = np.array(
        [1.0 + 0.22 * np.cos(angle), 1.0 + 0.18 * np.cos(angle + 2.1), 1.0 + 0.20 * np.cos(angle - 2.0)]
    )
    gamma = np.array(
        [0.82 + 0.28 * (0.5 + 0.5 * np.sin(angle)), 0.86 + 0.24 * (0.5 + 0.5 * np.sin(angle + 1.7)), 0.84 + 0.26 * (0.5 + 0.5 * np.sin(angle - 1.8))]
    )
    mix = np.eye(3) * 0.90
    mix += np.roll(np.eye(3), effect % 2 + 1, axis=1) * 0.10
    return gain, gamma, mix


def _apply(image: np.ndarray, effect: int) -> np.ndarray:
    gain, gamma, mix = _parameters(effect)
    powered = np.power(image, gamma[None, None, :]) * gain[None, None, :]
    mixed = powered @ mix.T
    # Intrinsic smooth compression, not clipping.
    return np.ascontiguousarray(mixed / (1.0 + mixed), dtype=np.float64)


def _descriptor(image: np.ndarray, bins: int) -> np.ndarray:
    parts = []
    for channel in range(3):
        hist, _ = np.histogram(image[..., channel], bins=bins, range=(0.0, 1.0))
        parts.append(hist.astype(np.float64) / image[..., channel].size)
    chroma = np.stack((image[..., 0] - image[..., 1], image[..., 2] - image[..., 1]), axis=-1)
    hist2, _, _ = np.histogram2d(
        chroma[..., 0].ravel(), chroma[..., 1].ravel(), bins=16, range=((-1.0, 1.0), (-1.0, 1.0))
    )
    parts.append(hist2.ravel() / image.shape[0] / image.shape[1])
    return np.concatenate(parts)


def _shuffle(image: np.ndarray, grid: int, seed: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h % grid or w % grid:
        raise ValueError("patch grid must divide image")
    patches = image.reshape(grid, h // grid, grid, w // grid, 3).transpose(0, 2, 1, 3, 4)
    flat = patches.reshape(grid * grid, h // grid, w // grid, 3)
    order = np.random.default_rng(seed).permutation(grid * grid)
    return flat[order].reshape(grid, grid, h // grid, w // grid, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, 3)


def evaluate(contract: Mapping[str, Any]) -> dict[str, Any]:
    p = contract["population"]
    scenes = [_scene(i, p["height"], p["width"], p["seed"]) for i in range(p["scene_count"])]
    dev = range(p["development_scene_count"])
    test = range(p["development_scene_count"], p["scene_count"])
    effects = range(p["effect_count"])
    prototypes = np.stack(
        [np.mean([_descriptor(_apply(scenes[s], e), p["histogram_bins"]) for s in dev], axis=0) for e in effects]
    )
    rows = []
    shuffle_error = 0.0
    improvements = []
    new_boundary = 0.0
    for s in test:
        source = scenes[s]
        for effect in effects:
            target = _apply(source, effect)
            descriptor = _descriptor(target, p["histogram_bins"])
            distances = np.mean((prototypes - descriptor[None, :]) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")
            selected = int(order[0])
            predicted = _apply(source, selected)
            identity_error = float(np.mean((source - target) ** 2))
            selected_error = float(np.mean((predicted - target) ** 2))
            improvement = 100.0 * (identity_error - selected_error) / max(identity_error, 1e-15)
            improvements.append(improvement)
            shuffled = _shuffle(target, p["patch_grid"], p["seed"] + 4099 * s + effect)
            shuffle_error = max(shuffle_error, float(np.max(np.abs(_descriptor(shuffled, p["histogram_bins"]) - descriptor))))
            eps = 1.0 / 65535.0
            source_edge = (source <= eps) | (source >= 1.0 - eps)
            output_edge = (predicted <= eps) | (predicted >= 1.0 - eps)
            new_boundary = max(new_boundary, float(np.mean(output_edge & ~source_edge)))
            rows.append({"scene": s, "effect": effect, "selected": selected, "top3": [int(x) for x in order[:3]], "improvement_percent": improvement})
    top1 = float(np.mean([r["selected"] == r["effect"] for r in rows]))
    top3 = float(np.mean([r["effect"] in r["top3"] for r in rows]))
    permuted = np.roll(prototypes, 5, axis=0)
    permuted_top1 = float(np.mean([int(np.argmin(np.mean((permuted - _descriptor(_apply(scenes[r["scene"]], r["effect"]), p["histogram_bins"])[None, :]) ** 2, axis=1))) == r["effect"] for r in rows]))
    metrics = {
        "row_count": len(rows), "top1_accuracy": top1, "top3_accuracy": top3,
        "cross_source_improvement_rate": float(np.mean(np.asarray(improvements) > 0.0)),
        "cross_source_median_improvement_percent": float(np.median(improvements)),
        "cross_source_worst_improvement_percent": float(np.min(improvements)),
        "patch_shuffle_descriptor_max_error": shuffle_error,
        "label_permutation_top1_accuracy": permuted_top1,
        "maximum_new_boundary_fraction": new_boundary,
    }
    g = contract["gates"]
    checks = {
        "top1": top1 >= g["minimum_top1_accuracy"], "top3": top3 >= g["minimum_top3_accuracy"],
        "improvement_rate": metrics["cross_source_improvement_rate"] >= g["minimum_cross_source_improvement_rate"],
        "median": metrics["cross_source_median_improvement_percent"] >= g["minimum_cross_source_median_improvement_percent"],
        "tail": metrics["cross_source_worst_improvement_percent"] >= g["minimum_cross_source_worst_improvement_percent"],
        "shuffle": shuffle_error <= g["maximum_patch_shuffle_descriptor_error"],
        "permutation": permuted_top1 <= g["maximum_label_permutation_top1_accuracy"],
        "boundary": new_boundary <= g["maximum_new_boundary_fraction"],
    }
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

