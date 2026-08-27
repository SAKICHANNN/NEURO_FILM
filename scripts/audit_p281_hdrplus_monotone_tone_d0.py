#!/usr/bin/env python3
"""Evaluate a tiny shared monotone tone representation on one HDR+ burst."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import load_working_image, working_image_to_srgb_float
from src.preprocess.dng_metadata import canonical_json_bytes

REPORT_SCHEMA = "neuro-film.p281-hdrplus-monotone-tone-d0-result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_pair(root: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    source = working_image_to_srgb_float(load_working_image(root / row["observation"]))
    with Image.open(root / row["target"]) as image:
        target = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    if source.shape != target.shape:
        raise ValueError(f"pair shape mismatch: {row['version']}")
    return source, target


def _fit_curves(
    source: np.ndarray, target: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    height, width, _ = source.shape
    yy, xx = np.indices((height, width))
    fit_mask = (
        ((yy + xx) % 2 == config["fit_parity"])
        & (yy % config["fit_stride"] == 0)
        & (xx % config["fit_stride"] == 0)
    )
    heldout_mask = (yy + xx) % 2 == config["heldout_parity"]
    knots = np.linspace(0.0, 1.0, config["knot_count"], dtype=np.float64)
    monotone = np.empty((3, config["knot_count"]), dtype=np.float64)
    control = np.empty_like(monotone)
    edges = (knots[:-1] + knots[1:]) * 0.5
    for channel in range(3):
        x = source[..., channel][fit_mask].astype(np.float64)
        y = target[..., channel][fit_mask].astype(np.float64)
        estimator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        estimator.fit(x, y)
        monotone[channel] = estimator.predict(knots)
        bins = np.digitize(x, edges)
        values = np.full(config["knot_count"], np.nan, dtype=np.float64)
        for index in range(config["knot_count"]):
            selected = y[bins == index]
            if selected.size:
                values[index] = np.median(selected)
        valid = np.flatnonzero(np.isfinite(values))
        if valid.size < 2:
            raise ValueError("insufficient unconstrained control support")
        control[channel] = np.interp(
            np.arange(config["knot_count"]), valid, values[valid]
        )
    return (
        monotone,
        control,
        {
            "fit_count": int(fit_mask.sum()),
            "heldout_count": int(heldout_mask.sum()),
            "fit_heldout_overlap": int(np.logical_and(fit_mask, heldout_mask).sum()),
        },
    )


def _apply(source: np.ndarray, curves: np.ndarray) -> np.ndarray:
    knots = np.linspace(0.0, 1.0, curves.shape[1], dtype=np.float64)
    output = np.empty_like(source, dtype=np.float32)
    for channel in range(3):
        output[..., channel] = np.interp(
            source[..., channel], knots, curves[channel]
        ).astype(np.float32)
    return output


def _metrics(
    source: np.ndarray,
    target: np.ndarray,
    candidate: np.ndarray,
    control: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    def rmse(value: np.ndarray) -> float:
        error = value[mask].astype(np.float64) - target[mask].astype(np.float64)
        return float(np.sqrt(np.mean(np.square(error))))

    identity_rmse = rmse(source)
    candidate_rmse = rmse(candidate)
    control_rmse = rmse(control)
    interior = (source > 0.0) & (source < 1.0)
    new_boundary = int(
        np.logical_and(
            interior, np.logical_or(candidate == 0.0, candidate == 1.0)
        ).sum()
    )
    return {
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "control_rmse_ratio": candidate_rmse / control_rmse,
        "identity_rmse": identity_rmse,
        "identity_rmse_reduction": (identity_rmse - candidate_rmse) / identity_rmse,
        "new_boundary_components": new_boundary,
    }


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_object(config_path)
    parent = ROOT / "docs/evidence/P280_HDRPLUS_RESULT_PAIR_GEOMETRY_RESULT.json"
    if _sha256_file(parent) != config["parent_p280_evidence_sha256"]:
        raise ValueError("P280 parent evidence SHA-256 mismatch")
    root = ROOT / config["source_root"]
    development_source, development_target = _load_pair(root, config["development"])
    monotone, control, split = _fit_curves(
        development_source, development_target, config
    )
    curves_sha = _sha256_bytes(
        monotone.astype("<f8").tobytes() + control.astype("<f8").tobytes()
    )
    dev_candidate = _apply(development_source, monotone)
    dev_control = _apply(development_source, control)
    yy, xx = np.indices(development_source.shape[:2])
    heldout = (yy + xx) % 2 == config["heldout_parity"]
    records = [
        {
            "role": "development_heldout",
            "version": config["development"]["version"],
            **_metrics(
                development_source,
                development_target,
                dev_candidate,
                dev_control,
                heldout,
            ),
        }
    ]

    confirmation_source, confirmation_target = _load_pair(root, config["confirmation"])
    confirmation_candidate = _apply(confirmation_source, monotone)
    confirmation_control = _apply(confirmation_source, control)
    records.append(
        {
            "role": "confirmation",
            "version": config["confirmation"]["version"],
            **_metrics(
                confirmation_source,
                confirmation_target,
                confirmation_candidate,
                confirmation_control,
                np.ones(confirmation_source.shape[:2], dtype=bool),
            ),
        }
    )
    if reverse:
        records.reverse()
    records.sort(key=lambda value: value["role"])
    gates = {
        "identity_reduction": all(
            row["identity_rmse_reduction"]
            >= config["gates"]["minimum_identity_rmse_reduction"]
            for row in records
        ),
        "control_noninferiority": all(
            row["control_rmse_ratio"] <= config["gates"]["maximum_control_rmse_ratio"]
            for row in records
        ),
        "curves_monotone": bool(np.all(np.diff(monotone, axis=1) >= 0.0)),
        "fit_heldout_overlap_zero": split["fit_heldout_overlap"] == 0,
        "finite_range": bool(
            np.isfinite(monotone).all()
            and np.min(monotone) >= 0.0
            and np.max(monotone) <= 1.0
        ),
        "new_boundary_zero": all(
            row["new_boundary_components"]
            <= config["gates"]["maximum_new_boundary_components"]
            for row in records
        ),
    }
    passed = all(gates.values())
    scientific = {
        "claim_ceiling": config["claim_ceiling"],
        "curves_sha256": curves_sha,
        "gates": gates,
        "monotone_knots": monotone.tolist(),
        "records": records,
        "split": split,
        "status": "PASS_PRIVATE_HDRPLUS_MONOTONE_TONE_D0"
        if passed
        else "FAIL_CLOSED_HDRPLUS_MONOTONE_TONE_D0",
    }
    scientific_bytes = canonical_json_bytes(scientific)
    return {
        "schema": REPORT_SCHEMA,
        "status": scientific["status"],
        "scientific": scientific,
        "execution": {"scientific_sha256": _sha256_bytes(scientific_bytes)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
