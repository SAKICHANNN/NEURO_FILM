#!/usr/bin/env python
"""Run the frozen U6.4A creative optical-diffusion representation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx.creative_diffusion import (  # noqa: E402
    CREATIVE_DIFFUSION_VERSION,
    CreativeDiffusionProfile,
    apply_creative_diffusion_linear,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _profile(row: dict[str, object]) -> CreativeDiffusionProfile:
    return CreativeDiffusionProfile(
        profile_id=str(row["profile_id"]),
        sigmas_px=tuple(float(value) for value in row["sigmas_px"]),
        scatter_fractions=tuple(float(value) for value in row["scatter_fractions"]),
    )


def run(config_path: Path) -> dict[str, object]:
    raw = config_path.read_bytes()
    config = json.loads(raw)
    shape = (257, 257, 3)
    centre = shape[0] // 2
    mild = _profile(config["profiles"]["mild"])
    strong = _profile(config["profiles"]["strong"])

    constant = np.full(shape, np.float32(0.37), dtype=np.float32)
    constant_out = apply_creative_diffusion_linear(constant, strong)

    impulse = np.zeros(shape, dtype=np.float32)
    impulse[centre, centre, :] = np.float32(1.0)
    mild_impulse = apply_creative_diffusion_linear(impulse, mild)
    strong_impulse = apply_creative_diffusion_linear(impulse, strong)
    repeated = apply_creative_diffusion_linear(impulse, strong)

    neutral_ramp = np.linspace(0.0, 1.75, shape[1], dtype=np.float32)
    neutral = np.repeat(neutral_ramp[None, :, None], shape[0], axis=0)
    neutral = np.repeat(neutral, 3, axis=2)
    neutral_out = apply_creative_diffusion_linear(neutral, strong)

    rng = np.random.default_rng(6401)
    bounded = rng.uniform(0.03, 1.6, size=(79, 113, 3)).astype(np.float32)
    bounded_out = apply_creative_diffusion_linear(bounded, strong)

    yy, xx = np.ogrid[: shape[0], : shape[1]]
    radius = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2)
    far_mask = (radius >= 32.0) & (radius < 48.0)
    mild_far = float(mild_impulse[..., 0][far_mask].mean())
    strong_far = float(strong_impulse[..., 0][far_mask].mean())

    metrics = {
        "constant_max_error": float(np.max(np.abs(constant_out - constant))),
        "impulse_relative_energy_error": float(
            abs(float(strong_impulse[..., 0].sum()) - 1.0)
        ),
        "neutral_channel_max_error": float(
            max(
                np.max(np.abs(neutral_out[..., 0] - neutral_out[..., 1])),
                np.max(np.abs(neutral_out[..., 1] - neutral_out[..., 2])),
            )
        ),
        "bounded_output_min": float(bounded_out.min()),
        "bounded_output_max": float(bounded_out.max()),
        "bounded_source_min": float(bounded.min()),
        "bounded_source_max": float(bounded.max()),
        "strong_peak_reduction": float(
            1.0 - strong_impulse[centre, centre, 0]
        ),
        "mild_far_tail_mean": mild_far,
        "strong_far_tail_mean": strong_far,
        "strong_to_mild_far_tail_ratio": float(strong_far / max(mild_far, 1e-30)),
        "repeat_byte_exact": strong_impulse.tobytes() == repeated.tobytes(),
    }
    gates = config["gates"]
    checks = {
        "constant": metrics["constant_max_error"]
        <= float(gates["constant_max_error"]),
        "energy": metrics["impulse_relative_energy_error"]
        <= float(gates["impulse_relative_energy_error"]),
        "neutral": metrics["neutral_channel_max_error"]
        <= float(gates["neutral_channel_max_error"]),
        "component_bounds": (
            metrics["bounded_output_min"]
            >= metrics["bounded_source_min"]
            - float(gates["component_bound_tolerance"])
            and metrics["bounded_output_max"]
            <= metrics["bounded_source_max"]
            + float(gates["component_bound_tolerance"])
        ),
        "peak_reduction": metrics["strong_peak_reduction"]
        >= float(gates["minimum_strong_peak_reduction"]),
        "far_tail": metrics["strong_far_tail_mean"]
        >= float(gates["minimum_far_tail_value"]),
        "strength_order": metrics["strong_to_mild_far_tail_ratio"]
        >= float(gates["minimum_strong_to_mild_far_tail_ratio"]),
        "repeat": metrics["repeat_byte_exact"],
    }
    evidence = {
        "schema": "neuro_film.u6_4a_creative_diffusion_representation.report.v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_version": CREATIVE_DIFFUSION_VERSION,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }
    evidence["stable_evidence_id"] = hashlib.sha256(
        _canonical_bytes(evidence)
    ).hexdigest()
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_4a_creative_diffusion_representation_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))
    print(json.dumps({"passed": report["passed"], "stable_evidence_id": report["stable_evidence_id"]}))


if __name__ == "__main__":
    main()
