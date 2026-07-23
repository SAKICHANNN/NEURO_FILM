#!/usr/bin/env python
"""Run the frozen U5.R2D representation and non-identifiability witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.lab_statistics import extract_lab_statistics  # noqa: E402
from src.roll2film.lut import DenseLUT3D  # noqa: E402


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity_lut() -> np.ndarray:
    axis = np.linspace(0.0, 1.0, 2)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def run_audit(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    gates = config["gates"]
    rng = np.random.default_rng(int(config["synthetic_seed"]))
    pixels = rng.random((257, 3))
    descriptor = extract_lab_statistics(pixels)
    permuted = extract_lab_statistics(pixels[rng.permutation(len(pixels))])
    permutation_error = float(np.max(np.abs(descriptor.vector() - permuted.vector())))

    red = np.tile(np.array([[0.85, 0.10, 0.08]]), (256, 1))
    blue = np.tile(np.array([[0.06, 0.12, 0.90]]), (256, 1))
    palette_distance = float(
        np.linalg.norm(
            extract_lab_statistics(red).vector()
            - extract_lab_statistics(blue).vector()
        )
    )

    identity_values = _identity_lut()
    changed_values = identity_values.copy()
    changed_values[0, 0, 1] = np.array([1.0, 0.0, 1.0])
    identity = DenseLUT3D(identity_values, np.zeros(3), np.ones(3), "tetrahedral")
    changed = DenseLUT3D(changed_values, np.zeros(3), np.ones(3), "tetrahedral")
    reference = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    probe = np.array([[[0.0, 0.0, 1.0]]])
    reference_identity = identity.apply(reference)
    reference_changed = changed.apply(reference)
    reference_pixel_error = float(
        np.max(np.abs(reference_identity - reference_changed))
    )
    reference_descriptor_error = float(
        np.max(
            np.abs(
                extract_lab_statistics(reference_identity).vector()
                - extract_lab_statistics(reference_changed).vector()
            )
        )
    )
    probe_difference = float(np.linalg.norm(identity.apply(probe) - changed.apply(probe)))
    lightness_mass_error = float(abs(descriptor.lightness_histogram.sum() - 1.0))
    chroma_mass_error = float(abs(descriptor.raw_chroma_histogram.sum() - 1.0))

    checks = {
        "descriptor_length": descriptor.vector().size == config["lab"]["descriptor_length"],
        "all_components_finite": bool(np.all(np.isfinite(descriptor.vector()))),
        "lightness_mass": lightness_mass_error <= gates["histogram_mass_absolute_error"],
        "chroma_mass": chroma_mass_error <= gates["histogram_mass_absolute_error"],
        "permutation_invariance": (
            permutation_error
            <= gates["permutation_component_maximum_absolute_error"]
        ),
        "palette_dependence_risk_witness": (
            palette_distance >= gates["palette_witness_minimum_descriptor_l2"]
        ),
        "reference_pixel_equivalence": (
            reference_pixel_error <= gates["reference_pixel_maximum_absolute_error"]
        ),
        "reference_descriptor_equivalence": (
            reference_descriptor_error
            <= gates["reference_descriptor_maximum_absolute_error"]
        ),
        "absent_colour_operator_divergence": (
            probe_difference >= gates["held_out_probe_minimum_rgb_l2_difference"]
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "representation_id": config["representation_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "synthetic_seed": config["synthetic_seed"],
        "metrics": {
            "descriptor_length": int(descriptor.vector().size),
            "lightness_mass_absolute_error": lightness_mass_error,
            "raw_chroma_mass_absolute_error": chroma_mass_error,
            "permutation_component_maximum_absolute_error": permutation_error,
            "palette_witness_descriptor_l2": palette_distance,
            "reference_pixel_maximum_absolute_error": reference_pixel_error,
            "reference_descriptor_maximum_absolute_error": reference_descriptor_error,
            "held_out_probe_rgb_l2_difference": probe_difference,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "interpretation": (
            "Permutation invariance coexists with strong palette dependence, and "
            "reference statistics do not identify an explicit global operator "
            "outside colours observed in that reference."
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2d_statistics_to_lut_shortcut_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2d_statistics_to_lut_shortcut_v1/audit.json",
    )
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    report = run_audit(config, _sha256(config_bytes))
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(encoded),
                "all_checks_passed": report["all_checks_passed"],
                "metrics": report["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
