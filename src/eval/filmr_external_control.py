"""Pinned external Filmr control runner and automatic evaluator.

The external executable and its exported preset remain ignored artifacts.  This
module records only the reproducible invocation boundary and evaluates decoded
outputs with the project's existing severe-artifact-first colour gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.eval.global_frontier import (
    load_frozen_samples,
    new_hard_clipping_fraction,
    sha256_file,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)


class FilmrExternalControlError(ValueError):
    """Raised when the external package, preset, replay, or output drifts."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_deterministic_preset(
    exported: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    zero_grain_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Return the exact exported preset with only stochastic amplitudes zeroed."""

    preset = copy.deepcopy(dict(exported))
    for key, expected in expected_identity.items():
        if preset.get(key) != expected:
            raise FilmrExternalControlError(f"exported preset identity mismatch: {key}")
    grain = preset.get("grain_model")
    if not isinstance(grain, dict):
        raise FilmrExternalControlError("exported preset has no grain_model object")
    allowed = {"alpha", "sigma_read", "shadow_noise", "highlight_coarseness"}
    if set(zero_grain_fields) != allowed:
        raise FilmrExternalControlError("deterministic override field set drifted")
    for field in zero_grain_fields:
        value = grain.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise FilmrExternalControlError(f"invalid grain field: {field}")
        grain[field] = 0.0
    return preset


def verify_external_runtime(root: Path, config: Mapping[str, Any]) -> Path:
    runtime = config["runtime"]
    executable = root / str(runtime["executable_path"])
    if not executable.is_file():
        raise FilmrExternalControlError("pinned Filmr executable is absent")
    if sha256_file(executable) != runtime["executable_sha256"]:
        raise FilmrExternalControlError("pinned Filmr executable hash mismatch")
    completed = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=float(runtime["per_call_timeout_seconds"]),
    )
    if completed.stdout.strip() != runtime["expected_version_stdout"]:
        raise FilmrExternalControlError("pinned Filmr version output mismatch")
    return executable


def export_deterministic_preset(
    *,
    root: Path,
    config: Mapping[str, Any],
    executable: Path,
    source_path: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_path = output_dir / "exported_preset.json"
    probe_path = output_dir / "export_probe.png"
    command = [
        str(executable),
        "--input",
        str(source_path),
        "--output",
        str(probe_path),
        "--preset",
        str(config["fixed_pipeline"]["preset_id"]),
        "--export-preset",
        str(exported_path),
    ]
    subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=float(config["runtime"]["per_call_timeout_seconds"]),
    )
    if sha256_file(exported_path) != config["fixed_pipeline"]["exported_preset_sha256"]:
        raise FilmrExternalControlError("exported Filmr preset hash mismatch")
    exported = json.loads(exported_path.read_text(encoding="utf-8"))
    deterministic = build_deterministic_preset(
        exported,
        expected_identity=config["fixed_pipeline"]["expected_preset_identity"],
        zero_grain_fields=tuple(config["fixed_pipeline"]["zero_grain_fields"]),
    )
    deterministic_path = output_dir / "deterministic_preset.json"
    deterministic_path.write_text(
        json.dumps(deterministic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return deterministic_path, {
        "exported_preset_sha256": sha256_file(exported_path),
        "deterministic_preset_sha256": sha256_file(deterministic_path),
        "deterministic_preset_canonical_sha256": canonical_sha256(deterministic),
        "overrides": {
            f"grain_model.{field}": 0.0
            for field in config["fixed_pipeline"]["zero_grain_fields"]
        },
    }


def render_pass(
    *,
    root: Path,
    config: Mapping[str, Any],
    executable: Path,
    preset_path: Path,
    pass_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    samples = load_frozen_samples(root, config["inputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for sample_id, sample in samples.items():
        source_path = root / str(sample["source_path"])
        output_path = output_dir / f"{sample_id}.png"
        command = [
            str(executable),
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--load-preset",
            str(preset_path),
            "--preset",
            "ignored",
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=float(config["runtime"]["per_call_timeout_seconds"]),
        )
        exposure_lines = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith("Exposure time:")
        ]
        if len(exposure_lines) != 1:
            raise FilmrExternalControlError(f"missing exposure receipt: {sample_id}")
        with Image.open(source_path) as source_image, Image.open(output_path) as output_image:
            source_size = ImageOps.exif_transpose(source_image).size
            output_rgb = ImageOps.exif_transpose(output_image)
            if output_rgb.mode != "RGB" or output_rgb.size != source_size:
                raise FilmrExternalControlError(
                    f"output decode/dimension mismatch: {sample_id}"
                )
        records.append(
            {
                "sample_id": sample_id,
                "split": sample["split"],
                "source_path": str(sample["source_path"]),
                "source_sha256": sample["source_sha256"],
                "output_path": output_path.relative_to(root).as_posix(),
                "output_sha256": sha256_file(output_path),
                "exposure_receipt": exposure_lines[0],
            }
        )
    return {
        "schema_version": "u5-r2av1-filmr-external-render-pass-v1",
        "pass_id": pass_id,
        "external_revision": config["external_source"]["revision"],
        "executable_sha256": config["runtime"]["executable_sha256"],
        "preset_sha256": sha256_file(preset_path),
        "frozen_set_sha256": config["inputs"]["frozen_set_sha256"],
        "records": records,
    }


def evaluate_passes(
    *,
    root: Path,
    config: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    samples = load_frozen_samples(root, config["inputs"])
    expected_ids = set(samples)
    rows_a = {str(row["sample_id"]): row for row in first["records"]}
    rows_b = {str(row["sample_id"]): row for row in second["records"]}
    if set(rows_a) != expected_ids or set(rows_b) != expected_ids:
        raise FilmrExternalControlError("render pass population mismatch")
    for manifest in (first, second):
        if (
            manifest.get("external_revision") != config["external_source"]["revision"]
            or manifest.get("executable_sha256")
            != config["runtime"]["executable_sha256"]
            or manifest.get("frozen_set_sha256")
            != config["inputs"]["frozen_set_sha256"]
        ):
            raise FilmrExternalControlError("render pass identity mismatch")
    exact_repeat = True
    exposure_repeat = True
    per_image: list[dict[str, Any]] = []
    budget = int(config["automatic_gates"]["metric_sample_pixels_per_image"])
    epsilon = float(config["automatic_gates"]["new_hard_clipping_epsilon"])
    for sample_id, sample in samples.items():
        first_row = rows_a[sample_id]
        second_row = rows_b[sample_id]
        exact_repeat &= first_row["output_sha256"] == second_row["output_sha256"]
        exposure_repeat &= (
            first_row["exposure_receipt"] == second_row["exposure_receipt"]
        )
        output_path = root / str(first_row["output_path"])
        if sha256_file(output_path) != first_row["output_sha256"]:
            raise FilmrExternalControlError(f"output hash mismatch: {sample_id}")
        source_pixels = sample_rgb_image(root / str(sample["source_path"]), budget)
        output_pixels = sample_rgb_image(output_path, budget)
        style, non_basic = style_and_basic_residual(source_pixels, output_pixels)
        per_image.append(
            {
                "sample_id": sample_id,
                "split": sample["split"],
                "style_delta_e76": style,
                "non_basic_residual_delta_e76": non_basic,
                "new_hard_clipping_fraction": new_hard_clipping_fraction(
                    source_pixels, output_pixels, epsilon
                ),
                "output_sha256": first_row["output_sha256"],
                "exposure_receipt": first_row["exposure_receipt"],
            }
        )
    gold = [row for row in per_image if row["split"] == "gold"]
    stress = [row for row in per_image if row["split"] == "stress"]
    summary = {
        "sample_count": len(per_image),
        "gold_sample_count": len(gold),
        "stress_sample_count": len(stress),
        "exact_output_repeat": bool(exact_repeat),
        "exact_exposure_receipt_repeat": bool(exposure_repeat),
        "gold_median_style_delta_e76": float(
            np.median([row["style_delta_e76"] for row in gold])
        ),
        "gold_median_non_basic_residual_delta_e76": float(
            np.median([row["non_basic_residual_delta_e76"] for row in gold])
        ),
        "worst_gold_new_hard_clipping_fraction": float(
            max(row["new_hard_clipping_fraction"] for row in gold)
        ),
        "worst_stress_new_hard_clipping_fraction": float(
            max(row["new_hard_clipping_fraction"] for row in stress)
        ),
        "distinct_exposure_receipts": sorted(
            {row["exposure_receipt"] for row in per_image}
        ),
    }
    gates = config["automatic_gates"]
    checks = {
        "exact_repeat": summary["exact_output_repeat"]
        and summary["exact_exposure_receipt_repeat"],
        "style": summary["gold_median_style_delta_e76"]
        >= float(gates["minimum_gold_median_style_delta_e76"]),
        "non_basic": summary["gold_median_non_basic_residual_delta_e76"]
        >= float(gates["minimum_gold_median_non_basic_residual_delta_e76"]),
        "gold_clipping": summary["worst_gold_new_hard_clipping_fraction"]
        <= float(gates["maximum_worst_gold_new_hard_clipping_fraction"]),
    }
    passed = all(checks.values())
    return {
        "summary": summary,
        "gate_checks": checks,
        "decision": (
            "automatic_pass_requires_blind_visual_review"
            if passed
            else "close_automatic_gate_failure"
        ),
        "visual_review_allowed": passed,
        "per_image": per_image,
    }
