#!/usr/bin/env python3
"""Evaluate the frozen P4GI exact-profile neutral gauge compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.native_cloud_spatial_partition import _build, _configure, render_physical_partition
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from src.film_physics.profile_bound_neutral_gauge import (
    apply_profile_bound_neutral_gauge,
    compile_profile_bound_neutral_gauge,
)
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latin_source(levels: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if height != levels.size or width != levels.size:
        raise ValueError("P4GI Latin layout requires square level geometry")
    y, x = np.indices((height, width), dtype=np.int64)
    labels = (x + 17 * y) % levels.size
    source = np.repeat(levels[labels, None], 3, axis=2)
    return np.ascontiguousarray(source, dtype=np.float32), labels


def _group_medians(values: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    return np.stack(
        [np.median(values[labels == index], axis=0) for index in range(count)],
        axis=0,
    ).astype(np.float64)


def _render(
    source: np.ndarray,
    *,
    tile_rows: int,
    library: Any,
    physical_contract: dict[str, Any],
    gaussian_library: Path,
    component_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()

    def provider(forward_rows, y0: int, count: int) -> np.ndarray:
        return render_physical_partition(
            library, physical_contract, forward_rows, y0, count
        )

    runtime = WindowedNativeCloudScanRuntime(
        gaussian_library=gaussian_library,
        forward_scatter_profile=_profile(),
        physical_rows=provider,
        physical_component_sha256=component_sha256,
        tile_rows=tile_rows,
    )
    parts: list[np.ndarray] = []
    receipt = runtime.render_to_sink(
        source, output_sink=lambda _y0, _y1, rows: parts.append(rows.copy())
    )
    if receipt["input_sha256"] != source_sha:
        raise RuntimeError("P4GI source identity drift")
    return np.concatenate(parts, axis=0), receipt


def evaluate(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent_path = ROOT / contract["parent"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    profile_spec = contract["physical_profile"]
    profile_path = ROOT / profile_spec["contract_path"]
    component_path = ROOT / profile_spec["component_path"]
    if (
        _sha(parent_path) != contract["parent"]["sha256"]
        or parent["stable"]["decision"] != contract["parent"]["required_decision"]
        or _sha(profile_path) != profile_spec["contract_sha256"]
        or _sha(component_path) != profile_spec["component_sha256"]
    ):
        raise RuntimeError("P4GI parent or profile identity drift")

    calibration = contract["calibration"]
    confirmation = contract["confirmation"]
    levels = np.linspace(0.0, 1.0, calibration["level_count"], dtype=np.float32)
    build_source, build_labels = _latin_source(
        levels, calibration["height"], calibration["width"]
    )
    midpoints = (
        (np.arange(confirmation["level_count"], dtype=np.float64) + 0.5)
        / confirmation["level_count"]
    ).astype(np.float32)
    confirm_source, confirm_labels = _latin_source(
        midpoints, confirmation["height"], confirmation["width"]
    )
    physical = json.loads(profile_path.read_text(encoding="utf-8"))
    physical["fixture"]["full_height"] = calibration["height"]
    physical["fixture"]["width"] = calibration["width"]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        build_root = Path(directory)
        gaussian_library = _build(ROOT, build_root / "native", None)
        library = _configure(gaussian_library)
        build_rows = [
            _render(
                build_source,
                tile_rows=int(tile_rows),
                library=library,
                physical_contract=physical,
                gaussian_library=gaussian_library,
                component_sha256=profile_spec["component_sha256"],
            )[0]
            for tile_rows in contract["execution"]["tile_rows"]
        ]
        partition_exact = bool(np.array_equal(build_rows[0], build_rows[1]))
        response = _group_medians(build_rows[0], build_labels, levels.size)
        compile_error = None
        payload = None
        compile_metrics = None
        try:
            payload, compile_metrics = compile_profile_bound_neutral_gauge(
                levels.astype(np.float64),
                response,
                base_component_sha256=profile_spec["component_sha256"],
            )
        except ValueError as error:
            compile_error = str(error)

        confirmation_metrics = None
        confirmation_exact = False
        if payload is not None:
            confirm_physical = json.loads(profile_path.read_text(encoding="utf-8"))
            confirm_physical["fixture"]["full_height"] = confirmation["height"]
            confirm_physical["fixture"]["width"] = confirmation["width"]
            confirm_rows = [
                _render(
                    confirm_source,
                    tile_rows=int(tile_rows),
                    library=library,
                    physical_contract=confirm_physical,
                    gaussian_library=gaussian_library,
                    component_sha256=profile_spec["component_sha256"],
                )[0]
                for tile_rows in contract["execution"]["tile_rows"]
            ]
            confirmation_exact = bool(np.array_equal(confirm_rows[0], confirm_rows[1]))
            gauged = apply_profile_bound_neutral_gauge(payload, confirm_rows[0])
            boundary = float(np.mean((gauged < 0.0) | (gauged > 1.0)))
            clipped = np.clip(gauged, 0.0, 1.0)
            encoded = linear_srgb_to_encoded(clipped)
            level_median = _group_medians(clipped, confirm_labels, midpoints.size)
            encoded_level_median = _group_medians(
                encoded, confirm_labels, midpoints.size
            )
            lab = rgb2lab(encoded)
            median_lab = rgb2lab(encoded_level_median[:, None, :])[:, 0]
            confirmation_metrics = {
                "maximum_level_absolute_error": float(
                    np.max(np.abs(level_median - midpoints[:, None]))
                ),
                "maximum_encoded_channel_spread": float(
                    np.max(np.ptp(encoded, axis=-1))
                ),
                "neutral_chroma_p99": float(
                    np.quantile(np.linalg.norm(lab[..., 1:3], axis=-1), 0.99)
                ),
                "minimum_lstar_step": float(np.min(np.diff(median_lab[:, 0]))),
                "new_boundary_fraction": boundary,
                "output_sha256": hashlib.sha256(
                    memoryview(np.ascontiguousarray(encoded, dtype=np.float32)).cast("B")
                ).hexdigest(),
            }

    gates = contract["gates"]
    channels = [] if compile_metrics is None else compile_metrics["channels"]
    checks = {
        "build_response_strict": bool(channels)
        and all(bool(row["strictly_increasing"]) for row in channels),
        "response_span": bool(channels)
        and min(float(row["response_span"]) for row in channels)
        >= gates["minimum_channel_response_span"],
        "inverse_slope": bool(channels)
        and max(float(row["maximum_inverse_secant_slope"]) for row in channels)
        <= gates["maximum_inverse_secant_slope"],
        "confirmation_error": confirmation_metrics is not None
        and confirmation_metrics["maximum_level_absolute_error"]
        <= gates["maximum_confirmation_level_absolute_error"],
        "confirmation_channel_spread": confirmation_metrics is not None
        and confirmation_metrics["maximum_encoded_channel_spread"]
        <= gates["maximum_confirmation_encoded_channel_spread"],
        "confirmation_neutral_chroma": confirmation_metrics is not None
        and confirmation_metrics["neutral_chroma_p99"]
        <= gates["maximum_confirmation_neutral_chroma_p99"],
        "confirmation_lstar": confirmation_metrics is not None
        and confirmation_metrics["minimum_lstar_step"]
        >= gates["minimum_confirmation_lstar_step"],
        "boundary": confirmation_metrics is not None
        and confirmation_metrics["new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
        "partition_exact": partition_exact and confirmation_exact,
    }
    passed = all(checks.values())
    stable = {
        "contract_sha256": _sha(contract_path),
        "build_source_sha256": hashlib.sha256(memoryview(build_source).cast("B")).hexdigest(),
        "confirmation_source_sha256": hashlib.sha256(memoryview(confirm_source).cast("B")).hexdigest(),
        "build_response_rgb": response.tolist(),
        "compile_error": compile_error,
        "compiled_gauge": payload,
        "compile_metrics": compile_metrics,
        "confirmation_metrics": confirmation_metrics,
        "gates": checks,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("-contract", "-result"),
        "automatic_pass": passed,
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = evaluate(arguments.contract)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if arguments.output is None:
        print(raw)
    else:
        arguments.output.write_text(raw, encoding="utf-8")


if __name__ == "__main__":
    main()
