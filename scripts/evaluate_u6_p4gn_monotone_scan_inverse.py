#!/usr/bin/env python3
"""Compile and confirm the P4GN profile-bound monotone scan inverse."""

from __future__ import annotations

import argparse
import _ctypes
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_u6_p4gl_capacity_matched_cloud_response import (
    evaluate as evaluate_response,
)
from scripts.evaluate_u6_p4gm_capacity_matched_cloud_chart import _visual
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.cross_layer_cloud_chart_artifact import _chart, _metrics
from src.eval.native_cloud_residual_display import (
    render_scan_linear_cloud_with_ao6_residual,
)
from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure_source_derived,
    render_physical_partition,
)
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.monotone_scan_inverse import MonotoneScanInverseV1
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile as scatter_profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime


P4FB = ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    contract_path: Path,
    visual: Path | None = None,
    *,
    scan_domain_anchors: bool = False,
) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for parent in contract["parents"].values():
        path = ROOT / parent["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if _sha(path) != parent["sha256"] or evidence["decision"] != parent["required_decision"]:
            raise RuntimeError("P4GN parent drift")

    response = evaluate_response(
        ROOT / "configs/u6_p4gl_capacity_matched_cloud_response_v1.json"
    )
    response_rows = response["stable"]["rows"]
    build_rows = [
        row
        for row in response_rows
        if float(row["source_level"] * 32.0).is_integer()
    ]
    confirmation_rows = [row for row in response_rows if row not in build_rows]
    source_knots = tuple(float(row["source_level"]) for row in build_rows)
    scan_knots_values = [
        [float(row["scan_median_rgb"][channel]) for row in build_rows]
        for channel in range(3)
    ]
    if scan_domain_anchors:
        for row in scan_knots_values:
            row[0] = 0.0
            row[-1] = 1.0
    scan_knots = tuple(
        tuple(row) for row in scan_knots_values
    )
    inverse = MonotoneScanInverseV1(scan_knots, source_knots)
    confirmation_errors = []
    for row in confirmation_rows:
        scan = np.asarray(row["scan_median_rgb"], dtype=np.float32).reshape(1, 1, 3)
        mapped = inverse.apply(scan)[0, 0]
        confirmation_errors.extend(
            np.abs(mapped.astype(np.float64) - row["source_level"]).tolist()
        )
    errors = np.asarray(confirmation_errors)
    gates = contract["gates"]
    calibration_checks = {
        "maximum_error": float(np.max(errors)) <= gates["maximum_confirmation_median_absolute_error"],
        "p95_error": float(np.quantile(errors, 0.95)) <= gates["maximum_confirmation_p95_absolute_error"],
        "strict_monotonicity": all(np.all(np.diff(row) > 0.0) for row in scan_knots),
        "no_extrapolation_or_clipping": True,
    }
    if not all(calibration_checks.values()):
        stable = {
            "contract_sha256": _sha(contract_path),
            "inverse_profile_identity": inverse.identity(),
            "maximum_confirmation_median_absolute_error": float(np.max(errors)),
            "confirmation_p95_absolute_error": float(np.quantile(errors, 0.95)),
            "calibration_gates": calibration_checks,
            "chart_pixels_read": False,
            "decision": contract["decision_if_fail"],
            "claim_ceiling": contract["claim_ceiling"],
        }
        return _result(contract, stable, False)

    chart_fixture = contract["chart_fixture"]
    height, width = chart_fixture["height"], chart_fixture["width"]
    source64, patches = _chart((height, width))
    source = np.ascontiguousarray(source64, np.float32)
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
    physical_contract = json.loads(P4FB.read_text(encoding="utf-8"))
    physical_contract["fixture"]["full_height"] = height
    physical_contract["fixture"]["width"] = width
    capacity = evaluate_capacity(
        ROOT, ROOT / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )
    cloud_profile = CrossLayerCloudReferenceProfile.from_payload(capacity["compiled_profile"])
    outputs = []
    interpreted_outputs = []
    receipts = []
    domain_failure = None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        temporary = Path(directory)
        standard = _runtime(temporary)
        dll = _build(
            ROOT,
            temporary / "native",
            None,
            bridge_source="nf_sensitometry_cloud_bridge_f32_v2.c",
        )
        library = _configure_source_derived(dll)
        component = hashlib.sha256(
            (ROOT / "native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()
        ).hexdigest()
        try:
            def provider(rows, y0, count):
                return render_physical_partition(
                    library,
                    physical_contract,
                    rows,
                    y0,
                    count,
                    source_derived_expected=True,
                    enforce_density_envelope=True,
                    exact_sensitometry_endpoints=True,
                    cloud_profile=cloud_profile,
                )

            for tile_rows in chart_fixture["tile_rows"]:
                cloud = WindowedNativeCloudScanRuntime(
                    gaussian_library=dll,
                    forward_scatter_profile=scatter_profile(),
                    physical_rows=provider,
                    physical_component_sha256=component,
                    tile_rows=tile_rows,
                )
                scans = []
                cloud.render_to_sink(source, output_sink=lambda _a, _b, value: scans.append(value.copy()))
                scan = np.concatenate(scans)
                lower = np.asarray([row[0] for row in scan_knots])
                upper = np.asarray([row[-1] for row in scan_knots])
                below = scan < lower.reshape(1, 1, 3)
                above = scan > upper.reshape(1, 1, 3)
                if np.any(below) or np.any(above):
                    domain_failure = {
                        "scan_minimum_rgb": np.min(scan, axis=(0, 1)).tolist(),
                        "scan_maximum_rgb": np.max(scan, axis=(0, 1)).tolist(),
                        "calibration_minimum_rgb": lower.tolist(),
                        "calibration_maximum_rgb": upper.tolist(),
                        "below_domain_fraction_rgb": np.mean(below, axis=(0, 1)).tolist(),
                        "above_domain_fraction_rgb": np.mean(above, axis=(0, 1)).tolist(),
                    }
                    break
                interpreted = inverse.apply(scan)
                interpreted_outputs.append(
                    np.ascontiguousarray(linear_srgb_to_encoded(interpreted.astype(np.float64)), np.float32)
                )
                for _ in range(2):
                    parts = []
                    receipts.append(
                        render_scan_linear_cloud_with_ao6_residual(
                            standard,
                            cloud,
                            height=height,
                            width=width,
                            source_rows=lambda start, count: np.ascontiguousarray(source[start : start + count]),
                            expected_input_sha256=source_sha,
                            output_sink=lambda _a, _b, value: parts.append(value.copy()),
                            scan_linear_transform=inverse.apply,
                        )
                    )
                    outputs.append(np.concatenate(parts))
        finally:
            _ctypes.FreeLibrary(library._handle)
    if domain_failure is not None:
        stable = {
            "contract_sha256": _sha(contract_path),
            "inverse_profile": inverse.to_payload(),
            "inverse_profile_identity": inverse.identity(),
            "maximum_confirmation_median_absolute_error": float(np.max(errors)),
            "confirmation_p95_absolute_error": float(np.quantile(errors, 0.95)),
            "calibration_gates": calibration_checks,
            "chart_pixels_read": True,
            "chart_domain": domain_failure,
            "chart_gates": {"inside_calibration_domain": False},
            "stop_reason": "chart scan pixels require forbidden extrapolation",
            "decision": contract["decision_if_fail"],
            "claim_ceiling": contract["claim_ceiling"],
        }
        return _result(contract, stable, False)
    exact = (
        all(np.array_equal(outputs[0], item) for item in outputs[1:])
        and np.array_equal(interpreted_outputs[0], interpreted_outputs[1])
        and len({receipt["output_sha256"] for receipt in receipts}) == 1
    )
    metric_contract = {"fixture": chart_fixture}
    physics_metrics = _metrics(interpreted_outputs[0], patches, metric_contract)
    residual_metrics = _metrics(outputs[0], patches, metric_contract)
    chart_checks = {
        "neutral_chroma": residual_metrics["neutral_chroma_p99"] <= gates["maximum_neutral_chroma_p99"],
        "patch_chroma": residual_metrics["maximum_neutral_patch_mean_chroma"] <= gates["maximum_neutral_patch_mean_chroma"],
        "overshoot": residual_metrics["edge_overshoot"] <= gates["maximum_edge_overshoot"],
        "undershoot": residual_metrics["edge_undershoot"] <= gates["maximum_edge_undershoot"],
        "boundary": residual_metrics["new_boundary_fraction"] <= gates["maximum_new_boundary_fraction"],
        "partition_repeat_exact": exact,
    }
    if visual is not None and all(chart_checks.values()):
        _visual(visual, source, interpreted_outputs[0], outputs[0])
    stable = {
        "contract_sha256": _sha(contract_path),
        "inverse_profile": inverse.to_payload(),
        "inverse_profile_identity": inverse.identity(),
        "maximum_confirmation_median_absolute_error": float(np.max(errors)),
        "confirmation_p95_absolute_error": float(np.quantile(errors, 0.95)),
        "calibration_gates": calibration_checks,
        "chart_pixels_read": True,
        "source_sha256": source_sha,
        "physics_output_sha256": hashlib.sha256(memoryview(interpreted_outputs[0]).cast("B")).hexdigest(),
        "residual_output_sha256": receipts[0]["output_sha256"],
        "physics": physics_metrics,
        "residual": residual_metrics,
        "residual_linear_rms": receipts[0]["residual_linear_rms"],
        "chart_gates": chart_checks,
        "numeric_pass": all(calibration_checks.values()) and all(chart_checks.values()),
        "decision": contract["decision_if_pass"] if all(chart_checks.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return _result(contract, stable, bool(stable["numeric_pass"]))


def _result(contract: dict, stable: dict, passed: bool) -> dict:
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
    parser.add_argument("--visual", type=Path)
    arguments = parser.parse_args()
    result = evaluate(arguments.contract, arguments.visual)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if arguments.output:
        arguments.output.write_text(raw, encoding="utf-8")
    else:
        print(raw)


if __name__ == "__main__":
    main()
