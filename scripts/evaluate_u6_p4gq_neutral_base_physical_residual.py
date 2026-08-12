#!/usr/bin/env python3
"""Run the P4GQ neutral-base physical-residual chart."""

from __future__ import annotations

import argparse
import _ctypes
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from src.film_physics.bounded_linear_residual import apply_bounded_linear_residual
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile as scatter_profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime


P4FB = ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visual(path: Path, source: np.ndarray, physics: np.ndarray, residual: np.ndarray) -> None:
    panels = []
    for label, value in (
        ("source neutral base", linear_srgb_to_encoded(source)),
        ("bounded physical residual", physics),
        ("physical residual + AO6", residual),
    ):
        image = Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 300, 30), fill=(0, 0, 0))
        draw.text((8, 8), label, fill=(255, 255, 255))
        panels.append(image)
    result = Image.new("RGB", (source.shape[1], source.shape[0] * 3))
    for index, image in enumerate(panels):
        result.paste(image, (0, index * source.shape[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)


def evaluate(contract_path: Path, visual: Path | None = None) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent_path = ROOT / contract["parent"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        _sha(parent_path) != contract["parent"]["sha256"]
        or parent["decision"] != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4GQ parent drift")
    fixture = contract["fixture"]
    height, width = fixture["height"], fixture["width"]
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
    physics_outputs = []
    receipts = []
    diagnostics = []
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
            (ROOT / "src/film_physics/bounded_linear_residual.py").read_bytes()
            + (ROOT / "native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()
        ).hexdigest()
        try:
            def provider(forward_rows, y0, count):
                cloud_scan = render_physical_partition(
                    library,
                    physical_contract,
                    forward_rows,
                    y0,
                    count,
                    source_derived_expected=True,
                    enforce_density_envelope=True,
                    exact_sensitometry_endpoints=True,
                    cloud_profile=cloud_profile,
                )
                baseline_outputs = []
                render_physical_partition(
                    library,
                    physical_contract,
                    source,
                    y0,
                    count,
                    source_derived_expected=True,
                    enforce_density_envelope=True,
                    exact_sensitometry_endpoints=True,
                    cloud_profile=cloud_profile,
                    physical_baseline_outputs=baseline_outputs,
                )
                result, row_diagnostics = apply_bounded_linear_residual(
                    source[y0 : y0 + count], cloud_scan, baseline_outputs[0]
                )
                diagnostics.append({"start": y0, "count": count, **row_diagnostics})
                return result

            for tile_rows in fixture["tile_rows"]:
                cloud = WindowedNativeCloudScanRuntime(
                    gaussian_library=dll,
                    forward_scatter_profile=scatter_profile(),
                    physical_rows=provider,
                    physical_component_sha256=component,
                    tile_rows=tile_rows,
                )
                scans = []
                cloud.render_to_sink(source, output_sink=lambda _a, _b, value: scans.append(value.copy()))
                physics_outputs.append(
                    np.ascontiguousarray(
                        linear_srgb_to_encoded(np.concatenate(scans).astype(np.float64)), np.float32
                    )
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
                        )
                    )
                    outputs.append(np.concatenate(parts))
        finally:
            _ctypes.FreeLibrary(library._handle)
    exact = (
        all(np.array_equal(outputs[0], item) for item in outputs[1:])
        and np.array_equal(physics_outputs[0], physics_outputs[1])
        and len({receipt["output_sha256"] for receipt in receipts}) == 1
    )
    metric_contract = {"fixture": fixture}
    physics_metrics = _metrics(physics_outputs[0], patches, metric_contract)
    residual_metrics = _metrics(outputs[0], patches, metric_contract)
    gates = contract["gates"]
    checks = {
        "physical_residual": min(row["bounded_residual_rms"] for row in diagnostics)
        >= gates["minimum_physical_residual_rms"],
        "limited_fraction": max(row["limited_fraction"] for row in diagnostics)
        <= gates["maximum_limited_fraction"],
        "neutral_chroma": residual_metrics["neutral_chroma_p99"] <= gates["maximum_neutral_chroma_p99"],
        "patch_chroma": residual_metrics["maximum_neutral_patch_mean_chroma"] <= gates["maximum_neutral_patch_mean_chroma"],
        "overshoot": residual_metrics["edge_overshoot"] <= gates["maximum_edge_overshoot"],
        "undershoot": residual_metrics["edge_undershoot"] <= gates["maximum_edge_undershoot"],
        "boundary": residual_metrics["new_boundary_fraction"] <= gates["maximum_new_boundary_fraction"],
        "partition_repeat_exact": exact,
        "no_hard_clipping": all(row["hard_clipping_used"] == 0.0 for row in diagnostics),
    }
    if visual is not None and all(checks.values()):
        _visual(visual, source, physics_outputs[0], outputs[0])
    stable = {
        "contract_sha256": _sha(contract_path),
        "source_sha256": source_sha,
        "compiled_profile_identity": cloud_profile.identity(),
        "physics_output_sha256": hashlib.sha256(memoryview(physics_outputs[0]).cast("B")).hexdigest(),
        "residual_output_sha256": receipts[0]["output_sha256"],
        "physics": physics_metrics,
        "residual": residual_metrics,
        "physical_residual_rms_minimum": min(row["bounded_residual_rms"] for row in diagnostics),
        "physical_residual_limited_fraction_maximum": max(row["limited_fraction"] for row in diagnostics),
        "ao6_residual_linear_rms": receipts[0]["residual_linear_rms"],
        "gates": checks,
        "decision": contract["decision_if_pass"] if all(checks.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("-contract", "-result"),
        "automatic_pass": all(checks.values()),
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
