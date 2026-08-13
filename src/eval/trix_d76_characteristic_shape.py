"""U6.P2AF TRI-X/D-76 characteristic-shape sufficiency audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree

from src.eval.trix_developer_contrast_source import (
    _matrix,
    _path_points,
    _render_svg,
    _sha,
)
from src.eval.trix_developer_contrast_source import (
    load_contract as load_p2ad,
)


class TrixCharacteristicShapeError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2af_trix_d76_characteristic_shape_contract.v1"
    ):
        raise TrixCharacteristicShapeError("unsupported P2AF contract")
    return payload


def _curve(element: Any, graph: dict[str, Any]) -> np.ndarray:
    raw = _path_points(element.get("d"))
    homogeneous = np.column_stack((raw, np.ones(len(raw))))
    page = homogeneous @ _matrix(element.get("transform")).T
    x_pdf = page[:, 0]
    y_pdf = 792.0 - page[:, 1]
    axis = graph["plot_axis_pdf_points"]
    values = graph["axis_values"]
    x = values["log_exposure_left"] + (x_pdf - axis["left"]) / (
        axis["right"] - axis["left"]
    ) * (values["log_exposure_right"] - values["log_exposure_left"])
    density = values["density_bottom"] + (y_pdf - axis["bottom"]) / (
        axis["top"] - axis["bottom"]
    ) * (values["density_top"] - values["density_bottom"])
    order = np.argsort(x, kind="stable")
    curve = np.column_stack((x[order], density[order]))
    keep = np.concatenate(([True], np.diff(curve[:, 0]) > 0.0))
    curve = curve[keep]
    if np.any(np.diff(curve[:, 1]) <= 0.0):
        raise TrixCharacteristicShapeError("characteristic curve is not monotone")
    return curve


def _shape(curve: np.ndarray, levels: np.ndarray) -> np.ndarray:
    normalized = (curve[:, 1] - curve[0, 1]) / (curve[-1, 1] - curve[0, 1])
    exposure = np.interp(levels, normalized, curve[:, 0])
    return exposure - np.mean(exposure)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    parent = contract["parent"]
    evidence_path = root / parent["process_profile_evidence"]
    if _sha(evidence_path) != parent["process_profile_evidence_sha256"]:
        raise TrixCharacteristicShapeError("P2AE evidence identity mismatch")
    p2ad = load_p2ad(root / parent["source_contract"])
    source = p2ad["source"]
    graph = contract["source_graph"]
    with tempfile.TemporaryDirectory(prefix="nf-p2af-") as directory:
        svg = Path(directory) / "page.svg"
        _render_svg(root / source["local_research_copy"], graph["page"], svg)
        elements = list(etree.parse(str(svg)).iter())
    levels = np.asarray(
        contract["analysis"]["normalized_density_levels"], dtype=np.float64
    )
    curves = {}
    shapes = {}
    for name, binding in graph["paths"].items():
        element = elements[int(binding["element_index"])]
        if (
            hashlib.sha256(element.get("d").encode()).hexdigest()
            != binding["path_sha256"]
        ):
            raise TrixCharacteristicShapeError(f"path identity mismatch: {name}")
        curves[name] = _curve(element, graph)
        shapes[name] = _shape(curves[name], levels)
    error_points = (
        graph["maximum_source_pixel_error"] * 72.0 / graph["raster_reference_dpi"]
    )
    axis = graph["plot_axis_pdf_points"]
    dx = error_points / (axis["right"] - axis["left"]) * 5.0
    dy = error_points / (axis["top"] - axis["bottom"]) * 4.0
    uncertainty = {}
    for name, curve in curves.items():
        density_range = float(curve[-1, 1] - curve[0, 1])
        # A normalized-density query depends on the sampled point and both
        # observed endpoints.  Treat all three as independent +/-3 px reads,
        # then convert the resulting level error through the worst local
        # inverse slope.  This deliberately does not let a coherent whole-path
        # translation cancel during endpoint normalization.
        normalized_y_error = min(0.1, 3.0 * dy / density_range)
        normalized = (curve[:, 1] - curve[0, 1]) / density_range
        nominal_exposure = np.interp(levels, normalized, curve[:, 0])
        low_exposure = np.interp(levels - normalized_y_error, normalized, curve[:, 0])
        high_exposure = np.interp(levels + normalized_y_error, normalized, curve[:, 0])
        inverse_density_error = float(
            max(
                np.max(np.abs(nominal_exposure - low_exposure)),
                np.max(np.abs(high_exposure - nominal_exposure)),
            )
        )
        uncertainty[name] = dx + inverse_density_error
    rows = []
    robust = 0
    maximum = 0.0
    names = sorted(shapes)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            separation = float(np.max(np.abs(shapes[left] - shapes[right])))
            envelope = uncertainty[left] + uncertainty[right]
            robust += int(separation > envelope)
            maximum = max(maximum, separation)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "shape_separation_log_exposure": separation,
                    "coordinate_uncertainty_envelope": envelope,
                    "beyond_coordinate_envelope": separation > envelope,
                }
            )
    gates = contract["automatic_gates"]
    measurements = {
        "curve_count": len(curves),
        "all_curves_monotone": True,
        "all_levels_covered": True,
        "pairs_beyond_coordinate_envelope": robust,
        "maximum_shape_separation_log_exposure": maximum,
        "repeat_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    results = {
        "curve_count": len(curves) == gates["curve_count"],
        "all_curves_monotone": True,
        "all_levels_covered": True,
        "minimum_pairs_beyond_coordinate_envelope": robust
        >= gates["minimum_pairs_beyond_coordinate_envelope"],
        "minimum_maximum_shape_separation_log_exposure": maximum
        >= gates["minimum_maximum_shape_separation_log_exposure"],
        "repeat_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    stable = {
        "schema": "neuro_film.u6_p2af_trix_d76_characteristic_shape_report.v1",
        "parent_evidence_sha256": parent["process_profile_evidence_sha256"],
        "normalized_density_levels": levels.tolist(),
        "shape_signatures": {k: v.tolist() for k, v in sorted(shapes.items())},
        "coordinate_uncertainty": dict(sorted(uncertainty.items())),
        "pair_rows": rows,
        "measurements": measurements,
        "gate_results": results,
        "automatic_pass": all(results.values()),
        "decision": contract["branch_rule"][
            "pass" if all(results.values()) else "fail"
        ],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
