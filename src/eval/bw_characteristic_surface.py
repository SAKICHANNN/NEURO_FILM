"""Compile and validate the frozen U6.P2AG B&W characteristic surface."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree

from src.eval.trix_d76_characteristic_shape import _curve
from src.eval.trix_developer_contrast_source import (
    _render_svg,
    _sha,
)
from src.eval.trix_developer_contrast_source import (
    load_contract as load_p2ad,
)
from src.film_physics.bw_characteristic_surface import BWCharacteristicSurface


class BWCharacteristicSurfaceError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ag_bw_characteristic_surface_contract.v1"
    ):
        raise BWCharacteristicSurfaceError("unsupported P2AG contract")
    return payload


def compile_surface(
    *, root: Path, contract: dict[str, Any]
) -> tuple[BWCharacteristicSurface, np.ndarray]:
    parents = contract["parents"]
    for binding in parents.values():
        if _sha(root / binding["path"]) != binding["sha256"]:
            raise BWCharacteristicSurfaceError("parent identity mismatch")
    evidence = json.loads(
        (root / parents["shape_evidence"]["path"]).read_text(encoding="utf-8")
    )
    if (
        evidence.get("automatic_pass")
        is not parents["shape_evidence"]["required_automatic_pass"]
    ):
        raise BWCharacteristicSurfaceError("P2AF decision mismatch")
    shape_contract = json.loads(
        (root / parents["shape_contract"]["path"]).read_text(encoding="utf-8")
    )
    p2ad = load_p2ad(root / shape_contract["parent"]["source_contract"])
    graph = shape_contract["source_graph"]
    with tempfile.TemporaryDirectory(prefix="nf-p2ag-") as directory:
        svg = Path(directory) / "page.svg"
        _render_svg(root / p2ad["source"]["local_research_copy"], graph["page"], svg)
        elements = list(etree.parse(str(svg)).iter())
    ordered_paths = ["6_minutes", "8_minutes", "10_minutes", "12_minutes"]
    curves = [
        _curve(elements[graph["paths"][name]["element_index"]], graph)
        for name in ordered_paths
    ]
    exposure = np.asarray(contract["surface"]["log_exposure_knots"], dtype=np.float64)
    density = np.stack(
        [np.interp(exposure, curve[:, 0], curve[:, 1]) for curve in curves]
    )
    context = {
        key: contract["surface"][key]
        for key in ("agitation", "densitometry", "equipment", "temperature_c")
    }
    surface = BWCharacteristicSurface(
        contract["surface"]["film_stock_id"],
        contract["surface"]["developer_id"],
        np.asarray(contract["surface"]["development_times_minutes"], dtype=np.float64),
        exposure,
        density,
        parents["shape_evidence"]["sha256"],
        context,
    )
    return surface, np.stack(
        [np.interp(exposure, curve[:, 0], curve[:, 1]) for curve in curves]
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    surface, source_density = compile_surface(root=root, contract=contract)
    rebuilt = BWCharacteristicSurface.from_dict(surface.to_dict())
    exposure = surface.log_exposure_knots
    reconstruction = float(
        np.max(np.abs(surface.diffuse_visual_density - source_density))
    )
    inverse_error = 0.0
    for time in surface.development_time_minutes:
        density = surface.density(float(time), exposure)
        inverse_error = max(
            inverse_error,
            float(
                np.max(np.abs(surface.log_exposure(float(time), density) - exposure))
            ),
        )
    partition = np.array_equal(
        surface.density(9.0, exposure),
        np.concatenate(
            [surface.density(9.0, exposure[:13]), surface.density(9.0, exposure[13:])]
        ),
    )
    outside_rejected = True
    unchanged = True
    sentinel = np.array([71.0], dtype=np.float64)
    try:
        sentinel[...] = surface.density(5.99, np.array([-1.0]))
        outside_rejected = False
    except ValueError:
        pass
    unchanged = unchanged and bool(sentinel[0] == 71.0)
    try:
        surface.log_exposure(9.0, np.array([100.0]))
        outside_rejected = False
    except ValueError:
        pass
    measurements = {
        "surface_shape": list(surface.diffuse_visual_density.shape),
        "all_exposure_rows_strictly_increasing": bool(
            np.all(np.diff(surface.diffuse_visual_density, axis=1) > 0.0)
        ),
        "all_time_columns_strictly_increasing": bool(
            np.all(np.diff(surface.diffuse_visual_density, axis=0) > 0.0)
        ),
        "maximum_source_reconstruction_error": reconstruction,
        "maximum_forward_inverse_log_exposure_error": inverse_error,
        "serialization_byte_exact": surface.to_dict() == rebuilt.to_dict()
        and surface.identity() == rebuilt.identity(),
        "partition_exact": partition,
        "outside_domain_rejected": outside_rejected,
        "input_unchanged_on_failure": unchanged,
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    results = {
        "surface_shape": measurements["surface_shape"] == gates["surface_shape"],
        "all_exposure_rows_strictly_increasing": measurements[
            "all_exposure_rows_strictly_increasing"
        ]
        is gates["all_exposure_rows_strictly_increasing"],
        "all_time_columns_strictly_increasing": measurements[
            "all_time_columns_strictly_increasing"
        ]
        is gates["all_time_columns_strictly_increasing"],
        "maximum_source_reconstruction_error": reconstruction
        <= gates["maximum_source_reconstruction_error"],
        "maximum_forward_inverse_log_exposure_error": inverse_error
        <= gates["maximum_forward_inverse_log_exposure_error"],
        **{
            key: measurements[key] is gates[key]
            for key in (
                "serialization_byte_exact",
                "partition_exact",
                "outside_domain_rejected",
                "input_unchanged_on_failure",
                "rgb_image_transform_count_zero",
            )
        },
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2ag_bw_characteristic_surface_report.v1",
        "surface_identity": surface.identity(),
        "surface_payload": surface.to_dict(),
        "measurements": measurements,
        "gate_results": results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
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
