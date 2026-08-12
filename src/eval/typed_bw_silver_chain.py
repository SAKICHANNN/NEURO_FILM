"""Frozen U6.P4DK typed B&W metallic-silver chain evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_primitive import build_operator
from src.film_physics.bw_silver_chain import (
    build_typed_bw_silver_chain,
    render_bw_silver_chain_region,
)
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)
from src.film_physics.exposure_development import (
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationRoute,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    for name, parent in contract["parents"].items():
        p = root / parent["path"]
        payload = json.loads(p.read_text(encoding="utf-8"))
        expected = parent.get("required_decision", parent.get("required_schema"))
        actual = payload.get("decision", payload.get("schema"))
        if _sha(p) != parent["sha256"] or actual != expected:
            raise RuntimeError(f"P4DK {name} parent drift")
    return contract


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    f = contract["fixture"]
    operator = build_operator(
        json.loads((root / "configs/u2_2a_sensitometry_primitive_v1.json").read_text())
    )
    yy, xx = np.indices((f["input_height"], f["input_width"]), dtype=np.float64)
    neutral = 0.03 + 2.0 * (0.25 + 0.75 * xx / f["input_width"]) * (
        0.5 + 0.5 * yy / f["input_height"]
    )
    exposure = PhysicalDomainArray(
        np.repeat(neutral[..., None], 3, axis=-1),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(f["output_zoom"] * f["output_pixel_pitch_um"]),
    )
    interpretation = DevelopmentInterpretationContract.for_operator(
        operator,
        emulsion_family=EmulsionFamily.BLACK_AND_WHITE,
        interpretation_route=InterpretationRoute.BW_DEVELOPER_SCAN,
    )
    result = build_typed_bw_silver_chain(
        exposure,
        operator,
        interpretation,
        radius_um=f["radius_um"],
        output_zoom=f["output_zoom"],
        output_pixel_pitch_um=f["output_pixel_pitch_um"],
        monte_carlo_samples=f["monte_carlo_samples"],
        seed=f["seed"],
    )
    repeat = build_typed_bw_silver_chain(
        exposure,
        operator,
        interpretation,
        radius_um=f["radius_um"],
        output_zoom=f["output_zoom"],
        output_pixel_pitch_um=f["output_pixel_pitch_um"],
        monte_carlo_samples=f["monte_carlo_samples"],
        seed=f["seed"],
    )
    full = result.transmittance.values
    split = full.shape[0] // 2
    partitioned = np.concatenate(
        (
            render_bw_silver_chain_region(
                result, output_origin_yx=(0, 0), output_shape=(split, full.shape[1])
            ).values,
            render_bw_silver_chain_region(
                result,
                output_origin_yx=(split, 0),
                output_shape=(full.shape[0] - split, full.shape[1]),
            ).values,
        )
    )
    margin = f["output_zoom"]
    observed = float(np.mean(full[margin:-margin, margin:-margin, 0], dtype=np.float64))
    expected = float(
        np.mean(
            np.power(10.0, -result.mean_silver_density)[1:-1, 1:-1], dtype=np.float64
        )
    )
    colour_rejected = False
    try:
        colour = DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=EmulsionFamily.SLIDE,
            interpretation_route=InterpretationRoute.SLIDE_DIRECT_SCAN,
        )
        build_typed_bw_silver_chain(
            exposure,
            operator,
            colour,
            radius_um=f["radius_um"],
            output_zoom=f["output_zoom"],
            output_pixel_pitch_um=f["output_pixel_pitch_um"],
            monte_carlo_samples=f["monte_carlo_samples"],
            seed=f["seed"],
        )
    except ValueError:
        colour_rejected = True
    error = abs(observed - expected)
    variance = float(np.var(full[margin:-margin, margin:-margin, 0], dtype=np.float64))
    gates = contract["gates"]
    results = {
        "mean": error <= gates["maximum_interior_mean_transmittance_absolute_error"],
        "variance": variance >= gates["minimum_interior_variance"],
        "partition_identity": bool(np.array_equal(full, partitioned)),
        "repeat_identity": bool(np.array_equal(full, repeat.transmittance.values)),
        "material_domain": result.context.material == "bw-metallic-silver"
        and result.transmittance.domain is PhysicalDomain.TRANSMITTANCE,
        "colour_rejection": colour_rejected,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "context_fingerprint": result.context.fingerprint(),
        "transmittance_sha256": hashlib.sha256(
            full.astype("<f4").tobytes()
        ).hexdigest(),
        "grain_count": len(result.context.centers_by_layer[0]),
        "interior_mean_transmittance": observed,
        "target_mean_transmittance": expected,
        "mean_absolute_error": error,
        "interior_variance": variance,
        "gates": results,
        "decision": contract["decision_if_pass"]
        if all(results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dk_typed_bw_silver_chain_report.v1",
        "automatic_pass": all(results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
