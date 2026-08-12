"""Frozen U6.P4DJ typed sensitometry/cloud-chain evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.exposure_development import (
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationRoute,
)
from src.film_physics.sensitometry_cloud_chain import (
    render_typed_sensitometry_cloud_chain,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if (
        _sha(parent) != contract["parent"]["sha256"]
        or evidence["decision"] != contract["parent"]["required_decision"]
        or evidence["compiled_profile_identity"]
        != contract["parent"]["compiled_profile_identity"]
    ):
        raise RuntimeError("P4DJ parent drift")
    return contract


def _exposure(shape: tuple[int, int], pitch: float) -> PhysicalDomainArray:
    yy, xx = np.indices(shape, dtype=np.float64)
    values = np.stack(
        (
            0.02 + 5.0 * np.square(xx / shape[1]),
            0.03 + 3.5 * np.square(yy / shape[0]),
            0.015 + 4.0 * (xx + yy) / sum(shape),
        ),
        axis=-1,
    )
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(pitch),
    )


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    f = contract["fixture"]
    operator = build_operator(
        json.loads((root / "configs/u2_2a_sensitometry_primitive_v1.json").read_text())
    )
    capacity = evaluate_capacity(
        root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )
    profile = CrossLayerCloudReferenceProfile.from_payload(capacity["compiled_profile"])
    exposure = _exposure((f["height"], f["width"]), f["pixel_pitch_um"])
    outputs = []
    for route in contract["routes"]:
        interpretation = DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=EmulsionFamily(route["family"]),
            interpretation_route=InterpretationRoute(route["route"]),
        )
        full = render_typed_sensitometry_cloud_chain(
            exposure,
            operator,
            interpretation,
            profile,
            seed=f["seed"],
            row_tile_height=f["height"],
        )
        tiled = render_typed_sensitometry_cloud_chain(
            exposure,
            operator,
            interpretation,
            profile,
            seed=f["seed"],
            row_tile_height=f["row_partition_height"],
        )
        repeat = render_typed_sensitometry_cloud_chain(
            exposure,
            operator,
            interpretation,
            profile,
            seed=f["seed"],
            row_tile_height=f["row_partition_height"],
        )
        outputs.append(
            {
                "family": route["family"],
                "route": route["route"],
                "developed_density_sha256": hashlib.sha256(
                    full.mean_developed_density.values.astype("<f8").tobytes()
                ).hexdigest(),
                "cloud_density_sha256": hashlib.sha256(
                    full.cloud_density.values.astype("<f4").tobytes()
                ).hexdigest(),
                "transmittance_sha256": hashlib.sha256(
                    full.transmittance.values.astype("<f4").tobytes()
                ).hexdigest(),
                "developed_exact_u2_2": bool(
                    np.array_equal(
                        full.mean_developed_density.values,
                        operator.apply(exposure.values),
                    )
                ),
                "partition_exact": bool(
                    np.array_equal(
                        full.cloud_density.values, tiled.cloud_density.values
                    )
                    and np.array_equal(
                        full.transmittance.values, tiled.transmittance.values
                    )
                ),
                "repeat_exact": bool(
                    np.array_equal(
                        tiled.cloud_density.values, repeat.cloud_density.values
                    )
                    and np.array_equal(
                        tiled.transmittance.values, repeat.transmittance.values
                    )
                ),
                "finite_physical": bool(
                    np.all(np.isfinite(full.cloud_density.values))
                    and np.all(
                        (full.transmittance.values > 0.0)
                        & (full.transmittance.values <= 1.0)
                    )
                ),
            }
        )
    bw_rejected = False
    try:
        bw = DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=EmulsionFamily.BLACK_AND_WHITE,
            interpretation_route=InterpretationRoute.BW_DEVELOPER_SCAN,
        )
        render_typed_sensitometry_cloud_chain(
            exposure, operator, bw, profile, seed=f["seed"], row_tile_height=127
        )
    except ValueError:
        bw_rejected = True
    gates = {
        "developed_exact": all(x["developed_exact_u2_2"] for x in outputs),
        "route_identity": outputs[0]["cloud_density_sha256"]
        == outputs[1]["cloud_density_sha256"]
        and outputs[0]["transmittance_sha256"] == outputs[1]["transmittance_sha256"],
        "partition_identity": all(x["partition_exact"] for x in outputs),
        "repeat_identity": all(x["repeat_exact"] for x in outputs),
        "black_and_white_rejection": bw_rejected,
        "finite_physical": all(x["finite_physical"] for x in outputs),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "cloud_profile_identity": profile.identity(),
        "routes": outputs,
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if all(gates.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dj_typed_sensitometry_cloud_chain_report.v1",
        "automatic_pass": all(gates.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
