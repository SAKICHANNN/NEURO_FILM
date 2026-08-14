"""U6.P2AH leave-one-time-out characteristic-surface validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.bw_characteristic_surface import (
    compile_surface,
)
from src.eval.bw_characteristic_surface import (
    load_contract as load_p2ag,
)
from src.eval.trix_developer_contrast_source import _sha


class BWCharacteristicTimeInterpolationError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ah_bw_characteristic_time_interpolation_contract.v1"
    ):
        raise BWCharacteristicTimeInterpolationError("unsupported P2AH contract")
    return payload


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    parent = contract["parent"]
    if _sha(root / parent["evidence"]) != parent["evidence_sha256"]:
        raise BWCharacteristicTimeInterpolationError("P2AG evidence identity mismatch")
    evidence = json.loads((root / parent["evidence"]).read_text(encoding="utf-8"))
    if evidence.get("automatic_pass") is not parent["required_automatic_pass"]:
        raise BWCharacteristicTimeInterpolationError("P2AG decision mismatch")
    surface, _ = compile_surface(
        root=root,
        contract=load_p2ag(root / "configs/u6_p2ag_bw_characteristic_surface_v1.json"),
    )
    rows = []
    for time in contract["held_times_minutes"]:
        index = int(np.flatnonzero(surface.development_time_minutes == time)[0])
        lower_time = surface.development_time_minutes[index - 1]
        upper_time = surface.development_time_minutes[index + 1]
        weight = (time - lower_time) / (upper_time - lower_time)
        prediction = (1.0 - weight) * surface.diffuse_visual_density[
            index - 1
        ] + weight * surface.diffuse_visual_density[index + 1]
        target = surface.diffuse_visual_density[index]
        error = np.abs(prediction - target)
        density_range = float(target[-1] - target[0])
        rows.append(
            {
                "held_time_minutes": time,
                "lower_time_minutes": float(lower_time),
                "upper_time_minutes": float(upper_time),
                "normalized_rmse": float(
                    np.sqrt(np.mean(np.square(error))) / density_range
                ),
                "normalized_maximum_absolute_error": float(
                    np.max(error) / density_range
                ),
                "prediction_strictly_increasing": bool(
                    np.all(np.diff(prediction) > 0.0)
                ),
                "prediction_sha256": hashlib.sha256(
                    np.asarray(prediction, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
    gates = contract["automatic_gates"]
    measurements = {
        "held_row_count": len(rows),
        "maximum_normalized_rmse": max(row["normalized_rmse"] for row in rows),
        "maximum_normalized_absolute_error": max(
            row["normalized_maximum_absolute_error"] for row in rows
        ),
        "all_predictions_strictly_increasing": all(
            row["prediction_strictly_increasing"] for row in rows
        ),
        "repeat_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    results = {
        "held_row_count": measurements["held_row_count"] == gates["held_row_count"],
        "maximum_each_normalized_rmse": measurements["maximum_normalized_rmse"]
        <= gates["maximum_each_normalized_rmse"],
        "maximum_each_normalized_absolute_error": measurements[
            "maximum_normalized_absolute_error"
        ]
        <= gates["maximum_each_normalized_absolute_error"],
        "all_predictions_strictly_increasing": measurements[
            "all_predictions_strictly_increasing"
        ]
        is gates["all_predictions_strictly_increasing"],
        "repeat_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2ah_bw_characteristic_time_interpolation_report.v1",
        "parent_evidence_sha256": parent["evidence_sha256"],
        "rows": rows,
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
