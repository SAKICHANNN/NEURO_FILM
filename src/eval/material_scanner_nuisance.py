"""Frozen U6.P4DL material-separated scanner nuisance evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_scanner_profile import _profile
from src.film_physics.scanner import apply_scanner_profile


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
            raise RuntimeError(f"P4DL {name} parent drift")
    return contract


def _materials(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    yy, xx = np.indices(shape, dtype=np.float64)
    x = xx / shape[1]
    y = yy / shape[0]
    colour_density = np.stack(
        (
            0.25 + 1.1 * x + 0.08 * np.sin(18.0 * y),
            0.32 + 0.9 * y + 0.06 * np.cos(16.0 * x),
            0.28 + 0.7 * (x + y) + 0.05 * np.sin(12.0 * (x + y)),
        ),
        axis=-1,
    )
    silver_density = 0.3 + 1.25 * (0.65 * x + 0.35 * y)
    return {
        "colour-dye-cloud": np.power(10.0, -colour_density),
        "bw-metallic-silver": np.repeat(
            np.power(10.0, -silver_density)[..., None], 3, axis=-1
        ),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    f = contract["fixture"]
    scanner_payload = json.loads(
        (root / contract["parents"]["scanner"]["path"]).read_text()
    )
    scanners = {
        name: _profile(row) for name, row in scanner_payload["profiles"].items()
    }
    materials = _materials((f["height"], f["width"]))
    rows: dict[str, Any] = {}
    identity_exact = True
    repeat_exact = True
    bounded = True
    scanner_separations = []
    for material, values in materials.items():
        outputs = {
            name: apply_scanner_profile(
                values, scanner, pixel_pitch_um=f["pixel_pitch_um"]
            )
            for name, scanner in scanners.items()
        }
        identity_exact &= bool(np.array_equal(outputs["identity"], values))
        repeat = apply_scanner_profile(
            values, scanners["scanner_a"], pixel_pitch_um=f["pixel_pitch_um"]
        )
        repeat_exact &= bool(np.array_equal(repeat, outputs["scanner_a"]))
        bounded &= all(
            np.all(np.isfinite(output))
            and np.all(output >= 0.0)
            and np.all(output <= 1.0)
            for output in outputs.values()
        )
        separation = float(np.max(np.abs(outputs["scanner_a"] - outputs["scanner_b"])))
        scanner_separations.append(separation)
        rows[material] = {
            "input_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
            "scanner_a_sha256": hashlib.sha256(
                outputs["scanner_a"].astype("<f8").tobytes()
            ).hexdigest(),
            "scanner_b_sha256": hashlib.sha256(
                outputs["scanner_b"].astype("<f8").tobytes()
            ).hexdigest(),
            "scanner_a_vs_b_max_abs": separation,
        }
    colour_a = apply_scanner_profile(
        materials["colour-dye-cloud"],
        scanners["scanner_a"],
        pixel_pitch_um=f["pixel_pitch_um"],
    )
    bw_a = apply_scanner_profile(
        materials["bw-metallic-silver"],
        scanners["scanner_a"],
        pixel_pitch_um=f["pixel_pitch_um"],
    )
    material_separation = float(np.max(np.abs(colour_a - bw_a)))
    identity_free = all(
        not hasattr(scanner, "stock_id") and not hasattr(scanner, "mode_id")
        for scanner in scanners.values()
    )
    gates = contract["gates"]
    results = {
        "identity": identity_exact,
        "scanner_separation": min(scanner_separations)
        >= gates["minimum_scanner_a_vs_b_max_abs"],
        "material_separation": material_separation
        >= gates["minimum_same_scanner_material_separation"],
        "identity_separation": identity_free,
        "repeat": repeat_exact,
        "domain": bounded,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "materials": rows,
        "same_scanner_material_separation_max_abs": material_separation,
        "scanner_profile_ids": [scanner.profile_id for scanner in scanners.values()],
        "gates": results,
        "decision": contract["decision_if_pass"]
        if all(results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dl_material_scanner_nuisance_report.v1",
        "automatic_pass": all(results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
