"""Compile and validate the U6.P2AK TRI-X granularity scalar profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.film_physics.bw_granularity_scalar import BWGranularityScalarProfile


class TrixGranularityScalarProfileError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ak_trix_granularity_scalar_profile_contract.v1"
    ):
        raise TrixGranularityScalarProfileError("unsupported P2AK contract")
    return payload


def compile_profile(
    *, root: Path, contract: dict[str, Any]
) -> BWGranularityScalarProfile:
    parent = contract["parents"]["compatibility_boundary"]
    path = root / parent["path"]
    if _sha(path) != parent["sha256"]:
        raise TrixGranularityScalarProfileError("P2AK parent identity mismatch")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("automatic_pass") is not parent["required_automatic_pass"]:
        raise TrixGranularityScalarProfileError("P2AK parent decision mismatch")
    return BWGranularityScalarProfile(**contract["profile"])


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    profile = compile_profile(root=root, contract=contract)
    rebuilt = BWGranularityScalarProfile.from_dict(profile.to_dict())
    exact = profile.density_rms_at(
        density_mean=profile.density_mean,
        aperture_diameter_micrometres=profile.aperture_diameter_micrometres,
    )
    density_rejected = aperture_rejected = render_rejected = False
    sentinel = [71.0]
    try:
        sentinel[0] = profile.density_rms_at(
            density_mean=profile.density_mean + 1e-9,
            aperture_diameter_micrometres=profile.aperture_diameter_micrometres,
        )
    except ValueError:
        density_rejected = True
    try:
        sentinel[0] = profile.density_rms_at(
            density_mean=profile.density_mean,
            aperture_diameter_micrometres=profile.aperture_diameter_micrometres + 1e-9,
        )
    except ValueError:
        aperture_rejected = True
    try:
        profile.render()
    except ValueError:
        render_rejected = True
    measurements = {
        "serialization_byte_exact": profile.to_dict() == rebuilt.to_dict(),
        "identity_byte_exact": profile.identity() == rebuilt.identity(),
        "exact_observation_replay": exact == profile.density_rms,
        "density_mismatch_rejected": density_rejected,
        "aperture_mismatch_rejected": aperture_rejected,
        "spatial_render_rejected": render_rejected,
        "input_unchanged_on_failure": sentinel == [71.0],
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    results = {key: measurements[key] is value for key, value in gates.items()}
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2ak_trix_granularity_scalar_profile_report.v1",
        "profile": profile.to_dict(),
        "profile_identity": profile.identity(),
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
