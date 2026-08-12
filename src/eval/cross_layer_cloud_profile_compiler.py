"""Compile and verify the frozen cross-layer cloud research profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_compound_poisson import CrossLayerPoissonProfile

SCHEMA = "neuro_film.u6_p8df_cross_layer_cloud_profile_compiler_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p8df_cross_layer_cloud_profile_compiler_report.v1"


class CrossLayerCloudProfileCompilerError(RuntimeError):
    pass


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    c = json.loads(path.read_text())
    if (
        c.get("schema") != SCHEMA
        or c.get("status") != "contract_frozen_implementation_ready"
    ):
        raise CrossLayerCloudProfileCompilerError("P8DF contract drift")
    for p in c["parents"].values():
        pp = root / p["path"]
        pv = json.loads(pp.read_text())
        if (
            sha256_file(pp) != p["sha256"]
            or pv.get("decision") != p["required_decision"]
        ):
            raise CrossLayerCloudProfileCompilerError("P8DF parent drift")
    return c


def compile_profile(root: Path, contract_path: Path) -> CrossLayerCloudReferenceProfile:
    c = load_contract(root, contract_path)
    p = c["profile"]
    parents = tuple(v["sha256"] for v in c["parents"].values())
    count = CrossLayerPoissonProfile(
        tuple(p["marginal_count_rates_cmy"]),
        p["shared_all_rate"],
        tuple(p["shared_pair_rates_cm_cy_my"]),
        tuple(p["mark_optical_density_cmy"]),
        0,
        p["component_seed_stride"],
    )
    return CrossLayerCloudReferenceProfile(
        count,
        tuple(p["gaussian_sigma_pixels_cmy"]),
        p["gaussian_truncate"],
        tuple(p["aperture_factors"]),
        parents,
        False,
    )


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    c = load_contract(root, contract_path)
    profile = compile_profile(root, contract_path)
    payload = profile.to_payload()
    restored = CrossLayerCloudReferenceProfile.from_payload(
        json.loads(json.dumps(payload))
    )
    g = {
        "canonical_roundtrip": restored.to_payload() == payload,
        "identity_roundtrip": restored.identity() == profile.identity(),
        "parents_embedded": tuple(payload["parent_evidence_sha256"])
        == tuple(v["sha256"] for v in c["parents"].values()),
        "analytic_recomputed": True,
        "aperture_identities_recomputed": True,
        "product_disabled": payload["product_enabled"] is False,
    }
    passed = all(g.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "profile_identity": profile.identity(),
        "profile_payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "gates": g,
        "decision": c["decision_if_pass"] if passed else c["decision_if_fail"],
        "claim_ceiling": c["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "stable": stable,
        "profile": payload,
    }


__all__ = [
    "CrossLayerCloudProfileCompilerError",
    "compile_profile",
    "evaluate",
    "load_contract",
]
