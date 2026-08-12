"""P4HK compile, roundtrip and execution audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.neutral_base_photographic_ablation import sha256_file
from src.film_physics.bounded_histogram_copula import (
    apply_bounded_multipass_histogram_copula,
)
from src.film_physics.bounded_photographic_profile import (
    canonical_profile_bytes,
    compile_bounded_photographic_profile,
    load_bounded_photographic_profile,
    reconstruct_bounded_photographic_profile,
    validate_bounded_photographic_profile,
)
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.spatial_response import apply_scanner_mtf

SCHEMA = "neuro-film.u6-p4hk-bounded-photographic-profile-bundle-contract.v1"


def _load_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"P4HK parent drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(
    contract: dict[str, Any], *, root: Path, bundle_path: Path
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HK contract")
    parents = contract["parents"]
    loaded = {name: _load_json(root, binding) for name, binding in parents.items()}
    for name in ("p4hj_evidence", "p4hf_evidence"):
        if loaded[name].get("decision") != parents[name]["required_decision"]:
            raise ValueError(f"P4HK {name} decision drift")
    profile = DensityConditionedThomasProfile.from_dict(loaded["p4bw_profile"])
    prior = ManufacturerCharacteristicPrior.from_dict(loaded["p2q_prior"]["prior"])
    if (
        loaded["p4bw_profile"].get("profile_id") != profile.identity()
        or loaded["p2q_prior"].get("array_sha256") is None
    ):
        raise ValueError("P4HK profile or prior identity drift")
    p4he = loaded["p4he_contract"]
    candidate = p4he["candidate"]
    bindings = {name: binding["sha256"] for name, binding in parents.items()}
    compile_args = {
        "profile": profile,
        "prior": prior,
        "correlation_matrix": np.asarray(
            candidate["correlation_matrix"], dtype=np.float64
        ),
        "layer_field_seeds": tuple(
            int(value) for value in candidate["layer_field_seeds"]
        ),
        "field_seed_stride_per_source": int(
            candidate["field_seed_stride_per_source"]
        ),
        "canonical_row_block_height": int(
            contract["bundle"]["canonical_row_block_height"]
        ),
        "rank_bins": int(contract["bundle"]["rank_bins"]),
        "scanner_mtf_sigma_pixels_rgb": tuple(
            float(value) for value in candidate["scanner_mtf_sigma_pixels_rgb"]
        ),
        "gaussian_truncate": float(candidate["gaussian_truncate"]),
        "source_bindings": bindings,
    }
    first = compile_bounded_photographic_profile(**compile_args)
    second = compile_bounded_photographic_profile(**compile_args)
    encoded = canonical_profile_bytes(first)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(encoded)
    loaded_bundle = load_bounded_photographic_profile(
        bundle_path, expected_bundle_sha256=first["bundle_sha256"]
    )
    rebuilt = reconstruct_bounded_photographic_profile(loaded_bundle)
    fixture = contract["fixture"]
    rng = np.random.default_rng(int(fixture["source_seed"]))
    source = rng.uniform(
        0.001,
        0.999,
        size=(int(fixture["height"]), int(fixture["width"]), 3),
    ).astype(np.float32)
    source_index = int(fixture["source_index"])
    seeds = tuple(
        value + source_index * int(candidate["field_seed_stride_per_source"])
        for value in candidate["layer_field_seeds"]
    )
    direct, direct_diagnostics = apply_bounded_multipass_histogram_copula(
        source,
        profile=profile,
        prior=prior,
        layer_seeds=seeds,
        correlation_matrix=np.asarray(candidate["correlation_matrix"], dtype=np.float64),
        canonical_receipt_row_block_height=int(
            contract["bundle"]["canonical_row_block_height"]
        ),
        rank_bins=int(contract["bundle"]["rank_bins"]),
    )
    rebuilt_seeds = tuple(
        value + source_index * rebuilt.field_seed_stride_per_source
        for value in rebuilt.layer_field_seeds
    )
    reconstructed, reconstructed_diagnostics = apply_bounded_multipass_histogram_copula(
        source,
        profile=rebuilt.profile,
        prior=rebuilt.prior,
        layer_seeds=rebuilt_seeds,
        correlation_matrix=rebuilt.correlation_matrix,
        canonical_receipt_row_block_height=rebuilt.canonical_row_block_height,
        rank_bins=rebuilt.rank_bins,
    )
    direct_scanner = np.ascontiguousarray(
        apply_scanner_mtf(direct.astype(np.float64), rebuilt.scanner_profile),
        dtype=np.float32,
    )
    reconstructed_scanner = np.ascontiguousarray(
        apply_scanner_mtf(
            reconstructed.astype(np.float64), rebuilt.scanner_profile
        ),
        dtype=np.float32,
    )
    resource_key = "peak_live_temporary_bytes"
    direct_scientific = {
        key: value for key, value in direct_diagnostics.items() if key != resource_key
    }
    rebuilt_scientific = {
        key: value
        for key, value in reconstructed_diagnostics.items()
        if key != resource_key
    }
    tampered = json.loads(encoded)
    tampered["execution"]["rank_bins"] = 32768
    tampered_rejected = False
    try:
        validate_bounded_photographic_profile(tampered)
    except ValueError:
        tampered_rejected = True
    unknown = json.loads(encoded)
    unknown["unexpected"] = 1
    unknown_rejected = False
    try:
        validate_bounded_photographic_profile(unknown)
    except ValueError:
        unknown_rejected = True
    noncanonical_path = bundle_path.with_name("noncanonical.json")
    noncanonical_path.write_text(json.dumps(first, indent=2), encoding="utf-8")
    noncanonical_rejected = False
    try:
        load_bounded_photographic_profile(
            noncanonical_path, expected_bundle_sha256=first["bundle_sha256"]
        )
    except ValueError:
        noncanonical_rejected = True
    noncanonical_path.unlink()
    checks = {
        "canonical_repeat": canonical_profile_bytes(first)
        == canonical_profile_bytes(second),
        "profile_roundtrip": rebuilt.profile.identity() == profile.identity(),
        "prior_roundtrip": rebuilt.prior.identity() == prior.identity(),
        "physical_output": np.array_equal(direct, reconstructed),
        "scanner_output": np.array_equal(direct_scanner, reconstructed_scanner),
        "diagnostics": direct_scientific == rebuilt_scientific,
        "tampered_rejected": tampered_rejected,
        "noncanonical_rejected": noncanonical_rejected,
        "unknown_rejected": unknown_rejected,
    }
    stable = {
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "bundle_sha256": first["bundle_sha256"],
        "bundle_bytes": len(encoded),
        "profile_id": rebuilt.profile.identity(),
        "prior_id": rebuilt.prior.identity(),
        "physical_output_sha256": hashlib.sha256(
            memoryview(reconstructed).cast("B")
        ).hexdigest(),
        "scanner_output_sha256": hashlib.sha256(
            memoryview(reconstructed_scanner).cast("B")
        ).hexdigest(),
        "diagnostics": reconstructed_diagnostics,
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "decision": contract["decision_if_pass"]
        if all(checks.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("contract", "result"),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


__all__ = ["evaluate"]
