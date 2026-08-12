"""P4EI native conditioned cloud row-chain evaluation."""

from __future__ import annotations

import _ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_cloud_spatial_response_conformance import (
    _build_llvm as build_spatial_llvm,
)
from src.eval.native_density_conditioned_poisson_conformance import (
    _build_llvm as build_count_llvm,
)
from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_density_conditioned_cross_layer_cloud_rows,
)
from src.film_physics.cross_layer_compound_poisson import CrossLayerPoissonProfile
from src.film_physics.native_conditioned_cloud import (
    iter_native_density_conditioned_cloud_rows,
    load_native_conditioned_cloud,
)

COUNT_SOURCE = "native/film_physics/nf_density_conditioned_poisson_u16_v2.c"
COUNT_HEADER = "native/film_physics/nf_density_conditioned_poisson_u16_v2.h"
SPATIAL_SOURCE = "native/film_physics/nf_cloud_spatial_response_f32_v1.c"
SPATIAL_HEADER = "native/film_physics/nf_cloud_spatial_response_f32_v1.h"


def _render(iterator) -> tuple[np.ndarray, np.ndarray]:
    rows = list(iterator)
    return np.concatenate([value.density for _, value in rows]), np.concatenate([value.transmittance for _, value in rows])


def evaluate(root: Path, contract_path: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if sha256_file(parent) != contract["parent"]["sha256"]:
        raise RuntimeError("P4EI parent drift")
    if evidence["decision"] != contract["parent"]["required_decision"]:
        raise RuntimeError("P4EI parent decision drift")
    builds = {
        "count": build_msvc_c11_dll(root=root, output_dir=output_dir / "count", source_relative=COUNT_SOURCE, header_relative=COUNT_HEADER, basename="nf_conditioned_count_p4ei"),
        "spatial": build_msvc_c11_dll(root=root, output_dir=output_dir / "spatial", source_relative=SPATIAL_SOURCE, header_relative=SPATIAL_HEADER, basename="nf_spatial_p4ei"),
    }
    # Compile the same sources with the second compiler; P4EG/P4EH already execute both.
    llvm_count = build_count_llvm(root, output_dir / "llvm_count", clang)
    llvm_spatial = build_spatial_llvm(root, output_dir / "llvm_spatial", clang)
    count_library, spatial_library = load_native_conditioned_cloud(Path(builds["count"]["dll_path"]), Path(builds["spatial"]["dll_path"]))
    fixture = contract["fixture"]
    shape = (fixture["height"], fixture["width"])
    rng = np.random.default_rng(fixture["scale_seed"])
    scale = np.ascontiguousarray(rng.uniform(0.0, 1.0, (*shape, 3)), dtype=np.float32)
    profile = CrossLayerCloudReferenceProfile(
        CrossLayerPoissonProfile(tuple(fixture["marginal_rates_cmy"]), fixture["shared_all_rate"], tuple(fixture["shared_pair_rates_cm_cy_my"]), tuple(fixture["mark_optical_density_cmy"]), fixture["profile_seed"]),
        tuple(fixture["sigma_pixels_cmy"]), fixture["truncate"], (1,), ("0" * 64,) * 3,
    )
    try:
        reference_d, reference_t = _render(iter_density_conditioned_cross_layer_cloud_rows(profile, scale, seed=fixture["profile_seed"], row_tile_height=fixture["partition_heights"][0]))
        native = {}
        for height in fixture["partition_heights"]:
            native[str(height)] = _render(iter_native_density_conditioned_cloud_rows(profile, scale, seed=fixture["profile_seed"], row_tile_height=height, count_library=count_library, spatial_library=spatial_library))
        repeat = _render(iter_native_density_conditioned_cloud_rows(profile, scale, seed=fixture["profile_seed"], row_tile_height=fixture["partition_heights"][0], count_library=count_library, spatial_library=spatial_library))
        bad = scale.copy(); bad[-1, -1, -1] = np.nan
        invalid_before_yield = False
        try:
            next(iter_native_density_conditioned_cloud_rows(profile, bad, seed=fixture["profile_seed"], row_tile_height=17, count_library=count_library, spatial_library=spatial_library))
        except ValueError:
            invalid_before_yield = True
    finally:
        _ctypes.FreeLibrary(count_library._handle); _ctypes.FreeLibrary(spatial_library._handle)
    first = native[str(fixture["partition_heights"][0])]
    max_d = max(float(np.max(np.abs(value[0].astype(np.float64) - reference_d.astype(np.float64)))) for value in native.values())
    max_t = max(float(np.max(np.abs(value[1].astype(np.float64) - reference_t.astype(np.float64)))) for value in native.values())
    partition_exact = all(np.array_equal(value[0], first[0]) and np.array_equal(value[1], first[1]) for value in native.values())
    repeat_exact = np.array_equal(first[0], repeat[0]) and np.array_equal(first[1], repeat[1])
    gates = contract["gates"]
    decisions = {"density": max_d <= gates["maximum_density_absolute_error"], "transmittance": max_t <= gates["maximum_transmittance_absolute_error"], "partition": partition_exact, "repeat": repeat_exact, "invalid": invalid_before_yield}
    stable = {
        "contract_sha256": sha256_file(contract_path), "count_source_sha256": sha256_file(root / COUNT_SOURCE), "spatial_source_sha256": sha256_file(root / SPATIAL_SOURCE),
        "llvm_count_dll_sha256": llvm_count["dll_sha256"], "llvm_spatial_dll_sha256": llvm_spatial["dll_sha256"],
        "scale_sha256": hashlib.sha256(scale.tobytes()).hexdigest(), "density_sha256": hashlib.sha256(first[0].tobytes()).hexdigest(), "transmittance_sha256": hashlib.sha256(first[1].tobytes()).hexdigest(),
        "maximum_density_absolute_error": max_d, "maximum_transmittance_absolute_error": max_t, "gates": decisions,
        "decision": contract["decision_if_pass"] if all(decisions.values()) else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"],
    }
    return {"schema": "neuro_film.u6_p4ei_native_conditioned_cloud_row_chain.v1", "automatic_pass": all(decisions.values()), "stable": stable, "stable_evidence_id": hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


__all__ = ["evaluate"]
