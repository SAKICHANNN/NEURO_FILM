"""Frozen P4ED Python/native cloud row integration audit."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_cloud_attenuation_runtime import (
    compile_cloud_attenuation_profile,
    iter_compiled_cloud_attenuation_rows,
    iter_native_compiled_cloud_attenuation_rows,
)
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import optical_density_capacity_cmy
from src.film_physics.native_cloud_attenuation import load_native_cloud_attenuation


def _render(iterator) -> tuple[np.ndarray, np.ndarray]:
    rows = tuple(iterator)
    return np.concatenate([x.density for _, x in rows]), np.concatenate(
        [x.transmittance for _, x in rows]
    )


def evaluate(root: Path, contract_path: Path, build_dir: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    parent = root / contract["parent"]["path"]
    if sha256_file(parent) != contract["parent"]["sha256"]:
        raise RuntimeError("P4ED parent drift")
    reference = CrossLayerCloudReferenceProfile.from_payload(
        evaluate_capacity(
            root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
        )["compiled_profile"]
    )
    p4dw = json.loads(
        (
            root
            / "docs/evidence/U6_P4DW_NPS_PRESERVING_CLOUD_RESIDUAL_ATTENUATION_V6_RESULT.json"
        ).read_text()
    )
    profile = compile_cloud_attenuation_profile(
        reference,
        aperture_factor=4,
        base_rate_multiplier=16.0,
        channel_residual_gain=tuple(p4dw["compiled_channel_gain"]),
    )
    build = build_msvc_c11_dll(
        root=root,
        output_dir=build_dir,
        source_relative="native/film_physics/nf_cloud_attenuation_f32_v1.c",
        header_relative="native/film_physics/nf_cloud_attenuation_f32_v1.h",
        basename="nf_cloud_attenuation_p4ed",
    )
    library = load_native_cloud_attenuation(Path(build["dll_path"]))
    shape = (contract["fixture"]["height"], contract["fixture"]["width"])
    yy, xx = np.indices(shape, dtype=np.float64)
    capacity = np.asarray(optical_density_capacity_cmy(profile.base_profile))
    target = (0.2 + 0.6 * (xx + yy) / (sum(shape) - 2))[..., None] * capacity
    seed = contract["fixture"]["seed"]
    py_d, py_t = _render(
        iter_compiled_cloud_attenuation_rows(
            profile, target, seed=seed, row_tile_height=64
        )
    )
    rows = []
    max_d = max_t = 0.0
    for partition in contract["fixture"]["partition_heights"]:
        d, t = _render(
            iter_native_compiled_cloud_attenuation_rows(
                profile, target, seed=seed, row_tile_height=partition, library=library
            )
        )
        max_d = max(max_d, float(np.max(np.abs(d - py_d))))
        max_t = max(max_t, float(np.max(np.abs(t - py_t))))
        rows.append(
            {
                "partition": partition,
                "density_sha256": hashlib.sha256(d.tobytes()).hexdigest(),
                "transmittance_sha256": hashlib.sha256(t.tobytes()).hexdigest(),
            }
        )
    native64_d, native64_t = _render(
        iter_native_compiled_cloud_attenuation_rows(
            profile, target, seed=seed, row_tile_height=64, library=library
        )
    )
    repeat_d, repeat_t = _render(
        iter_native_compiled_cloud_attenuation_rows(
            profile, target, seed=seed, row_tile_height=64, library=library
        )
    )

    def timed(native: bool) -> float:
        start = time.perf_counter()
        iterator = (
            iter_native_compiled_cloud_attenuation_rows(
                profile, target, seed=seed, row_tile_height=64, library=library
            )
            if native
            else iter_compiled_cloud_attenuation_rows(
                profile, target, seed=seed, row_tile_height=64
            )
        )
        _render(iterator)
        return time.perf_counter() - start

    py_times = [timed(False) for _ in range(contract["fixture"]["timed_replays"])]
    native_times = [timed(True) for _ in range(contract["fixture"]["timed_replays"])]
    ratio = float(np.median(native_times) / np.median(py_times))
    g = contract["gates"]
    invalid = target.copy()
    invalid[-1, -1, 0] = np.nan
    try:
        _render(
            iter_native_compiled_cloud_attenuation_rows(
                profile, invalid, seed=seed, row_tile_height=64, library=library
            )
        )
    except ValueError:
        atomic = True
    else:
        atomic = False
    decisions = {
        "density": max_d <= g["maximum_native_python_density_absolute_error"],
        "transmittance": max_t
        <= g["maximum_native_python_transmittance_absolute_error"],
        "partition": len(
            {(r["density_sha256"], r["transmittance_sha256"]) for r in rows}
        )
        == 1,
        "repeat": np.array_equal(repeat_d, native64_d)
        and np.array_equal(repeat_t, native64_t),
        "performance": ratio <= g["maximum_native_to_python_wall_ratio"],
        "atomic": atomic,
    }
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "native_density_max_error": max_d,
        "native_transmittance_max_error": max_t,
        "native_to_python_wall_ratio": ratio,
        "partition_rows": rows,
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4ed_native_cloud_row_integration.v1",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate"]
