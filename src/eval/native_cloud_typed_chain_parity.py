"""Frozen P4EE native-cloud typed-chain parity audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.eval.physical_scanner_profile import _profile as scanner_profile
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.sensitometry_primitive import build_operator
from src.eval.typed_sensitometry_cloud_chain import _exposure
from src.film_physics.cross_layer_cloud_attenuation_runtime import (
    compile_cloud_attenuation_profile,
    iter_compiled_cloud_attenuation_rows,
    iter_native_compiled_cloud_attenuation_rows,
)
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.native_cloud_attenuation import load_native_cloud_attenuation
from src.film_physics.scanner import apply_scanner_profile


def _render(iterator) -> np.ndarray:
    return np.concatenate([result.transmittance for _, result in iterator])


def evaluate(root: Path, contract_path: Path, build_dir: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    parent = root / contract["parent"]["path"]
    if sha256_file(parent) != contract["parent"]["sha256"]:
        raise RuntimeError("P4EE parent drift")
    fixture = contract["fixture"]
    shape = (fixture["height"], fixture["width"])
    exposure = _exposure(shape, fixture["pixel_pitch_um"])
    operator = build_operator(
        json.loads((root / "configs/u2_2a_sensitometry_primitive_v1.json").read_text())
    )
    developed = operator.apply(exposure.values)
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
        basename="nf_cloud_attenuation_p4ee",
    )
    library = load_native_cloud_attenuation(Path(build["dll_path"]))
    rows = fixture["row_partition_height"]
    seed = fixture["seed"]
    python_t = _render(
        iter_compiled_cloud_attenuation_rows(
            profile, developed, seed=seed, row_tile_height=rows
        )
    )
    native_t = _render(
        iter_native_compiled_cloud_attenuation_rows(
            profile, developed, seed=seed, row_tile_height=rows, library=library
        )
    )
    repeat_t = _render(
        iter_native_compiled_cloud_attenuation_rows(
            profile, developed, seed=seed, row_tile_height=rows, library=library
        )
    )
    scanner_contract = json.loads(
        (root / "configs/u6_p6a_scanner_profile_boundary_v1.json").read_text()
    )
    scanner = scanner_profile(scanner_contract["profiles"]["scanner_a"])
    python_final = apply_scanner_profile(
        python_t.astype(np.float64), scanner, pixel_pitch_um=fixture["pixel_pitch_um"]
    )
    native_final = apply_scanner_profile(
        native_t.astype(np.float64), scanner, pixel_pitch_um=fixture["pixel_pitch_um"]
    )
    diff = native_final - python_final
    maximum = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mean = np.power(10.0, -developed)
    stage = float(
        np.sqrt(
            np.mean(
                np.square(
                    python_final
                    - apply_scanner_profile(
                        mean, scanner, pixel_pitch_um=fixture["pixel_pitch_um"]
                    )
                )
            )
        )
    )
    boundary = float(
        np.mean(
            ((native_final <= 0) | (native_final >= 1))
            & ~((python_final <= 0) | (python_final >= 1))
        )
    )
    g = contract["gates"]
    gates = {
        "maximum": maximum <= g["maximum_final_scan_absolute_error"],
        "rmse": rmse <= g["maximum_final_scan_rmse"],
        "stage": stage > 0.0,
        "boundary": boundary == 0.0,
        "repeat": np.array_equal(native_t, repeat_t),
    }
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "maximum_final_scan_absolute_error": maximum,
        "final_scan_rmse": rmse,
        "cloud_stage_rms": stage,
        "new_boundary_fraction": boundary,
        "python_final_sha256": hashlib.sha256(
            python_final.astype("<f8").tobytes()
        ).hexdigest(),
        "native_final_sha256": hashlib.sha256(
            native_final.astype("<f8").tobytes()
        ).hexdigest(),
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if all(gates.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4ee_native_cloud_typed_chain_parity.v1",
        "automatic_pass": all(gates.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate"]
