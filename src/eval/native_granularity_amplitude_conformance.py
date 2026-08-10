"""Build and evaluate U6.P8BV native granularity-amplitude compilation."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.density_conditioned_thomas import DensityConditionedThomasProfile
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_granularity_amplitude import (
    NativeGranularityAmplitudeProfileV1,
    apply_native_granularity_amplitude,
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)

SCHEMA = "neuro_film.u6_p8bv_native_granularity_amplitude_contract.v1"
SOURCE = "native/film_physics/nf_granularity_amplitude_f32_v1.c"
HEADER = "native/film_physics/nf_granularity_amplitude_f32_v1.h"


class NativeGranularityAmplitudeConformanceError(RuntimeError):
    """Raised when the P8BV contract or source bindings drift."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_contract(root: Path, contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "contract_frozen_implementation_ready"
        or contract["candidate"].get("profile_refit_allowed") is not False
        or contract["candidate"].get("spatial_field_generation_allowed") is not False
    ):
        raise NativeGranularityAmplitudeConformanceError("P8BV contract drift")
    for binding in contract["parents"].values():
        path = root / str(binding["path"])
        if not path.is_file() or sha256_file(path) != str(binding["sha256"]):
            raise NativeGranularityAmplitudeConformanceError("P8BV parent hash drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "required_decision" in binding and payload.get("decision") != binding[
            "required_decision"
        ]:
            raise NativeGranularityAmplitudeConformanceError(
                "P8BV parent decision drift"
            )
        if "required_profile_id" in binding and payload.get("profile_id") != binding[
            "required_profile_id"
        ]:
            raise NativeGranularityAmplitudeConformanceError(
                "P8BV parent profile drift"
            )
        if "required_prior_id" in binding and payload["prior"]["source_evidence_id"] == "":
            raise NativeGranularityAmplitudeConformanceError("P8BV prior is empty")
    p4bw = json.loads(
        (root / contract["parents"]["p4bw_bundle"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    prior_payload = json.loads(
        (root / contract["parents"]["p2q_bundle"]["path"]).read_text(encoding="utf-8")
    )
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if (
        p4bw["amplitude_profile"]["characteristic_prior_identity"]
        != contract["parents"]["p2q_bundle"]["required_prior_id"]
        or prior.identity() != contract["parents"]["p2q_bundle"]["required_prior_id"]
    ):
        raise NativeGranularityAmplitudeConformanceError("P8BV prior identity drift")


def _bundles(root: Path, contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads(
            (root / contract["parents"]["p4bw_bundle"]["path"]).read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (root / contract["parents"]["p2q_bundle"]["path"]).read_text(
                encoding="utf-8"
            )
        ),
    )


def fixture_exposure(
    prior: ManufacturerCharacteristicPrior, samples_per_layer: int
) -> np.ndarray:
    result = np.empty((3, samples_per_layer), dtype=np.float32)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        lower_f32 = np.float32(lower)
        if float(lower_f32) < lower:
            lower_f32 = np.nextafter(lower_f32, np.float32(np.inf))
        upper_f32 = np.float32(upper)
        if float(upper_f32) > upper:
            upper_f32 = np.nextafter(upper_f32, np.float32(-np.inf))
        values = np.linspace(
            lower_f32, upper_f32, samples_per_layer, dtype=np.float32
        )
        probes: list[np.float32] = []
        for knot in curve.log_exposure_knots:
            value = np.float32(knot)
            for candidate in (
                np.nextafter(value, np.float32(-np.inf)),
                value,
                np.nextafter(value, np.float32(np.inf)),
            ):
                if lower <= float(candidate) <= upper:
                    probes.append(candidate)
        values[: len(probes)] = probes
        result[channel] = values
    return result


def reference_outputs(
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    exposure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    density = np.stack(
        [
            prior.curves[channel].apply(exposure[channel].astype(np.float64))
            for channel in range(3)
        ]
    )
    sigma = np.stack(
        [
            profile.amplitude_profile.evaluate_channel(
                prior,
                ("red", "green", "blue")[channel],
                exposure[channel].astype(np.float64),
            )
            / np.sqrt(profile.measurement_energy())
            for channel in range(3)
        ]
    )
    return (
        np.ascontiguousarray(density, dtype=np.float64),
        np.ascontiguousarray(sigma, dtype=np.float64),
    )


def _build_llvm(root: Path, output_dir: Path, clang: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = output_dir / "nf_granularity_amplitude_llvm_v1.dll"
    completed = subprocess.run(
        [
            str(clang),
            "--target=x86_64-w64-windows-gnu",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-model=strict",
            "-shared",
            str(root / SOURCE),
            "-o",
            str(dll),
            "-Wl,--no-insert-timestamp",
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not dll.is_file():
        raise NativeGranularityAmplitudeConformanceError(
            "LLVM P8BV build failed:\n"
            + (completed.stdout + completed.stderr).decode(errors="replace")
        )
    return {
        "toolchain": "llvm-mingw-20260616-clang-22.1.8-x64",
        "clang_sha256": sha256_file(clang),
        "source_sha256": sha256_file(root / SOURCE),
        "header_sha256": sha256_file(root / HEADER),
        "dll_sha256": sha256_file(dll),
        "dll_path": str(dll.resolve()),
    }


def _failure_atomic(
    library: ctypes.CDLL, profile: NativeGranularityAmplitudeProfileV1
) -> bool:
    exposure = np.zeros((3, 4), dtype=np.float32)
    exposure[2, 3] = np.float32(-100.0)
    density = np.full((3, 4), np.float32(-13.0))
    sigma = np.full((3, 4), np.float32(-17.0))
    pointer = ctypes.POINTER(ctypes.c_float)
    status = library.nf_granularity_amplitude_f32_apply_v1(
        ctypes.byref(profile),
        exposure.ctypes.data_as(pointer),
        exposure[0].size,
        density.ctypes.data_as(pointer),
        sigma.ctypes.data_as(pointer),
    )
    return bool(
        status != 0 and np.all(density == -13.0) and np.all(sigma == -17.0)
    )


def evaluate_conformance(
    root: Path,
    contract: Mapping[str, Any],
    *,
    output_dir: Path,
    clang: Path,
) -> dict[str, Any]:
    validate_contract(root, contract)
    p4bw_payload, prior_payload = _bundles(root, contract)
    profile = DensityConditionedThomasProfile.from_dict(p4bw_payload)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    native_profile = compile_native_granularity_amplitude_profile(
        p4bw_payload, prior_payload
    )
    builds = {
        "msvc": build_msvc_c11_dll(
            root=root,
            output_dir=output_dir / "msvc",
            source_relative=SOURCE,
            header_relative=HEADER,
            basename="nf_granularity_amplitude_msvc_v1",
        ),
        "llvm_mingw": _build_llvm(root, output_dir / "llvm", clang),
    }
    samples = int(contract["conformance"]["samples_per_layer"])
    exposure = fixture_exposure(prior, samples)
    reference_density, reference_sigma = reference_outputs(
        profile, prior, exposure
    )
    rows: dict[str, Any] = {}
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, build in builds.items():
        library = load_native_granularity_amplitude_library(Path(build["dll_path"]))
        density, sigma = apply_native_granularity_amplitude(
            library, native_profile, exposure
        )
        repeat_density, repeat_sigma = apply_native_granularity_amplitude(
            library, native_profile, exposure
        )
        outputs[name] = (density, sigma)
        rows[name] = {
            "density_sha256": _sha256_bytes(density.tobytes()),
            "point_sigma_sha256": _sha256_bytes(sigma.tobytes()),
            "density_repeat_exact": bool(np.array_equal(density, repeat_density)),
            "point_sigma_repeat_exact": bool(np.array_equal(sigma, repeat_sigma)),
            "maximum_density_absolute_error": float(
                np.max(np.abs(density.astype(np.float64) - reference_density))
            ),
            "maximum_point_sigma_absolute_error": float(
                np.max(np.abs(sigma.astype(np.float64) - reference_sigma))
            ),
            "failure_atomic": _failure_atomic(library, native_profile),
        }
    density_cross = float(
        np.max(np.abs(outputs["msvc"][0] - outputs["llvm_mingw"][0]))
    )
    sigma_cross = float(
        np.max(np.abs(outputs["msvc"][1] - outputs["llvm_mingw"][1]))
    )
    gates = contract["conformance"]
    gate_results = {
        "density_reference": max(
            row["maximum_density_absolute_error"] for row in rows.values()
        )
        <= float(gates["maximum_density_absolute_error_vs_float64"]),
        "sigma_reference": max(
            row["maximum_point_sigma_absolute_error"] for row in rows.values()
        )
        <= float(gates["maximum_point_sigma_absolute_error_vs_float64"]),
        "cross_compiler": max(density_cross, sigma_cross)
        <= float(gates["maximum_cross_compiler_absolute_error"]),
        "repeat": all(
            row["density_repeat_exact"] and row["point_sigma_repeat_exact"]
            for row in rows.values()
        ),
        "failure_atomic": all(row["failure_atomic"] for row in rows.values()),
    }
    report = {
        "schema": "neuro_film.u6_p8bv_native_granularity_amplitude_conformance.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(
            root / "configs/u6_p8bv_native_granularity_amplitude_v1.json"
        ),
        "samples_per_layer": samples,
        "exposure_sha256": _sha256_bytes(exposure.tobytes()),
        "reference_density_sha256": _sha256_bytes(reference_density.tobytes()),
        "reference_point_sigma_sha256": _sha256_bytes(reference_sigma.tobytes()),
        "toolchains": builds,
        "results": rows,
        "maximum_cross_compiler_absolute_error": max(density_cross, sigma_cross),
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "claim_ceiling": contract["claim_ceiling"],
    }
    identity = json.loads(json.dumps(report))
    for toolchain in identity["toolchains"].values():
        toolchain.pop("dll_path", None)
        toolchain.pop("dll_sha256", None)
        toolchain.pop("compiler_output", None)
    stable_id = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return {**report, "stable_evidence_id": stable_id}


__all__ = [
    "NativeGranularityAmplitudeConformanceError",
    "evaluate_conformance",
    "fixture_exposure",
    "reference_outputs",
    "validate_contract",
]
