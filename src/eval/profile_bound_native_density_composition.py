"""P4HQ dual-compiler profile-bound native density composition evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gamma

from src.eval.hybrid_native_gamma_density_inverse import _build_llvm as build_gamma_llvm
from src.eval.native_histogram_copula_transport import _build_llvm as build_copula_llvm
from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.bounded_photographic_profile import (
    reconstruct_bounded_photographic_profile,
)
from src.film_physics.calibrated_native_histogram_copula import (
    apply_source_observable_calibrated_copula,
)
from src.film_physics.native_histogram_copula import (
    load_native_histogram_copula_library,
)
from src.film_physics.native_hybrid_gamma_density import (
    load_native_hybrid_gamma_density_library,
)
from src.film_physics.profile_bound_native_density import (
    apply_profile_bound_native_density,
)

SCHEMA = "neuro-film.u6-p4hq-profile-bound-native-density-composition-contract.v1"


def _fixtures(contract: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fixture = contract["fixture"]
    count = int(fixture["height"]) * int(fixture["width"])
    base_rng = np.random.default_rng(int(fixture["base_seed"]))
    base = base_rng.uniform(
        float(fixture["minimum_base_value"]),
        float(fixture["maximum_base_value"]),
        size=(int(fixture["height"]), int(fixture["width"]), 3),
    ).astype(np.float64)
    field_rng = np.random.default_rng(int(fixture["field_seed"]))
    latent = field_rng.standard_normal((count, 4))
    fields = np.empty((count, 3), dtype=np.float32)
    fields[:, 0] = latent[:, 0] + 0.19 * latent[:, 3] ** 2
    fields[:, 1] = 0.23 * latent[:, 0] + latent[:, 1] - 0.11 * latent[:, 3] ** 2
    fields[:, 2] = -0.17 * latent[:, 0] + 0.29 * latent[:, 1] + latent[:, 2]
    return base, fields


def _reference(
    base: np.ndarray, uniforms: np.ndarray, components: Any
) -> tuple[np.ndarray, np.ndarray]:
    flat = base.reshape(-1, 3)
    density = -np.log10(flat)
    sigma = np.empty_like(flat)
    for index, channel in enumerate(("red", "green", "blue")):
        lower, upper = components.prior.curves[index].domain
        exposure = lower + flat[:, index] * (upper - lower)
        sigma_d = components.profile.amplitude_profile.evaluate_channel(
            components.prior, channel, exposure
        )
        sigma[:, index] = sigma_d * (4.0 * flat[:, index] * (1.0 - flat[:, index]))
    shape = np.square(density / sigma)
    scale = np.square(sigma) / density
    developed = gamma.ppf(uniforms.reshape(flat.shape), a=shape, scale=scale)
    output = (flat * np.power(10.0, -(developed - density))).reshape(base.shape)
    return developed, output.astype(np.float32)


def _stable(report: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(report))
    payload.pop("stable_evidence_id", None)
    for toolchains in payload["toolchains"].values():
        for build in toolchains.values():
            for key in ("dll_path", "dll_sha256", "compiler_output"):
                build.pop(key, None)
    return payload


def evaluate(
    *, root: Path, contract: dict[str, Any], output_dir: Path, clang: Path
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HQ contract")
    for name in ("p4hn_evidence", "p4hp_evidence", "profile"):
        binding = contract["parents"][name]
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"P4HQ parent drift: {name}")
    for name in ("p4hn_evidence", "p4hp_evidence"):
        binding = contract["parents"][name]
        evidence = json.loads((root / binding["path"]).read_text("utf-8"))
        if evidence.get("decision") != binding["required_decision"]:
            raise ValueError(f"P4HQ {name} decision drift")
    profile = json.loads(
        (root / contract["parents"]["profile"]["path"]).read_text("utf-8")
    )
    components = reconstruct_bounded_photographic_profile(profile)
    base, fields = _fixtures(contract)
    before = base.tobytes()
    copula_dir = output_dir / "copula"
    gamma_dir = output_dir / "gamma"
    builds = {
        "msvc": {
            "copula": build_msvc_c11_dll(
                root=root,
                output_dir=copula_dir,
                source_relative="native/film_physics/nf_histogram_copula_f32_v1.c",
                header_relative="native/film_physics/nf_histogram_copula_f32_v1.h",
                basename="nf_histogram_copula_f32_msvc_v1",
            ),
            "gamma": build_msvc_c11_dll(
                root=root,
                output_dir=gamma_dir,
                source_relative="native/film_physics/nf_gamma_density_hybrid_f64_v1.c",
                header_relative="native/film_physics/nf_gamma_density_hybrid_f64_v1.h",
                basename="nf_gamma_density_hybrid_f64_msvc_v1",
            ),
        },
        "llvm_mingw": {
            "copula": build_copula_llvm(root, copula_dir, clang),
            "gamma": build_gamma_llvm(root, gamma_dir, clang),
        },
    }
    candidate = contract["candidate"]
    rows: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    developed_rows: dict[str, np.ndarray] = {}
    for name, build in builds.items():
        copula = load_native_histogram_copula_library(Path(build["copula"]["dll_path"]))
        gamma_library = load_native_hybrid_gamma_density_library(
            Path(build["gamma"]["dll_path"])
        )
        first, first_developed, receipt = apply_profile_bound_native_density(
            copula,
            gamma_library,
            base,
            fields,
            components=components,
            copula_iterations=int(candidate["copula_iterations"]),
            gamma_inverse_iterations=int(candidate["gamma_inverse_iterations"]),
            high_shape_threshold=float(candidate["high_shape_threshold"]),
        )
        second, second_developed, second_receipt = apply_profile_bound_native_density(
            copula,
            gamma_library,
            base,
            fields,
            components=components,
            copula_iterations=int(candidate["copula_iterations"]),
            gamma_inverse_iterations=int(candidate["gamma_inverse_iterations"]),
            high_shape_threshold=float(candidate["high_shape_threshold"]),
        )
        uniforms = np.asarray(
            receipt["copula"]["final_native_diagnostics"]["output_uniform_mean"]
        )
        del uniforms
        reference_developed, reference_output = _reference(
            base,
            # The final native uniforms are recovered deterministically by rerunning the retained copula.
            apply_source_observable_calibrated_copula(
                copula,
                fields,
                profile_correlation=components.correlation_matrix,
                rank_bins=components.rank_bins,
                iterations=int(candidate["copula_iterations"]),
            )[0],
            components,
        )
        outputs[name] = first
        developed_rows[name] = first_developed
        rows[name] = {
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "repeat_exact": bool(
                np.array_equal(first, second)
                and np.array_equal(first_developed, second_developed)
            ),
            "receipt_repeat_exact": receipt == second_receipt,
            "input_unchanged": base.tobytes() == before,
            "finite_unit": bool(
                np.all(np.isfinite(first)) and np.all(first >= 0) and np.all(first <= 1)
            ),
            "maximum_transmittance_absolute_error": float(
                np.max(
                    np.abs(
                        first.astype(np.float64) - reference_output.astype(np.float64)
                    )
                )
            ),
            "maximum_developed_density_error": float(
                np.max(np.abs(first_developed.reshape(-1, 3) - reference_developed))
            ),
            "final_uniform_correlation_error": float(
                receipt["copula"]["iteration_maximum_correlation_errors"][-1]
            ),
            "receipt": receipt,
        }
    cross = float(
        np.max(
            np.abs(outputs["msvc"].astype(float) - outputs["llvm_mingw"].astype(float))
        )
    )
    gates = contract["automatic_gates"]
    checks = {
        "developed_density": max(
            row["maximum_developed_density_error"] for row in rows.values()
        )
        <= gates["maximum_developed_density_error"],
        "transmittance": max(
            row["maximum_transmittance_absolute_error"] for row in rows.values()
        )
        <= gates["maximum_transmittance_absolute_error"],
        "cross_compiler": cross <= gates["maximum_cross_compiler_absolute_error"],
        "uniform_correlation": max(
            row["final_uniform_correlation_error"] for row in rows.values()
        )
        <= gates["maximum_final_uniform_correlation_error"],
        "active_fraction": min(
            row["receipt"]["active_fraction"] for row in rows.values()
        )
        >= gates["minimum_active_fraction"],
        "both_gamma_branches": all(
            row["receipt"]["gamma"]["direct_branch_count"] > 0
            and row["receipt"]["gamma"]["asymptotic_branch_count"] > 0
            for row in rows.values()
        ),
        "repeat": all(
            row["repeat_exact"] and row["receipt_repeat_exact"] for row in rows.values()
        ),
        "input_unchanged": all(row["input_unchanged"] for row in rows.values()),
        "finite_unit": all(row["finite_unit"] for row in rows.values()),
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.u6-p4hq-profile-bound-native-density-composition-result.v1",
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "profile_bundle_sha256": contract["parents"]["profile"]["bundle_sha256"],
        "toolchains": builds,
        "results": rows,
        "maximum_cross_compiler_absolute_error": cross,
        "gates": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(_stable(report), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


__all__ = ["evaluate"]
