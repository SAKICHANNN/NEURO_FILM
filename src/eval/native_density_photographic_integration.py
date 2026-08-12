"""P4HR native density composition in the frozen P4HE photographic chain."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.eval.paired_scanner_mtf_photographic_development import _evaluate_paired
from src.film_physics.bounded_photographic_profile import (
    reconstruct_bounded_photographic_profile,
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
from src.film_physics.thomas_dc_projection import (
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)

SCHEMA = "neuro-film.u6-p4hr-native-density-photographic-integration-contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HR contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"P4HR parent drift: {path}")
    return json.loads(path.read_text("utf-8"))


def evaluate(
    contract: dict[str, Any],
    *,
    root: Path,
    build_dir: Path,
    contact_sheet_path: Path,
) -> dict[str, Any]:
    parents = contract["parents"]
    p4hq = _load_bound(root, parents["p4hq_evidence"])
    p4he_contract = _load_bound(root, parents["p4he_contract"])
    p4he_evidence = _load_bound(root, parents["p4he_evidence"])
    _load_bound(root, parents["p4hj_evidence"])
    p4hj_report = _load_bound(root, parents["p4hj_report"])
    profile_payload = _load_bound(root, parents["profile"])
    if (
        p4hq.get("decision") != parents["p4hq_evidence"]["required_decision"]
        or p4he_evidence.get("decision")
        != parents["p4he_evidence"]["required_decision"]
    ):
        raise ValueError("P4HR parent decision drift")
    components = reconstruct_bounded_photographic_profile(profile_payload)
    if profile_payload["bundle_sha256"] != parents["profile"]["bundle_sha256"]:
        raise ValueError("P4HR profile bundle identity drift")
    copula_build = build_msvc_c11_dll(
        root=root,
        output_dir=build_dir / "copula",
        source_relative="native/film_physics/nf_histogram_copula_f32_v1.c",
        header_relative="native/film_physics/nf_histogram_copula_f32_v1.h",
        basename="nf_histogram_copula_f32_msvc_v1",
    )
    gamma_build = build_msvc_c11_dll(
        root=root,
        output_dir=build_dir / "gamma",
        source_relative="native/film_physics/nf_gamma_density_hybrid_f64_v1.c",
        header_relative="native/film_physics/nf_gamma_density_hybrid_f64_v1.h",
        basename="nf_gamma_density_hybrid_f64_msvc_v1",
    )
    copula_library = load_native_histogram_copula_library(
        Path(copula_build["dll_path"])
    )
    gamma_library = load_native_hybrid_gamma_density_library(
        Path(gamma_build["dll_path"])
    )
    candidate = contract["candidate"]

    def apply_physical(
        source: np.ndarray,
        profile: Any,
        prior: Any,
        seeds: tuple[int, int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if (
            profile.identity() != profile_payload["density_profile_id"]
            or prior.identity() != profile_payload["manufacturer_prior_id"]
        ):
            raise ValueError("P4HR photographic component identity drift")
        height, width = source.shape[:2]
        receipts = tuple(
            build_thomas_dc_receipt(
                (height, width),
                profile_id=profile.spatial_profile_id,
                particle_sigma_pixels=profile.particle_sigma_samples,
                cluster_sigma_pixels=profile.cluster_sigma_samples,
                mean_offspring=profile.mean_offspring,
                component_seeds=profile.component_seeds,
                realization_seed=seed,
                truncate=profile.truncate,
                canonical_row_block_height=components.canonical_row_block_height,
            )
            for seed in seeds
        )
        fields = np.stack(
            [
                render_dc_projected_thomas_region(
                    receipt, origin_yx=(0, 0), shape=(height, width)
                )
                for receipt in receipts
            ],
            axis=-1,
        ).astype(np.float32)
        physical, developed, native = apply_profile_bound_native_density(
            copula_library,
            gamma_library,
            source,
            fields.reshape(-1, 3),
            components=components,
            copula_iterations=int(candidate["copula_iterations"]),
            gamma_inverse_iterations=int(candidate["gamma_inverse_iterations"]),
            high_shape_threshold=float(candidate["high_shape_threshold"]),
        )
        difference = physical.astype(np.float64) - source.astype(np.float64)
        minimum_sigma = math.inf
        maximum_sigma = 0.0
        for index, channel in enumerate(("red", "green", "blue")):
            values = source[..., index].astype(np.float64)
            lower, upper = prior.curves[index].domain
            exposure = lower + values * (upper - lower)
            sigma_d = profile.amplitude_profile.evaluate_channel(
                prior, channel, exposure
            )
            sigma = sigma_d * (4.0 * values * (1.0 - values))
            minimum_sigma = min(minimum_sigma, float(np.min(sigma)))
            maximum_sigma = max(maximum_sigma, float(np.max(sigma)))
        return physical, {
            "receipt_ids": [receipt.receipt_id for receipt in receipts],
            "bounded_residual_rms": float(
                np.sqrt(np.mean(difference * difference, dtype=np.float64))
            ),
            "minimum_target_sigma_d": minimum_sigma,
            "maximum_target_sigma_d": maximum_sigma,
            "support_degenerate_fraction": 1.0 - native["active_fraction"],
            "minimum_developed_density": native["minimum_developed_density"],
            "limited_fraction": 0.0,
            "hard_clipping_used": 0.0,
            "rank_bins": components.rank_bins,
            "pass_count": int(candidate["copula_iterations"]) + 1,
            "canonical_row_block_height": components.canonical_row_block_height,
            "peak_live_temporary_bytes": int(
                fields.nbytes + developed.nbytes + physical.nbytes
            ),
            "full_frame_intermediate_count_excluding_input_output": 2,
        }

    result = _evaluate_paired(
        p4he_contract,
        root=root,
        contact_sheet_path=contact_sheet_path,
        result_parent_name="p4hd_result",
        apply_physical_override=apply_physical,
        compact_visual_rows=True,
    )
    stable = result["stable"]
    p4hj_rows = {
        row["id"]: row
        for row in p4hj_report["stable"]["scientific_result"]["stable"]["rows"]
    }
    stable["p4hj_comparison"] = {
        "same_row_count": len(p4hj_rows) == len(stable["rows"]),
        "physical_pixel_identity_count": sum(
            row["physical_output_sha256"]
            == p4hj_rows[row["id"]]["physical_output_sha256"]
            for row in stable["rows"]
        ),
        "pixel_identity_required": False,
    }
    stable["profile_bundle_sha256"] = profile_payload["bundle_sha256"]
    stable["native_toolchains"] = {"copula": copula_build, "gamma": gamma_build}
    stable["claim_ceiling"] = contract["claim_ceiling"]
    result["schema"] = contract["schema"].replace("contract", "worker-result")
    return result


__all__ = ["evaluate", "load_contract"]
