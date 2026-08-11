"""Opt-in runtime for the frozen CB56 analytic Y/chromaticity research profile."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.analytic_y_chromaticity_streaming import (
    select_analytic_y_chromaticity_candidate_streamed,
)
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.fujifilm_characteristic_photographic import _compiled_curve
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.eval.nonexpansive_fraction_transport_streaming import (
    nonexpansive_fraction_transport_target_row_materialized,
)
from src.film_physics.profile_consumer import validate_standalone_profile_artifact
from src.preprocess.types import WorkingImage

PROFILE_SCHEMA = "kmcfm.research-render-profile.v1"
ENGINE_ID = "analytic_y_chromaticity_cb56_v1"
_HASH = re.compile(r"[0-9a-f]{64}")


class AnalyticYChromaticityProfileError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_asset(root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    path = root / str(asset["path"])
    expected = str(asset["sha256"])
    if (
        not _HASH.fullmatch(expected)
        or not path.is_file()
        or _sha256_file(path) != expected
    ):
        raise AnalyticYChromaticityProfileError(
            f"research profile asset drift: {asset.get('role')}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class AnalyticYChromaticityRuntime:
    profile: dict[str, Any]
    profile_sha256: str
    cb52: dict[str, Any]
    cb11: dict[str, Any]
    ao6_config: dict[str, Any]
    artifact: dict[str, Any]
    curve: Any


def load_analytic_y_chromaticity_profile(
    path: Path, *, root: Path
) -> AnalyticYChromaticityRuntime:
    if not path.is_file():
        raise AnalyticYChromaticityProfileError("research profile is missing")
    profile = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_id",
        "profile_id",
        "profile_version",
        "display_name",
        "engine_id",
        "style",
        "identity",
        "evidence",
        "assets",
        "execution",
        "output",
    }
    if set(profile) != required or profile["schema_id"] != PROFILE_SCHEMA:
        raise AnalyticYChromaticityProfileError("research profile structure drift")
    profile_identity = (profile["profile_id"], profile["profile_version"])
    if (
        profile_identity
        not in {
            ("analytic-y-chromaticity-cb56-v1", "1.0.0"),
            ("analytic-y-chromaticity-cb61-v2", "2.0.0"),
            ("analytic-y-chromaticity-cb66-v3", "3.0.0"),
        }
        or profile["engine_id"] != ENGINE_ID
        or profile["style"] != "velvia_50"
        or profile["identity"]
        != {
            "film_stock_id": None,
            "interpretation": "look_approximation",
            "research_champion": True,
            "product_default": False,
        }
    ):
        raise AnalyticYChromaticityProfileError("research profile identity drift")
    execution = profile["execution"]
    if (
        execution.get("input_working_space") != "linear_srgb"
        or execution.get("input_color_state") != "display_linear"
        or execution.get("row_chunk")
        != (128 if profile_identity[0].endswith("cb66-v3") else 64)
        or execution.get("recipe_v1_allowed") is not False
        or execution.get("target_materializer", "legacy_full_frame_v1")
        not in {
            "legacy_full_frame_v1",
            "row_bounded_float64_v1",
            "row_safe_base_external_sorted_v1",
        }
    ):
        raise AnalyticYChromaticityProfileError("research execution drift")
    assets = profile["assets"]
    expected_asset_count = (
        12
        if profile_identity[0].endswith("cb66-v3")
        else 10 if profile_identity[0].endswith("cb61-v2") else 8
    )
    if not isinstance(assets, list) or len(assets) != expected_asset_count:
        raise AnalyticYChromaticityProfileError("research asset inventory drift")
    by_role = {str(asset.get("role")): asset for asset in assets}
    required_roles = {
        "cb11_operator_contract",
        "cb12_ao6_contract",
        "cb52_operator_contract",
        "cb56_decision",
        "cb6_curve_contract",
        "ao6_artifact_report",
        "streaming_selector_code",
        "target_transport_code",
    }
    if profile_identity[0].endswith("cb61-v2"):
        required_roles |= {"cb61_decision", "row_target_materializer_code"}
    if profile_identity[0].endswith("cb66-v3"):
        required_roles |= {
            "cb66_decision",
            "memory_optimized_renderer_code",
            "external_fraction_map_code",
            "renderer_adapter_code",
        }
    if len(by_role) != len(assets) or set(by_role) != required_roles:
        raise AnalyticYChromaticityProfileError("research asset roles drift")
    code_roles = {"streaming_selector_code", "target_transport_code"}
    if profile_identity[0].endswith("cb61-v2"):
        code_roles.add("row_target_materializer_code")
    if profile_identity[0].endswith("cb66-v3"):
        code_roles |= {
            "memory_optimized_renderer_code",
            "external_fraction_map_code",
            "renderer_adapter_code",
        }
    for role in code_roles:
        asset = by_role[role]
        asset_path = root / str(asset["path"])
        if not asset_path.is_file() or _sha256_file(asset_path) != asset["sha256"]:
            raise AnalyticYChromaticityProfileError(
                f"research profile asset drift: {role}"
            )
    cb11 = _load_asset(root, by_role["cb11_operator_contract"])
    cb12 = _load_asset(root, by_role["cb12_ao6_contract"])
    cb52 = _load_asset(root, by_role["cb52_operator_contract"])
    decision = _load_asset(root, by_role["cb56_decision"])
    _load_asset(root, by_role["cb6_curve_contract"])
    artifact_report = _load_asset(root, by_role["ao6_artifact_report"])
    if profile_identity[0].endswith("cb61-v2"):
        resource_decision = _load_asset(root, by_role["cb61_decision"])
        if (
            resource_decision.get("decision")
            != "pass_cb61_output_exact_row_bounded_hue_target_memory"
            or resource_decision.get("automatic_pass") is not True
        ):
            raise AnalyticYChromaticityProfileError("CB61 resource decision drift")
    if profile_identity[0].endswith("cb66-v3"):
        resource_decision = _load_asset(root, by_role["cb66_decision"])
        if (
            resource_decision.get("decision")
            != "pass_cb66_output_exact_row_bounded_safe_base_complete_memory"
            or resource_decision.get("automatic_pass") is not True
        ):
            raise AnalyticYChromaticityProfileError("CB66 resource decision drift")
    if (
        decision.get("decision")
        != "pass_cb56_decoded_face_severe_retain_analytic_y_chromaticity_research_champion"
        or decision.get("product_default_changed") is not False
        or cb11["parents"]["cb6_contract_path"] != by_role["cb6_curve_contract"]["path"]
    ):
        raise AnalyticYChromaticityProfileError("research evidence boundary drift")
    ao6_config = cb12["ao6"]
    artifact = artifact_report["artifact"]
    if (
        artifact_report.get("artifact_canonical_sha256")
        != ao6_config["frozen_artifact_canonical_sha256"]
        or artifact.get("bundle_sha256") != ao6_config["frozen_bundle_sha256"]
    ):
        raise AnalyticYChromaticityProfileError("AO6 artifact identity drift")
    validate_standalone_profile_artifact(artifact)
    return AnalyticYChromaticityRuntime(
        profile=profile,
        profile_sha256=_sha256_file(path),
        cb52=cb52,
        cb11=cb11,
        ao6_config=ao6_config,
        artifact=artifact,
        curve=_compiled_curve(load_cb6(root / by_role["cb6_curve_contract"]["path"])),
    )


def render_analytic_y_chromaticity_profile(
    working: WorkingImage,
    runtime: AnalyticYChromaticityRuntime,
    *,
    scratch_root: Path | None = None,
    target_builder: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if (
        working.working_space != "linear_srgb"
        or working.transfer_state != "display_linear"
    ):
        raise AnalyticYChromaticityProfileError(
            "analytic Y/chromaticity profile requires display-linear linear-sRGB"
        )
    if (
        target_builder is None
        and runtime.profile["execution"].get("target_materializer")
        == "row_safe_base_external_sorted_v1"
    ):
        from src.eval.analytic_y_chromaticity_memory_optimized import (
            render_analytic_y_chromaticity_memory_candidate,
        )

        if scratch_root is not None:
            return render_analytic_y_chromaticity_memory_candidate(
                working,
                runtime,
                scratch_root=scratch_root,
                row_chunk=int(runtime.profile["execution"]["row_chunk"]),
            )
        with tempfile.TemporaryDirectory(prefix="cb66-") as temporary:
            return render_analytic_y_chromaticity_memory_candidate(
                working,
                runtime,
                scratch_root=Path(temporary),
                row_chunk=int(runtime.profile["execution"]["row_chunk"]),
            )
    source = np.asarray(working.pixels, dtype=np.float32)
    operator = runtime.cb11["operator"]
    weights = np.asarray(operator["luminance_weights"], dtype=np.float64)
    epsilon = float(operator["boundary_epsilon"])
    ao6 = render_fixed_pair(source, runtime.artifact, runtime.ao6_config["component"])[
        runtime.ao6_config["arm_id"]
    ]
    safe_base, _, _ = apply_characteristic_luma_chroma(
        source,
        runtime.curve,
        weights=weights,
        strength=float(operator["nominal_strength"]),
        boundary_epsilon=epsilon,
    )
    config = runtime.cb52
    if target_builder is None:
        build_target = (
            nonexpansive_fraction_transport_target_row_materialized
            if runtime.profile["execution"].get("target_materializer")
            == "row_bounded_float64_v1"
            else nonexpansive_fraction_transport_target
        )
    else:
        build_target = target_builder
    target = build_target(
        safe_base,
        ao6,
        weights=weights,
        minimum_valid_fraction=float(config["operator"]["minimum_valid_fraction"]),
        fraction_knots=int(config["operator"]["fraction_knots"]),
        maximum_fraction_slope=float(config["operator"]["maximum_fraction_slope"]),
    )
    candidate, _, _, facts = select_analytic_y_chromaticity_candidate_streamed(
        source,
        target,
        curve=runtime.curve,
        strength=float(operator["nominal_strength"]),
        boundary_epsilon=epsilon,
        dose_grid=config["operator"]["dose_grid"],
        maximum_gradient_ratio=float(
            config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
        ),
        maximum_lstar_inversion_fraction=float(
            config["automatic_gates"][
                "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
            ]
        ),
        lstar_order_epsilon=float(config["operator"]["lstar_order_epsilon"]),
        row_chunk=int(runtime.profile["execution"]["row_chunk"]),
        scratch_root=scratch_root,
    )
    return candidate, facts


__all__ = [
    "ENGINE_ID",
    "AnalyticYChromaticityProfileError",
    "AnalyticYChromaticityRuntime",
    "load_analytic_y_chromaticity_profile",
    "render_analytic_y_chromaticity_profile",
]
