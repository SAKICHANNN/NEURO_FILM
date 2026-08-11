"""Versioned CB69 profile adapter without mutating the frozen CB66 adapter."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from src.eval.analytic_y_chromaticity_throughput_candidate import (
    render_analytic_y_chromaticity_throughput_candidate,
)
from src.inference.analytic_y_chromaticity_profile import (
    ENGINE_ID,
    PROFILE_SCHEMA,
    AnalyticYChromaticityProfileError,
    AnalyticYChromaticityRuntime,
)
from src.inference.analytic_y_chromaticity_profile import (
    load_analytic_y_chromaticity_profile as load_v1_v3_profile,
)
from src.inference.analytic_y_chromaticity_profile import (
    render_analytic_y_chromaticity_profile as render_v1_v3_profile,
)
from src.preprocess.types import WorkingImage

_HASH = re.compile(r"[0-9a-f]{64}")
_V4_IDENTITY = ("analytic-y-chromaticity-cb69-v4", "4.0.0")
_V4_ROLES = {
    "cb66_base_profile",
    "cb69_decision",
    "single_target_ao6_code",
    "throughput_renderer_code",
    "v4_adapter_code",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_analytic_y_chromaticity_profile(
    path: Path, *, root: Path
) -> AnalyticYChromaticityRuntime:
    """Load v4, delegating immutable v1-v3 profiles to their frozen adapter."""
    if not path.is_file():
        raise AnalyticYChromaticityProfileError("research profile is missing")
    profile = json.loads(path.read_text(encoding="utf-8"))
    identity = (profile.get("profile_id"), profile.get("profile_version"))
    if identity != _V4_IDENTITY:
        return load_v1_v3_profile(path, root=root)
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
    execution = profile.get("execution", {})
    if (
        set(profile) != required
        or profile.get("schema_id") != PROFILE_SCHEMA
        or profile.get("engine_id") != ENGINE_ID
        or profile.get("style") != "velvia_50"
        or profile.get("identity")
        != {
            "film_stock_id": None,
            "interpretation": "look_approximation",
            "research_champion": True,
            "product_default": False,
        }
        or execution
        != {
            "input_working_space": "linear_srgb",
            "input_color_state": "display_linear",
            "row_chunk": 128,
            "target_materializer": "parallel_ao6_row_safe_base_external_sorted_v1",
            "ao6_workers": 4,
            "effects_after_colour_allowed": True,
            "recipe_v1_allowed": False,
        }
    ):
        raise AnalyticYChromaticityProfileError("CB69 profile drift")
    assets = profile.get("assets")
    if not isinstance(assets, list) or len(assets) != len(_V4_ROLES):
        raise AnalyticYChromaticityProfileError("CB69 asset inventory drift")
    by_role = {str(asset.get("role")): asset for asset in assets}
    if len(by_role) != len(assets) or set(by_role) != _V4_ROLES:
        raise AnalyticYChromaticityProfileError("CB69 asset roles drift")
    for role, asset in by_role.items():
        expected = asset.get("sha256")
        asset_path = root / str(asset.get("path"))
        if (
            not isinstance(expected, str)
            or not _HASH.fullmatch(expected)
            or not asset_path.is_file()
            or _sha256_file(asset_path) != expected
        ):
            raise AnalyticYChromaticityProfileError(
                f"CB69 profile asset drift: {role}"
            )
    base = load_v1_v3_profile(
        root / str(by_role["cb66_base_profile"]["path"]), root=root
    )
    decision = json.loads(
        (root / str(by_role["cb69_decision"]["path"])).read_text(encoding="utf-8")
    )
    if (
        base.profile.get("profile_id") != "analytic-y-chromaticity-cb66-v3"
        or decision.get("decision")
        != "pass_cb69_output_exact_parallel_ao6_complete_throughput"
        or decision.get("automatic_pass") is not True
    ):
        raise AnalyticYChromaticityProfileError("CB69 evidence drift")
    return AnalyticYChromaticityRuntime(
        profile=profile,
        profile_sha256=_sha256_file(path),
        cb52=base.cb52,
        cb11=base.cb11,
        ao6_config=base.ao6_config,
        artifact=base.artifact,
        curve=base.curve,
    )


def render_analytic_y_chromaticity_profile(
    working: WorkingImage,
    runtime: AnalyticYChromaticityRuntime,
    *,
    scratch_root: Path | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render v4 or delegate older profiles without changing their behavior."""
    if runtime.profile.get("profile_id") != _V4_IDENTITY[0]:
        return render_v1_v3_profile(working, runtime, scratch_root=scratch_root)
    if scratch_root is not None:
        return render_analytic_y_chromaticity_throughput_candidate(
            working,
            runtime,
            scratch_root=scratch_root,
            row_chunk=128,
            ao6_workers=4,
        )
    with tempfile.TemporaryDirectory(prefix="cb69-") as temporary:
        return render_analytic_y_chromaticity_throughput_candidate(
            working,
            runtime,
            scratch_root=Path(temporary),
            row_chunk=128,
            ao6_workers=4,
        )


__all__ = [
    "ENGINE_ID",
    "load_analytic_y_chromaticity_profile",
    "render_analytic_y_chromaticity_profile",
]
