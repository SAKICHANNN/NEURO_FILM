"""Strict post-encode recipe for the opt-in analytic colour renderer."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.inference.analytic_y_chromaticity_profile import (
    ENGINE_ID,
    AnalyticYChromaticityRuntime,
    load_analytic_y_chromaticity_profile,
)
from src.inference.render_contract import sha256_file

SCHEMA_ID = "kmcfm.analytic-render-recipe.v1"
_HASH = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SELECTOR_KEYS = {
    "characteristic_strength",
    "global_dose",
    "selected_gradient_ratio",
    "selected_lstar_inversion_fraction",
    "selected_new_boundary_fraction",
    "median_gamut_scale",
    "fraction_gamut_scale_below_0p8",
    "maximum_luminance_error",
}


class AnalyticRenderRecipeError(ValueError):
    pass


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise AnalyticRenderRecipeError(f"{label} keys drift")


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise AnalyticRenderRecipeError(f"{label} is not SHA-256")
    return value


def _finite_tree(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise AnalyticRenderRecipeError(f"{label} is non-finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AnalyticRenderRecipeError(f"{label} key is not text")
            _finite_tree(item, f"{label}.{key}")
        return
    raise AnalyticRenderRecipeError(f"{label} is not JSON-compatible")


def validate_analytic_render_recipe(recipe: Mapping[str, Any]) -> None:
    _exact_keys(
        recipe,
        {
            "schema_id",
            "profile",
            "assets",
            "input",
            "render",
            "output",
            "claim",
            "software",
        },
        "recipe",
    )
    if recipe["schema_id"] != SCHEMA_ID:
        raise AnalyticRenderRecipeError("unsupported analytic recipe schema")
    profile = recipe["profile"]
    _exact_keys(profile, {"profile_id", "profile_version", "sha256"}, "profile")
    profile_identity = (profile["profile_id"], profile["profile_version"])
    if profile_identity not in {
        ("analytic-y-chromaticity-cb56-v1", "1.0.0"),
        ("analytic-y-chromaticity-cb61-v2", "2.0.0"),
    }:
        raise AnalyticRenderRecipeError("analytic profile identity drift")
    _hash(profile["sha256"], "profile.sha256")
    assets = recipe["assets"]
    expected_assets = 10 if profile_identity[0].endswith("cb61-v2") else 8
    if not isinstance(assets, list) or len(assets) != expected_assets:
        raise AnalyticRenderRecipeError("analytic recipe asset inventory drift")
    roles: set[str] = set()
    for asset in assets:
        _exact_keys(asset, {"role", "path", "sha256"}, "asset")
        if not isinstance(asset["role"], str) or asset["role"] in roles:
            raise AnalyticRenderRecipeError("analytic recipe asset role drift")
        roles.add(asset["role"])
        if not isinstance(asset["path"], str) or Path(asset["path"]).is_absolute():
            raise AnalyticRenderRecipeError("analytic recipe asset path drift")
        _hash(asset["sha256"], "asset.sha256")
    source = recipe["input"]
    _exact_keys(
        source,
        {
            "path",
            "sha256",
            "source_color_state",
            "runtime_transfer_state",
            "working_space",
            "source_profile_kind",
            "bit_depth",
            "warnings",
        },
        "input",
    )
    if (
        not isinstance(source["path"], str)
        or source["source_color_state"] not in {"display_referred", "display_linear"}
        or source["runtime_transfer_state"] != "display_linear"
        or source["working_space"] != "linear_srgb"
        or not isinstance(source["warnings"], list)
    ):
        raise AnalyticRenderRecipeError("analytic recipe input drift")
    _hash(source["sha256"], "input.sha256")
    render = recipe["render"]
    _exact_keys(render, {"engine_id", "style", "selector_facts", "effects"}, "render")
    if render["engine_id"] != ENGINE_ID or render["style"] != "velvia_50":
        raise AnalyticRenderRecipeError("analytic recipe engine/style drift")
    if (
        not isinstance(render["selector_facts"], Mapping)
        or set(render["selector_facts"]) != _SELECTOR_KEYS
    ):
        raise AnalyticRenderRecipeError("analytic selector facts drift")
    _finite_tree(render["selector_facts"], "selector_facts")
    _finite_tree(render["effects"], "effects")
    output = recipe["output"]
    _exact_keys(
        output,
        {
            "path",
            "sha256",
            "format",
            "bit_depth",
            "transfer",
            "icc_profile_fingerprint_sha256",
        },
        "output",
    )
    if (
        not isinstance(output["path"], str)
        or output["format"] not in {"PNG", "JPEG", "TIFF"}
        or output["bit_depth"] not in {8, 16}
        or output["transfer"] != "sRGB"
    ):
        raise AnalyticRenderRecipeError("analytic recipe output drift")
    _hash(output["sha256"], "output.sha256")
    _hash(output["icc_profile_fingerprint_sha256"], "output.icc")
    claim = recipe["claim"]
    _exact_keys(
        claim,
        {
            "render_mode",
            "output_label",
            "evidence_grade",
            "input_color_state",
            "color_state_policy",
            "calibrated_reference_allowed",
            "claim_ceiling",
        },
        "claim",
    )
    if (
        claim["render_mode"] != "Style-safe"
        or claim["output_label"] != "film-inspired"
        or claim["evidence_grade"] != "look-approximation"
        or claim["input_color_state"] != source["source_color_state"]
        or claim["color_state_policy"] != "look_approximation_only"
        or claim["calibrated_reference_allowed"] is not False
    ):
        raise AnalyticRenderRecipeError("analytic recipe claim escalation")
    software = recipe["software"]
    _exact_keys(software, {"commit"}, "software")
    if not isinstance(software["commit"], str) or not _COMMIT.fullmatch(
        software["commit"]
    ):
        raise AnalyticRenderRecipeError("analytic recipe commit drift")


def build_analytic_render_recipe(
    *,
    runtime: AnalyticYChromaticityRuntime,
    input_path: Path,
    input_metadata: Mapping[str, Any],
    selector_facts: Mapping[str, Any],
    effects: Mapping[str, Any],
    output_path: Path,
    output_format: str,
    output_bit_depth: int,
    output_icc_fingerprint_sha256: str,
    output_claim: Mapping[str, Any],
    software_commit: str,
) -> dict[str, Any]:
    claim = dict(output_claim)
    claim["claim_ceiling"] = runtime.profile["evidence"]["claim_ceiling"]
    recipe = {
        "schema_id": SCHEMA_ID,
        "profile": {
            "profile_id": runtime.profile["profile_id"],
            "profile_version": runtime.profile["profile_version"],
            "sha256": runtime.profile_sha256,
        },
        "assets": [dict(asset) for asset in runtime.profile["assets"]],
        "input": {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            **dict(input_metadata),
        },
        "render": {
            "engine_id": ENGINE_ID,
            "style": runtime.profile["style"],
            "selector_facts": dict(selector_facts),
            "effects": dict(effects),
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "format": output_format,
            "bit_depth": output_bit_depth,
            "transfer": "sRGB",
            "icc_profile_fingerprint_sha256": output_icc_fingerprint_sha256,
        },
        "claim": claim,
        "software": {"commit": software_commit},
    }
    validate_analytic_render_recipe(recipe)
    return recipe


def verify_analytic_render_recipe_files(
    recipe: Mapping[str, Any], *, profile_path: Path, root: Path
) -> None:
    validate_analytic_render_recipe(recipe)
    runtime = load_analytic_y_chromaticity_profile(profile_path, root=root)
    if runtime.profile_sha256 != recipe["profile"]["sha256"]:
        raise AnalyticRenderRecipeError("analytic recipe profile file drift")
    if runtime.profile["assets"] != recipe["assets"]:
        raise AnalyticRenderRecipeError("analytic recipe asset ledger drift")
    for side in ("input", "output"):
        path = Path(recipe[side]["path"])
        if not path.is_file() or sha256_file(path) != recipe[side]["sha256"]:
            raise AnalyticRenderRecipeError(f"analytic recipe {side} file drift")


__all__ = [
    "SCHEMA_ID",
    "AnalyticRenderRecipeError",
    "build_analytic_render_recipe",
    "validate_analytic_render_recipe",
    "verify_analytic_render_recipe_files",
]
