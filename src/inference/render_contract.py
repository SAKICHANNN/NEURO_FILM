"""Strict versioned profile and replay-recipe contracts for deterministic rendering."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


PROFILE_SCHEMA_ID = "kmcfm.render-profile.v1"
RECIPE_SCHEMA_ID = "kmcfm.render-recipe.v1"
PROFILE_EVIDENCE_SUMMARY_SCHEMA_ID = "kmcfm.profile-evidence-summary.v1"
LEGACY_STYLE_EVIDENCE_INVENTORY_SCHEMA_ID = (
    "kmcfm.legacy-style-evidence-inventory.v1"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_STYLE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")

COLOR_PARAMETER_KEYS = {
    "strength",
    "luma_strength",
    "grain",
    "gamut_safe",
    "gamut_mode",
    "output_margin",
    "use_guardrails",
    "chroma_curve_strength",
    "tone_rolloff",
    "shadow_floor_l",
    "highlight_ceiling_l",
    "preserve_luma_detail",
    "dither",
}


class RenderContractError(ValueError):
    """Raised when a render profile or recipe violates its frozen schema."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RenderContractError(f"{label} keys mismatch: {sorted(actual ^ expected)}")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RenderContractError(f"{label} must be an object")
    return value


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise RenderContractError(f"{label} must be a non-empty string")
    return value


def _number(value: object, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RenderContractError(f"{label} must be finite")
    result = float(value)
    if not low <= result <= high:
        raise RenderContractError(f"{label} must be in [{low}, {high}]")
    return result


def _integer(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise RenderContractError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _hash(value: object, label: str) -> str:
    result = _string(value, label)
    if not _HASH.fullmatch(result):
        raise RenderContractError(f"{label} must be a lowercase SHA-256")
    return result


def _nullable_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _hash(value, label)


def _identifier(value: object, label: str) -> str:
    result = _string(value, label)
    if not _IDENTIFIER.fullmatch(result):
        raise RenderContractError(f"{label} is not a safe identifier")
    return result


def _relative_path(value: object, label: str) -> str:
    result = _string(value, label)
    path = Path(result)
    if path.is_absolute() or ".." in path.parts or result != path.as_posix():
        raise RenderContractError(f"{label} must be a normalized repository-relative POSIX path")
    return result


def _validate_color_parameters(value: object, label: str) -> None:
    parameters = _mapping(value, label)
    _keys(parameters, COLOR_PARAMETER_KEYS, label)
    _number(parameters["strength"], f"{label}.strength", 0.0, 1.0)
    _number(parameters["luma_strength"], f"{label}.luma_strength", 0.0, 1.0)
    _number(parameters["grain"], f"{label}.grain", 0.0, 1.0)
    if not isinstance(parameters["gamut_safe"], bool):
        raise RenderContractError(f"{label}.gamut_safe must be boolean")
    if parameters["gamut_mode"] not in {"off", "source", "chroma"}:
        raise RenderContractError(f"{label}.gamut_mode is invalid")
    _integer(parameters["output_margin"], f"{label}.output_margin", 0, 32)
    if not isinstance(parameters["use_guardrails"], bool):
        raise RenderContractError(f"{label}.use_guardrails must be boolean")
    _number(parameters["chroma_curve_strength"], f"{label}.chroma_curve_strength", 0.0, 1.0)
    _number(parameters["tone_rolloff"], f"{label}.tone_rolloff", 0.0, 1.0)
    shadow = _number(parameters["shadow_floor_l"], f"{label}.shadow_floor_l", 0.0, 49.0)
    highlight = _number(parameters["highlight_ceiling_l"], f"{label}.highlight_ceiling_l", 51.0, 100.0)
    if shadow >= highlight:
        raise RenderContractError(f"{label} shadow floor must be below highlight ceiling")
    _number(parameters["preserve_luma_detail"], f"{label}.preserve_luma_detail", 0.0, 1.0)
    _number(parameters["dither"], f"{label}.dither", 0.0, 2.0)


def validate_render_profile(profile: Mapping[str, Any], *, root: Path | None = None) -> None:
    """Validate profile structure and optionally verify every referenced asset."""
    _keys(
        profile,
        {
            "schema_id",
            "profile_id",
            "profile_version",
            "display_name",
            "engine_id",
            "identity",
            "evidence",
            "assets",
            "style_parameters",
            "effect_defaults",
            "output",
            "fallback_profile_id",
        },
        "profile",
    )
    if profile["schema_id"] != PROFILE_SCHEMA_ID:
        raise RenderContractError("unsupported profile schema_id")
    profile_id = _identifier(profile["profile_id"], "profile.profile_id")
    if not _VERSION.fullmatch(_string(profile["profile_version"], "profile.profile_version")):
        raise RenderContractError("profile.profile_version must be semantic x.y.z")
    _string(profile["display_name"], "profile.display_name")
    if profile["engine_id"] != "safe_lab_v1":
        raise RenderContractError("profile.engine_id is unsupported")

    identity = _mapping(profile["identity"], "profile.identity")
    _keys(identity, {"film_stock_id", "latent_mode_id", "interpretation"}, "profile.identity")
    if identity["film_stock_id"] is not None:
        _identifier(identity["film_stock_id"], "profile.identity.film_stock_id")
    if identity["latent_mode_id"] is not None:
        if identity["film_stock_id"] is None or not re.fullmatch(r"Mode [A-Z]", str(identity["latent_mode_id"])):
            raise RenderContractError("latent mode requires a stock and legal Mode A/B/C naming")
    interpretations = {
        "look_approximation",
        "color_negative_neutral_scan",
        "color_negative_print",
        "slide_direct_scan",
        "bw_developer_scan",
    }
    if identity["interpretation"] not in interpretations:
        raise RenderContractError("profile.identity.interpretation is invalid")

    evidence = _mapping(profile["evidence"], "profile.evidence")
    _keys(
        evidence,
        {"data_grade", "expert_grade", "method", "claim_ceiling", "calibrated_reference_allowed"},
        "profile.evidence",
    )
    if evidence["data_grade"] not in {"none", "S0", "S1", "S2", "S3", "H"}:
        raise RenderContractError("profile.evidence.data_grade is invalid")
    if evidence["expert_grade"] not in {"none", "S2", "S3"}:
        raise RenderContractError("profile.evidence.expert_grade is invalid")
    if evidence["method"] not in {"heuristic", "measured", "paired", "held-out"}:
        raise RenderContractError("profile.evidence.method is invalid")
    _string(evidence["claim_ceiling"], "profile.evidence.claim_ceiling")
    if not isinstance(evidence["calibrated_reference_allowed"], bool):
        raise RenderContractError("profile.evidence.calibrated_reference_allowed must be boolean")
    if evidence["calibrated_reference_allowed"]:
        if not (
            evidence["data_grade"] == "S3"
            and evidence["expert_grade"] == "S3"
            and evidence["method"] == "held-out"
            and identity["interpretation"] != "look_approximation"
        ):
            raise RenderContractError("calibrated Reference requires held-out S3 evidence")

    assets = profile["assets"]
    if not isinstance(assets, list) or not assets:
        raise RenderContractError("profile.assets must be a non-empty list")
    roles: set[str] = set()
    for index, raw in enumerate(assets):
        asset = _mapping(raw, f"profile.assets[{index}]")
        _keys(asset, {"role", "path", "sha256"}, f"profile.assets[{index}]")
        role = _identifier(asset["role"], f"profile.assets[{index}].role")
        if role in roles:
            raise RenderContractError(f"duplicate asset role: {role}")
        roles.add(role)
        relative = _relative_path(asset["path"], f"profile.assets[{index}].path")
        expected_hash = _hash(asset["sha256"], f"profile.assets[{index}].sha256")
        if root is not None:
            path = root.resolve() / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RenderContractError(f"profile asset hash mismatch: {relative}")
    if roles != {"legacy_profile_config", "style_statistics", "color_guardrails"}:
        raise RenderContractError("profile asset roles are incomplete")

    styles = _mapping(profile["style_parameters"], "profile.style_parameters")
    if not styles:
        raise RenderContractError("profile.style_parameters cannot be empty")
    for style, parameters in styles.items():
        if not isinstance(style, str) or not _STYLE.fullmatch(style):
            raise RenderContractError(f"invalid style ID: {style}")
        _validate_color_parameters(parameters, f"profile.style_parameters.{style}")

    effects = _mapping(profile["effect_defaults"], "profile.effect_defaults")
    _keys(effects, {"grain", "halation", "dust", "halation_model"}, "profile.effect_defaults")
    for key in ("grain", "halation", "dust"):
        _number(effects[key], f"profile.effect_defaults.{key}", 0.0, 1.0)
    if effects["halation_model"] not in {"simple", "physical"}:
        raise RenderContractError("profile.effect_defaults.halation_model is invalid")

    output = _mapping(profile["output"], "profile.output")
    _keys(output, {"working_space", "transfer", "icc_profile", "supported_bit_depths"}, "profile.output")
    if output["working_space"] != "linear_srgb" or output["transfer"] != "sRGB":
        raise RenderContractError("profile output space is unsupported")
    if output["icc_profile"] != "standard_sRGB":
        raise RenderContractError("profile output ICC contract is unsupported")
    if output["supported_bit_depths"] != [8, 16]:
        raise RenderContractError("profile supported bit depths must be [8, 16]")
    if _identifier(profile["fallback_profile_id"], "profile.fallback_profile_id") != profile_id:
        raise RenderContractError("v1 safe profile must fall back to itself")


def load_render_profile(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    validate_render_profile(profile, root=root)
    return profile


def summarize_render_profile_evidence(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated, non-escalating profile evidence summary."""
    validate_render_profile(profile)
    identity = _mapping(profile["identity"], "profile.identity")
    evidence = _mapping(profile["evidence"], "profile.evidence")
    return {
        "schema_id": PROFILE_EVIDENCE_SUMMARY_SCHEMA_ID,
        "profile_id": profile["profile_id"],
        "profile_version": profile["profile_version"],
        "film_stock_id": identity["film_stock_id"],
        "latent_mode_id": identity["latent_mode_id"],
        "interpretation": identity["interpretation"],
        "data_grade": evidence["data_grade"],
        "expert_grade": evidence["expert_grade"],
        "method": evidence["method"],
        "calibrated_reference_allowed": evidence["calibrated_reference_allowed"],
        "claim_ceiling": evidence["claim_ceiling"],
    }


def summarize_legacy_style_evidence_inventory(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose legacy style names without implying stock-specific evidence."""
    validate_render_profile(profile)
    identity = _mapping(profile["identity"], "profile.identity")
    evidence = _mapping(profile["evidence"], "profile.evidence")
    if identity["film_stock_id"] is not None:
        raise RenderContractError(
            "legacy style inventory requires a stock-neutral render profile"
        )
    styles = []
    for style_id in sorted(profile["style_parameters"]):
        styles.append(
            {
                "style_id": style_id,
                "evidence_role": "legacy_named_look_proxy",
                "film_stock_id": None,
                "interpretation": identity["interpretation"],
                "data_grade": evidence["data_grade"],
                "method": evidence["method"],
                "stock_specific_operator_admitted": False,
                "target_film_closeness_established": False,
                "stock_distinguishability_established": False,
                "calibrated_reference_allowed": False,
                "claim_ceiling": evidence["claim_ceiling"],
            }
        )
    return {
        "schema_id": LEGACY_STYLE_EVIDENCE_INVENTORY_SCHEMA_ID,
        "profile_id": profile["profile_id"],
        "profile_version": profile["profile_version"],
        "style_count": len(styles),
        "styles": styles,
    }


def _asset(root: Path, path: Path, role: str) -> dict[str, str]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return {"role": role, "path": relative, "sha256": sha256_file(path)}


def migrate_legacy_safe_rich(
    profile_config: Path,
    stats: Path,
    guardrails: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    """Convert current safe-rich YAML into the exact v1 heuristic profile."""
    document = OmegaConf.to_container(OmegaConf.load(profile_config), resolve=True)
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise RenderContractError("legacy profile config schema is unsupported")
    profiles = _mapping(document.get("profiles"), "legacy.profiles")
    safe_rich = _mapping(profiles.get("safe_rich"), "legacy.profiles.safe_rich")
    defaults = dict(_mapping(safe_rich.get("defaults"), "legacy.safe_rich.defaults"))
    styles = _mapping(safe_rich.get("styles"), "legacy.safe_rich.styles")
    resolved = {str(style): {**defaults, **dict(_mapping(values, f"legacy.style.{style}"))} for style, values in styles.items()}
    profile: dict[str, Any] = {
        "schema_id": PROFILE_SCHEMA_ID,
        "profile_id": "safe-rich-v1",
        "profile_version": "1.0.0",
        "display_name": "Safe Rich v1",
        "engine_id": "safe_lab_v1",
        "identity": {"film_stock_id": None, "latent_mode_id": None, "interpretation": "look_approximation"},
        "evidence": {
            "data_grade": "none",
            "expert_grade": "none",
            "method": "heuristic",
            "claim_ceiling": "film-inspired deterministic look approximation; no stock response, calibration or authenticity claim",
            "calibrated_reference_allowed": False,
        },
        "assets": [
            _asset(root, profile_config, "legacy_profile_config"),
            _asset(root, stats, "style_statistics"),
            _asset(root, guardrails, "color_guardrails"),
        ],
        "style_parameters": resolved,
        "effect_defaults": {"grain": 0.0, "halation": 0.0, "dust": 0.0, "halation_model": "simple"},
        "output": {
            "working_space": "linear_srgb",
            "transfer": "sRGB",
            "icc_profile": "standard_sRGB",
            "supported_bit_depths": [8, 16],
        },
        "fallback_profile_id": "safe-rich-v1",
    }
    validate_render_profile(profile, root=root)
    return profile


def _validate_json_parameters(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise RenderContractError(f"{label} contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_parameters(item, f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not _IDENTIFIER.fullmatch(key):
                raise RenderContractError(f"{label} contains an unsafe parameter key")
            _validate_json_parameters(item, f"{label}.{key}")
        return
    raise RenderContractError(f"{label} contains a non-JSON value")


def validate_render_recipe(recipe: Mapping[str, Any]) -> None:
    _keys(recipe, {"schema_id", "profile", "assets", "input", "render", "output", "claim", "software"}, "recipe")
    if recipe["schema_id"] != RECIPE_SCHEMA_ID:
        raise RenderContractError("unsupported recipe schema_id")
    profile = _mapping(recipe["profile"], "recipe.profile")
    _keys(profile, {"profile_id", "profile_version", "sha256"}, "recipe.profile")
    _identifier(profile["profile_id"], "recipe.profile.profile_id")
    if not _VERSION.fullmatch(_string(profile["profile_version"], "recipe.profile.profile_version")):
        raise RenderContractError("recipe profile version is invalid")
    _hash(profile["sha256"], "recipe.profile.sha256")
    assets = recipe["assets"]
    if not isinstance(assets, list) or not assets:
        raise RenderContractError("recipe.assets must be non-empty")
    asset_roles: set[str] = set()
    for index, raw in enumerate(assets):
        asset = _mapping(raw, f"recipe.assets[{index}]")
        _keys(asset, {"role", "path", "sha256"}, f"recipe.assets[{index}]")
        role = _identifier(asset["role"], f"recipe.assets[{index}].role")
        if role in asset_roles:
            raise RenderContractError(f"duplicate recipe asset role: {role}")
        asset_roles.add(role)
        _relative_path(asset["path"], f"recipe.assets[{index}].path")
        _hash(asset["sha256"], f"recipe.assets[{index}].sha256")

    source = _mapping(recipe["input"], "recipe.input")
    _keys(
        source,
        {"path", "sha256", "color_state", "working_space", "source_profile_kind", "source_profile_fingerprint_sha256", "bit_depth", "warnings"},
        "recipe.input",
    )
    _string(source["path"], "recipe.input.path")
    _hash(source["sha256"], "recipe.input.sha256")
    if source["color_state"] not in {"scene_linear", "display_linear", "display_referred", "unknown"}:
        raise RenderContractError("recipe.input.color_state is invalid")
    if source["working_space"] != "linear_srgb":
        raise RenderContractError("recipe.input.working_space is unsupported")
    if source["source_profile_kind"] not in {"icc", "cicp", "nclx", "raw_metadata", "assumed_srgb", "unknown"}:
        raise RenderContractError("recipe.input.source_profile_kind is invalid")
    _nullable_hash(source["source_profile_fingerprint_sha256"], "recipe.input.source_profile_fingerprint_sha256")
    _integer(source["bit_depth"], "recipe.input.bit_depth", 1, 32)
    if not isinstance(source["warnings"], list):
        raise RenderContractError("recipe.input.warnings must be a list")
    for index, raw in enumerate(source["warnings"]):
        warning = _mapping(raw, f"recipe.input.warnings[{index}]")
        _keys(warning, {"code", "message"}, f"recipe.input.warnings[{index}]")
        _identifier(warning["code"], f"recipe.input.warnings[{index}].code")
        _string(warning["message"], f"recipe.input.warnings[{index}].message")

    render = _mapping(recipe["render"], "recipe.render")
    _keys(render, {"engine_id", "preset", "style", "seed", "color_parameters", "effects"}, "recipe.render")
    if render["engine_id"] != "safe_lab_v1" or render["preset"] != "safe-rich":
        raise RenderContractError("recipe render engine/preset is unsupported")
    if not isinstance(render["style"], str) or not _STYLE.fullmatch(render["style"]):
        raise RenderContractError("recipe.render.style is invalid")
    _integer(render["seed"], "recipe.render.seed", -(2**31), 2**31 - 1)
    _validate_color_parameters(render["color_parameters"], "recipe.render.color_parameters")
    effects = _mapping(render["effects"], "recipe.render.effects")
    _keys(effects, {"grain", "halation", "dust"}, "recipe.render.effects")
    grain = _mapping(effects["grain"], "recipe.render.effects.grain")
    _keys(grain, {"strength", "seed", "color"}, "recipe.render.effects.grain")
    _number(grain["strength"], "recipe.render.effects.grain.strength", 0.0, 1.0)
    _integer(grain["seed"], "recipe.render.effects.grain.seed", -(2**31), 2**31 - 1)
    if not isinstance(grain["color"], bool):
        raise RenderContractError("recipe.render.effects.grain.color must be boolean")
    halation = _mapping(effects["halation"], "recipe.render.effects.halation")
    _keys(halation, {"strength", "model", "preset", "control_mode", "resolved_parameters"}, "recipe.render.effects.halation")
    _number(halation["strength"], "recipe.render.effects.halation.strength", 0.0, 1.0)
    if halation["model"] not in {"simple", "physical"} or halation["control_mode"] not in {"locked", "expert"}:
        raise RenderContractError("recipe halation model/control mode is invalid")
    if halation["preset"] is not None:
        _identifier(halation["preset"], "recipe.render.effects.halation.preset")
    _validate_json_parameters(halation["resolved_parameters"], "recipe.render.effects.halation.resolved_parameters")
    dust = _mapping(effects["dust"], "recipe.render.effects.dust")
    _keys(dust, {"strength", "seed"}, "recipe.render.effects.dust")
    _number(dust["strength"], "recipe.render.effects.dust.strength", 0.0, 1.0)
    _integer(dust["seed"], "recipe.render.effects.dust.seed", -(2**31), 2**31 - 1)

    output = _mapping(recipe["output"], "recipe.output")
    _keys(output, {"path", "sha256", "format", "bit_depth", "transfer", "icc_profile_fingerprint_sha256"}, "recipe.output")
    _string(output["path"], "recipe.output.path")
    _hash(output["sha256"], "recipe.output.sha256")
    if output["format"] not in {"PNG", "JPEG", "TIFF"}:
        raise RenderContractError("recipe.output.format is invalid")
    if output["bit_depth"] not in {8, 16} or output["transfer"] != "sRGB":
        raise RenderContractError("recipe output encoding is unsupported")
    _hash(output["icc_profile_fingerprint_sha256"], "recipe.output.icc_profile_fingerprint_sha256")

    claim = _mapping(recipe["claim"], "recipe.claim")
    _keys(
        claim,
        {"render_mode", "output_label", "evidence_grade", "input_color_state", "color_state_policy", "calibrated_reference_allowed", "claim_ceiling"},
        "recipe.claim",
    )
    if claim["render_mode"] != "Style-safe" or claim["output_label"] != "film-inspired" or claim["evidence_grade"] != "look-approximation":
        raise RenderContractError("recipe claim exceeds the current renderer")
    if claim["input_color_state"] != source["color_state"]:
        raise RenderContractError("recipe claim/input color state mismatch")
    expected_policy = "look_approximation_fail_closed" if source["color_state"] == "unknown" else "look_approximation_only"
    if claim["color_state_policy"] != expected_policy:
        raise RenderContractError("recipe color-state policy mismatch")
    if claim["calibrated_reference_allowed"] is not False:
        raise RenderContractError("current recipe cannot allow calibrated Reference")
    _string(claim["claim_ceiling"], "recipe.claim.claim_ceiling")
    software = _mapping(recipe["software"], "recipe.software")
    _keys(software, {"commit"}, "recipe.software")
    if not _COMMIT.fullmatch(_string(software["commit"], "recipe.software.commit")):
        raise RenderContractError("recipe software commit must be a full Git SHA")


def build_render_recipe(
    *,
    profile_path: Path,
    profile: Mapping[str, Any],
    input_path: Path,
    input_metadata: Mapping[str, Any],
    render_metadata: Mapping[str, Any],
    output_path: Path,
    output_format: str,
    output_bit_depth: int,
    output_icc_fingerprint_sha256: str,
    output_claim: Mapping[str, Any],
    software_commit: str,
) -> dict[str, Any]:
    """Build and validate one post-encode replay recipe."""
    validate_render_profile(profile)
    if render_metadata.get("style") not in profile["style_parameters"]:
        raise RenderContractError("recipe style is absent from the selected profile")
    claim = dict(output_claim)
    claim["claim_ceiling"] = profile["evidence"]["claim_ceiling"]
    recipe = {
        "schema_id": RECIPE_SCHEMA_ID,
        "profile": {
            "profile_id": profile["profile_id"],
            "profile_version": profile["profile_version"],
            "sha256": sha256_file(profile_path),
        },
        "assets": [dict(value) for value in profile["assets"]],
        "input": {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            **dict(input_metadata),
        },
        "render": dict(render_metadata),
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
    validate_render_recipe(recipe)
    return recipe


def verify_render_recipe_inputs(
    recipe: Mapping[str, Any], *, profile_path: Path, root: Path
) -> None:
    """Verify a recipe against its immutable profile, assets and input file."""
    validate_render_recipe(recipe)
    if not profile_path.is_file() or sha256_file(profile_path) != recipe["profile"]["sha256"]:
        raise RenderContractError("recipe profile file hash mismatch")
    profile = load_render_profile(profile_path, root=root)
    if recipe["profile"]["profile_id"] != profile["profile_id"] or recipe["profile"]["profile_version"] != profile["profile_version"]:
        raise RenderContractError("recipe profile identity mismatch")
    if recipe["assets"] != profile["assets"]:
        raise RenderContractError("recipe asset ledger differs from profile")
    path = Path(str(recipe["input"]["path"]))
    if not path.is_file() or sha256_file(path) != recipe["input"]["sha256"]:
        raise RenderContractError("recipe input file hash mismatch")


def verify_render_recipe_files(recipe: Mapping[str, Any], *, profile_path: Path, root: Path) -> None:
    """Verify a recipe against its immutable profile, assets and local I/O files."""
    verify_render_recipe_inputs(recipe, profile_path=profile_path, root=root)
    path = Path(str(recipe["output"]["path"]))
    if not path.is_file() or sha256_file(path) != recipe["output"]["sha256"]:
        raise RenderContractError("recipe output file hash mismatch")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()
