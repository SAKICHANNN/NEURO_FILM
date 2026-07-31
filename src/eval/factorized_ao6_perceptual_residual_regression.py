"""BK16 synthetic gates and frozen BK2 severe-failure regression."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.color import rgb2lab

from src.eval.density_witness_frontier import encoded_srgb_to_linear
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.global_frontier import sha256_file
from src.eval.log_chroma_fresh_comparison import _safe_rich
from src.film_physics.profile_consumer import compile_standalone_profile_artifact
from src.preprocess import save_srgb16_png
from src.roll2film.factorized_ao6_perceptual_residual import (
    FactorizedAO6PerceptualResidual,
)


SCHEMA = "neuro_film.u5_r2bk16_factorized_ao6_perceptual_residual_report.v1"
AO6_ARM = "fixed_ao6_colour_only_t15_c35"


class FactorizedAO6ResidualRegressionError(RuntimeError):
    """Raised when a frozen BK16 identity or invariant drifts."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_exact(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise FactorizedAO6ResidualRegressionError(f"identity drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _operator(config: Mapping[str, Any]) -> FactorizedAO6PerceptualResidual:
    params = config["operator"]
    return FactorizedAO6PerceptualResidual(
        residual_strength=float(params["residual_strength"]),
        lightness_soft_cap=float(params["lightness_soft_cap"]),
        chroma_soft_cap=float(params["chroma_soft_cap"]),
        maximum_final_delta_e76=float(params["maximum_final_delta_e76"]),
        minimum_lightness=float(params["minimum_lightness"]),
        maximum_lightness=float(params["maximum_lightness"]),
        residual_epsilon=float(params["residual_epsilon"]),
        gamut_iterations=int(params["gamut_iterations"]),
        encoded_margin=float(params["encoded_margin"]),
        row_chunk=int(params["row_chunk"]),
        numeric_gamut_tolerance=float(params["numeric_gamut_tolerance"]),
        perceptual_tolerance=float(params["perceptual_tolerance"]),
    )


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    operator = _operator(config)
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk16_factorized_ao6_perceptual_residual.v1"
        or config.get("status") != "primitive_contract_frozen"
        or operator != FactorizedAO6PerceptualResidual()
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("production_default_changed")
        or config.get("stock_or_authenticity_claim_allowed")
    ):
        raise FactorizedAO6ResidualRegressionError("BK16 contract drift")
    parent = config["parent"]
    decision = _load_exact(
        root, parent["decision"], parent["decision_sha256"]
    )
    if decision.get("decision") != parent["required_decision"]:
        raise FactorizedAO6ResidualRegressionError("BK16 branch is not open")

    fixed = config["fixed_inputs"]
    base_binding = fixed["base"]
    profile = _load_exact(
        root, base_binding["profile"], base_binding["profile_sha256"]
    )
    statistics = _load_exact(
        root,
        base_binding["style_statistics"],
        base_binding["style_statistics_sha256"],
    )
    guardrails = _load_exact(
        root,
        base_binding["guardrails"],
        base_binding["guardrails_sha256"],
    )
    style = str(base_binding["style"])
    if (
        profile.get("profile_id") != "safe-rich-v1"
        or style not in profile["style_parameters"]
        or style not in statistics["styles"]
        or style not in guardrails["styles"]
    ):
        raise FactorizedAO6ResidualRegressionError("safe-rich base drift")

    ao6_binding = fixed["ao6_candidate"]
    ao6_config = _load_exact(
        root,
        ao6_binding["profile_compiler_config"],
        ao6_binding["profile_compiler_config_sha256"],
    )
    severe = fixed["known_severe_regression"]
    severe_path = root / str(severe["source"])
    if sha256_file(severe_path) != severe["source_sha256"]:
        raise FactorizedAO6ResidualRegressionError(
            "known severe source drift"
        )
    return {
        "operator": operator,
        "ao6_config": ao6_config,
        "ao6_component": str(ao6_binding["component"]),
        "safe": {
            "safe_profile": profile["style_parameters"][style],
            "safe_statistics": statistics["styles"][style],
            "safe_guardrails": {
                **guardrails["defaults"],
                **guardrails["styles"][style],
            },
            "safe_style": style,
            "safe_seed": int(base_binding["seed"]),
        },
        "source_path": severe_path,
        "source_sha256": str(severe["source_sha256"]),
    }


def _cube() -> np.ndarray:
    axis = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    red, green, blue = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((red, green, blue), axis=-1).reshape(-1, 17, 3)


def _neutral_ramp() -> np.ndarray:
    axis = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    return np.repeat(axis[:, None], 3, axis=1).reshape(1, -1, 3)


def _style_delta(source: np.ndarray, output: np.ndarray) -> np.ndarray:
    return np.linalg.norm(rgb2lab(output) - rgb2lab(source), axis=-1)


def _new_boundary_fraction(source: np.ndarray, output: np.ndarray) -> float:
    source_code = np.rint(source * 65535.0).astype(np.uint16)
    output_code = np.rint(output * 65535.0).astype(np.uint16)
    source_boundary = np.any(
        (source_code == 0) | (source_code == 65535), axis=-1
    )
    output_boundary = np.any(
        (output_code == 0) | (output_code == 65535), axis=-1
    )
    return float(np.mean(output_boundary & ~source_boundary))


def _render(
    source: np.ndarray,
    *,
    validated: Mapping[str, Any],
    ao6_artifact: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _safe_rich(source, validated["safe"]).astype(np.float32)
    ao6 = render_fixed_pair(
        encoded_srgb_to_linear(source).astype(np.float32),
        ao6_artifact,
        validated["ao6_component"],
    )[AO6_ARM].astype(np.float32)
    output = validated["operator"].apply(source, base, ao6)
    return base, ao6, output


def _contact_sheet(
    paths: list[tuple[str, Path]], output_path: Path
) -> None:
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (1600, 650), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(paths):
        with Image.open(path) as opened:
            image = ImageOps.contain(
                ImageOps.exif_transpose(opened).convert("RGB"),
                (390, 600),
                method=Image.Resampling.LANCZOS,
            )
        x = index * 400 + (390 - image.width) // 2
        sheet.paste(image, (x, 30 + (600 - image.height) // 2))
        draw.text((index * 400 + 5, 8), label, fill="black", font=font)
    sheet.save(output_path, "PNG")


def run_regression(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    if output_dir.exists():
        raise FileExistsError("BK16 regression is create-only")
    output_dir.mkdir(parents=True)
    artifact = compile_standalone_profile_artifact(
        root=root, config=validated["ao6_config"]
    )
    operator = validated["operator"]

    cube = _cube()
    cube_base, cube_ao6, cube_output = _render(
        cube, validated=validated, ao6_artifact=artifact
    )
    cube_style = _style_delta(cube, cube_output)
    cube_increment = _style_delta(cube_base, cube_output)
    _, factorized = operator._factorized_residual(
        cube, cube_base, cube_ao6
    )

    neutral = _neutral_ramp()
    neutral_base, _, neutral_output = _render(
        neutral, validated=validated, ao6_artifact=artifact
    )
    neutral_lab = rgb2lab(neutral_output)[0]
    neutral_base_lab = rgb2lab(neutral_base)[0]
    minimum_lightness_step = float(np.min(np.diff(neutral_lab[:, 0])))
    maximum_neutral_chroma = float(
        np.max(np.linalg.norm(neutral_lab[:, 1:], axis=-1))
    )
    maximum_base_neutral_chroma = float(
        np.max(np.linalg.norm(neutral_base_lab[:, 1:], axis=-1))
    )

    with Image.open(validated["source_path"]) as opened:
        source = np.asarray(
            ImageOps.exif_transpose(opened).convert("RGB"),
            dtype=np.float32,
        )
    source /= 255.0
    base, ao6, output = _render(
        source, validated=validated, ao6_artifact=artifact
    )
    paths = {
        "source": output_dir / "source_phaseone_p25plus.png",
        "base": output_dir / "safe_rich_phaseone_p25plus.png",
        "bk16": output_dir / "bk16_phaseone_p25plus.png",
        "ao6": output_dir / "ao6_phaseone_p25plus.png",
    }
    for name, pixels in (
        ("source", source),
        ("base", base),
        ("bk16", output),
        ("ao6", ao6),
    ):
        save_srgb16_png(pixels, paths[name])

    gates = config["primitive_gates"]
    maximum_increment = float(np.max(cube_increment))
    final_cap_fraction = float(
        np.mean(
            cube_increment
            > operator.maximum_final_delta_e76
            - 10.0 * operator.perceptual_tolerance
        )
    )
    automatic_gates = {
        "identity_strength_zero_returns_base_exact": np.array_equal(
            operator.apply(cube, cube_base, cube_ao6, strength=0.0),
            cube_base,
        ),
        "black_white_source_endpoints_exact": bool(
            np.array_equal(cube_output[0, 0, 0], cube[0, 0, 0])
            and np.array_equal(cube_output[-1, -1, -1], cube[-1, -1, -1])
        ),
        "partition_output_exact_given_fixed_inputs": np.array_equal(
            np.concatenate(
                (
                    operator.apply(
                        cube[:120], cube_base[:120], cube_ao6[:120]
                    ),
                    operator.apply(
                        cube[120:], cube_base[120:], cube_ao6[120:]
                    ),
                ),
                axis=0,
            ),
            cube_output,
        ),
        "cube_style": float(np.median(cube_style))
        >= float(gates["minimum_17_cube_median_style_delta_e76"]),
        "cube_increment": float(np.median(cube_increment))
        >= float(
            gates["minimum_17_cube_median_increment_vs_base_delta_e76"]
        ),
        "final_increment_bound": maximum_increment
        <= float(gates["maximum_final_increment_vs_base_delta_e76"]),
        "final_cap_fraction": final_cap_fraction
        <= float(gates["maximum_17_cube_final_cap_fraction"]),
        "soft_lightness_bound": float(
            np.max(np.abs(factorized[..., 0]))
        )
        <= float(gates["maximum_soft_lightness_residual"]),
        "soft_chroma_bound": float(
            np.max(np.linalg.norm(factorized[..., 1:], axis=-1))
        )
        <= float(gates["maximum_soft_chroma_residual"]),
        "new_boundary": _new_boundary_fraction(cube, cube_output)
        == float(gates["new_uint16_boundary_fraction_on_17_cube"]),
        "neutral_lightness_monotone": minimum_lightness_step >= 0.0,
        "neutral_chroma_not_increased": maximum_neutral_chroma
        - maximum_base_neutral_chroma
        <= float(gates["maximum_neutral_chroma_increase_vs_base"]),
        "phaseone_new_boundary": _new_boundary_fraction(source, output)
        <= float(gates["maximum_phaseone_new_uint16_boundary_fraction"]),
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "source_sha256": validated["source_sha256"],
        "ao6_bundle_sha256": artifact["bundle_sha256"],
        "outputs": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "cube_median_style_delta_e76": float(np.median(cube_style)),
        "cube_p95_style_delta_e76": float(np.percentile(cube_style, 95)),
        "cube_median_increment_vs_base_delta_e76": float(
            np.median(cube_increment)
        ),
        "cube_p95_increment_vs_base_delta_e76": float(
            np.percentile(cube_increment, 95)
        ),
        "cube_maximum_increment_vs_base_delta_e76": maximum_increment,
        "cube_final_cap_fraction": final_cap_fraction,
        "maximum_soft_lightness_residual": float(
            np.max(np.abs(factorized[..., 0]))
        ),
        "maximum_soft_chroma_residual": float(
            np.max(np.linalg.norm(factorized[..., 1:], axis=-1))
        ),
        "cube_new_uint16_boundary_fraction": _new_boundary_fraction(
            cube, cube_output
        ),
        "neutral_ramp_minimum_lightness_step": minimum_lightness_step,
        "neutral_ramp_maximum_chroma": maximum_neutral_chroma,
        "safe_base_neutral_ramp_maximum_chroma": (
            maximum_base_neutral_chroma
        ),
        "phaseone_style": {
            "safe_rich_median_delta_e76": float(
                np.median(_style_delta(source, base))
            ),
            "bk16_median_delta_e76": float(
                np.median(_style_delta(source, output))
            ),
            "ao6_median_delta_e76": float(
                np.median(_style_delta(source, ao6))
            ),
            "bk16_median_increment_vs_base_delta_e76": float(
                np.median(_style_delta(base, output))
            ),
            "bk16_maximum_increment_vs_base_delta_e76": float(
                np.max(_style_delta(base, output))
            ),
            "bk16_new_boundary_fraction": _new_boundary_fraction(
                source, output
            ),
        },
        "automatic_gates": automatic_gates,
        "automatic_pass": all(automatic_gates.values()),
        "visual_review_allowed": all(automatic_gates.values()),
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sheet_path = output_dir / "visual_regression.png"
    if report["visual_review_allowed"]:
        _contact_sheet(
            [(name.upper(), paths[name]) for name in ("source", "base", "bk16", "ao6")],
            sheet_path,
        )
    return {
        "report": report,
        "report_sha256": sha256_file(report_path),
        "visual_sheet": (
            {
                "path": sheet_path.relative_to(root).as_posix(),
                "sha256": sha256_file(sheet_path),
            }
            if sheet_path.is_file()
            else None
        ),
    }


__all__ = [
    "FactorizedAO6ResidualRegressionError",
    "run_regression",
    "validate_contract",
]
