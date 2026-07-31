"""BK10 synthetic gates and frozen BK2 severe-failure regression."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.color import rgb2lab

from src.eval.global_frontier import sha256_file
from src.eval.log_chroma_fresh_comparison import _safe_rich
from src.preprocess import save_srgb16_png
from src.roll2film.encoded_safe_log_chroma import (
    EncodedSafeLogChromaFilmResponse,
)
from src.roll2film.orthogonal_perceptual_residual import (
    OrthogonalPerceptualResidual,
)
from src.roll2film.smooth_perceptual_hue_density import (
    SmoothPerceptualHueDensityResponse,
)


SCHEMA = "neuro_film.u5_r2bk10_orthogonal_residual_regression_report.v1"


class OrthogonalResidualRegressionError(RuntimeError):
    """Raised when the frozen BK10 regression contract drifts."""


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
        raise OrthogonalResidualRegressionError(f"identity drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _operator(config: Mapping[str, Any]) -> OrthogonalPerceptualResidual:
    params = config["operator"]
    return OrthogonalPerceptualResidual(
        residual_strength=float(params["residual_strength"]),
        maximum_orthogonal_delta_e76=float(
            params["maximum_orthogonal_delta_e76"]
        ),
        minimum_lightness=float(params["minimum_lightness"]),
        maximum_lightness=float(params["maximum_lightness"]),
        projection_epsilon=float(params["projection_epsilon"]),
        gamut_iterations=int(params["gamut_iterations"]),
        encoded_margin=float(params["encoded_margin"]),
        row_chunk=int(params["row_chunk"]),
    )


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    operator = _operator(config)
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk10_orthogonal_perceptual_residual.v1"
        or config.get("status") != "primitive_contract_frozen"
        or operator != OrthogonalPerceptualResidual()
    ):
        raise OrthogonalResidualRegressionError("BK10 contract drift")
    parent = config["parent"]
    decision = _load_exact(
        root, parent["decision"], parent["decision_sha256"]
    )
    if decision.get("decision") != parent["required_decision"]:
        raise OrthogonalResidualRegressionError("BK10 branch is not open")

    fixed = config["fixed_inputs"]
    profile_binding = fixed["base"]
    profile = _load_exact(
        root,
        profile_binding["profile"],
        profile_binding["profile_sha256"],
    )
    assets = {row["role"]: row for row in profile["assets"]}
    statistics = _load_exact(
        root,
        assets["style_statistics"]["path"],
        assets["style_statistics"]["sha256"],
    )
    guardrails = _load_exact(
        root,
        assets["color_guardrails"]["path"],
        assets["color_guardrails"]["sha256"],
    )
    style = str(profile_binding["style"])
    if (
        profile.get("profile_id") != "safe-rich-v1"
        or style not in profile["style_parameters"]
        or style not in statistics["styles"]
        or style not in guardrails["styles"]
    ):
        raise OrthogonalResidualRegressionError("safe-rich base drift")

    residual_binding = fixed["residual_source"]
    residual_config = _load_exact(
        root,
        residual_binding["config"],
        residual_binding["config_sha256"],
    )
    if residual_config.get("status") != "primitive_contract_frozen":
        raise OrthogonalResidualRegressionError("BK7 residual source drift")

    bk3 = json.loads(
        (
            root
            / "configs/u5_r2bk3_encoded_log_chroma_fresh_comparison_decision_v1.json"
        ).read_text(encoding="utf-8")
    )
    severe = bk3["result"]["confirmed_severe_rows"]
    if (
        bk3.get("decision") != "reject_fixed_bk2_severe_artifact"
        or len(severe) != 1
        or severe[0]["source_id"] != "phaseone_p25plus"
    ):
        raise OrthogonalResidualRegressionError("BK2 regression drift")
    source_path = (
        root
        / "outputs/u5_r2bk3s_encoded_log_chroma_fresh_source_v1/inputs"
        / "phaseone_p25plus.png"
    )
    if sha256_file(source_path) != severe[0]["source_sha256"]:
        raise OrthogonalResidualRegressionError("regression source drift")
    return {
        "operator": operator,
        "bk7": SmoothPerceptualHueDensityResponse(),
        "safe": {
            "safe_profile": profile["style_parameters"][style],
            "safe_statistics": statistics["styles"][style],
            "safe_guardrails": {
                **guardrails["defaults"],
                **guardrails["styles"][style],
            },
            "safe_style": style,
            "safe_seed": int(profile_binding["seed"]),
        },
        "source_path": source_path,
        "source_sha256": severe[0]["source_sha256"],
    }


def _cube() -> np.ndarray:
    axis = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    red, green, blue = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((red, green, blue), axis=-1).reshape(-1, 17, 3)


def _style_delta(source: np.ndarray, output: np.ndarray) -> float:
    delta = rgb2lab(output) - rgb2lab(source)
    return float(np.median(np.linalg.norm(delta, axis=-1)))


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
    source: np.ndarray, validated: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _safe_rich(source, validated["safe"]).astype(np.float32)
    bk7 = validated["bk7"].apply(source)
    bk10 = validated["operator"].apply(source, base, bk7)
    return base, bk7, bk10


def _contact_sheet(
    *,
    source_path: Path,
    bk2_path: Path,
    base_path: Path,
    bk10_path: Path,
    output_path: Path,
) -> None:
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (1920, 560), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (5, 6),
        "U5.R2BK10 frozen BK2 severe-failure regression",
        fill="black",
        font=font,
    )
    rows = (
        (source_path, "SOURCE"),
        (bk2_path, "BK2 REJECTED"),
        (base_path, "SAFE-RICH BASE"),
        (bk10_path, "BK10"),
    )
    for index, (path, label) in enumerate(rows):
        with Image.open(path) as opened:
            image = ImageOps.contain(
                ImageOps.exif_transpose(opened).convert("RGB"),
                (470, 510),
                method=Image.Resampling.LANCZOS,
            )
        x = index * 480 + (470 - image.width) // 2
        sheet.paste(image, (x, 36 + (510 - image.height) // 2))
        draw.text((index * 480 + 5, 22), label, fill="black", font=font)
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
        raise FileExistsError("BK10 regression is create-only")
    output_dir.mkdir(parents=True)
    operator = validated["operator"]

    cube = _cube()
    cube_base, cube_bk7, cube_output = _render(cube, validated)
    cube_style = _style_delta(cube, cube_output)
    cube_increment = _style_delta(cube_base, cube_output)
    cube_boundary = _new_boundary_fraction(cube, cube_output)
    _, residual = operator._orthogonal_lab_residual(
        cube, cube_base, cube_bk7
    )
    maximum_increment = float(
        np.max(
            np.linalg.norm(
                operator.residual_strength * residual,
                axis=-1,
            )
        )
    )

    with Image.open(validated["source_path"]) as opened:
        source = np.asarray(
            ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.float32
        )
    source /= 255.0
    bk2 = EncodedSafeLogChromaFilmResponse().apply(source)
    base, _, bk10 = _render(source, validated)
    paths = {
        "bk2": output_dir / "bk2_phaseone_p25plus.png",
        "base": output_dir / "safe_rich_phaseone_p25plus.png",
        "bk10": output_dir / "bk10_phaseone_p25plus.png",
    }
    save_srgb16_png(bk2, paths["bk2"])
    save_srgb16_png(base, paths["base"])
    save_srgb16_png(bk10, paths["bk10"])

    gates = config["primitive_gates"]
    automatic_gates = {
        "identity_strength_zero_returns_base_exact": np.array_equal(
            operator.apply(cube, cube_base, cube_bk7, strength=0.0),
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
                        cube[:8], cube_base[:8], cube_bk7[:8]
                    ),
                    operator.apply(
                        cube[8:], cube_base[8:], cube_bk7[8:]
                    ),
                ),
                axis=0,
            ),
            cube_output,
        ),
        "cube_style": cube_style
        >= float(gates["minimum_17_cube_median_style_delta_e76"]),
        "cube_increment": cube_increment
        >= float(
            gates["minimum_17_cube_median_increment_vs_base_delta_e76"]
        ),
        "bounded_increment": maximum_increment
        <= float(gates["maximum_increment_before_gamut_delta_e76"]),
        "new_boundary": cube_boundary
        == float(gates["new_uint16_boundary_fraction_on_17_cube"]),
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "source_sha256": validated["source_sha256"],
        "outputs": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "cube_median_style_delta_e76": cube_style,
        "cube_median_increment_vs_base_delta_e76": cube_increment,
        "maximum_increment_before_gamut_delta_e76": maximum_increment,
        "cube_new_uint16_boundary_fraction": cube_boundary,
        "bk2_style_delta_e76": _style_delta(source, bk2),
        "safe_rich_style_delta_e76": _style_delta(source, base),
        "bk10_style_delta_e76": _style_delta(source, bk10),
        "bk2_new_boundary_fraction": _new_boundary_fraction(source, bk2),
        "bk10_new_boundary_fraction": _new_boundary_fraction(source, bk10),
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
            source_path=validated["source_path"],
            bk2_path=paths["bk2"],
            base_path=paths["base"],
            bk10_path=paths["bk10"],
            output_path=sheet_path,
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
    "OrthogonalResidualRegressionError",
    "run_regression",
    "validate_contract",
]
