"""BK7 primitive gates and frozen BK2 severe-failure regression."""

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
from src.preprocess import save_srgb16_png
from src.roll2film.encoded_safe_log_chroma import (
    EncodedSafeLogChromaFilmResponse,
)
from src.roll2film.smooth_perceptual_hue_density import (
    SmoothPerceptualHueDensityResponse,
)


SCHEMA = (
    "neuro_film.u5_r2bk7_smooth_perceptual_hue_density_regression_report.v1"
)


class SmoothPerceptualRegressionError(RuntimeError):
    """Raised when the frozen BK7 regression contract drifts."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _operator(
    config: Mapping[str, Any],
) -> SmoothPerceptualHueDensityResponse:
    params = config["operator"]
    return SmoothPerceptualHueDensityResponse(
        tone_power=float(params["tone_power"]),
        luminance_floor=float(params["luminance_floor"]),
        base_chroma_gain=float(params["base_chroma_gain"]),
        first_harmonic_gain=float(params["first_harmonic_gain"]),
        first_harmonic_center_degrees=float(
            params["first_harmonic_center_degrees"]
        ),
        second_harmonic_gain=float(params["second_harmonic_gain"]),
        second_harmonic_center_degrees=float(
            params["second_harmonic_center_degrees"]
        ),
        hue_warp=float(params["hue_warp"]),
        hue_warp_center_degrees=float(
            params["hue_warp_center_degrees"]
        ),
        luminance_hue_tilt=float(params["luminance_hue_tilt"]),
        split_tone_a=float(params["split_tone_a"]),
        split_tone_b=float(params["split_tone_b"]),
        gamut_iterations=int(params["gamut_iterations"]),
        encoded_margin=float(params["encoded_margin"]),
    )


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk7_smooth_perceptual_hue_density.v1"
        or config.get("status") != "primitive_contract_frozen"
        or float(config["operator"]["strength"]) != 1.0
    ):
        raise SmoothPerceptualRegressionError("BK7 contract drift")
    parent = config["parent"]
    decision_path = root / parent["decision"]
    if (
        not decision_path.is_file()
        or sha256_file(decision_path) != parent["decision_sha256"]
    ):
        raise SmoothPerceptualRegressionError("BK6 decision drift")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != parent["required_decision"]:
        raise SmoothPerceptualRegressionError("BK7 branch is not open")
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
        raise SmoothPerceptualRegressionError("BK2 regression drift")
    source_path = (
        root
        / "outputs/u5_r2bk3s_encoded_log_chroma_fresh_source_v1/inputs"
        / "phaseone_p25plus.png"
    )
    if sha256_file(source_path) != severe[0]["source_sha256"]:
        raise SmoothPerceptualRegressionError("regression source drift")
    return {
        "operator": _operator(config),
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


def _zero_to_one_delta(
    operator: SmoothPerceptualHueDensityResponse,
) -> float:
    axis = np.arange(0.0, 256.0, 16.0, dtype=np.float32)
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    for channel in range(3):
        for first in axis:
            for second in axis:
                zero = np.asarray((first, second, first), dtype=np.float32)
                zero[channel] = 0.0
                one = zero.copy()
                one[channel] = 1.0
                lower.append(zero)
                upper.append(one)
    low = operator.apply((np.stack(lower) / 255.0)[:, None, :])
    high = operator.apply((np.stack(upper) / 255.0)[:, None, :])
    delta = np.linalg.norm(rgb2lab(high) - rgb2lab(low), axis=-1)
    return float(np.max(delta))


def _neutral_range(operator: SmoothPerceptualHueDensityResponse) -> float:
    gray = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    source = np.stack((gray, gray, gray), axis=-1)[:, None, :]
    return float(np.max(np.ptp(operator.apply(source), axis=-1)))


def _contact_sheet(
    *, source_path: Path, bk2_path: Path, bk7_path: Path, output_path: Path
) -> None:
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (1440, 560), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (5, 6),
        "U5.R2BK7 frozen BK2 severe-failure regression",
        fill="black",
        font=font,
    )
    for index, (path, label) in enumerate(
        ((source_path, "SOURCE"), (bk2_path, "BK2"), (bk7_path, "BK7"))
    ):
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
        raise FileExistsError("BK7 regression is create-only")
    output_dir.mkdir(parents=True)
    operator = validated["operator"]

    cube = _cube()
    cube_output = operator.apply(cube)
    cube_style = _style_delta(cube, cube_output)
    cube_boundary = _new_boundary_fraction(cube, cube_output)
    zero_delta = _zero_to_one_delta(operator)
    neutral_range = _neutral_range(operator)

    with Image.open(validated["source_path"]) as opened:
        source = np.asarray(
            ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.float32
        )
    source /= 255.0
    bk2 = EncodedSafeLogChromaFilmResponse().apply(source)
    bk7 = operator.apply(source)
    bk2_path = output_dir / "bk2_phaseone_p25plus.png"
    bk7_path = output_dir / "bk7_phaseone_p25plus.png"
    save_srgb16_png(bk2, bk2_path)
    save_srgb16_png(bk7, bk7_path)

    gates = config["primitive_gates"]
    automatic_gates = {
        "identity_strength_zero_exact": np.array_equal(
            operator.apply(cube, strength=0.0), cube
        ),
        "black_white_endpoints_exact": bool(
            np.array_equal(cube_output[0, 0, 0], cube[0, 0, 0])
            and np.array_equal(cube_output[-1, -1, -1], cube[-1, -1, -1])
        ),
        "partition_output_exact": np.array_equal(
            np.concatenate(
                (operator.apply(cube[:8]), operator.apply(cube[8:])),
                axis=0,
            ),
            cube_output,
        ),
        "cube_style": cube_style
        >= float(gates["minimum_17_cube_median_style_delta_e76"]),
        "zero_to_one_continuity": zero_delta
        <= float(gates["maximum_zero_to_one_axis_delta_e76"]),
        "neutral_axis": neutral_range
        <= float(gates["maximum_neutral_axis_channel_range"]),
        "new_boundary": cube_boundary
        == float(gates["new_uint16_boundary_fraction_on_17_cube"]),
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "source_sha256": validated["source_sha256"],
        "bk2_output_sha256": sha256_file(bk2_path),
        "bk7_output_sha256": sha256_file(bk7_path),
        "cube_median_style_delta_e76": cube_style,
        "maximum_zero_to_one_axis_delta_e76": zero_delta,
        "maximum_neutral_axis_channel_range": neutral_range,
        "cube_new_uint16_boundary_fraction": cube_boundary,
        "bk2_style_delta_e76": _style_delta(source, bk2),
        "bk7_style_delta_e76": _style_delta(source, bk7),
        "bk2_new_boundary_fraction": _new_boundary_fraction(source, bk2),
        "bk7_new_boundary_fraction": _new_boundary_fraction(source, bk7),
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
            bk2_path=bk2_path,
            bk7_path=bk7_path,
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
    "SmoothPerceptualRegressionError",
    "run_regression",
    "validate_contract",
]
