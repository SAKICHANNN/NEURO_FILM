"""BK2 severe-failure regression for the fixed BK5 opponent response."""

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
from src.roll2film.bounded_opponent_response import (
    BoundedOpponentFilmResponse,
)
from src.roll2film.encoded_safe_log_chroma import (
    EncodedSafeLogChromaFilmResponse,
)


SCHEMA = "neuro_film.u5_r2bk5_bounded_opponent_regression_report.v1"


class BoundedOpponentRegressionError(RuntimeError):
    """Raised when the fixed BK5 regression contract drifts."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _operator(config: Mapping[str, Any]) -> BoundedOpponentFilmResponse:
    params = config["operator"]
    return BoundedOpponentFilmResponse(
        contrast=float(params["contrast"]),
        red_green_gain=float(params["red_green_gain"]),
        blue_yellow_gain=float(params["blue_yellow_gain"]),
        midtone_chroma_lift=float(params["midtone_chroma_lift"]),
        opponent_rotation=float(params["opponent_rotation"]),
        hue_bend=float(params["hue_bend"]),
        encoded_margin=float(params["encoded_margin"]),
    )


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk5_bounded_opponent_response.v1"
        or config.get("status") != "primitive_contract_frozen"
        or float(config["operator"]["strength"]) != 1.0
    ):
        raise BoundedOpponentRegressionError("BK5 contract drift")
    parent = config["parent"]
    decision_path = root / parent["decision"]
    if (
        not decision_path.is_file()
        or sha256_file(decision_path) != parent["decision_sha256"]
    ):
        raise BoundedOpponentRegressionError("BK4 decision drift")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != parent["required_decision"]:
        raise BoundedOpponentRegressionError("BK4 branch is not open")
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
        raise BoundedOpponentRegressionError("BK2 severe regression drift")
    source_path = (
        root
        / "outputs/u5_r2bk3s_encoded_log_chroma_fresh_source_v1/inputs"
        / "phaseone_p25plus.png"
    )
    if sha256_file(source_path) != severe[0]["source_sha256"]:
        raise BoundedOpponentRegressionError("regression source drift")
    return {
        "operator": _operator(config),
        "source_path": source_path,
        "source_sha256": severe[0]["source_sha256"],
    }


def _style_delta_e76(source: np.ndarray, output: np.ndarray) -> float:
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


def _zero_to_one_delta(operator: BoundedOpponentFilmResponse) -> float:
    green = np.arange(64, 193, 8, dtype=np.float64)
    blue = np.clip(green + 55, 0, 255)
    red_zero = np.stack((np.zeros_like(green), green, blue), axis=-1) / 255.0
    red_one = np.stack((np.ones_like(green), green, blue), axis=-1) / 255.0
    delta = np.linalg.norm(
        rgb2lab(operator.apply(red_one[None]))
        - rgb2lab(operator.apply(red_zero[None])),
        axis=-1,
    )
    return float(np.max(delta))


def _contact_sheet(
    *, source_path: Path, bk2_path: Path, bk5_path: Path, output_path: Path
) -> None:
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (1440, 560), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (5, 6),
        "U5.R2BK5 BK2 severe-failure regression",
        fill="black",
        font=font,
    )
    for index, (path, label) in enumerate(
        ((source_path, "SOURCE"), (bk2_path, "BK2"), (bk5_path, "BK5"))
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
        raise FileExistsError("BK5 regression is create-only")
    output_dir.mkdir(parents=True)
    with Image.open(validated["source_path"]) as opened:
        source = np.asarray(
            ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.float32
        )
    source /= 255.0
    bk2 = EncodedSafeLogChromaFilmResponse().apply(source)
    bk5 = validated["operator"].apply(source)
    bk2_path = output_dir / "bk2_phaseone_p25plus.png"
    bk5_path = output_dir / "bk5_phaseone_p25plus.png"
    save_srgb16_png(bk2, bk2_path)
    save_srgb16_png(bk5, bk5_path)

    gate = config["primitive_gates"]
    zero_delta = _zero_to_one_delta(validated["operator"])
    bk5_new_boundary = _new_boundary_fraction(source, bk5)
    automatic_gates = {
        "zero_to_one_continuity": zero_delta
        <= float(gate["maximum_zero_to_one_red_code_delta_e76"]),
        "regression_new_boundary": bk5_new_boundary
        == float(
            gate[
                "new_uint16_boundary_fraction_on_interior_synthetic_cube"
            ]
        ),
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "source_sha256": validated["source_sha256"],
        "bk2_output_sha256": sha256_file(bk2_path),
        "bk5_output_sha256": sha256_file(bk5_path),
        "bk2_style_delta_e76": _style_delta_e76(source, bk2),
        "bk5_style_delta_e76": _style_delta_e76(source, bk5),
        "bk2_new_boundary_fraction": _new_boundary_fraction(source, bk2),
        "bk5_new_boundary_fraction": bk5_new_boundary,
        "maximum_zero_to_one_red_code_delta_e76": zero_delta,
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
            bk5_path=bk5_path,
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
    "BoundedOpponentRegressionError",
    "run_regression",
    "validate_contract",
]
