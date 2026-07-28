#!/usr/bin/env python
"""Render fixed material-map diagnostics after the U6.P4C3 automatic gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p4c3_material import load_contract  # noqa: E402
from src.eval.density_witness_frontier import (  # noqa: E402
    linear_srgb_to_encoded,
)
from src.film_physics import (  # noqa: E402
    CompoundPoissonProfile,
    render_compound_poisson,
)
from src.preprocess.output_encode import save_srgb8  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(row: dict, seed: int) -> CompoundPoissonProfile:
    return CompoundPoissonProfile(
        family=str(row["family"]),
        poisson_rate=float(row["poisson_rate"]),
        correlation_sigma_pixels=float(row["correlation_sigma_pixels"]),
        baseline=float(row["baseline"]),
        scale=float(row["scale"]),
        seed=seed,
    )


def render_diagnostics(contract: dict, output_dir: Path) -> dict:
    shape = tuple(int(value) for value in contract["diagnostic"]["shape"])
    seed = int(contract["diagnostic"]["seed"])
    density = np.stack(
        [
            render_compound_poisson(
                _profile(row, seed + layer), shape
            )
            for layer, row in enumerate(contract["compiled_profiles"][:3])
        ],
        axis=-1,
    )
    colour_linear = np.power(10.0, -density.astype(np.float64))
    colour_encoded = linear_srgb_to_encoded(colour_linear)
    bw = render_compound_poisson(
        _profile(contract["compiled_profiles"][3], seed + 3), shape
    )
    bw_encoded = linear_srgb_to_encoded(
        np.repeat(bw[..., None].astype(np.float64), 3, axis=-1)
    )
    colour_path = output_dir / "colour_material_transmittance.png"
    bw_path = output_dir / "bw_material_transmittance.png"
    save_srgb8(colour_encoded, colour_path)
    save_srgb8(bw_encoded, bw_path)
    colour_image = Image.open(colour_path).convert("RGB")
    bw_image = Image.open(bw_path).convert("RGB")
    header = 48
    contact = Image.new(
        "RGB",
        (colour_image.width * 2, colour_image.height + header),
        "white",
    )
    contact.paste(colour_image, (0, header))
    contact.paste(bw_image, (colour_image.width, header))
    draw = ImageDraw.Draw(contact)
    draw.text(
        (12, 14),
        "Colour dye-density -> transmittance (diagnostic only)",
        fill="black",
    )
    draw.text(
        (colour_image.width + 12, 14),
        "B&W silver transmittance (diagnostic only)",
        fill="black",
    )
    contact_path = output_dir / "material_contact_sheet.png"
    temporary = contact_path.with_suffix(".png.tmp")
    try:
        contact.save(temporary, "PNG")
        os.replace(temporary, contact_path)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema": "neuro_film.u6_p4c3_material_diagnostic.v1",
        "claim_ceiling": contract["diagnostic"]["allowed_interpretation"],
        "shape": list(shape),
        "colour_density_min": float(np.min(density)),
        "colour_density_max": float(np.max(density)),
        "colour_transmittance_min": float(np.min(colour_linear)),
        "colour_transmittance_max": float(np.max(colour_linear)),
        "bw_transmittance_min": float(np.min(bw)),
        "bw_transmittance_max": float(np.max(bw)),
        "files": {
            path.name: _sha256(path)
            for path in (colour_path, bw_path, contact_path)
        },
        "forbidden_claim": contract["diagnostic"]["forbidden_claim"],
    }
    report_path = output_dir / "diagnostic.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p4c3_streamed_material_benchmark_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = render_diagnostics(load_contract(args.contract), args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
