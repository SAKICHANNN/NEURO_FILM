#!/usr/bin/env python3
"""Run the frozen U7.2C three-stock bounded-amount audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config
from src.inference import load_render_profile, render_resolved_safe_lab_rgb
from src.inference.three_stock_look import (
    list_three_stock_looks,
    render_three_stock_look_rgb,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u7_2c_three_stock_look_amount_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    return parser.parse_args()


def _delta_e_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    values: list[np.ndarray] = []
    for y0 in range(0, left.shape[0], 128):
        left_lab = rgb2lab(left[y0 : y0 + 128]).astype(np.float32)
        right_lab = rgb2lab(right[y0 : y0 + 128]).astype(np.float32)
        values.append(
            np.sqrt(np.sum(np.square(left_lab - right_lab), axis=2)).reshape(-1)
        )
    delta = np.concatenate(values)
    return {
        "median_delta_e76": float(np.median(delta)),
        "p95_delta_e76": float(np.quantile(delta, 0.95)),
    }


def main() -> int:
    args = _parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("schema") != "neuro-film.u7-2c-three-stock-look-amount-contract.v1":
        raise ValueError("unsupported U7.2C contract")
    source_path = ROOT / contract["source"]["path"]
    if _sha256_file(source_path) != contract["source"]["sha256"]:
        raise RuntimeError("U7.2C source identity drifted")
    width, height = (int(value) for value in contract["dimensions"])
    with Image.open(source_path) as image:
        rgb8 = np.asarray(
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    source = np.ascontiguousarray(rgb8.astype(np.float32) / 255.0)
    profile = load_render_profile(
        ROOT / "configs/render_profiles/safe_rich_v1.json", root=ROOT
    )
    all_statistics = json.loads(
        (ROOT / "configs/film_color_stats.json").read_text(encoding="utf-8")
    )["styles"]
    catalog = {row["film_stock_id"]: row for row in list_three_stock_looks()}
    stock_ids = tuple(str(value) for value in contract["film_stock_ids"])
    if set(catalog) != set(stock_ids):
        raise RuntimeError("U7.2C catalog drifted")
    if args.order == "reverse":
        stock_ids = tuple(reversed(stock_ids))
    amounts = tuple(float(value) for value in contract["amounts"])
    if args.order == "reverse":
        amounts = tuple(reversed(amounts))

    rows: list[dict[str, object]] = []
    full_outputs: dict[str, np.ndarray] = {}
    by_stock_residual: dict[str, dict[str, float]] = {}
    for stock_id in stock_ids:
        style = catalog[stock_id]["style_id"]
        statistics = all_statistics[style]
        guardrails = load_guardrail_config(
            ROOT / "configs/color_guardrails.json", style
        )
        residuals: dict[str, float] = {}
        for amount in amounts:
            start = time.perf_counter()
            output = render_three_stock_look_rgb(
                source,
                profile=profile,
                film_stock_id=stock_id,
                look_amount=amount,
                style_statistics=statistics,
                guardrails=guardrails,
                seed=int(contract["seed"]),
                tile_size=int(contract["tile_size"]),
                tile_workers=int(contract["tile_workers"]),
            )
            seconds = time.perf_counter() - start
            if amount == 0.0 and not np.array_equal(output, source):
                raise RuntimeError(f"{stock_id} zero amount is not identity")
            if amount == 1.0:
                expected = render_resolved_safe_lab_rgb(
                    source,
                    style=style,
                    style_statistics=statistics,
                    style_parameters=profile["style_parameters"][style],
                    guardrails=guardrails,
                    seed=int(contract["seed"]),
                    tile_size=int(contract["tile_size"]),
                    tile_workers=int(contract["tile_workers"]),
                )
                if not np.array_equal(output, expected):
                    raise RuntimeError(f"{stock_id} full amount parity failed")
                full_outputs[stock_id] = output.copy()
            difference = output.astype(np.float64) - source.astype(np.float64)
            residual_rmse = float(np.sqrt(np.mean(np.square(difference))))
            residuals[str(amount)] = residual_rmse
            boundary_fraction = float(np.mean((output <= 0.0) | (output >= 1.0)))
            rows.append(
                {
                    "film_stock_id": stock_id,
                    "style_id": style,
                    "look_amount": amount,
                    "output_float32_sha256": _sha256_bytes(output.tobytes()),
                    "output_srgb16_sha256": _sha256_bytes(
                        np.rint(output * 65535.0).astype(np.uint16).tobytes()
                    ),
                    "residual_rmse": residual_rmse,
                    "output_minimum": float(output.min()),
                    "output_maximum": float(output.max()),
                    "output_boundary_fraction": boundary_fraction,
                    "wall_seconds": seconds,
                }
            )
        by_stock_residual[stock_id] = residuals

    pairs: list[dict[str, object]] = []
    canonical_stocks = tuple(str(value) for value in contract["film_stock_ids"])
    for index, left_id in enumerate(canonical_stocks):
        for right_id in canonical_stocks[index + 1 :]:
            pairs.append(
                {
                    "left_film_stock_id": left_id,
                    "right_film_stock_id": right_id,
                    **_delta_e_summary(full_outputs[left_id], full_outputs[right_id]),
                }
            )
    maximum_boundary = max(
        float(row["output_boundary_fraction"])
        for row in rows
        if float(row["look_amount"]) > 0.0
    )
    minimum_pairwise = min(float(row["median_delta_e76"]) for row in pairs)
    residual_order_pass = all(
        values["0.0"] == 0.0 < values["0.5"] < values["1.0"]
        for values in by_stock_residual.values()
    )
    gates = {
        "zero_amount_exact_identity": all(
            float(row["look_amount"]) != 0.0
            or row["output_float32_sha256"] == _sha256_bytes(source.tobytes())
            for row in rows
        ),
        "one_amount_exact_existing_baseline": len(full_outputs) == 3,
        "half_residual_rmse_strictly_between_zero_and_full": residual_order_pass,
        "nonzero_output_boundary": maximum_boundary
        <= float(contract["gates"]["maximum_nonzero_output_boundary_fraction"]),
        "pairwise_full_output_separation": minimum_pairwise
        >= float(
            contract["gates"][
                "minimum_pairwise_full_output_population_median_delta_e76"
            ]
        ),
    }
    stable = {
        "schema": "neuro-film.u7-2c-three-stock-look-amount-report.v1",
        "contract_sha256": _sha256_file(args.contract),
        "source_file_sha256": _sha256_file(source_path),
        "resized_rgb8_sha256": _sha256_bytes(rgb8.tobytes()),
        "dimensions": [width, height],
        "tile_size": int(contract["tile_size"]),
        "tile_workers": int(contract["tile_workers"]),
        "outputs": {
            f"{row['film_stock_id']}:{row['look_amount']}": {
                "output_float32_sha256": row["output_float32_sha256"],
                "output_srgb16_sha256": row["output_srgb16_sha256"],
                "residual_rmse": row["residual_rmse"],
                "output_minimum": row["output_minimum"],
                "output_maximum": row["output_maximum"],
                "output_boundary_fraction": row["output_boundary_fraction"],
            }
            for row in rows
        },
        "pairwise_full_output_delta_e76": pairs,
        "gates": gates,
        "automatic_pass": all(gates.values()),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report = {
        **stable,
        "order": args.order,
        "rows": rows,
        "stable_identity": _sha256_bytes(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
