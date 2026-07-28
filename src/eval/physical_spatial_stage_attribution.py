"""U6.P7C cumulative physical spatial-stage attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.color import rgb2lab

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import (
    new_hard_clipping_fraction,
    sha256_file,
)
from src.eval.physical_joint_ablation import (
    _canonicalize_endpoint_roundoff,
    load_contracts,
)
from src.film_physics import (
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
)


SCHEMA = "neuro_film.u6_p7c_spatial_stage_attribution_contract.v1"
ARMS = (
    "cheap_no_spatial",
    "forward_scatter",
    "development_adjacency",
    "dye_diffusion",
    "scanner_mtf",
)
LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or tuple(config.get("cumulative_arms", ())) != ARMS
        or config.get("post_result_retuning_allowed")
    ):
        raise ValueError("unsupported U6.P7C contract")
    _load_exact_json(
        root, config["parent_contract"], config["parent_contract_sha256"]
    )
    decision = _load_exact_json(
        root, config["parent_decision"], config["parent_decision_sha256"]
    )
    runtime_parent = _load_exact_json(
        root, config["runtime_parent"], config["runtime_parent_sha256"]
    )
    if (
        decision.get("next_leaf")
        != "U6.P7C stagewise spatial-response attribution with fixed AO6 context"
        or decision.get("production_default_changed")
        or config["execution"].get("promotion_allowed")
        or config["execution"].get("preference_review")
    ):
        raise ValueError("U6.P7C parent or execution boundary drift")
    return runtime_parent


def render_cumulative_stages(
    encoded: np.ndarray, runtime: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return pre- and post-AO6 cumulative arms for one source frame."""

    source = np.asarray(encoded, dtype=np.float64)
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise ValueError("source must be finite encoded HxWx3")
    linear = encoded_srgb_to_linear(source)
    cheap = _canonicalize_endpoint_roundoff(runtime.print_operator.apply(linear))
    exposure = apply_forward_scatter(linear, runtime.profile)
    forward_density = runtime.print_operator.sensitometry.apply(exposure)
    forward = _canonicalize_endpoint_roundoff(
        runtime.print_operator.interpretation.apply(forward_density)
    )
    adjacency_density = runtime.apply_adjacency(
        forward_density, runtime.profile
    )
    adjacency = _canonicalize_endpoint_roundoff(
        runtime.print_operator.interpretation.apply(adjacency_density)
    )
    dye_density = apply_dye_diffusion(adjacency_density, runtime.profile)
    dye = _canonicalize_endpoint_roundoff(
        runtime.print_operator.interpretation.apply(dye_density)
    )
    scanner = _canonicalize_endpoint_roundoff(
        apply_scanner_mtf(dye, runtime.profile)
    )
    pre = dict(
        zip(ARMS, (cheap, forward, adjacency, dye, scanner), strict=True)
    )
    apply_colour = runtime.build_source_context_colour(source)
    post = {
        name: apply_colour(linear_srgb_to_encoded(values))
        for name, values in pre.items()
    }
    for group in (pre, post):
        for name, values in group.items():
            if (
                values.shape != source.shape
                or not np.all(np.isfinite(values))
                or np.any(values < 0.0)
                or np.any(values > 1.0)
            ):
                raise RuntimeError(f"{name} left its RGB domain")
    return pre, post


def _gradient_energy(linear_rgb: np.ndarray) -> np.ndarray:
    luma = np.asarray(linear_rgb, dtype=np.float64) @ LUMA_WEIGHTS
    dy, dx = np.gradient(luma)
    return dx * dx + dy * dy


def _sample(values: np.ndarray, maximum: int = 65_536) -> np.ndarray:
    flat = np.asarray(values).reshape(-1, 3)
    if len(flat) <= maximum:
        return flat
    indices = np.linspace(0, len(flat) - 1, maximum, dtype=np.int64)
    return flat[indices]


def _median_delta_e76(left: np.ndarray, right: np.ndarray) -> float:
    first = rgb2lab(_sample(left).reshape(-1, 1, 3))
    second = rgb2lab(_sample(right).reshape(-1, 1, 3))
    return float(np.median(np.linalg.norm(first - second, axis=-1)))


def _save_rgb(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        path, format="PNG", compress_level=6
    )
    return sha256_file(path)


def _tile(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    return ImageOps.fit(
        Image.fromarray(pixels, mode="RGB"),
        size,
        method=Image.Resampling.LANCZOS,
    )


def _save_contact_sheet(
    rows: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> str:
    columns = ("source", *ARMS)
    tile_size = (240, 160)
    header = 24
    canvas = Image.new(
        "RGB",
        (
            len(columns) * tile_size[0],
            len(rows) * (tile_size[1] + header),
        ),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (sample_id, values) in enumerate(rows.items()):
        y = row_index * (tile_size[1] + header)
        for column_index, name in enumerate(columns):
            canvas.paste(
                _tile(values[name], tile_size),
                (column_index * tile_size[0], y + header),
            )
            draw.text(
                (column_index * tile_size[0] + 4, y + 4),
                sample_id if column_index == 0 else name,
                fill=(235, 235, 235),
            )
    path = output_dir / "diagnostic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=6)
    return sha256_file(path)


def evaluate_stage_attribution(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    runtime_parent = validate_contract(root, config)
    _, runtime = load_contracts(root, runtime_parent)
    metric = config["metrics"]["source_edge_mask"]
    percentile = float(metric["minimum_percentile"])
    epsilon = float(metric["epsilon"])
    rows: list[dict[str, Any]] = []
    visual_rows: dict[str, dict[str, np.ndarray]] = {}
    visual_ids = {
        "canon_eos_kiss_f",
        "nikon_d2x",
        "sony_nex_3n",
        "fujifilm_finepix_s5000",
        "olympus_sp550uz",
        "panasonic_dmc_gf2",
        "pentax_k_r",
        "leica_d_lux_6",
        "kodak_dcs_pro_14n",
    }
    for sample_id in runtime.eligible_ids:
        source_row = runtime.source_rows[sample_id]
        source_path = root / source_row["decoded_path"]
        if sha256_file(source_path) != source_row["decoded_sha256"]:
            raise ValueError(f"source hash drift: {sample_id}")
        with Image.open(source_path) as image:
            source = (
                np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    dtype=np.float64,
                )
                / 255.0
            )
        first_pre, first_post = render_cumulative_stages(source, runtime)
        second_pre, second_post = render_cumulative_stages(source, runtime)
        repeat = {
            name: (
                np.array_equal(first_pre[name], second_pre[name])
                and np.array_equal(first_post[name], second_post[name])
            )
            for name in ARMS
        }
        source_energy = _gradient_energy(encoded_srgb_to_linear(source))
        threshold = float(np.percentile(source_energy, percentile))
        edge_mask = source_energy >= max(threshold, epsilon)
        if not np.any(edge_mask):
            raise RuntimeError("source edge mask is empty")
        pre_energy = {
            name: float(np.mean(_gradient_energy(values)[edge_mask]))
            for name, values in first_pre.items()
        }
        post_energy = {
            name: float(
                np.mean(
                    _gradient_energy(encoded_srgb_to_linear(values))[edge_mask]
                )
            )
            for name, values in first_post.items()
        }
        cheap_pre = max(pre_energy["cheap_no_spatial"], epsilon)
        cheap_post = max(post_energy["cheap_no_spatial"], epsilon)
        metrics: dict[str, Any] = {}
        previous: np.ndarray | None = None
        for name in ARMS:
            metrics[name] = {
                "pre_ao6_edge_energy_ratio_to_cheap": (
                    pre_energy[name] / cheap_pre
                ),
                "post_ao6_edge_energy_ratio_to_cheap": (
                    post_energy[name] / cheap_post
                ),
                "median_delta_e76_to_previous": (
                    0.0
                    if previous is None
                    else _median_delta_e76(previous, first_post[name])
                ),
                "new_hard_boundary_fraction": new_hard_clipping_fraction(
                    _sample(source),
                    _sample(first_post[name]),
                    0.5 / 255.0,
                ),
                "repeat_identity": repeat[name],
            }
            previous = first_post[name]
        hashes = {
            name: _save_rgb(
                output_dir / "renders" / name / f"{sample_id}.png",
                first_post[name],
            )
            for name in ARMS
        }
        rows.append(
            {
                "sample_id": sample_id,
                "make": source_row["make"],
                "source_sha256": source_row["decoded_sha256"],
                "edge_mask_pixel_count": int(np.count_nonzero(edge_mask)),
                "output_sha256": hashes,
                "metrics": metrics,
            }
        )
        if sample_id in visual_ids:
            visual_rows[sample_id] = {"source": source, **first_post}
    if len(rows) != 16 or set(visual_rows) != visual_ids:
        raise ValueError("U6.P7C population drift")

    median_ratios = {
        name: float(
            np.median(
                [
                    row["metrics"][name][
                        "post_ao6_edge_energy_ratio_to_cheap"
                    ]
                    for row in rows
                ]
            )
        )
        for name in ARMS
    }
    drops: dict[str, float] = {}
    for previous, current in zip(ARMS[:-1], ARMS[1:], strict=True):
        drops[current] = median_ratios[previous] - median_ratios[current]
    dominant = max(drops, key=drops.get)
    core = {
        "schema": "neuro_film.u6_p7c_spatial_stage_attribution_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "rows": rows,
        "summary": {
            "median_post_ao6_edge_energy_ratio_to_cheap": median_ratios,
            "cumulative_stage_edge_energy_drop": drops,
            "dominant_edge_loss_stage": dominant,
            "dominant_edge_loss": drops[dominant],
            "maximum_new_hard_boundary_fraction": float(
                max(
                    row["metrics"][name]["new_hard_boundary_fraction"]
                    for row in rows
                    for name in ARMS
                )
            ),
            "all_repeat_identity": all(
                row["metrics"][name]["repeat_identity"]
                for row in rows
                for name in ARMS
            ),
        },
        "visual_evidence": {
            "diagnostic_sha256": _save_contact_sheet(
                visual_rows, output_dir
            ),
            "preference_review_performed": False,
        },
        "branch": config["branch_rule"][
            "single_dominant_stage"
            if drops[dominant] > 0.0
            else "no_measured_loss"
        ],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_stage_attribution",
    "render_cumulative_stages",
    "validate_contract",
    "write_report",
]
