"""U6.P5H deterministic visibility metrics and blind-pair construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.eval.physical_interimage_photographic_stress import (
    _combined_pipeline,
    _crop_bounds,
    _linear_image,
)
from src.eval.physical_spatial_photographic_stress import _load_source
from src.eval.physical_spatial_response import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    apply_bounded_development_adjacency,
    apply_spatial_response_pipeline,
)
from src.film_physics.interimage_adjacency import (
    interimage_adjacency_profile_from_contract,
)


SCHEMA = "neuro_film.u6_p5h_interimage_visible_value_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P5H contract")
    return payload


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(values, dtype=np.float64)
    dy = np.zeros_like(values, dtype=np.float64)
    dx[:, 1:] = np.diff(values, axis=1)
    dy[1:, :] = np.diff(values, axis=0)
    if values.ndim == 2:
        return np.sqrt(dx * dx + dy * dy)
    return np.sqrt(np.sum(dx * dx + dy * dy, axis=-1))


def _opponent_gradient(values: np.ndarray) -> np.ndarray:
    opponent = values - np.mean(values, axis=-1, keepdims=True)
    return _gradient_magnitude(opponent)


def _encoded_srgb8(values: np.ndarray) -> np.ndarray:
    encoded = linear_srgb_to_encoded(np.clip(values, 0.0, 1.0))
    return np.rint(encoded * 255.0).astype(np.uint8)


def _visibility_metrics(
    source: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    baseline_u8 = _encoded_srgb8(baseline)
    candidate_u8 = _encoded_srgb8(candidate)
    changed = np.any(baseline_u8 != candidate_u8, axis=-1)
    baseline_opponent = _opponent_gradient(baseline)
    candidate_opponent = _opponent_gradient(candidate)
    strong_threshold = float(np.quantile(baseline_opponent, 0.9))
    strong = baseline_opponent >= strong_threshold
    source_luma = source @ np.array([0.2126, 0.7152, 0.0722])
    source_gradient = _gradient_magnitude(source_luma)
    flat_threshold = float(np.quantile(source_gradient, 0.25))
    flat = source_gradient <= flat_threshold
    baseline_edge = float(np.mean(baseline_opponent[strong]))
    candidate_edge = float(np.mean(candidate_opponent[strong]))
    gain = candidate_edge / max(baseline_edge, np.finfo(np.float64).tiny) - 1.0
    strong_changed = float(np.mean(changed[strong]))
    flat_changed = float(np.mean(changed[flat]))
    flat_floor = 1.0 / max(int(np.count_nonzero(flat)), 1)
    return {
        "changed_pixel_fraction": float(np.mean(changed)),
        "strong_edge_changed_pixel_fraction": strong_changed,
        "flat_changed_pixel_fraction": flat_changed,
        "strong_edge_to_flat_changed_ratio": strong_changed
        / max(flat_changed, flat_floor),
        "strong_edge_opponent_gain_fraction": gain,
        "maximum_encoded_uint8_channel_delta": float(
            np.max(
                np.abs(
                    candidate_u8.astype(np.int16)
                    - baseline_u8.astype(np.int16)
                )
            )
        ),
    }


def _pair_crop(
    values: np.ndarray,
    bounds: tuple[slice, slice],
    size: int,
) -> Image.Image:
    image = _linear_image(values[bounds])
    if image.size == (size, size):
        return image
    return ImageOps.pad(
        image,
        (size, size),
        method=Image.Resampling.NEAREST,
        color=(24, 24, 24),
        centering=(0.5, 0.5),
    )


def _write_blind_round(
    visual_rows: list[dict[str, Any]],
    *,
    seed: int,
    size: int,
    path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(visual_rows))
    candidate_on_left = rng.integers(0, 2, size=len(visual_rows)).astype(bool)
    header = 28
    canvas = Image.new(
        "RGB",
        (2 * size, len(visual_rows) * (size + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    key_rows = []
    for display_row, source_index in enumerate(order):
        row = visual_rows[int(source_index)]
        magnitude = np.max(np.abs(row["candidate"] - row["baseline"]), axis=-1)
        centre = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        bounds = _crop_bounds(magnitude.shape, centre, size)
        left_is_candidate = bool(candidate_on_left[display_row])
        left = row["candidate"] if left_is_candidate else row["baseline"]
        right = row["baseline"] if left_is_candidate else row["candidate"]
        y = display_row * (size + header)
        draw.text(
            (4, y + 6),
            f"row {display_row + 1}: A",
            fill=(235, 235, 235),
        )
        draw.text(
            (size + 4, y + 6),
            "B",
            fill=(235, 235, 235),
        )
        canvas.paste(_pair_crop(left, bounds, size), (0, y + header))
        canvas.paste(_pair_crop(right, bounds, size), (size, y + header))
        key_rows.append(
            {
                "display_row": display_row + 1,
                "id": row["id"],
                "candidate_side": "A" if left_is_candidate else "B",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest(), key_rows


def evaluate_interimage_visible_value(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    p5g_decision: dict[str, Any],
    p5c_contract: dict[str, Any],
    p5f_contract: dict[str, Any],
    p5a_contract: dict[str, Any],
    sensitometry_config: dict[str, Any],
    *,
    root: Path,
    blind_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    parent_passed = p5g_decision["status"] == (
        "automatic-and-autonomous-visual-pass-development-only"
    ) and (
        int(
            p5g_decision["autonomous_visual_evidence"][
                "confirmed_severe_failures"
            ]
        )
        == 0
    )
    if not parent_passed:
        raise ValueError("P5G parent safety is not fully passed")

    profile = _profile(p5a_contract)
    candidate = p5c_contract["candidate"]

    def p5c_apply(density: np.ndarray, spatial_profile: Any) -> np.ndarray:
        return apply_bounded_development_adjacency(
            density,
            spatial_profile,
            maximum_absolute_transmittance_delta=float(
                candidate["maximum_absolute_transmittance_delta"]
            ),
            maximum_absolute_density_delta=float(
                candidate["maximum_absolute_density_delta"]
            ),
        )

    p5f_profile = interimage_adjacency_profile_from_contract(p5f_contract)
    operator = build_operator(sensitometry_config)
    rows = []
    visual_rows = []
    fixed_ids = contract["blind_visual_protocol"]["fixed_ids"]
    for source_row in manifest:
        source, input_sha = _load_source(source_row, root)
        baseline = apply_spatial_response_pipeline(
            source,
            profile,
            sensitometry_apply=operator.apply,
            adjacency_apply=p5c_apply,
        )
        combined = _combined_pipeline(
            source,
            profile,
            operator,
            p5c_apply,
            p5f_profile,
        )
        metrics = _visibility_metrics(source, baseline, combined)
        if metrics != _visibility_metrics(source, baseline, combined):
            raise RuntimeError("visibility metrics are not exact on replay")
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "input_sha256": input_sha,
                **metrics,
            }
        )
        if source_row["id"] in fixed_ids:
            visual_rows.append(
                {
                    "id": source_row["id"],
                    "baseline": baseline,
                    "candidate": combined,
                }
            )

    if sorted(row["id"] for row in visual_rows) != sorted(fixed_ids):
        raise ValueError("fixed blind IDs are incomplete")
    gates = contract["automatic_gates"]
    changed = [row["changed_pixel_fraction"] for row in rows]
    gains = [row["strong_edge_opponent_gain_fraction"] for row in rows]
    ratios = [row["strong_edge_to_flat_changed_ratio"] for row in rows]
    decisions = {
        "parent_safety": True,
        "changed_population": float(np.median(changed))
        >= float(gates["minimum_population_median_changed_pixel_fraction"]),
        "changed_breadth": sum(value >= 0.005 for value in changed)
        >= int(gates["minimum_images_with_changed_pixel_fraction_at_least_0_005"]),
        "edge_gain_population": float(np.median(gains))
        >= float(
            gates[
                "minimum_population_median_strong_edge_opponent_gain_fraction"
            ]
        ),
        "edge_gain_breadth": sum(value > 0.0 for value in gains)
        >= int(gates["minimum_images_with_positive_strong_edge_opponent_gain"]),
        "edge_concentration": float(np.median(ratios))
        >= float(
            gates[
                "minimum_population_median_strong_edge_to_flat_changed_ratio"
            ]
        ),
        "code_delta": max(
            row["maximum_encoded_uint8_channel_delta"] for row in rows
        )
        <= int(gates["maximum_encoded_uint8_channel_delta"]),
        "repeat_metrics": True,
    }
    automatic_pass = all(decisions.values())
    sheet_rows = []
    blind_key = None
    if automatic_pass:
        key_rounds = []
        for round_index, seed in enumerate(
            contract["blind_visual_protocol"]["round_seeds"],
            start=1,
        ):
            sheet_path = blind_directory / f"round_{round_index}.png"
            sheet_sha, key_rows = _write_blind_round(
                visual_rows,
                seed=int(seed),
                size=int(contract["blind_visual_protocol"]["crop_size"]),
                path=sheet_path,
            )
            sheet_rows.append(
                {
                    "round": round_index,
                    "sheet_sha256": sheet_sha,
                }
            )
            key_rounds.append({"round": round_index, "rows": key_rows})
        blind_key = {
            "schema": "neuro_film.u6_p5h_interimage_blind_key.v1",
            "rounds": key_rounds,
        }

    core = {
        "schema": "neuro_film.u6_p5h_interimage_visible_value_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "rows": rows,
        "aggregate": {
            "median_changed_pixel_fraction": float(np.median(changed)),
            "images_changed_fraction_at_least_0_005": sum(
                value >= 0.005 for value in changed
            ),
            "median_strong_edge_opponent_gain_fraction": float(
                np.median(gains)
            ),
            "images_positive_strong_edge_opponent_gain": sum(
                value > 0.0 for value in gains
            ),
            "median_strong_edge_to_flat_changed_ratio": float(
                np.median(ratios)
            ),
            "maximum_encoded_uint8_channel_delta": max(
                row["maximum_encoded_uint8_channel_delta"] for row in rows
            ),
        },
        "automatic_decisions": decisions,
        "automatic_pass": automatic_pass,
        "blind_sheets": sheet_rows,
        "blind_review_status": (
            "ready-key-withheld" if automatic_pass else "forbidden"
        ),
        "branch_after_automatic": (
            "run frozen blind visual protocol"
            if automatic_pass
            else contract["branch_rule"]["fail"]
        ),
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return ({**core, "stable_evidence_id": evidence_id}, blind_key)


def score_blind_choices(
    contract: dict[str, Any],
    key: dict[str, Any],
    choices: dict[int, list[str]],
) -> dict[str, Any]:
    round_rows = []
    severe_total = 0
    for round_payload in key["rounds"]:
        round_index = int(round_payload["round"])
        selected = choices[round_index]
        if len(selected) != len(round_payload["rows"]):
            raise ValueError("blind choice count drift")
        candidate_preferences = sum(
            choice == row["candidate_side"]
            for choice, row in zip(
                selected,
                round_payload["rows"],
                strict=True,
            )
            if choice in {"A", "B"}
        )
        round_rows.append(
            {
                "round": round_index,
                "candidate_preferences": candidate_preferences,
                "total_pairs": len(selected),
                "pass": candidate_preferences
                >= int(
                    contract["blind_visual_protocol"][
                        "minimum_candidate_preferences_per_round"
                    ]
                ),
            }
        )
    passed = all(row["pass"] for row in round_rows) and severe_total <= int(
        contract["blind_visual_protocol"]["maximum_confirmed_severe_failures"]
    )
    return {
        "rounds": round_rows,
        "confirmed_severe_failures": severe_total,
        "visual_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
    }


def write_json(payload: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_interimage_visible_value",
    "load_contract",
    "score_blind_choices",
    "write_json",
]
