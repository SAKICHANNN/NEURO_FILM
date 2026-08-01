"""U6.P5M measured total-film MTF ownership and double-counting audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.film_physics.measured_mtf import compiled_channel_response
from src.film_physics.measured_mtf_budget import (
    RGB_CHANNELS,
    MeasuredTotalFilmSpatialBudget,
)

SCHEMA = "neuro_film.u6_p5m_measured_mtf_spatial_budget_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p5m_measured_mtf_spatial_budget_report.v1"


class MeasuredMtfBudgetError(RuntimeError):
    """Raised when a frozen P5M input or contract drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MeasuredMtfBudgetError("P5M paths must be repository-relative")
    return path


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    relative = _relative(path)
    resolved = root / relative
    if _hash(resolved) != expected:
        raise MeasuredMtfBudgetError(f"P5M parent hash mismatch: {relative}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(config: Mapping[str, Any]) -> None:
    domain = config.get("domain", {})
    policy = config.get("ownership_policy", {})
    gates = config.get("gates", {})
    if (
        config.get("schema") != SCHEMA
        or domain.get("sampling_dpi") != 4000.0
        or domain.get("pixel_pitch_um") != 6.35
        or domain.get("channels") != ["blue", "green", "red"]
        or not domain.get("small_signal_only")
        or policy.get("measured_total_replaces_if_selected")
        != ["forward_scatter", "development_adjacency", "dye_diffusion"]
        or not policy.get("scanner_mtf_is_separate_downstream_nuisance")
        or not policy.get(
            "generic_and_measured_film_spatial_budgets_may_not_be_active_together"
        )
        or gates.get("row_partitions") != [7, 17, 31]
        or gates.get("positive_residual_maximum") != 1.000000000001
        or not gates.get("mixed_residual_directions_close_positive_composition")
    ):
        raise MeasuredMtfBudgetError("P5M frozen contract drift")


def _gaussian_response(sigma_um: float, frequencies: np.ndarray) -> np.ndarray:
    return np.exp(-2.0 * math.pi**2 * (sigma_um * 1e-3) ** 2 * frequencies**2)


def _legacy_response(
    profile: Mapping[str, Any], channel_index: int, frequencies: np.ndarray
) -> np.ndarray:
    forward = _gaussian_response(
        float(profile["forward_scatter_sigma_um_rgb"][channel_index]), frequencies
    )
    adjacency_blur = _gaussian_response(
        float(profile["development_adjacency_sigma_um_rgb"][channel_index]),
        frequencies,
    )
    gain = float(profile["development_adjacency_gain_rgb"][channel_index])
    adjacency = 1.0 + gain * (1.0 - adjacency_blur)
    dye = _gaussian_response(
        float(profile["dye_diffusion_sigma_um_rgb"][channel_index]), frequencies
    )
    return forward * adjacency * dye


def _save_diagnostic(
    output: Path,
    frequencies: np.ndarray,
    rows: Mapping[str, Mapping[str, np.ndarray]],
) -> str:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for axis, name in zip(axes, ("blue", "green", "red"), strict=True):
        values = rows[name]
        axis.plot(frequencies, values["measured"], label="measured total", lw=2.2)
        axis.plot(frequencies, values["legacy"], label="legacy total", lw=1.8)
        axis.plot(frequencies, values["stacked"], label="stacked", lw=1.8)
        axis.set_title(name)
        axis.set_xlabel("cycles/mm")
        axis.grid(alpha=0.25)
        axis.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("MTF")
    axes[-1].legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, metadata={"Software": "neuro-film"})
    plt.close(figure)
    return _hash(output)


def evaluate_budget(
    *, root: Path, config: Mapping[str, Any], diagnostic_path: Path
) -> dict[str, Any]:
    validate_contract(config)
    parents = config["parents"]
    lod_payload = _load_exact(root, parents["p5l_bundle"], parents["p5l_bundle_sha256"])
    p5l_decision = _load_exact(
        root, parents["p5l_decision"], parents["p5l_decision_sha256"]
    )
    generic = _load_exact(
        root,
        parents["generic_spatial_contract"],
        parents["generic_spatial_contract_sha256"],
    )
    bounded = _load_exact(
        root,
        parents["bounded_adjacency_contract"],
        parents["bounded_adjacency_contract_sha256"],
    )
    long_range = _load_exact(
        root,
        parents["long_range_response_decision"],
        parents["long_range_response_decision_sha256"],
    )
    if (
        not p5l_decision.get("automatic_pass")
        or lod_payload.get("lod_bundle_id") != p5l_decision.get("lod_bundle_id")
        or long_range.get("decision") != "close_product_value_retain_research_primitive"
    ):
        raise MeasuredMtfBudgetError("P5M parent decision drift")
    profile = generic["profile"]
    candidate = bounded["candidate"]
    if (
        profile["development_adjacency_sigma_um_rgb"]
        != candidate["chemical_spread_sigma_um_rgb"]
        or profile["development_adjacency_gain_rgb"] != candidate["unbounded_gain_rgb"]
    ):
        raise MeasuredMtfBudgetError("P5M P5A/P5C linearization drift")

    budget = MeasuredTotalFilmSpatialBudget.from_lod_bundle(lod_payload)
    frequency_spec = config["domain"]["frequencies"]
    frequencies = np.arange(
        frequency_spec["start"],
        frequency_spec["stop"] + frequency_spec["step"] * 0.5,
        frequency_spec["step"],
        dtype=np.float64,
    )
    channel_rows: dict[str, dict[str, np.ndarray]] = {}
    summaries = {}
    residual_above = []
    residual_below = []
    stacked_errors = []
    for rgb_index, name in enumerate(RGB_CHANNELS):
        measured = compiled_channel_response(
            budget.compiled_rgb[rgb_index],
            frequencies,
            pixel_pitch_um=float(config["domain"]["pixel_pitch_um"]),
        )
        legacy = _legacy_response(profile, rgb_index, frequencies)
        stacked = measured * legacy
        residual = measured / legacy
        channel_rows[name] = {
            "measured": measured,
            "legacy": legacy,
            "stacked": stacked,
            "residual": residual,
        }
        margin = float(config["gates"]["minimum_residual_direction_margin"])
        if float(np.max(residual)) > 1.0 + margin:
            residual_above.append(name)
        if float(np.min(residual)) < 1.0 - margin:
            residual_below.append(name)
        error = np.abs(stacked - measured)
        stacked_errors.extend(error.tolist())
        summaries[name] = {
            "measured_min": float(np.min(measured)),
            "measured_max": float(np.max(measured)),
            "legacy_min": float(np.min(legacy)),
            "legacy_max": float(np.max(legacy)),
            "residual_min": float(np.min(residual)),
            "residual_max": float(np.max(residual)),
            "stacked_max_absolute_error": float(np.max(error)),
        }

    rng = np.random.default_rng(20260801)
    values = rng.random((127, 191, 3))
    direct = budget.apply(values)
    repeat = np.array_equal(direct, budget.apply(values))
    partitions = {
        str(rows): bool(
            np.array_equal(direct, budget.apply_row_tiled(values, tile_rows=rows))
        )
        for rows in config["gates"]["row_partitions"]
    }
    rejection_exact = False
    try:
        budget.validate_active_stages(["forward_scatter", "scanner_mtf"])
    except ValueError:
        rejection_exact = True
    budget.validate_active_stages(["scanner_mtf"])

    stacked_max = float(max(stacked_errors))
    mixed = bool(residual_above and residual_below)
    positive_residual_necessary_condition = not residual_above
    gate_results = {
        "measured_response_finite_inside_zero_one": all(
            np.all(np.isfinite(row["measured"]))
            and np.all(row["measured"] >= 0.0)
            and np.all(row["measured"] <= 1.0 + 1e-12)
            for row in channel_rows.values()
        ),
        "legacy_response_finite_positive": all(
            np.all(np.isfinite(row["legacy"])) and np.all(row["legacy"] > 0.0)
            for row in channel_rows.values()
        ),
        "residual_direction_classified": bool(residual_above or residual_below),
        "stacked_error_material": stacked_max
        >= float(config["gates"]["minimum_stacked_max_absolute_error_to_measured"]),
        "budget_rejects_double_count": rejection_exact,
        "budget_application_repeat_exact": repeat,
        "budget_application_row_partition_exact": all(partitions.values()),
    }
    diagnostic_sha = _save_diagnostic(diagnostic_path, frequencies, channel_rows)
    stable = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _hash(
            root / "configs/u6_p5m_measured_mtf_spatial_budget_v1.json"
        ),
        "source_lod_bundle_id": budget.source_lod_bundle_id,
        "spatial_budget_id": budget.budget_id,
        "frequency_count": int(frequencies.size),
        "frequency_min": float(frequencies[0]),
        "frequency_max": float(frequencies[-1]),
        "channel_summaries": summaries,
        "residual_above_unity_channels": residual_above,
        "residual_below_unity_channels": residual_below,
        "stacked_max_absolute_error": stacked_max,
        "repeat_exact": repeat,
        "row_partition_exact": partitions,
        "diagnostic_sha256": diagnostic_sha,
        "gate_results": gate_results,
    }
    passed = all(gate_results.values())
    if passed and positive_residual_necessary_condition:
        decision = "open_positive_residual_compiler"
    elif passed and (mixed or residual_above):
        decision = "replace_generic_film_spatial_stages_with_measured_total"
    else:
        decision = "close_measured_spatial_budget_integration"
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "positive_residual_necessary_condition": positive_residual_necessary_condition,
        "decision": decision,
        "active_film_spatial_stages": [],
        "retained_downstream_stages": ["scanner_mtf"] if passed else [],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "MeasuredMtfBudgetError",
    "evaluate_budget",
    "validate_contract",
]
