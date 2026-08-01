"""U6.P5P synthetic audit of the analytical measured-MTF minimax envelope."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.eval.cave_conditional_variability import array_sha256
from src.eval.physical_measured_mtf_domain_placement import _apply_arms
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.measured_mtf_budget import MeasuredTotalFilmSpatialBudget
from src.film_physics.measured_mtf_envelope import (
    PLACEMENT_ARMS,
    MinimaxTransmittanceEnvelope,
    build_minimax_transmittance_envelope,
)

SCHEMA = "neuro_film.u6_p5p_measured_mtf_minimax_envelope_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p5p_measured_mtf_minimax_envelope_report.v1"


class MeasuredMtfEnvelopeError(RuntimeError):
    """Raised when a frozen P5P input or invariant drifts."""


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
        raise MeasuredMtfEnvelopeError("P5P paths must be repository-relative")
    return path


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    relative = _relative(path)
    resolved = root / relative
    if _hash(resolved) != expected:
        raise MeasuredMtfEnvelopeError(f"P5P parent hash mismatch: {relative}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(config: Mapping[str, Any]) -> None:
    algorithm = config.get("algorithm", {})
    probes = config.get("probes", {})
    gates = config.get("gates", {})
    if (
        config.get("schema") != SCHEMA
        or algorithm.get("common_domain") != "scan_transmittance"
        or algorithm.get("output") != "0.5 * (lower + upper)"
        or algorithm.get("fitted_parameters") != 0
        or algorithm.get("hard_arm_selected") is not False
        or probes.get("relative_modulation_amplitudes") != [0.1, 0.5]
        or probes.get("shapes") != [[63, 95], [128, 191]]
        or probes.get("log_exposure")
        != {"toe": -2.5, "mid": -0.5, "shoulder": 1.5}
        or probes.get("patterns")
        != [
            "impulse",
            "vertical_step",
            "checker_3x5",
            "seeded_random",
            "chromatic_edge",
        ]
        or probes.get("row_partitions") != [7, 17, 31]
        or gates.get("nonidentity_transmittance_max_absolute_difference_minimum")
        != 0.0000152587890625
    ):
        raise MeasuredMtfEnvelopeError("P5P frozen contract drift")


def _signed_pattern(name: str, shape: tuple[int, int], seed: int) -> np.ndarray:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    if name == "impulse":
        pattern = np.zeros((height, width, 3), dtype=np.float64)
        pattern[height // 2, width // 2] = 1.0
        return pattern
    if name == "vertical_step":
        plane = np.where(x < width // 2, -1.0, 1.0)
        return np.repeat(plane[..., None], 3, axis=2)
    if name == "checker_3x5":
        plane = np.where(((x // 3) + (y // 5)) % 2 == 0, -1.0, 1.0)
        return np.repeat(plane[..., None], 3, axis=2)
    if name == "seeded_random":
        return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(height, width, 3))
    if name == "chromatic_edge":
        return np.stack(
            (
                np.where(x < width // 2, -1.0, 1.0),
                np.where(y < height // 2, -1.0, 1.0),
                np.where(x + y < (width + height) // 2, -1.0, 1.0),
            ),
            axis=-1,
        )
    raise MeasuredMtfEnvelopeError(f"unsupported P5P pattern: {name}")


def _envelope(arms: Mapping[str, Mapping[str, np.ndarray]]) -> MinimaxTransmittanceEnvelope:
    return build_minimax_transmittance_envelope(
        {name: arms[name]["transmittance"] for name in PLACEMENT_ARMS}
    )


def _minimax_metrics(
    envelope: MinimaxTransmittanceEnvelope,
    arms: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, float]:
    transmittances = [arms[name]["transmittance"] for name in PLACEMENT_ARMS]
    interval_violation = max(
        0.0,
        float(np.max(envelope.lower - envelope.transmittance)),
        float(np.max(envelope.transmittance - envelope.upper)),
    )
    midpoint_losses = np.stack(
        [np.abs(envelope.transmittance - value) for value in transmittances], axis=0
    )
    midpoint_worst = np.max(midpoint_losses, axis=0)
    identity_error = float(
        np.max(np.abs(midpoint_worst - envelope.uncertainty_half_range))
    )
    fixed_global_losses = []
    for candidate in transmittances:
        fixed_global_losses.append(
            float(
                np.max(
                    np.stack([np.abs(candidate - other) for other in transmittances])
                )
            )
        )
    midpoint_global = float(np.max(midpoint_worst))
    best_fixed_global = min(fixed_global_losses)
    return {
        "arm_interval_violation_max": interval_violation,
        "chebyshev_identity_max_absolute_error": identity_error,
        "midpoint_global_worst_loss": midpoint_global,
        "best_fixed_arm_global_worst_loss": best_fixed_global,
        "midpoint_worst_loss_excess_over_best_fixed_arm": midpoint_global
        - best_fixed_global,
        "uncertainty_half_range_max": float(
            np.max(envelope.uncertainty_half_range)
        ),
        "uncertainty_half_range_mean": float(
            np.mean(envelope.uncertainty_half_range)
        ),
    }


def _save_diagnostic(
    path: Path,
    base_transmittance: np.ndarray,
    envelope: MinimaxTransmittanceEnvelope,
    arms: Mapping[str, Mapping[str, np.ndarray]],
) -> str:
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    images = [
        base_transmittance,
        envelope.transmittance,
        envelope.uncertainty_half_range
        / max(float(np.max(envelope.uncertainty_half_range)), np.finfo(float).eps),
        arms[PLACEMENT_ARMS[0]]["transmittance"],
        arms[PLACEMENT_ARMS[1]]["transmittance"],
        arms[PLACEMENT_ARMS[2]]["transmittance"],
    ]
    titles = [
        "unfiltered transmittance",
        "minimax envelope",
        "normalized uncertainty",
        "exposure arm",
        "density arm",
        "transmittance arm",
    ]
    for axis, image, title in zip(axes.flat, images, titles, strict=True):
        displayed = image if title == "normalized uncertainty" else np.clip(image, 0, 1) ** (1 / 2.2)
        axis.imshow(displayed, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("P5P analytical minimax measured-MTF envelope")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, metadata={"Software": "neuro-film"})
    plt.close(figure)
    return _hash(path)


def evaluate_minimax_envelope(
    *, root: Path, config: Mapping[str, Any], diagnostic_path: Path
) -> dict[str, Any]:
    validate_contract(config)
    parents = config["parents"]
    p5o = _load_exact(root, parents["p5o_decision"], parents["p5o_decision_sha256"])
    if p5o.get("decision") != "placement_unidentified_open_bounded_envelope":
        raise MeasuredMtfEnvelopeError("P5P requires the exact P5O ambiguity branch")
    lod = _load_exact(root, parents["p5l_bundle"], parents["p5l_bundle_sha256"])
    sens_config = _load_exact(
        root,
        parents["sensitometry_contract"],
        parents["sensitometry_contract_sha256"],
    )
    budget = MeasuredTotalFilmSpatialBudget.from_lod_bundle(lod)
    sensitometry = build_operator(sens_config)
    rows = []
    hashes = {}
    maxima = {
        "arm_interval_violation_max": 0.0,
        "chebyshev_identity_max_absolute_error": 0.0,
        "midpoint_worst_loss_excess_over_best_fixed_arm": -float("inf"),
        "nonidentity_transmittance_max_absolute_difference": 0.0,
        "uncertainty_half_range_max": 0.0,
    }
    density_min = float("inf")
    density_max = -float("inf")
    trans_min = float("inf")
    trans_max = -float("inf")
    partitions_exact = True
    diagnostic_payload = None
    for regime, log_exposure in config["probes"]["log_exposure"].items():
        baseline = float(sensitometry.encoder.inverse(np.asarray(log_exposure)))
        for amplitude in config["probes"]["relative_modulation_amplitudes"]:
            for shape_values in config["probes"]["shapes"]:
                shape = tuple(int(value) for value in shape_values)
                for pattern in config["probes"]["patterns"]:
                    seeds = (
                        config["probes"]["random_seeds"]
                        if pattern == "seeded_random"
                        else [config["probes"]["random_seeds"][0]]
                    )
                    for seed in seeds:
                        signed = _signed_pattern(pattern, shape, int(seed))
                        exposure = baseline * (1.0 + float(amplitude) * signed)
                        arms = _apply_arms(
                            exposure, sensitometry=sensitometry, budget=budget
                        )
                        envelope = _envelope(arms)
                        unfiltered_density = sensitometry.apply(exposure)
                        unfiltered_transmittance = np.power(10.0, -unfiltered_density)
                        metrics = _minimax_metrics(envelope, arms)
                        nonidentity = float(
                            np.max(
                                np.abs(
                                    envelope.transmittance - unfiltered_transmittance
                                )
                            )
                        )
                        metrics["nonidentity_transmittance_max_absolute_difference"] = nonidentity
                        partition_rows = {}
                        for tile_rows in config["probes"]["row_partitions"]:
                            tiled_arms = _apply_arms(
                                exposure,
                                sensitometry=sensitometry,
                                budget=budget,
                                tile_rows=int(tile_rows),
                            )
                            tiled = _envelope(tiled_arms)
                            partition_rows[str(tile_rows)] = all(
                                np.array_equal(getattr(envelope, field), getattr(tiled, field))
                                for field in (
                                    "lower",
                                    "upper",
                                    "transmittance",
                                    "density",
                                    "uncertainty_half_range",
                                )
                            )
                        partitions_exact &= all(partition_rows.values())
                        for name in maxima:
                            maxima[name] = max(maxima[name], metrics[name])
                        density_min = min(density_min, float(np.min(envelope.density)))
                        density_max = max(density_max, float(np.max(envelope.density)))
                        trans_min = min(trans_min, float(np.min(envelope.transmittance)))
                        trans_max = max(trans_max, float(np.max(envelope.transmittance)))
                        key = f"{regime}-{amplitude:.2f}-{shape[0]}x{shape[1]}-{pattern}-{seed}"
                        hashes[key] = {
                            field: array_sha256(getattr(envelope, field))
                            for field in (
                                "transmittance",
                                "density",
                                "uncertainty_half_range",
                            )
                        }
                        rows.append(
                            {
                                "probe": key,
                                "metrics": metrics,
                                "row_partition_exact": partition_rows,
                            }
                        )
                        if (
                            regime == "mid"
                            and float(amplitude) == 0.5
                            and shape == (128, 191)
                            and pattern == "chromatic_edge"
                        ):
                            diagnostic_payload = (
                                unfiltered_transmittance,
                                envelope,
                                arms,
                            )
    if diagnostic_payload is None:
        raise MeasuredMtfEnvelopeError("P5P diagnostic probe missing")

    constant_trans_error = 0.0
    constant_density_error = 0.0
    for value in config["probes"]["constant_values"]:
        exposure = np.full((37, 53, 3), float(value), dtype=np.float64)
        arms = _apply_arms(exposure, sensitometry=sensitometry, budget=budget)
        envelope = _envelope(arms)
        expected_density = sensitometry.apply(exposure)
        expected_trans = np.power(10.0, -expected_density)
        constant_trans_error = max(
            constant_trans_error,
            float(np.max(np.abs(envelope.transmittance - expected_trans))),
        )
        constant_density_error = max(
            constant_density_error,
            float(np.max(np.abs(envelope.density - expected_density))),
        )

    diagnostic_sha = _save_diagnostic(
        diagnostic_path,
        diagnostic_payload[0],
        diagnostic_payload[1],
        diagnostic_payload[2],
    )
    gates = config["gates"]
    gate_results = {
        "arm_interval": maxima["arm_interval_violation_max"]
        <= float(gates["arm_interval_violation_max"]),
        "chebyshev_identity": maxima["chebyshev_identity_max_absolute_error"]
        <= float(gates["chebyshev_identity_max_absolute_error"]),
        "minimax_vs_fixed_hard_arm": maxima[
            "midpoint_worst_loss_excess_over_best_fixed_arm"
        ]
        <= float(gates["midpoint_worst_loss_excess_over_best_fixed_arm_max"]),
        "constant_transmittance": constant_trans_error
        <= float(gates["constant_transmittance_max_absolute_error"]),
        "constant_density": constant_density_error
        <= float(gates["constant_density_max_absolute_error"]),
        "density_range": density_min >= float(gates["density_minimum"]),
        "transmittance_range": trans_min
        > float(gates["transmittance_minimum_exclusive"])
        and trans_max <= float(gates["transmittance_maximum"]),
        "nonidentity": maxima["nonidentity_transmittance_max_absolute_difference"]
        >= float(
            gates["nonidentity_transmittance_max_absolute_difference_minimum"]
        ),
        "row_partition_exact": bool(partitions_exact),
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _hash(
            root / "configs/u6_p5p_measured_mtf_minimax_envelope_v1.json"
        ),
        "parent_spatial_budget_id": budget.budget_id,
        "probe_rows": rows,
        "array_hashes": dict(sorted(hashes.items())),
        **maxima,
        "constant_transmittance_max_absolute_error": constant_trans_error,
        "constant_density_max_absolute_error": constant_density_error,
        "density_range": [density_min, density_max],
        "transmittance_range": [trans_min, trans_max],
        "row_partition_exact": partitions_exact,
        "diagnostic_sha256": diagnostic_sha,
        "gate_results": gate_results,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": automatic_pass,
        "decision": "retain_minimax_envelope_open_photographic_challenger"
        if automatic_pass
        else "close_minimax_envelope",
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "MeasuredMtfEnvelopeError",
    "evaluate_minimax_envelope",
    "validate_contract",
]
