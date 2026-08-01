"""U6.P5O measured-total MTF nonlinear-domain placement audit."""

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
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.measured_mtf_budget import MeasuredTotalFilmSpatialBudget

SCHEMA = "neuro_film.u6_p5o_measured_mtf_domain_placement_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p5o_measured_mtf_domain_placement_report.v1"
ARMS = ("exposure_domain", "developed_density_domain", "scan_transmittance_domain")


class MeasuredMtfPlacementError(RuntimeError):
    """Raised when the frozen P5O contract or its exact parents drift."""


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
        raise MeasuredMtfPlacementError("P5O paths must be repository-relative")
    return path


def _load_exact(root: Path, path: str, expected: str) -> Any:
    relative = _relative(path)
    resolved = root / relative
    if _hash(resolved) != expected:
        raise MeasuredMtfPlacementError(f"P5O parent hash mismatch: {relative}")
    if resolved.suffix.lower() == ".json":
        return json.loads(resolved.read_text(encoding="utf-8"))
    return resolved


def validate_contract(config: Mapping[str, Any]) -> None:
    probes = config.get("probes", {})
    gates = config.get("gates", {})
    if (
        config.get("schema") != SCHEMA
        or tuple(config.get("arms", {})) != ARMS
        or probes.get("log_exposure")
        != {"toe": -2.5, "mid": -0.5, "shoulder": 1.5}
        or probes.get("relative_modulation_amplitudes") != [0.05, 0.5]
        or probes.get("patterns")
        != ["vertical_step", "checker_3x5", "seeded_random"]
        or probes.get("row_partitions") != [7, 17, 31]
        or gates.get("material_nonlinear_density_max_absolute_difference_minimum")
        != 0.002
        or gates.get(
            "material_nonlinear_transmittance_max_absolute_difference_minimum"
        )
        != 0.0000152587890625
        or config.get("source_boundary", {}).get(
            "measurement_protocol_identifies_exact_operator_domain"
        )
        is not False
    ):
        raise MeasuredMtfPlacementError("P5O frozen contract drift")
    if [list(pair) for pair in config["comparison"]["pairwise_arms"]] != [
        [ARMS[0], ARMS[1]],
        [ARMS[0], ARMS[2]],
        [ARMS[1], ARMS[2]],
    ]:
        raise MeasuredMtfPlacementError("P5O pairwise arm contract drift")


def _density_to_transmittance(density: np.ndarray) -> np.ndarray:
    return np.power(10.0, -np.asarray(density, dtype=np.float64))


def _transmittance_to_density(transmittance: np.ndarray) -> np.ndarray:
    values = np.asarray(transmittance, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise MeasuredMtfPlacementError("P5O transmittance must be finite and positive")
    return -np.log10(values)


def _apply_arms(
    exposure: np.ndarray,
    *,
    sensitometry: Any,
    budget: MeasuredTotalFilmSpatialBudget,
    tile_rows: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    blur = budget.apply if tile_rows is None else (
        lambda values: budget.apply_row_tiled(values, tile_rows=tile_rows)
    )
    exposure_blurred = blur(exposure)
    density_unblurred = sensitometry.apply(exposure)
    density_exposure = sensitometry.apply(exposure_blurred)
    density_density = blur(density_unblurred)
    transmittance_unblurred = _density_to_transmittance(density_unblurred)
    transmittance_trans = blur(transmittance_unblurred)
    return {
        "exposure_domain": {
            "density": density_exposure,
            "transmittance": _density_to_transmittance(density_exposure),
        },
        "developed_density_domain": {
            "density": density_density,
            "transmittance": _density_to_transmittance(density_density),
        },
        "scan_transmittance_domain": {
            "density": _transmittance_to_density(transmittance_trans),
            "transmittance": transmittance_trans,
        },
    }


def _signed_pattern(name: str, shape: tuple[int, int], seed: int) -> np.ndarray:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    if name == "vertical_step":
        pattern = np.where(x < width // 2, -1.0, 1.0)
    elif name == "checker_3x5":
        pattern = np.where(((x // 3) + (y // 5)) % 2 == 0, -1.0, 1.0)
    elif name == "seeded_random":
        pattern = np.random.default_rng(seed).uniform(-1.0, 1.0, size=shape)
    else:
        raise MeasuredMtfPlacementError(f"unsupported P5O pattern: {name}")
    return np.repeat(pattern[..., None], 3, axis=2)


def _pair_metrics(
    arms: Mapping[str, Mapping[str, np.ndarray]], pairs: list[list[str]]
) -> dict[str, dict[str, float]]:
    metrics = {}
    for first, second in pairs:
        key = f"{first}--{second}"
        metrics[key] = {}
        for domain in ("density", "transmittance"):
            difference = np.abs(arms[first][domain] - arms[second][domain])
            metrics[key][f"{domain}_max_absolute_difference"] = float(
                np.max(difference)
            )
            metrics[key][f"{domain}_mean_absolute_difference"] = float(
                np.mean(difference)
            )
    return metrics


def _save_diagnostic(
    path: Path, rows: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]
) -> str:
    figure, axes = plt.subplots(3, 4, figsize=(12.5, 9.0))
    for row_index, regime in enumerate(("toe", "mid", "shoulder")):
        arms = rows[regime]
        transmittances = [arms[name]["transmittance"] for name in ARMS]
        for column, (name, image) in enumerate(zip(ARMS, transmittances, strict=True)):
            axes[row_index, column].imshow(
                np.clip(image, 0.0, 1.0) ** (1.0 / 2.2), interpolation="nearest"
            )
            axes[row_index, column].set_title(f"{regime}: {name.replace('_domain', '')}")
            axes[row_index, column].axis("off")
        spread = np.max(np.stack(transmittances), axis=0) - np.min(
            np.stack(transmittances), axis=0
        )
        scale = max(float(np.max(spread)), np.finfo(np.float64).eps)
        axes[row_index, 3].imshow(np.clip(spread / scale, 0.0, 1.0))
        axes[row_index, 3].set_title(f"arm spread / {scale:.4g}")
        axes[row_index, 3].axis("off")
    figure.suptitle("P5O measured-total MTF placement sensitivity (50% checker)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, metadata={"Software": "neuro-film"})
    plt.close(figure)
    return _hash(path)


def evaluate_domain_placement(
    *, root: Path, config: Mapping[str, Any], diagnostic_path: Path
) -> dict[str, Any]:
    validate_contract(config)
    parents = config["parents"]
    p5n = _load_exact(root, parents["p5n_decision"], parents["p5n_decision_sha256"])
    if p5n.get("decision") != "close_residual_use_measured_total_replacement":
        raise MeasuredMtfPlacementError("P5O requires the exact P5N closure")
    lod = _load_exact(root, parents["p5l_bundle"], parents["p5l_bundle_sha256"])
    sens_config = _load_exact(
        root,
        parents["sensitometry_contract"],
        parents["sensitometry_contract_sha256"],
    )
    _load_exact(
        root,
        parents["kodak_250d_technical_data"],
        parents["kodak_250d_technical_data_sha256"],
    )
    budget = MeasuredTotalFilmSpatialBudget.from_lod_bundle(lod)
    sensitometry = build_operator(sens_config)
    shape = tuple(int(value) for value in config["probes"]["shape"])
    pairs = config["comparison"]["pairwise_arms"]
    rows = []
    diagnostic_rows: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    density_max = 0.0
    transmittance_max = 0.0
    all_density = []
    all_transmittance = []
    all_partition_exact = True
    array_hashes = {}
    for regime, log_exposure in config["probes"]["log_exposure"].items():
        baseline = float(sensitometry.encoder.inverse(np.asarray(log_exposure)))
        for amplitude in config["probes"]["relative_modulation_amplitudes"]:
            for pattern_name in config["probes"]["patterns"]:
                signed = _signed_pattern(
                    pattern_name,
                    shape,
                    int(config["probes"]["random_seed"]),
                )
                exposure = baseline * (1.0 + float(amplitude) * signed)
                arms = _apply_arms(
                    exposure, sensitometry=sensitometry, budget=budget
                )
                partitions = {}
                for tile_rows in config["probes"]["row_partitions"]:
                    tiled = _apply_arms(
                        exposure,
                        sensitometry=sensitometry,
                        budget=budget,
                        tile_rows=int(tile_rows),
                    )
                    partitions[str(tile_rows)] = all(
                        np.array_equal(arms[arm][domain], tiled[arm][domain])
                        for arm in ARMS
                        for domain in ("density", "transmittance")
                    )
                all_partition_exact &= all(partitions.values())
                metrics = _pair_metrics(arms, pairs)
                density_max = max(
                    density_max,
                    max(
                        item["density_max_absolute_difference"]
                        for item in metrics.values()
                    ),
                )
                transmittance_max = max(
                    transmittance_max,
                    max(
                        item["transmittance_max_absolute_difference"]
                        for item in metrics.values()
                    ),
                )
                key = f"{regime}-{float(amplitude):.2f}-{pattern_name}"
                for arm in ARMS:
                    all_density.append(arms[arm]["density"])
                    all_transmittance.append(arms[arm]["transmittance"])
                    array_hashes[f"{key}-{arm}-density"] = array_sha256(
                        arms[arm]["density"]
                    )
                    array_hashes[f"{key}-{arm}-transmittance"] = array_sha256(
                        arms[arm]["transmittance"]
                    )
                rows.append(
                    {
                        "probe": key,
                        "linear_exposure_baseline": baseline,
                        "pairwise": metrics,
                        "row_partition_exact": partitions,
                    }
                )
                if float(amplitude) == 0.5 and pattern_name == "checker_3x5":
                    diagnostic_rows[regime] = arms

    constant_density_max = 0.0
    constant_transmittance_max = 0.0
    for value in config["probes"]["constant_values"]:
        field = np.full((37, 53, 3), float(value), dtype=np.float64)
        arms = _apply_arms(field, sensitometry=sensitometry, budget=budget)
        metrics = _pair_metrics(arms, pairs)
        constant_density_max = max(
            constant_density_max,
            max(row["density_max_absolute_difference"] for row in metrics.values()),
        )
        constant_transmittance_max = max(
            constant_transmittance_max,
            max(
                row["transmittance_max_absolute_difference"]
                for row in metrics.values()
            ),
        )

    random_exposure = np.random.default_rng(20260816).uniform(
        0.0, 1.0, size=(63, 79, 3)
    )
    affine_after = 0.2 + 0.4 * budget.apply(random_exposure)
    affine_before = budget.apply(0.2 + 0.4 * random_exposure)
    affine_commutation = float(np.max(np.abs(affine_after - affine_before)))
    density_min = float(min(np.min(value) for value in all_density))
    trans_min = float(min(np.min(value) for value in all_transmittance))
    trans_max = float(max(np.max(value) for value in all_transmittance))
    finite = all(
        np.all(np.isfinite(value)) for value in all_density + all_transmittance
    )
    diagnostic_sha = _save_diagnostic(diagnostic_path, diagnostic_rows)
    gates = config["gates"]
    gate_results = {
        "finite": bool(finite),
        "density_range": density_min >= float(gates["density_minimum"]),
        "transmittance_range": trans_min
        > float(gates["transmittance_minimum_exclusive"])
        and trans_max <= float(gates["transmittance_maximum"]),
        "constant_density_equivalence": constant_density_max
        <= float(gates["constant_arm_density_max_absolute_difference"]),
        "constant_transmittance_equivalence": constant_transmittance_max
        <= float(gates["constant_arm_transmittance_max_absolute_difference"]),
        "affine_exposure_density_commutation": affine_commutation
        <= float(gates["affine_exposure_density_commutation_max_absolute_difference"]),
        "material_nonlinear_density_noncommutation": density_max
        >= float(
            gates["material_nonlinear_density_max_absolute_difference_minimum"]
        ),
        "material_nonlinear_transmittance_noncommutation": transmittance_max
        >= float(
            gates[
                "material_nonlinear_transmittance_max_absolute_difference_minimum"
            ]
        ),
        "row_partition_exact": bool(all_partition_exact),
    }
    automatic_pass = all(gate_results.values())
    if automatic_pass:
        decision = "placement_unidentified_open_bounded_envelope"
    elif all(
        value
        for name, value in gate_results.items()
        if not name.startswith("material_nonlinear_")
    ):
        decision = "retain_developed_density_numerical_arm"
    else:
        decision = "close_measured_total_domain_placement"
    stable = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _hash(
            root / "configs/u6_p5o_measured_mtf_domain_placement_v1.json"
        ),
        "parent_lod_bundle_id": lod["lod_bundle_id"],
        "parent_spatial_budget_id": budget.budget_id,
        "sensitometry_schema": sensitometry.to_dict()["schema"],
        "probe_rows": rows,
        "array_hashes": dict(sorted(array_hashes.items())),
        "material_density_max_absolute_difference": density_max,
        "material_transmittance_max_absolute_difference": transmittance_max,
        "constant_density_max_absolute_difference": constant_density_max,
        "constant_transmittance_max_absolute_difference": constant_transmittance_max,
        "affine_exposure_density_commutation_max_absolute_difference": affine_commutation,
        "density_range": [density_min, float(max(np.max(value) for value in all_density))],
        "transmittance_range": [trans_min, trans_max],
        "row_partition_exact": all_partition_exact,
        "diagnostic_sha256": diagnostic_sha,
        "gate_results": gate_results,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "ARMS",
    "REPORT_SCHEMA",
    "SCHEMA",
    "MeasuredMtfPlacementError",
    "evaluate_domain_placement",
    "validate_contract",
]
