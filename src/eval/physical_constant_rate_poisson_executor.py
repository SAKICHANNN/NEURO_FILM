"""U6.P4P exact-semantics constant-rate Poisson executor candidate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.film_physics.density_conditioned_structure import (
    counter_poisson_rate_field,
)
from src.film_physics.structure_compiler import counter_uniform_region


SCHEMA = "neuro_film.u6_p4p_constant_rate_poisson_executor_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u6_p4p_constant_rate_poisson_executor_report.v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4P parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4P contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    candidate = contract["candidate"]
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or candidate["general_spatial_rate_path_change_allowed"]
        or candidate["default_integration_allowed"]
    ):
        raise ValueError("unsupported U6.P4P contract")
    parents = contract["parents"]
    decision = _load_exact(
        root,
        parents["p4o_decision"],
        parents["p4o_decision_sha256"],
    )
    if (
        decision["decision"]
        != "close_python_scale_retain_small_stationary_p4h_open_executor_optimization"
        or not decision["next_leaf"].startswith("U6.P4P")
    ):
        raise ValueError("U6.P4O did not open executor optimization")
    return contract


def _component_constant_counts(
    rate_value: float,
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    uniform = counter_uniform_region(
        full_shape,
        origin_yx=origin_yx,
        shape=shape,
        seed=seed,
    )
    probability = float(np.exp(-np.float64(rate_value)))
    cumulative = probability
    maximum_uniform = float(np.max(uniform, initial=0.0))
    thresholds = [cumulative]
    order = 0
    while maximum_uniform > cumulative:
        order += 1
        if order > 1024:
            raise RuntimeError(
                "constant-rate Poisson recurrence did not converge"
            )
        probability *= rate_value / order
        cumulative += probability
        thresholds.append(cumulative)
    counts = np.searchsorted(
        np.asarray(thresholds, dtype=np.float64),
        uniform,
        side="left",
    ).astype(np.uint16)
    counts.setflags(write=False)
    return counts


def counter_poisson_constant_rate_field(
    rate_value: float,
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    seed: int,
    maximum_rate: float | None = None,
) -> np.ndarray:
    """Sample a bit-constant rate field using the legacy decomposition."""

    value = float(rate_value)
    if (
        not math.isfinite(value)
        or value < 0.0
        or value > 4096.0
        or len(full_shape) != 2
        or len(origin_yx) != 2
        or len(shape) != 2
        or any(
            not isinstance(item, int) or item < 0
            for item in (*full_shape, *origin_yx, *shape)
        )
        or origin_yx[0] + shape[0] > full_shape[0]
        or origin_yx[1] + shape[1] > full_shape[1]
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("invalid constant-rate Poisson inputs")
    decomposition_maximum = (
        value if maximum_rate is None else float(maximum_rate)
    )
    if (
        not math.isfinite(decomposition_maximum)
        or decomposition_maximum < value
        or decomposition_maximum > 4096.0
    ):
        raise ValueError("Poisson decomposition maximum is invalid")
    components = max(1, int(math.ceil(decomposition_maximum / 64.0)))
    if components == 1:
        return _component_constant_counts(
            value,
            full_shape,
            origin_yx=origin_yx,
            shape=shape,
            seed=seed,
        )
    component_rate = value / float(components)
    total = np.zeros(shape, dtype=np.uint32)
    for component in range(components):
        component_seed = (
            seed + component * 0x9E3779B97F4A7C15
        ) % (2**64)
        total += _component_constant_counts(
            component_rate,
            full_shape,
            origin_yx=origin_yx,
            shape=shape,
            seed=component_seed,
        ).astype(np.uint32)
    if np.any(total > np.iinfo(np.uint16).max):
        raise RuntimeError("Poisson superposition exceeds uint16")
    output = total.astype(np.uint16)
    output.setflags(write=False)
    return output


def _legacy(
    rate_value: float,
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    seed: int,
    maximum_rate: float | None,
) -> np.ndarray:
    return counter_poisson_rate_field(
        np.full(shape, rate_value, dtype=np.float64),
        full_shape,
        origin_yx=origin_yx,
        seed=seed,
        maximum_rate=maximum_rate,
    )


def _evaluate_exactness(contract: dict[str, Any]) -> list[dict[str, Any]]:
    spec = contract["exactness_matrix"]
    padding = tuple(int(value) for value in spec["full_shape_padding"])
    rows = []
    for value in (float(item) for item in spec["rate_values"]):
        for maximum in (
            float(item) for item in spec["decomposition_maxima"]
        ):
            if maximum < value:
                continue
            for shape_row in spec["shapes"]:
                shape = tuple(int(item) for item in shape_row)
                for origin_row in spec["origins"]:
                    origin = tuple(int(item) for item in origin_row)
                    full_shape = (
                        origin[0] + shape[0] + padding[0],
                        origin[1] + shape[1] + padding[1],
                    )
                    for seed in (int(item) for item in spec["seeds"]):
                        legacy = _legacy(
                            value,
                            full_shape,
                            origin_yx=origin,
                            shape=shape,
                            seed=seed,
                            maximum_rate=maximum,
                        )
                        candidate = counter_poisson_constant_rate_field(
                            value,
                            full_shape,
                            origin_yx=origin,
                            shape=shape,
                            seed=seed,
                            maximum_rate=maximum,
                        )
                        repeat = counter_poisson_constant_rate_field(
                            value,
                            full_shape,
                            origin_yx=origin,
                            shape=shape,
                            seed=seed,
                            maximum_rate=maximum,
                        )
                        rows.append(
                            {
                                "rate_value": value,
                                "decomposition_maximum": maximum,
                                "shape": list(shape),
                                "origin_yx": list(origin),
                                "seed": seed,
                                "byte_equal": bool(
                                    np.array_equal(legacy, candidate)
                                ),
                                "repeat_exact": bool(
                                    np.array_equal(candidate, repeat)
                                ),
                                "maximum_count": int(
                                    np.max(candidate, initial=0)
                                ),
                                "output_sha256": hashlib.sha256(
                                    candidate.tobytes(order="C")
                                ).hexdigest(),
                            }
                        )
    return rows


def _failure_parity() -> list[dict[str, Any]]:
    cases = [
        (-1.0, 1.0),
        (math.nan, 1.0),
        (2.0, 1.0),
        (0.0, math.inf),
    ]
    rows = []
    for value, maximum in cases:
        legacy_failed = False
        candidate_failed = False
        try:
            _legacy(
                value,
                (5, 7),
                origin_yx=(0, 0),
                shape=(5, 7),
                seed=0,
                maximum_rate=maximum,
            )
        except ValueError:
            legacy_failed = True
        try:
            counter_poisson_constant_rate_field(
                value,
                (5, 7),
                origin_yx=(0, 0),
                shape=(5, 7),
                seed=0,
                maximum_rate=maximum,
            )
        except ValueError:
            candidate_failed = True
        rows.append(
            {
                "rate_value": repr(value),
                "maximum_rate": repr(maximum),
                "legacy_failed": legacy_failed,
                "candidate_failed": candidate_failed,
            }
        )
    return rows


def _performance(contract: dict[str, Any]) -> dict[str, Any]:
    spec = contract["performance"]
    shape = tuple(int(value) for value in spec["shape"])
    value = float(spec["rate_value"])
    maximum = float(spec["decomposition_maximum"])
    seed = int(spec["seed"])
    legacy_seconds = []
    candidate_seconds = []
    hashes = []
    for _ in range(int(spec["interleaved_runs"])):
        started = perf_counter()
        legacy = _legacy(
            value,
            shape,
            origin_yx=(0, 0),
            shape=shape,
            seed=seed,
            maximum_rate=maximum,
        )
        legacy_seconds.append(perf_counter() - started)
        started = perf_counter()
        candidate = counter_poisson_constant_rate_field(
            value,
            shape,
            origin_yx=(0, 0),
            shape=shape,
            seed=seed,
            maximum_rate=maximum,
        )
        candidate_seconds.append(perf_counter() - started)
        if not np.array_equal(legacy, candidate):
            raise RuntimeError("U6.P4P performance outputs differ")
        hashes.append(
            hashlib.sha256(candidate.tobytes(order="C")).hexdigest()
        )
    legacy_median = float(np.median(legacy_seconds))
    candidate_median = float(np.median(candidate_seconds))
    return {
        "legacy_seconds": legacy_seconds,
        "candidate_seconds": candidate_seconds,
        "legacy_median_seconds": legacy_median,
        "candidate_median_seconds": candidate_median,
        "median_speedup": legacy_median / candidate_median,
        "candidate_hashes": hashes,
    }


def evaluate_constant_rate_executor(
    contract: dict[str, Any],
) -> dict[str, Any]:
    exactness = _evaluate_exactness(contract)
    failures = _failure_parity()
    performance = _performance(contract)
    gates = contract["automatic_gates"]
    checks = {
        "exact_matrix_byte_equal": all(
            row["byte_equal"] for row in exactness
        )
        == bool(gates["exact_matrix_byte_equal"]),
        "candidate_repeat_exact": all(
            row["repeat_exact"] for row in exactness
        )
        == bool(gates["candidate_repeat_exact"]),
        "invalid_input_failure_parity": all(
            row["legacy_failed"] and row["candidate_failed"]
            for row in failures
        )
        == bool(gates["invalid_input_failure_parity"]),
        "median_speedup": performance["median_speedup"]
        >= float(gates["minimum_median_speedup"]),
        "candidate_wall": max(performance["candidate_seconds"])
        <= float(gates["maximum_candidate_wall_seconds"]),
        "maximum_count": max(
            row["maximum_count"] for row in exactness
        )
        <= int(gates["maximum_output_count"]),
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "exactness_rows": exactness,
        "failure_parity": failures,
        "performance": performance,
        "checks": checks,
        "automatic_pass": passed,
        "decision": (
            contract["branch_rule"]["pass"]
            if passed
            else contract["branch_rule"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "counter_poisson_constant_rate_field",
    "evaluate_constant_rate_executor",
    "load_contract",
]
