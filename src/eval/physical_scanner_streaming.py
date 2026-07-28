"""Exact context-bound row streaming audit for U6.P6B scanner profiles."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_scanner_profile import _profile
from src.film_physics import (
    apply_scanner_profile,
    apply_scanner_profile_row_tiled,
    compile_scanner_context,
    required_scanner_halo,
)


SCHEMA = "neuro_film.u6_p6b_scanner_streaming_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6B contract")
    return payload


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def evaluate_scanner_streaming(
    contract: dict[str, Any], p6a_contract: dict[str, Any]
) -> dict[str, Any]:
    inputs = contract["synthetic_inputs"]
    rng = np.random.default_rng(int(inputs["seed"]))
    rows: list[dict[str, Any]] = []
    context_ids: set[str] = set()
    wrong_context_rejected = True
    for shape_row in inputs["shapes"]:
        shape = tuple(int(value) for value in shape_row)
        transmittance = rng.uniform(
            float(inputs["minimum_transmittance"]),
            float(inputs["maximum_transmittance"]),
            size=(*shape, 3),
        )
        for profile_name in inputs["profiles"]:
            profile = _profile(p6a_contract["profiles"][profile_name])
            context = compile_scanner_context(transmittance, profile)
            repeated_context = compile_scanner_context(transmittance, profile)
            context_exact = context == repeated_context
            context_ids.add(context.context_id)
            full = apply_scanner_profile(
                transmittance,
                profile,
                pixel_pitch_um=1.0,
                context=context,
            )
            repeat = apply_scanner_profile(
                transmittance,
                profile,
                pixel_pitch_um=1.0,
                context=context,
            )
            partitions: dict[str, dict[str, Any]] = {}
            for tile_rows in inputs["row_partitions"]:
                tiled = apply_scanner_profile_row_tiled(
                    transmittance,
                    profile,
                    pixel_pitch_um=1.0,
                    context=context,
                    tile_rows=int(tile_rows),
                )
                partitions[str(tile_rows)] = {
                    "exact": np.array_equal(full, tiled),
                    "sha256": _array_sha256(tiled),
                }
            try:
                apply_scanner_profile_row_tiled(
                    transmittance,
                    profile,
                    pixel_pitch_um=1.0,
                    context=replace(context, profile_id="wrong-profile"),
                    tile_rows=int(inputs["row_partitions"][0]),
                )
            except ValueError:
                pass
            else:
                wrong_context_rejected = False
            try:
                apply_scanner_profile_row_tiled(
                    transmittance,
                    profile,
                    pixel_pitch_um=1.0,
                    context=replace(
                        context, full_shape=(shape[0] + 1, shape[1])
                    ),
                    tile_rows=int(inputs["row_partitions"][0]),
                )
            except ValueError:
                pass
            else:
                wrong_context_rejected = False
            rows.append(
                {
                    "shape": list(shape),
                    "profile": profile_name,
                    "context_id": context.context_id,
                    "context_repeat_exact": context_exact,
                    "required_halo_rows": required_scanner_halo(
                        profile, pixel_pitch_um=1.0
                    ),
                    "full_sha256": _array_sha256(full),
                    "repeat_exact": np.array_equal(full, repeat),
                    "partitions": partitions,
                    "finite_bounded": bool(
                        np.all(np.isfinite(full))
                        and np.all(full >= 0.0)
                        and np.all(full <= 1.0)
                    ),
                }
            )
    gates = contract["automatic_gates"]
    decisions = {
        "full_vs_tiled": all(
            all(item["exact"] for item in row["partitions"].values())
            for row in rows
        ),
        "repeat": all(row["repeat_exact"] for row in rows),
        "context": all(row["context_repeat_exact"] for row in rows)
        and len(context_ids) == len(rows),
        "domain": all(row["finite_bounded"] for row in rows),
        "wrong_context": wrong_context_rejected,
        "halo": max(row["required_halo_rows"] for row in rows)
        <= int(gates["maximum_required_halo_rows"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p6b_scanner_streaming_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "rows": rows,
        "wrong_context_rejected": wrong_context_rejected,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": evidence_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
