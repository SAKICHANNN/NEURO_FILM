"""P4HL caller-held-profile CPU consumer audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.neutral_base_photographic_ablation import sha256_file
from src.film_physics.bounded_photographic_runtime import (
    BoundedPhotographicCpuRuntime,
)

SCHEMA = "neuro-film.u6-p4hl-bounded-photographic-cpu-consumer-contract.v1"


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HL contract")
    parents = contract["parents"]
    evidence_path = root / parents["p4hk_evidence"]["path"]
    profile_path = root / parents["profile"]["path"]
    if (
        sha256_file(evidence_path) != parents["p4hk_evidence"]["sha256"]
        or sha256_file(profile_path) != parents["profile"]["sha256"]
    ):
        raise ValueError("P4HL parent drift")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("decision") != parents["p4hk_evidence"]["required_decision"]:
        raise ValueError("P4HL P4HK decision drift")
    expected_bundle = parents["profile"]["bundle_sha256"]
    runtime = BoundedPhotographicCpuRuntime.load(
        profile_path, expected_bundle_sha256=expected_bundle
    )
    fixture = contract["fixture"]
    rng = np.random.default_rng(int(fixture["source_seed"]))
    source = rng.uniform(
        0.001,
        0.999,
        size=(int(fixture["height"]), int(fixture["width"]), 3),
    ).astype(np.float32)
    original = source.copy()
    first_output, first_receipt = runtime.render(
        source, source_index=int(fixture["source_index"])
    )
    second_output, second_receipt = runtime.render(
        source, source_index=int(fixture["source_index"])
    )
    reloaded = BoundedPhotographicCpuRuntime.load(
        profile_path, expected_bundle_sha256=expected_bundle
    )
    third_output, third_receipt = reloaded.render(
        source, source_index=int(fixture["source_index"])
    )
    wrong_bundle_rejected = False
    try:
        BoundedPhotographicCpuRuntime.load(
            profile_path, expected_bundle_sha256="0" * 64
        )
    except ValueError:
        wrong_bundle_rejected = True
    invalid_rejected = True
    invalid_sources = [
        source.astype(np.float64),
        source[:, ::2],
        np.full((2, 2, 3), np.nan, dtype=np.float32),
        np.full((2, 2, 3), 1.1, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ]
    for invalid in invalid_sources:
        try:
            runtime.render(invalid, source_index=0)
        except ValueError:
            pass
        else:
            invalid_rejected = False
    for invalid_index in (-1, True, 2**32):
        try:
            runtime.render(source, source_index=invalid_index)
        except ValueError:
            pass
        else:
            invalid_rejected = False
    expected_physical = fixture["expected_physical_output_sha256"]
    expected_scanner = fixture["expected_scanner_output_sha256"]
    receipt_bound = (
        first_receipt["bundle_sha256"] == expected_bundle
        and first_receipt["source_index"] == fixture["source_index"]
        and first_receipt["input_float32_sha256"]
        == hashlib.sha256(memoryview(source).cast("B")).hexdigest()
        and first_receipt["physical_float32_sha256"] == expected_physical
        and first_receipt["scanner_float32_sha256"] == expected_scanner
        and len(first_receipt["layer_field_seeds"]) == 3
        and len(first_receipt["receipt_id"]) == 64
    )
    checks = {
        "expected_fixture": first_receipt["physical_float32_sha256"]
        == expected_physical
        and first_receipt["scanner_float32_sha256"] == expected_scanner,
        "repeat_apply": np.array_equal(first_output, second_output),
        "reload_apply": np.array_equal(first_output, third_output),
        "receipt_repeat": first_receipt == second_receipt == third_receipt,
        "receipt_bound": receipt_bound,
        "input_unchanged": np.array_equal(source, original),
        "wrong_bundle_rejected": wrong_bundle_rejected,
        "invalid_rejected": invalid_rejected,
        "output_not_alias": not np.shares_memory(source, first_output),
    }
    stable = {
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "bundle_sha256": expected_bundle,
        "output_sha256": first_receipt["scanner_float32_sha256"],
        "receipt_id": first_receipt["receipt_id"],
        "receipt": first_receipt,
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "decision": contract["decision_if_pass"]
        if all(checks.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("contract", "result"),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


__all__ = ["evaluate"]
