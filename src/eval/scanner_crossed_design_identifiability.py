"""Design-rank audit for scanner/material identifiability after P6AO."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro-film.u6-p6ap-scanner-crossed-design-identifiability-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p6ap-scanner-crossed-design-identifiability-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    ids = [row["id"] for row in value.get("pipelines", [])]
    if (
        value.get("schema") != SCHEMA
        or len(ids) != 4
        or len(set(ids)) != 4
        or value.get("candidate_effects", [None])[0] != "intercept"
        or value["repeat_noise_requirement"]["minimum_same_physical_slide_scans"] < 3
    ):
        raise ValueError("unsupported P6AP contract")
    return value


def _row(pipeline: Mapping[str, Any]) -> list[float]:
    return [
        1.0,
        float(pipeline["scanner_model"] == "Nikon LS 9000"),
        float(str(pipeline["software"]).startswith("VueScan")),
        float(pipeline["device_instance"] == "ls50_device_b"),
        float(pipeline["operator"] == "Olivier Desmaison"),
        float(pipeline["operator"] == "Harald Lampe"),
        float(int(pipeline["sampling_dpi"]) == 2800),
        float(pipeline["output_encoding"] == "48_bit_adobe_rgb_label"),
    ]


def _in_row_space(vector: np.ndarray, matrix: np.ndarray) -> tuple[bool, float]:
    projection = vector @ np.linalg.pinv(matrix) @ matrix
    error = float(np.max(np.abs(projection - vector)))
    return error <= 1e-10, error


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError("P6AP parent drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if binding.get("required_all_checks_passed") and payload.get("all_checks_passed") is not True:
            raise ValueError("P6AP source audit decision drift")
        if "required_decision" in binding and payload.get("decision") != binding["required_decision"]:
            raise ValueError("P6AP parent decision drift")

    pipelines = {row["id"]: row for row in contract["pipelines"]}
    matrix = np.asarray([_row(row) for row in contract["pipelines"]], dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix))
    singular = np.linalg.svd(matrix, compute_uv=False)
    contrasts = []
    for item in contract["frozen_contrasts"]:
        vector = np.asarray(_row(pipelines[item["left"]])) - np.asarray(_row(pipelines[item["right"]]))
        estimable, error = _in_row_space(vector, matrix)
        # A row contrast can be observed, but it is a pure named effect only when
        # exactly one non-intercept design column changes.
        changed = [contract["candidate_effects"][i] for i, value in enumerate(vector) if i and value != 0]
        pure_effect = len(changed) == 1
        observed_as_named_effect = estimable and pure_effect
        if observed_as_named_effect != bool(item["expected_estimable"]):
            raise ValueError("P6AP frozen contrast expectation drift")
        contrasts.append({
            "id": item["id"],
            "changed_effects": changed,
            "row_contrast_estimable": estimable,
            "row_space_projection_error": error,
            "pure_named_effect_estimable": observed_as_named_effect,
        })

    repeated_cells = {}
    fixed = ("scanner_model", "device_instance", "software", "operator", "sampling_dpi", "output_encoding")
    for row in contract["pipelines"]:
        key = tuple(str(row[name]) for name in fixed)
        repeated_cells.setdefault(key, []).append(row["id"])
    maximum_repeats = max(map(len, repeated_cells.values()))
    repeat_requirement = int(contract["repeat_noise_requirement"]["minimum_same_physical_slide_scans"])
    pure = [row["id"] for row in contrasts if row["pure_named_effect_estimable"]]
    identified = maximum_repeats >= repeat_requirement
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "design": {
            "observations": int(matrix.shape[0]),
            "candidate_effects": int(matrix.shape[1]),
            "rank": rank,
            "nullity": int(matrix.shape[1] - rank),
            "singular_values": singular.tolist(),
            "matrix": matrix.astype(int).tolist(),
        },
        "contrasts": contrasts,
        "pure_named_effects_identified": pure,
        "random_repeat_noise": {
            "maximum_fixed_condition_repeats": maximum_repeats,
            "required_repeats": repeat_requirement,
            "identified": identified,
            "missing_acquisition": contract["repeat_noise_requirement"],
        },
        "automatic_pass": identified,
        "decision": contract["decision_if_identified"] if identified else contract["decision_if_underidentified"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

