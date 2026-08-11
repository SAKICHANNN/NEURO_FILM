"""CB14 fixed CB11 versus fixed AO6 on a disjoint OOD population."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.characteristic_vs_ao6_fresh import _sheet
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import _new_boundary_fraction
from src.eval.kci_velvia_tone_photographic_stress import _load_rgb, _save_rgb
from src.film_physics.profile_consumer import validate_standalone_profile_artifact

SCHEMA = "neuro_film.u5_r2cb14_characteristic_vs_ao6_ood_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb14_characteristic_vs_ao6_ood_report.v1"
EXPERIMENT_ID = "U5.R2CB14"


class CharacteristicVsAo6OodError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicVsAo6OodError("CB14 contract structure drift")
    return payload


def _validate(config: Mapping[str, Any], root: Path):
    parents = config["parents"]
    cb12_decision = _load_exact_json(
        root, parents["cb12_decision_path"], parents["cb12_decision_sha256"]
    )
    cb13_decision = _load_exact_json(
        root, parents["cb13_decision_path"], parents["cb13_decision_sha256"]
    )
    cb12 = _load_exact_json(
        root, parents["cb12_contract_path"], parents["cb12_contract_sha256"]
    )
    if (
        cb12_decision.get("status") != parents["cb12_required_status"]
        or not cb12_decision.get("pass")
        or cb13_decision.get("status") != parents["cb13_required_status"]
        or not cb13_decision.get("partial_available_cohort_pass")
    ):
        raise CharacteristicVsAo6OodError("CB14 parent decision drift")

    candidate = cb12["candidate"]
    cb11 = _load_exact_json(
        root, candidate["contract_path"], candidate["contract_sha256"]
    )
    curve = _compiled_curve(load_cb6(root / cb11["parents"]["cb6_contract_path"]))
    ao6 = cb12["ao6"]
    artifact_report = _load_exact_json(
        root,
        ao6["frozen_artifact_report_path"],
        ao6["frozen_artifact_report_sha256"],
    )
    artifact = artifact_report["artifact"]
    if (
        artifact_report.get("artifact_canonical_sha256")
        != ao6["frozen_artifact_canonical_sha256"]
        or artifact.get("bundle_sha256") != ao6["frozen_bundle_sha256"]
    ):
        raise CharacteristicVsAo6OodError("AO6 artifact drift")
    validate_standalone_profile_artifact(artifact)

    population = config["population"]
    decision = _load_exact_json(
        root, population["decision_path"], population["decision_sha256"]
    )
    if decision.get("status") != population["required_status"]:
        raise CharacteristicVsAo6OodError("OOD source decision drift")
    rows = _load_exact_json(
        root, population["manifest_path"], population["manifest_sha256"]
    )
    if (
        len(rows) != population["source_count_exact"]
        or len({row["make"] for row in rows}) != population["camera_make_count_exact"]
        or len({row["id"] for row in rows}) != len(rows)
    ):
        raise CharacteristicVsAo6OodError("OOD population drift")
    for row in rows:
        if hash_file(root / row["decoded_path"]) != row["decoded_sha256"]:
            raise CharacteristicVsAo6OodError(f"OOD source hash drift: {row['id']}")
    return cb11, ao6, artifact, rows, curve


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    cb11, ao6_config, artifact, source_rows, curve = _validate(config, root)
    if output_dir.exists():
        raise FileExistsError("CB14 is create-only")
    output_dir.mkdir(parents=True)
    operator = cb11["operator"]
    protocol = config["protocol"]
    rows = []
    for source_row in source_rows:
        source = _load_rgb(
            root / source_row["decoded_path"],
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        ao6 = render_fixed_pair(source, artifact, ao6_config["component"])[
            ao6_config["arm_id"]
        ]
        candidate, _, _ = apply_characteristic_luma_chroma(
            source,
            curve,
            weights=np.asarray(operator["luminance_weights"], dtype=np.float64),
            strength=float(operator["nominal_strength"]),
            boundary_epsilon=float(operator["boundary_epsilon"]),
        )
        row_dir = output_dir / "renders" / source_row["id"]
        source_path = row_dir / "source.png"
        ao6_path = row_dir / "ao6.png"
        candidate_path = row_dir / "candidate.png"
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "source_path": str(source_path),
                "source_sha256": _save_rgb(source, source_path),
                "ao6_path": str(ao6_path),
                "ao6_sha256": _save_rgb(ao6, ao6_path),
                "candidate_path": str(candidate_path),
                "candidate_sha256": _save_rgb(candidate, candidate_path),
                "candidate_new_boundary_fraction": _new_boundary_fraction(
                    source, candidate, float(operator["boundary_epsilon"])
                ),
                "ao6_new_boundary_fraction": _new_boundary_fraction(
                    source, ao6, float(operator["boundary_epsilon"])
                ),
            }
        )
    automatic = (
        len(rows) == protocol["aggregate_choice_denominator"] // protocol["rounds"]
        and max(row["candidate_new_boundary_fraction"] for row in rows)
        <= protocol["maximum_new_hard_boundary_fraction"]
        and max(row["ao6_new_boundary_fraction"] for row in rows)
        <= protocol["maximum_new_hard_boundary_fraction"]
    )
    sheets = []
    mappings = {}
    if automatic:
        for round_index in range(protocol["rounds"]):
            path = output_dir / "blind" / f"round_{round_index + 1}.png"
            digest, mapping = _sheet(
                rows, path, seed=20260811 + 1400, round_index=round_index
            )
            sheets.append(
                {
                    "round": round_index + 1,
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": digest,
                }
            )
            mappings[str(round_index + 1)] = mapping
    for row in rows:
        for key in ("source_path", "ao6_path", "candidate_path"):
            row[key] = Path(row[key]).relative_to(output_dir).as_posix()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": hash_file(
            root / "configs/u5_r2cb14_characteristic_vs_ao6_ood_v1.json"
        ),
        "rows": rows,
        "automatic_pass": automatic,
        "blind_sheets": sheets,
        "sealed_mappings": mappings,
        "decision": (
            "open_severe_review_then_blind_adjudication"
            if automatic
            else "close_before_visual_review"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = ["CharacteristicVsAo6OodError", "evaluate", "load_contract", "write_report"]
