"""U5.R2CB12 fixed CB11 versus fixed AO6 on BH1S display proxies."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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
from src.film_physics.profile_consumer import compile_standalone_profile_artifact

SCHEMA = "neuro_film.u5_r2cb12_characteristic_vs_ao6_fresh_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb12_characteristic_vs_ao6_fresh_report.v1"
EXPERIMENT_ID = "U5.R2CB12"
CONTRACT_SHA256 = "3178bc089348c32ac306ce1114ed34089ea26baf674078ea98f0c8f1cc5eea6b"


class CharacteristicVsAo6FreshError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise CharacteristicVsAo6FreshError("CB12 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicVsAo6FreshError("CB12 contract structure drift")
    return payload


def _validate(config: Mapping[str, Any], root: Path):
    candidate = config["candidate"]
    decision = _load_exact_json(
        root, candidate["decision_path"], candidate["decision_sha256"]
    )
    cb11 = _load_exact_json(
        root, candidate["contract_path"], candidate["contract_sha256"]
    )
    if decision.get("decision") != candidate["required_decision"]:
        raise CharacteristicVsAo6FreshError("CB11 decision drift")
    population = config["population"]
    source_decision = _load_exact_json(
        root, population["decision_path"], population["decision_sha256"]
    )
    manifest_path = root / population["manifest_path"]
    review_path = root / population["visual_review_path"]
    if (
        hash_file(manifest_path) != population["manifest_sha256"]
        or hash_file(review_path) != population["visual_review_sha256"]
    ):
        raise CharacteristicVsAo6FreshError("BH1S evidence drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    eligible = list(review["eligible_ids"])
    rows = {row["id"]: row for row in manifest}
    if (
        source_decision["result"]["decision"] != population["required_decision"]
        or len(eligible) != population["source_count_exact"]
        or len({rows[key]["make"] for key in eligible})
        != population["camera_make_count_exact"]
    ):
        raise CharacteristicVsAo6FreshError("BH1S population drift")
    ao6 = config["ao6"]
    _load_exact_json(root, ao6["parent_contract_path"], ao6["parent_contract_sha256"])
    compiler = _load_exact_json(
        root,
        ao6["profile_compiler_config_path"],
        ao6["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(root=root, config=compiler)
    curve = _compiled_curve(load_cb6(root / cb11["parents"]["cb6_contract_path"]))
    return cb11, artifact, eligible, rows, curve


def _sheet(
    rows, output: Path, *, seed: int, round_index: int
) -> tuple[str, dict[str, list[str]]]:
    rng = random.Random(seed + round_index * 1009)
    mappings = {}
    width, height, header = 420, 280, 28
    canvas = Image.new("RGB", (width * 3, header + height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(("SOURCE", "A", "B")):
        draw.text((col * width + 8, 8), label, fill="black")
    for index, row in enumerate(rows):
        arms = ["ao6", "candidate"]
        rng.shuffle(arms)
        mappings[row["id"]] = arms
        paths = [row["source_path"], row[f"{arms[0]}_path"], row[f"{arms[1]}_path"]]
        for col, path in enumerate(paths):
            with Image.open(path) as image:
                thumb = image.convert("RGB")
            thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas.paste(
                thumb,
                (
                    col * width + (width - thumb.width) // 2,
                    header + index * height + (height - thumb.height) // 2,
                ),
            )
        draw.text(
            (4, header + index * height + 4),
            row["id"],
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, compress_level=6)
    return hash_file(output), mappings


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    cb11, artifact, eligible, source_rows, curve = _validate(config, root)
    if output_dir.exists():
        raise FileExistsError("CB12 is create-only")
    output_dir.mkdir(parents=True)
    operator = cb11["operator"]
    protocol = config["protocol"]
    rows = []
    for source_id in eligible:
        source_row = source_rows[source_id]
        source_path = root / source_row["decoded_path"]
        if hash_file(source_path) != source_row["decoded_sha256"]:
            raise CharacteristicVsAo6FreshError("decoded source drift")
        source = _load_rgb(
            source_path,
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        ao6 = render_fixed_pair(source, artifact, config["ao6"]["component"])[
            config["ao6"]["arm_id"]
        ]
        candidate, _, _ = apply_characteristic_luma_chroma(
            source,
            curve,
            weights=np.asarray(operator["luminance_weights"], dtype=np.float64),
            strength=float(operator["nominal_strength"]),
            boundary_epsilon=float(operator["boundary_epsilon"]),
        )
        row_dir = output_dir / "renders" / source_id
        source_out = row_dir / "source.png"
        ao6_out = row_dir / "ao6.png"
        candidate_out = row_dir / "candidate.png"
        rows.append(
            {
                "id": source_id,
                "make": source_row["make"],
                "source_path": str(source_out),
                "source_sha256": _save_rgb(source, source_out),
                "ao6_path": str(ao6_out),
                "ao6_sha256": _save_rgb(ao6, ao6_out),
                "candidate_path": str(candidate_out),
                "candidate_sha256": _save_rgb(candidate, candidate_out),
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
        and max(r["candidate_new_boundary_fraction"] for r in rows)
        <= protocol["maximum_new_hard_boundary_fraction"]
    )
    sheets = []
    mappings = {}
    if automatic:
        for round_index in range(protocol["rounds"]):
            path = output_dir / "blind" / f"round_{round_index + 1}.png"
            digest, mapping = _sheet(rows, path, seed=20260811, round_index=round_index)
            sheets.append(
                {"round": round_index + 1, "path": str(path), "sha256": digest}
            )
            mappings[str(round_index + 1)] = mapping
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "rows": rows,
        "automatic_pass": automatic,
        "blind_sheets": sheets,
        "sealed_mappings": mappings,
        "decision": "open_severe_review_then_blind_adjudication"
        if automatic
        else "close_before_visual_review",
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
