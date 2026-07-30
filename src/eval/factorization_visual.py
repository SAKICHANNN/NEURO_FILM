"""Blind and severe-review sheets for two fixed factorization reports."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from src.eval.density_residual_visual import _portable_path, _sheet
from src.eval.global_frontier import load_frozen_samples, sha256_file


def _report_paths(
    root: Path,
    row: dict[str, Any],
    gold_ids: set[str],
    *,
    require_automatic_pass: bool,
) -> dict[str, Path]:
    report_path = root / row["report"]
    if sha256_file(report_path) != row["report_sha256"]:
        raise ValueError("factorization report hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if require_automatic_pass and not report.get("automatic_pass"):
        raise ValueError("automatic gate forbids visual review")
    paths: dict[str, Path] = {}
    for record in report["records"]:
        if record["split"] != "gold":
            continue
        if record["candidate_id"] != row["candidate_id"]:
            raise ValueError("factorization candidate identity drift")
        sample_id = str(record["sample_id"])
        path = report_path.parent / record["output"]
        if (
            sample_id not in gold_ids
            or sample_id in paths
            or sha256_file(path) != record["output_sha256"]
        ):
            raise ValueError("factorization output identity drift")
        paths[sample_id] = path
    if paths.keys() != gold_ids:
        raise ValueError("factorization gold coverage drift")
    return paths


def build_visual_evidence(
    *,
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if (
        config.get("experiment_id")
        != "u5.r2ba1v-hue-value-residual-visual-v1"
        or config.get("status")
        != "contract_frozen_before_visual_sheet_build"
        or config.get("production_integration_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
        or config["blind"].get("score_before_mapping_reveal") is not True
        or int(config["blind"]["round_count"]) != 3
    ):
        raise ValueError("BA1V frozen boundary drift")
    samples = load_frozen_samples(root, config)
    gold_ids = [
        sample_id
        for sample_id, row in samples.items()
        if row["split"] == "gold"
    ]
    if len(gold_ids) != int(config["expected_gold_samples"]):
        raise ValueError("BA1V gold population drift")
    gold_set = set(gold_ids)
    candidate = _report_paths(
        root, config["candidate"], gold_set, require_automatic_pass=True
    )
    comparator = _report_paths(
        root, config["comparator"], gold_set, require_automatic_pass=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_path = output_dir / "direct_severe_review.png"
    _sheet(
        [
            (
                sample_id,
                [
                    ("INPUT", root / samples[sample_id]["source_path"]),
                    ("DENSITY", comparator[sample_id]),
                    ("HUE_VALUE", candidate[sample_id]),
                ],
            )
            for sample_id in gold_ids
        ],
        direct_path,
    )

    mappings: dict[str, Any] = {}
    blind_sheets = []
    for round_index in range(1, 4):
        round_mapping: dict[str, dict[str, str]] = {}
        rows = []
        for sample_index, sample_id in enumerate(gold_ids):
            roles = ["candidate", "comparator"]
            random.Random(
                int(config["blind"]["seed"])
                + round_index * 1009
                + sample_index * 9176
            ).shuffle(roles)
            round_mapping[sample_id] = {
                "A": roles[0],
                "B": roles[1],
            }
            paths = {
                "candidate": candidate[sample_id],
                "comparator": comparator[sample_id],
            }
            rows.append(
                (
                    sample_id,
                    [
                        ("INPUT", root / samples[sample_id]["source_path"]),
                        ("A", paths[roles[0]]),
                        ("B", paths[roles[1]]),
                    ],
                )
            )
        mappings[f"round_{round_index}"] = round_mapping
        sheet_path = output_dir / f"blind_round_{round_index}.png"
        _sheet(rows, sheet_path)
        blind_sheets.append(
            {
                "round": round_index,
                "path": _portable_path(sheet_path, root),
                "sha256": sha256_file(sheet_path),
            }
        )
    if len(
        {json.dumps(value, sort_keys=True) for value in mappings.values()}
    ) != 3:
        raise RuntimeError("BA1V blind mappings are not distinct")
    mapping_raw = (
        json.dumps(mappings, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (output_dir / "private_mapping.json").write_bytes(mapping_raw)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "direct_severe_review": {
            "path": _portable_path(direct_path, root),
            "sha256": sha256_file(direct_path),
        },
        "blind_sheets": blind_sheets,
        "private_mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "gold_sample_count": len(gold_ids),
        "claim_ceiling": config["claim_ceiling"],
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["build_visual_evidence"]
