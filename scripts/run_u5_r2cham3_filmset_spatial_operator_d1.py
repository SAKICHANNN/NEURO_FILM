#!/usr/bin/env python
"""Run the frozen FilmSet fixed-spatial explicit-operator experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2w2f0_filmset_recipe_global_explainability import (
    _contract,
    _group_internal_rows,
    _ids_sha256,
    _load_partition_samples,
    _validate_preflight,
)
from src.eval.filmset_spatial_operator_d1 import (
    evaluate_arrays,
    finalize_report,
    load_contract,
    write_report,
)
from src.roll2film.ct5_data import load_ct5_internal_dev_rows
from src.roll2film.filmset_reference_preflight import (
    partition_filmset_content_ids,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(config_path: Path) -> dict:
    config = load_contract(config_path)
    parent_ids = {}
    for name, binding in config["parents"].items():
        path = ROOT / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"CHAM3 parent drift: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "required_decision" in binding and payload.get("decision") != binding["required_decision"]:
            raise ValueError(f"CHAM3 parent decision drift: {name}")
        if "required_branch" in binding and payload.get("decision_branch") != binding["required_branch"]:
            raise ValueError(f"CHAM3 parent branch drift: {name}")
        parent_ids[name] = payload.get("stable_evidence_id", binding["sha256"])
    preflight_sha = _validate_preflight(config)
    contract = _contract(config, config_sha256=sha256_file(config_path))
    all_rows = load_ct5_internal_dev_rows(contract)
    grouped = _group_internal_rows(
        [row for row in all_rows if str(row["domain"]) in {"input", "classneg"}],
        ("classneg",),
    )
    p = config["partition"]
    order = partition_filmset_content_ids(grouped, seed=int(p["seed"]), pool="internal")
    dev_pool = int(p["preflight_internal_development_identities"])
    development_ids = order[: int(p["development_identities_used"])]
    confirmatory_ids = order[dev_pool : dev_pool + int(p["confirmatory_identities_used"])]
    if set(development_ids) & set(confirmatory_ids):
        raise ValueError("CHAM3 partition leakage")
    dev_fit, dev_target_fit, _dev_eval, _dev_target_eval, dev_cells = _load_partition_samples(
        contract, grouped, development_ids, config=config, progress_label="development"
    )
    _conf_fit, _conf_target_fit, conf_eval, conf_target_eval, conf_cells = _load_partition_samples(
        contract, grouped, confirmatory_ids, config=config, progress_label="confirmatory"
    )
    if not np.array_equal(dev_cells, conf_cells):
        raise ValueError("CHAM3 cell drift")
    requested = config["operator"]["device"]
    device = "cuda" if requested == "cuda_if_available_else_cpu" and torch.cuda.is_available() else "cpu"
    result = evaluate_arrays(
        config,
        dev_fit,
        dev_target_fit["classneg"],
        conf_eval,
        conf_target_eval["classneg"],
        conf_cells,
        device=device,
    )
    return finalize_report(
        config,
        result,
        parent_ids=parent_ids,
        dataset={
            "preflight_sha256": preflight_sha,
            "internal_manifest_sha256": config["dataset"]["internal_dev_manifest_sha256"],
            "development_count": len(development_ids),
            "confirmatory_count": len(confirmatory_ids),
            "development_ids_sha256": _ids_sha256(development_ids),
            "confirmatory_ids_sha256": _ids_sha256(confirmatory_ids),
            "development_confirmatory_intersection": 0,
            "selected_payload_image_count": 80,
            "unselected_payload_images_read": 0,
            "final_manifest_payload_rows_parsed": 0,
            "device": device,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u5_r2cham3_filmset_spatial_operator_d1_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config.resolve())
    digest = write_report(report, args.output.resolve())
    print(json.dumps({"report_sha256": digest, "stable_evidence_id": report["stable_evidence_id"], "automatic_pass": report["automatic_pass"], "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
