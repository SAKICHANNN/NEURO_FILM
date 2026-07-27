#!/usr/bin/env python
"""Run frozen U5.R2Z0 within-recipe explicit case-bank retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2w2f0_filmset_recipe_global_explainability import (  # noqa: E402
    _contract,
    _group_internal_rows,
    _ids_sha256,
    _load_partition_samples,
    _validate_preflight,
)
from src.roll2film.ct5_data import load_ct5_internal_dev_rows  # noqa: E402
from src.roll2film.filmset_case_retrieval import (  # noqa: E402
    evaluate_within_recipe_case_bank,
    evaluate_wrong_recipe_oracle,
)
from src.roll2film.filmset_reference_preflight import (  # noqa: E402
    partition_filmset_content_ids,
)
from src.roll2film.manifests import FilmSetManifestError  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _software_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_activation(
    config: dict[str, Any],
    decision: dict[str, Any],
    *,
    decision_sha256: str,
) -> None:
    gate = config["activation_gate"]
    if config["status"] != "implemented_frozen_pending_formal_run":
        raise ValueError("unexpected Z0 config status")
    if decision_sha256 != gate["required_w2f0_decision_sha256"]:
        raise RuntimeError("W2F0 decision hash drift")
    if (
        decision.get("decision_branch") != "no_global_recipe_champion"
        or decision.get("repeat_report_sha256_equal") is not True
        or decision.get("report_a_sha256")
        != gate["required_w2f0_report_sha256"]
        or decision.get("report_b_sha256")
        != gate["required_w2f0_report_sha256"]
    ):
        raise RuntimeError("W2F0 repeat evidence does not activate Z0")
    required = gate["required_domain_branches"]
    for domain, branch in required.items():
        observed = decision["domain_decisions"][domain]["decision_branch"]
        if observed != branch:
            raise RuntimeError(
                f"W2F0 branch drift for {domain}: {observed} != {branch}"
            )
    if gate["w2f1_output_only_allowed"] is not False:
        raise ValueError("Z0 must not reopen output-only reference recovery")


def run_experiment(
    config: dict[str, Any],
    w2f0_decision: dict[str, Any],
    *,
    config_sha256: str,
    decision_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    _validate_activation(
        config, w2f0_decision, decision_sha256=decision_sha256
    )
    preflight_sha256 = _validate_preflight(config)
    contract = _contract(config, config_sha256=config_sha256)
    all_rows = load_ct5_internal_dev_rows(contract)
    allowed_domains = {"input", *contract.domains}
    rows = [
        row for row in all_rows if str(row["domain"]) in allowed_domains
    ]
    grouped = _group_internal_rows(rows, contract.domains)
    partition = config["partition"]
    order = partition_filmset_content_ids(
        grouped, seed=int(partition["seed"]), pool="internal"
    )
    development_count = int(
        partition["preflight_internal_development_identities"]
    )
    confirmatory_count = int(
        partition["preflight_internal_confirmatory_identities"]
    )
    stress_count = int(partition["preflight_internal_stress_identities"])
    if development_count + confirmatory_count + stress_count != len(order):
        raise FilmSetManifestError("Z0 partition does not exhaust identities")
    development_partition = order[:development_count]
    confirmatory_partition = order[
        development_count : development_count + confirmatory_count
    ]
    stress_partition = order[development_count + confirmatory_count :]
    development_ids = development_partition[
        : int(partition["development_identities_used"])
    ]
    confirmatory_ids = confirmatory_partition[
        : int(partition["confirmatory_identities_used"])
    ]
    if set(development_ids) & set(confirmatory_ids):
        raise FilmSetManifestError("Z0 development/confirmatory leakage")

    (
        development_input_fit,
        development_target_fit,
        development_input_evaluation,
        development_target_evaluation,
        development_cells,
    ) = _load_partition_samples(
        contract,
        grouped,
        development_ids,
        config=config,
        progress_label="development",
    )
    (
        _confirmatory_input_fit,
        _confirmatory_target_fit,
        confirmatory_input_evaluation,
        confirmatory_target_evaluation,
        confirmatory_cells,
    ) = _load_partition_samples(
        contract,
        grouped,
        confirmatory_ids,
        config=config,
        progress_label="confirmatory",
    )
    if not np.array_equal(development_cells, confirmatory_cells):
        raise RuntimeError("Z0 spatial cell IDs differ across partitions")

    requested_device = str(config["operators"]["o0"]["device"])
    device = (
        "cuda"
        if requested_device == "cuda_if_available_else_cpu"
        and torch.cuda.is_available()
        else "cpu"
    )
    operator_settings = dict(config["operators"]["o0"])
    evaluations = {}
    for domain_index, domain in enumerate(contract.domains):
        print(f"fitting case bank {domain}", flush=True)
        evaluations[domain] = evaluate_within_recipe_case_bank(
            development_input_fit,
            development_target_fit[domain],
            development_input_evaluation,
            development_target_evaluation[domain],
            confirmatory_input_evaluation,
            confirmatory_target_evaluation[domain],
            confirmatory_cells,
            operator_settings=operator_settings,
            gates=config["gates"],
            shared_seed=int(operator_settings["shared_fit_seed_base"])
            + 1000 * domain_index,
            case_seed_base=int(operator_settings["case_fit_seed_base"])
            + 1000 * domain_index,
            shuffle_seed=int(config["controls"]["shuffle_seed"])
            + domain_index,
            bootstrap_seed=int(config["controls"]["bootstrap_seed"])
            + domain_index,
            device=device,
        )

    domains: dict[str, Any] = {}
    for domain, evaluation in evaluations.items():
        wrong_operators = tuple(
            operator
            for other_domain, other_evaluation in evaluations.items()
            if other_domain != domain
            for operator in other_evaluation.eligible_operators
        )
        domains[domain] = {
            **evaluation.report,
            "wrong_recipe_control": evaluate_wrong_recipe_oracle(
                evaluation, wrong_operators
            ),
        }

    selected_ids = development_ids + confirmatory_ids
    selected_payload_count = len(selected_ids) * (
        1 + len(contract.domains)
    )
    expected_payload_count = int(
        config["dataset"]["selected_payload_image_count_expected"]
    )
    if selected_payload_count != expected_payload_count:
        raise FilmSetManifestError(
            f"Z0 payload count {selected_payload_count} "
            f"!= {expected_payload_count}"
        )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": "formal_run_pending_exact_repeat",
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "activation_evidence": {
            "w2f0_decision_sha256": decision_sha256,
            "w2f0_report_sha256": config["activation_gate"][
                "required_w2f0_report_sha256"
            ],
            "w2f0_repeat_exact": True,
            "w2f1_output_only_allowed": False,
            "w2f_preflight_report_sha256": preflight_sha256,
        },
        "dataset": {
            "dataset_id": config["dataset"]["dataset_id"],
            "archive_version": config["dataset"]["archive_version"],
            "internal_manifest_sha256": config["dataset"][
                "internal_dev_manifest_sha256"
            ],
            "final_manifest_sha256_verified": config["dataset"][
                "forbidden_final_manifest_sha256"
            ],
            "internal_manifest_rows_parsed": len(all_rows),
            "selected_identity_count": len(selected_ids),
            "selected_payload_image_count": selected_payload_count,
            "unselected_payload_images_read": 0,
            "final_manifest_payload_rows_parsed": 0,
            "full_raster_outputs_generated": 0,
        },
        "partition": {
            "algorithm": partition["algorithm"],
            "seed": int(partition["seed"]),
            "development_partition_count": len(development_partition),
            "confirmatory_partition_count": len(confirmatory_partition),
            "stress_partition_count": len(stress_partition),
            "development_partition_sha256": _ids_sha256(
                development_partition
            ),
            "confirmatory_partition_sha256": _ids_sha256(
                confirmatory_partition
            ),
            "stress_partition_sha256": _ids_sha256(stress_partition),
            "selected_development_count": len(development_ids),
            "selected_confirmatory_count": len(confirmatory_ids),
            "selected_development_sha256": _ids_sha256(development_ids),
            "selected_confirmatory_sha256": _ids_sha256(confirmatory_ids),
            "development_confirmatory_intersection": 0,
        },
        "sampling": {
            **config["sampling"],
            "fit_evaluation_indices_disjoint": True,
            "same_coordinates_across_aligned_domains": True,
        },
        "device": {
            "selected": device,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "domains": domains,
        "decision_branch_before_repeat": {
            domain: result["decision_branch_before_repeat"]
            for domain, result in domains.items()
        },
        "filmset_recipe_operator_fits_performed": len(contract.domains)
        * (1 + len(development_ids)),
        "automatic_visual_shortlist_generated": False,
        "current_stock_pixels_accessed": False,
        "current_stock_training_or_operator_fitting_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2z0_filmset_case_retrieval_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decision_path = (
        ROOT / config["activation_gate"]["w2f0_decision_path"]
    ).resolve()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    report = run_experiment(
        config,
        decision,
        config_sha256=_sha256_file(config_path),
        decision_sha256=_sha256_file(decision_path),
        software_commit=_software_commit(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
