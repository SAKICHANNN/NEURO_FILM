#!/usr/bin/env python
"""Run frozen U5.R2W2F0 paired FilmSet recipe explainability."""

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

from src.roll2film.ct5_data import (  # noqa: E402
    CT5DataContract,
    load_ct5_internal_dev_rows,
    load_ct5_working_image,
)
from src.roll2film.filmset_recipe_explainability import (  # noqa: E402
    apply_spatial_indices,
    evaluate_recipe_global_explainability,
    sample_aligned_spatial_pixels,
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


def _ids_sha256(content_ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{content_id}\n" for content_id in content_ids).encode()
    ).hexdigest()


def _software_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _validate_activation(
    config: dict[str, Any], w1_decision: dict[str, Any]
) -> None:
    if config.get("status") != "implemented_frozen_pending_formal_run":
        raise ValueError("unexpected W2F0 contract status")
    gate = config["activation_gate"]
    reports_equal = (
        w1_decision.get("report_a_sha256")
        == w1_decision.get("report_b_sha256")
    )
    if (
        w1_decision.get("decision_branch") != gate["required_w1_branch"]
        or w1_decision.get("repeat_report_sha256_equal") is not True
        or not reports_equal
    ):
        raise RuntimeError("W2F0 activation rejected by repeated W1 evidence")
    if gate.get("w2f1_output_only_allowed") is not False:
        raise ValueError("W2F0 must not open output-only reference recovery")


def _validate_preflight(config: dict[str, Any]) -> str:
    gate = config["activation_gate"]
    expected = str(gate["required_w2f_preflight_report_sha256"])
    paths = [
        (ROOT / gate[key]).resolve()
        for key in (
            "w2f_preflight_report_a_path",
            "w2f_preflight_report_b_path",
        )
    ]
    observed = [_sha256_file(path) for path in paths]
    if observed != [expected, expected]:
        raise RuntimeError("W2F0 preflight reports are absent or not byte-identical")
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    dataset = config["dataset"]
    for report in reports:
        if (
            report.get("image_payloads_read") != 0
            or report.get("final_manifest_payload_rows_parsed") != 0
            or report.get("partition", {}).get("seed")
            != config["partition"]["seed"]
            or report.get("manifest_sha256", {}).get("internal")
            != dataset["internal_dev_manifest_sha256"]
            or report.get("manifest_sha256", {}).get("final")
            != dataset["forbidden_final_manifest_sha256"]
        ):
            raise RuntimeError("W2F0 preflight evidence violates the frozen boundary")
    return expected


def _contract(
    config: dict[str, Any], *, config_sha256: str
) -> CT5DataContract:
    dataset = config["dataset"]
    return CT5DataContract(
        project_root=ROOT.resolve(),
        dataset_root=(ROOT / dataset["root"]).resolve(),
        evidence_dir=(ROOT / dataset["evidence_dir"]).resolve(),
        experiment_id=str(config["experiment_id"]),
        config_sha256=config_sha256,
        seed=int(config["partition"]["seed"]),
        pixels_per_training_image=16,
        pixels_per_dev_image=16,
        pilot_fraction=0.5,
        domains=tuple(str(value) for value in dataset["domains"]),
        manifest_hashes={
            "source_train": "",
            "target_train": "",
            "internal_dev_lockbox": str(
                dataset["internal_dev_manifest_sha256"]
            ),
            "final_628_lockbox": str(
                dataset["forbidden_final_manifest_sha256"]
            ),
        },
    )


def _group_internal_rows(
    rows: list[dict[str, Any]], domains: tuple[str, ...]
) -> dict[str, dict[str, dict[str, Any]]]:
    required = {"input", *domains}
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        content_id = str(row["content_id"])
        domain = str(row["domain"])
        if domain not in required:
            raise FilmSetManifestError(f"unexpected W2F0 domain: {domain}")
        by_domain = grouped.setdefault(content_id, {})
        if domain in by_domain:
            raise FilmSetManifestError(
                f"duplicate W2F0 content/domain: {content_id}/{domain}"
            )
        by_domain[domain] = row
    for content_id, by_domain in grouped.items():
        if set(by_domain) != required:
            raise FilmSetManifestError(
                f"incomplete W2F0 aligned identity: {content_id}"
            )
        clusters = {
            str(row["duplicate_cluster_id"]) for row in by_domain.values()
        }
        if len(clusters) != 1:
            raise FilmSetManifestError(
                f"inconsistent W2F0 duplicate cluster: {content_id}"
            )
    return grouped


def _load_partition_samples(
    contract: CT5DataContract,
    grouped: dict[str, dict[str, dict[str, Any]]],
    content_ids: list[str],
    *,
    config: dict[str, Any],
    progress_label: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    sampling = config["sampling"]
    input_fit: list[np.ndarray] = []
    input_evaluation: list[np.ndarray] = []
    target_fit = {domain: [] for domain in contract.domains}
    target_evaluation = {domain: [] for domain in contract.domains}
    cell_ids: np.ndarray | None = None
    for index, content_id in enumerate(content_ids, start=1):
        rows = grouped[content_id]
        decoded = {
            domain: load_ct5_working_image(contract, row).pixels
            for domain, row in rows.items()
        }
        shape = decoded["input"].shape
        if any(image.shape != shape for image in decoded.values()):
            raise FilmSetManifestError(
                f"W2F0 paired dimensions differ: {content_id}"
            )
        sampled = sample_aligned_spatial_pixels(
            decoded["input"],
            identity=content_id,
            grid_rows=int(sampling["grid_rows"]),
            grid_columns=int(sampling["grid_columns"]),
            pixels_per_cell_fit=int(sampling["pixels_per_cell_fit"]),
            pixels_per_cell_evaluation=int(
                sampling["pixels_per_cell_evaluation"]
            ),
            fit_seed=int(sampling["fit_seed"]),
            evaluation_seed=int(sampling["evaluation_seed"]),
        )
        input_fit.append(sampled.fit_pixels)
        input_evaluation.append(sampled.evaluation_pixels)
        if cell_ids is None:
            cell_ids = sampled.evaluation_cell_ids
        elif not np.array_equal(cell_ids, sampled.evaluation_cell_ids):
            raise RuntimeError("W2F0 cell layout changed across identities")
        for domain in contract.domains:
            target_fit[domain].append(
                apply_spatial_indices(decoded[domain], sampled.fit_flat_indices)
            )
            target_evaluation[domain].append(
                apply_spatial_indices(
                    decoded[domain], sampled.evaluation_flat_indices
                )
            )
        print(
            f"{progress_label} {index}/{len(content_ids)} {content_id}",
            flush=True,
        )
    if cell_ids is None:
        raise RuntimeError("W2F0 selected an empty partition")
    return (
        np.stack(input_fit),
        {domain: np.stack(values) for domain, values in target_fit.items()},
        np.stack(input_evaluation),
        {
            domain: np.stack(values)
            for domain, values in target_evaluation.items()
        },
        cell_ids,
    )


def run_experiment(
    config: dict[str, Any],
    w1_decision: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    _validate_activation(config, w1_decision)
    preflight_sha256 = _validate_preflight(config)
    contract = _contract(config, config_sha256=config_sha256)
    rows = load_ct5_internal_dev_rows(contract)
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
        raise FilmSetManifestError("W2F0 partition does not exhaust manifest")
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
        raise FilmSetManifestError("W2F0 development/confirmatory leakage")

    (
        development_input_fit,
        development_target_fit,
        _development_input_evaluation,
        _development_target_evaluation,
        development_cells,
    ) = _load_partition_samples(
        contract,
        grouped,
        development_ids,
        config=config,
        progress_label="development",
    )
    (
        confirmatory_input_fit,
        confirmatory_target_fit,
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
        raise RuntimeError("W2F0 spatial cell IDs differ by partition")

    requested_device = str(config["operators"]["o0"]["device"])
    device = (
        "cuda"
        if requested_device == "cuda_if_available_else_cpu"
        and torch.cuda.is_available()
        else "cpu"
    )
    operator_settings = dict(config["operators"]["o0"])
    domains: dict[str, Any] = {}
    for domain_index, domain in enumerate(contract.domains):
        print(f"fitting recipe {domain}", flush=True)
        domains[domain] = evaluate_recipe_global_explainability(
            development_input_fit,
            development_target_fit[domain],
            confirmatory_input_fit,
            confirmatory_target_fit[domain],
            confirmatory_input_evaluation,
            confirmatory_target_evaluation[domain],
            confirmatory_cells,
            operator_settings=operator_settings,
            gates=config["gates"],
            shared_seed=int(operator_settings["shared_fit_seed_base"])
            + 1000 * domain_index,
            per_pair_seed_base=int(
                operator_settings["per_pair_fit_seed_base"]
            )
            + 1000 * domain_index,
            device=device,
        )
    selected_identities = development_ids + confirmatory_ids
    payload_images = len(selected_identities) * (1 + len(contract.domains))
    if payload_images != 160:
        raise FilmSetManifestError(
            f"W2F0 payload budget changed: expected 160, got {payload_images}"
        )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": "formal_run_pending_exact_repeat",
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "activation_evidence": {
            "w1_decision_branch": w1_decision["decision_branch"],
            "w1_report_sha256": w1_decision["report_a_sha256"],
            "w1_repeat_report_sha256_equal": True,
            "w2f_preflight_report_sha256": preflight_sha256,
            "w2f_preflight_repeat_report_sha256_equal": True,
            "w2f1_output_only_allowed": False,
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
            "internal_manifest_rows_parsed": len(rows),
            "final_manifest_payload_rows_parsed": 0,
            "selected_identity_count": len(selected_identities),
            "selected_payload_image_count": payload_images,
            "selected_payload_image_count_expected": 160,
            "unselected_payload_images_read": 0,
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
            "pixels_per_identity_per_split": int(
                config["sampling"]["grid_rows"]
            )
            * int(config["sampling"]["grid_columns"])
            * int(config["sampling"]["pixels_per_cell_fit"]),
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
        "automatic_visual_shortlist_generated": False,
        "current_stock_pixels_accessed": False,
        "filmset_recipe_operator_fits_performed": len(contract.domains)
        * (1 + len(confirmatory_ids)),
        "current_stock_training_or_operator_fitting_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2w2f0_filmset_recipe_global_explainability_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "eval"
        / "u5_r2w2f0_filmset_recipe_global_explainability_report.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    w1_path = (ROOT / config["activation_gate"]["w1_decision_path"]).resolve()
    w1_decision = json.loads(w1_path.read_text(encoding="utf-8"))
    report = run_experiment(
        config,
        w1_decision,
        config_sha256=_sha256_file(config_path),
        software_commit=_software_commit(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(_sha256_file(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
