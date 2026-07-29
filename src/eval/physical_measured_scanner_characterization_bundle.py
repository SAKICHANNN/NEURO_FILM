"""U6.P6J compiler for research-only measured scanner characterization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from zipfile import ZipFile

import numpy as np

from src.eval.physical_measured_scanner_characterization import (
    _load_target_table,
    _metrics,
    _select_common_power,
    _sha256,
    _slide_member,
)
from src.film_physics.scanner_characterization import (
    MeasuredScannerCharacterization,
    MeasuredScannerCharacterizationBundle,
    apply_measured_scanner_characterization,
    measured_scanner_bundle_from_payload,
)
from src.real_film.scanner_nuisance import (
    ScannerNuisanceError,
    _native_rgb,
    align_source_to_scan,
    patch_medians,
)


SCHEMA = (
    "neuro_film.u6_p6j_measured_scanner_characterization_bundle_contract.v1"
)


def _stable_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(value.shape.__repr__().encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6J contract")
    return payload


def _build_patch_bank(
    root: Path,
    p6i_contract: dict[str, Any],
    retry_contract: dict[str, Any],
) -> tuple[
    dict[str, dict[int, np.ndarray]],
    dict[str, dict[str, Any]],
    dict[int, dict[int, dict[str, np.ndarray | tuple[str, ...]]]],
    list[dict[str, Any]],
]:
    parents = p6i_contract["parents"]
    p6h_decision = json.loads(
        (root / parents["p6h_decision_path"]).read_text(encoding="utf-8")
    )
    p6h_report = json.loads(
        (root / parents["p6h_report_path"]).read_text(encoding="utf-8")
    )
    p6h_contract_path = root / p6h_decision["evidence"]["contract"]["path"]
    if _sha256(p6h_contract_path) != p6h_decision["evidence"]["contract"][
        "sha256"
    ]:
        raise ValueError("P6H contract hash mismatch")
    p6h_contract = json.loads(p6h_contract_path.read_text(encoding="utf-8"))
    acquisition = p6h_contract["acquisition"]
    data_root = root / acquisition["data_root"]

    alignment_contract = json.loads(
        (root / parents["alignment_contract_path"]).read_text(encoding="utf-8")
    )
    alignment = alignment_contract["alignment"]
    retry_alignment = dict(alignment)
    retry_alignment["ratio_test"] = float(
        retry_contract["retry"]["ratio_test"]
    )
    grid = alignment_contract["source_grid"]
    target_sets = [
        int(value)
        for value in p6i_contract["measurement_contract"]["target_sets"]
    ]
    target_table = _load_target_table(
        root / parents["pair_table_path"], target_sets
    )

    source_images = {}
    for row in p6i_contract["source_images"]:
        path = root / row["path"]
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"source slide {row['slide']} hash mismatch")
        source_images[int(row["slide"])] = _native_rgb(path.read_bytes())

    assets_by_role = {row["role"]: row for row in p6h_report["assets"]}
    scan_bank: dict[str, dict[int, np.ndarray]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    alignment_records = []
    for asset_contract in acquisition["archives"]:
        role = str(asset_contract["role"])
        asset = assets_by_role[role]
        archive_path = data_root / str(asset["path"])
        if _sha256(archive_path) != asset["sha256"]:
            raise ValueError(f"archive {role} hash mismatch")
        target_set = int(asset_contract["target_set"])
        metadata[role] = {
            "scanner": str(asset_contract["scanner"]),
            "software": str(asset_contract["software"]),
            "target_set": target_set,
            "archive_sha256": str(asset["sha256"]),
        }
        scan_bank[role] = {}
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            for slide in range(1, 6):
                member = _slide_member(names, target_set, slide)
                scan = _native_rgb(archive.read(member))
                try:
                    homography, diagnostics = align_source_to_scan(
                        source_images[slide], scan, alignment
                    )
                    attempt = "primary"
                except ScannerNuisanceError:
                    homography, diagnostics = align_source_to_scan(
                        source_images[slide], scan, retry_alignment
                    )
                    attempt = "ratio-0.75-retry"
                patches = patch_medians(scan, homography, grid)
                if patches.shape != (264, 3):
                    raise ValueError(f"{role}/{slide} patch shape mismatch")
                scan_bank[role][slide] = patches
                alignment_records.append(
                    {
                        "role": role,
                        "target_set": target_set,
                        "slide": slide,
                        "member": member,
                        "alignment_attempt": attempt,
                        **diagnostics,
                    }
                )
    return scan_bank, metadata, target_table, alignment_records


def compile_measured_scanner_characterization_bundle(
    root: Path,
    contract_path: Path,
) -> tuple[MeasuredScannerCharacterizationBundle, dict[str, Any]]:
    contract_bytes = contract_path.read_bytes()
    contract = _load_contract(contract_path)
    parents = contract["parents"]
    for path_key, hash_key in (
        ("p6i_decision_path", "p6i_decision_sha256"),
        ("p6i_contract_path", "p6i_contract_sha256"),
        ("p6i_alignment_retry_path", "p6i_alignment_retry_sha256"),
        ("p6i_report_path", "p6i_report_sha256"),
    ):
        if _sha256(root / parents[path_key]) != parents[hash_key]:
            raise ValueError(f"{path_key} hash mismatch")
    decision = json.loads(
        (root / parents["p6i_decision_path"]).read_text(encoding="utf-8")
    )
    p6i_report = json.loads(
        (root / parents["p6i_report_path"]).read_text(encoding="utf-8")
    )
    if (
        decision.get("decision")
        != (
            "retain_common_power_positive_matrix_per_device_"
            "characterization_family_independent_target_validation_required"
        )
        or decision.get("automatic_pass") is not True
        or p6i_report.get("automatic_pass") is not True
    ):
        raise ValueError("P6I pass decision is not exact")
    p6i_contract = json.loads(
        (root / parents["p6i_contract_path"]).read_text(encoding="utf-8")
    )
    retry_contract = json.loads(
        (root / parents["p6i_alignment_retry_path"]).read_text(encoding="utf-8")
    )
    scan_bank, metadata, target_table, alignment_records = _build_patch_bank(
        root, p6i_contract, retry_contract
    )

    compiler = contract["compiler"]
    entries = []
    per_entry = {}
    pooled_errors = []
    output_chunks = []
    max_apply_error = 0.0
    for role in sorted(scan_bank):
        row = metadata[role]
        target_set = int(row["target_set"])
        source = np.concatenate(
            [scan_bank[role][slide] for slide in range(1, 6)]
        )
        target_xyz = np.concatenate(
            [
                np.asarray(
                    target_table[target_set][slide]["xyz"], dtype=np.float64
                )
                for slide in range(1, 6)
            ]
        )
        target_lab = np.concatenate(
            [
                np.asarray(
                    target_table[target_set][slide]["lab"], dtype=np.float64
                )
                for slide in range(1, 6)
            ]
        )
        power, matrix, development_loss = _select_common_power(
            source,
            target_xyz,
            compiler["common_power_grid"],
            compiler["matrix_coefficient_interval"],
        )
        fit_support_sha256 = _array_sha256(source, target_xyz, target_lab)
        entry = MeasuredScannerCharacterization(
            scanner=row["scanner"],
            software=row["software"],
            target_set=target_set,
            archive_sha256=row["archive_sha256"],
            common_power=power,
            matrix=tuple(tuple(float(value) for value in line) for line in matrix),
            fit_patch_count=int(source.shape[0]),
            fit_support_sha256=fit_support_sha256,
        )
        reference = np.power(source, power, dtype=np.float64) @ matrix.T
        applied = apply_measured_scanner_characterization(source, entry)
        max_apply_error = max(
            max_apply_error, float(np.max(np.abs(reference - applied)))
        )
        metrics, errors = _metrics(applied, target_xyz, target_lab)
        per_entry[role] = {
            "entry_id": entry.entry_id,
            "scanner": entry.scanner,
            "software": entry.software,
            "target_set": entry.target_set,
            "archive_sha256": entry.archive_sha256,
            "common_power": entry.common_power,
            "matrix": [list(line) for line in entry.matrix],
            "fit_patch_count": entry.fit_patch_count,
            "fit_support_sha256": entry.fit_support_sha256,
            "development_mean_xyz_l2": development_loss,
            "metrics": metrics,
            "output_minimum": float(np.min(applied)),
            "output_maximum": float(np.max(applied)),
        }
        entries.append(entry)
        pooled_errors.append(errors)
        output_chunks.append(applied)

    entries.sort(
        key=lambda entry: (
            entry.scanner,
            entry.software,
            entry.target_set,
            entry.archive_sha256,
        )
    )
    bundle = MeasuredScannerCharacterizationBundle(
        contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),
        evidence_report_sha256=parents["p6i_report_sha256"],
        entries=tuple(entries),
        claim_ceiling=contract["claim_ceiling"],
    )
    roundtrip = measured_scanner_bundle_from_payload(bundle.to_payload())
    errors = np.concatenate(pooled_errors)
    outputs = np.concatenate(output_chunks)
    gates = contract["automatic_gates"]
    checks = {
        "entry_count": len(entries) == int(gates["expected_entries"]),
        "fit_patch_count": all(
            entry.fit_patch_count
            == int(gates["expected_fit_patches_per_entry"])
            for entry in entries
        ),
        "bundle_roundtrip_identity": roundtrip.bundle_id == bundle.bundle_id
        and roundtrip.to_payload() == bundle.to_payload(),
        "apply_matches_compiler_reference": max_apply_error
        <= float(gates["apply_matches_compiler_reference_max_abs"]),
        "aggregate_development_median": float(np.median(errors))
        <= float(gates["aggregate_development_median_delta_e76_max"]),
        "aggregate_development_p95": float(np.percentile(errors, 95.0))
        <= float(gates["aggregate_development_p95_delta_e76_max"]),
        "each_entry_development_median": all(
            row["metrics"]["median_delta_e76"]
            <= float(gates["each_entry_development_median_delta_e76_max"])
            for row in per_entry.values()
        ),
        "output_xyz_domain": float(np.min(outputs))
        >= float(gates["output_xyz_minimum"]) - 1e-15
        and float(np.max(outputs))
        <= float(gates["output_xyz_maximum"]) + 1e-15,
        "alignment_support": len(alignment_records) == 65
        and min(int(row["ransac_inliers"]) for row in alignment_records) >= 40,
    }
    stable_payload = {
        "schema": (
            "neuro_film.u6_p6j_measured_scanner_characterization_bundle_report.v1"
        ),
        "node": contract["node"],
        "config_sha256": bundle.contract_sha256,
        "parent_hashes_verified": {
            key: parents[key]
            for key in sorted(parents)
            if key.endswith("_sha256")
        },
        "bundle_id": bundle.bundle_id,
        "support": {
            "entries": len(entries),
            "target_sets": sorted(
                {entry.target_set for entry in entries}
            ),
            "fit_patches_per_entry": sorted(
                {entry.fit_patch_count for entry in entries}
            ),
            "alignment_cells": len(alignment_records),
            "alignment_retry_cells": sum(
                row["alignment_attempt"] != "primary"
                for row in alignment_records
            ),
            "scanner_target_graph_connected_across_sets": False,
        },
        "aggregate_development": {
            "median_delta_e76": float(np.median(errors)),
            "p95_delta_e76": float(np.percentile(errors, 95.0)),
            "maximum_delta_e76": float(np.max(errors)),
            "output_minimum": float(np.min(outputs)),
            "output_maximum": float(np.max(outputs)),
        },
        "maximum_apply_vs_compiler_reference_abs_error": max_apply_error,
        "per_entry": per_entry,
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "claim_ceiling": contract["claim_ceiling"],
        "forbidden_claims": contract["forbidden_claims"],
    }
    report = {
        **stable_payload,
        "stable_evidence_id": _stable_id(stable_payload),
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "alignment_records": alignment_records,
    }
    return bundle, report


def write_bundle(
    bundle: MeasuredScannerCharacterizationBundle, path: Path
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            bundle.to_payload(), indent=2, sort_keys=True, ensure_ascii=True
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
