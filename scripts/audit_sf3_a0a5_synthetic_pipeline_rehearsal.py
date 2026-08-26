"""Run a disposable structural rehearsal of the controlled three-stock chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import srgb_icc_profile
from src.real_film.three_stock_acquisition import (
    LEDGER_SCHEMA,
    compile_acquisition_ledger,
)
from src.real_film.three_stock_blind_package import adjudicate_package, build_package
from src.real_film.three_stock_capture_receipts import (
    build_ledger_template,
    build_receipt_template,
    evaluate_ledger_binding,
    evaluate_receipts,
)
from src.real_film.three_stock_confirmation_render import (
    evaluate_single_stock_and_materialize,
)
from src.real_film.three_stock_k1_file_runner import evaluate_single_stock_files
from src.real_film.three_stock_scan_integrity import build_alignment_evidence

DEFAULT_CONTRACT = ROOT / "configs/sf3_a0a5_synthetic_pipeline_rehearsal_v1.json"
STIMULUS_ROOT = ROOT / "data/real_film/sf3_a0k_three_stock_display_stimulus_v1"
STOCKS = (
    "fujifilm_velvia_50",
    "kodak_portra_400",
    "kodak_ektar_100",
)
GAINS = {
    "fujifilm_velvia_50": np.array([0.95, 0.72, 0.62], dtype=np.float32),
    "kodak_portra_400": np.array([0.82, 0.90, 0.72], dtype=np.float32),
    "kodak_ektar_100": np.array([0.65, 0.82, 0.97], dtype=np.float32),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(path: Path, value: Any) -> bytes:
    raw = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _repo_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe(value: str) -> str:
    return value.replace(":", "_")


def _copy_bound_parents(contract: dict[str, Any], root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, binding in contract["parents"].items():
        source = ROOT / binding["path"]
        if _sha_file(source) != binding["sha256"]:
            raise RuntimeError(f"parent identity drift: {name}")
        destination = root / binding["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        paths[name] = destination
    return paths


def _derive_contracts(paths: dict[str, Path]) -> None:
    integrity = json.loads(paths["integrity_contract"].read_text(encoding="utf-8"))
    integrity["decode"].update(required_scan_width=320, required_scan_height=240)
    integrity["alignment"].update(
        maximum_features=1000,
        minimum_good_matches=4,
        minimum_ransac_inliers=4,
        maximum_median_inlier_reprojection_error_px=1.0,
    )
    _write(paths["integrity_contract"], integrity)

    k1 = json.loads(paths["k1_contract"].read_text(encoding="utf-8"))
    k1["parent"]["integrity_contract"]["sha256"] = _sha_file(
        paths["integrity_contract"]
    )
    _write(paths["k1_contract"], k1)

    confirmation = json.loads(
        paths["confirmation_contract"].read_text(encoding="utf-8")
    )
    confirmation["parent"]["k1_contract"]["sha256"] = _sha_file(paths["k1_contract"])
    _write(paths["confirmation_contract"], confirmation)

    blind = json.loads(paths["blind_contract"].read_text(encoding="utf-8"))
    blind["parents"]["k1_baseline_contract"]["sha256"] = _sha_file(paths["k1_contract"])
    blind["parents"]["confirmation_render_contract"]["sha256"] = _sha_file(
        paths["confirmation_contract"]
    )
    _write(paths["blind_contract"], blind)


def _fill_receipts(paths: dict[str, Path], root: Path) -> tuple[Path, dict[str, Any]]:
    packet = build_receipt_template(paths["receipt_contract"], root=root)
    for row in packet["common_condition_records"]:
        row.update(
            display_device_id="procedural-display-v1",
            display_profile_sha256="a" * 64,
            display_luminance_cd_m2=120.0,
            ambient_illuminance_lux=5.0,
            camera_system_id="procedural-camera-v1",
            lens_id="procedural-lens-v1",
            aperture_f_number=8.0,
            focus_distance_m=2.0,
            framing_id="procedural-framing-v1",
        )
    for row in packet["exposure_receipts"]:
        row.update(
            shutter_seconds=1.0 / float(row["nominal_iso"]),
            meter_reading_ev100=10.0,
            exposure_compensation_ev=0.0,
            meter_id="procedural-meter-v1",
            meter_calibration_sha256="b" * 64,
        )
    path = root / "data/rehearsal/receipt_packet.json"
    _write(path, packet)
    evaluate_receipts(paths["receipt_contract"], path, root=root)
    return path, packet


def _materialize_packet(
    paths: dict[str, Path], root: Path, packet_path: Path, packet: dict[str, Any]
) -> tuple[Path, Path, dict[str, Any]]:
    work = json.loads(paths["work_order"].read_text(encoding="utf-8"))
    conditions = {
        row["condition_slot_id"]: row for row in work["common_condition_records"]
    }
    condition_receipts = {
        row["condition_slot_id"]: row for row in packet["common_condition_records"]
    }
    exposure_by_id = {row["exposure_slot_id"]: row for row in work["exposure_rows"]}
    row_by_id = {
        row["row_id"]: row
        for row in build_ledger_template(
            paths["receipt_contract"],
            packet_path,
            paths["acquisition_contract"],
            root=root,
        )["rows"]
    }
    source_paths: dict[str, Path] = {}
    for condition_id, condition in conditions.items():
        source = STIMULUS_ROOT / condition["stimulus_relative_path"]
        if _sha_file(source) != condition["stimulus_sha256"]:
            raise RuntimeError(f"stimulus identity drift: {condition_id}")
        destination = (
            root / "data/rehearsal/stimuli" / condition["stimulus_relative_path"]
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_paths[condition_id] = destination

    integrity = json.loads(paths["integrity_contract"].read_text(encoding="utf-8"))
    profile = srgb_icc_profile()
    for task in work["scan_tasks"]:
        if task["counts_toward_evidence_minimum"] is not True:
            continue
        row = row_by_id[task["scan_task_id"]]
        exposure = exposure_by_id[task["exposure_slot_id"]]
        condition_id = exposure["common_condition_slot_id"]
        source_path = source_paths[condition_id]
        source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise RuntimeError(f"failed to decode stimulus: {condition_id}")
        source_rgb = source_bgr[..., ::-1]
        resized = cv2.resize(source_rgb, (320, 240), interpolation=cv2.INTER_AREA)
        scan_rgb = np.clip(
            resized.astype(np.float32) * GAINS[row["stock_id"]][None, None, :],
            0.0,
            255.0,
        ).astype(np.uint8)
        identity_bytes = hashlib.sha256(row["row_id"].encode("utf-8")).digest()
        scan_rgb[0, 0] = np.frombuffer(identity_bytes[:3], dtype=np.uint8)
        base = root / "data/rehearsal"
        condition_path = base / "conditions" / f"{_safe(condition_id)}.json"
        process_path = base / "process" / f"{_safe(row['process_session_id'])}.json"
        scanner_path = base / "scanner" / f"{_safe(row['scanner_session_id'])}.json"
        rights_path = base / "rights" / f"{row['role']}.json"
        scan_path = base / "scans" / f"{_safe(row['row_id'])}.tif"
        alignment_path = base / "alignment" / f"{_safe(row['row_id'])}.json"
        if not condition_path.exists():
            _write(condition_path, condition_receipts[condition_id])
        if not process_path.exists():
            _write(process_path, {"process_session_id": row["process_session_id"]})
        if not scanner_path.exists():
            _write(scanner_path, {"scanner_session_id": row["scanner_session_id"]})
        if not rights_path.exists():
            _write(
                rights_path,
                {
                    "schema": integrity["record_schemas"]["rights"],
                    "rights_record_id": f"procedural-rights-{row['role']}",
                    "source_owner_id": "project-owner",
                    "rights_scope": "project_owned_internal_research_and_commercial_derivatives",
                    "allow_internal_training": True,
                    "allow_commercial_derivatives": True,
                    "allow_released_weights": True,
                },
            )
        scan_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            scan_path,
            scan_rgb.astype(np.uint16) * 257,
            photometric="rgb",
            planarconfig="contig",
            metadata=None,
            extratags=[(34675, "B", len(profile), profile, False)],
        )
        row.update(
            scene_content_group=f"procedural:{row['role']}:{row['scene_id']}",
            capture_session_id=f"procedural-capture:{row['role']}",
            source_owner_id="project-owner",
            digital_reference_path=_repo_path(root, source_path),
            capture_condition_record_path=_repo_path(root, condition_path),
            lab_id=f"procedural-lab:{row['role']}",
            process_recipe_record_path=_repo_path(root, process_path),
            scanner_profile_record_path=_repo_path(root, scanner_path),
            scan_file_path=_repo_path(root, scan_path),
            scan_sample_path=_repo_path(root, scan_path),
            alignment_evidence_path=_repo_path(root, alignment_path),
            rights_record_path=_repo_path(root, rights_path),
            rights_scope="project_owned_internal_research_and_commercial_derivatives",
            rights_allow_internal_training=True,
            rights_allow_commercial_derivatives=True,
            rights_allow_released_weights=True,
        )
        evidence_row = {
            **row,
            "digital_reference_sha256": _sha_file(source_path),
            "scan_sample_sha256": _sha_file(scan_path),
        }
        evidence = build_alignment_evidence(
            evidence_row,
            source_rgb,
            scan_rgb,
            alignment=integrity["alignment"],
            schema=integrity["record_schemas"]["alignment"],
        )
        _write(alignment_path, evidence)

    ledger_path = root / "data/rehearsal/ledger.json"
    rows = [
        row_by_id[task["scan_task_id"]]
        for task in work["scan_tasks"]
        if task["counts_toward_evidence_minimum"] is True
    ]
    _write(ledger_path, {"schema": LEDGER_SCHEMA, "rows": rows})
    binding = evaluate_ledger_binding(
        paths["receipt_contract"],
        packet_path,
        paths["acquisition_contract"],
        ledger_path,
        root=root,
    )
    manifest_path = root / "data/rehearsal/manifest.json"
    _write(manifest_path, binding["compiled_manifest"])
    return ledger_path, manifest_path, binding


def _stock_lane(
    stock: str,
    *,
    root: Path,
    paths: dict[str, Path],
    ledger: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    stock_ledger = {
        **ledger,
        "rows": [row for row in ledger["rows"] if row["stock_id"] == stock],
    }
    ledger_path = root / "data/rehearsal" / stock / "ledger.json"
    _write(ledger_path, stock_ledger)
    manifest = compile_acquisition_ledger(
        paths["acquisition_contract"], ledger_path, root=root
    )
    manifest_path = ledger_path.with_name("manifest.json")
    _write(manifest_path, manifest)
    a2 = evaluate_single_stock_files(
        root=root,
        integrity_contract_path=paths["integrity_contract"],
        k1_contract_path=paths["k1_contract"],
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        stock=stock,
    )
    a2_path = root / "outputs/rehearsal" / stock / "a2.json"
    _write(a2_path, a2)
    render_root = root / "outputs/rehearsal" / stock / "a4"
    render = evaluate_single_stock_and_materialize(
        paths["confirmation_contract"],
        root=root,
        a2_report_path=a2_path,
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        output_dir=render_root,
        stock=stock,
    )
    review_path = root / "outputs/rehearsal" / stock / "severe.json"
    _write(
        review_path,
        {
            "schema": "neuro-film.sf3-a5-three-stock-severe-review.v1",
            "render_report_sha256": _sha_file(render_root / "report.json"),
            "confirmed_severe_count": 0,
            "reviewed_outputs": [
                {
                    "scene_id": row["scene_id"],
                    "stock_id": stock,
                    "png_sha256": row["png_sha256"],
                    "confirmed_severe": False,
                }
                for row in render["outputs"]
            ],
        },
    )
    return a2_path, render_root / "report.json", render_root, review_path


def _blind_observations(root: Path, package_root: Path) -> tuple[Path, Path]:
    sheet_path = package_root / "public_sheet.json"
    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    assignments = []
    for row in sheet["rows"]:
        label_means: dict[str, np.ndarray] = {}
        for facts in row["labels"]:
            label = facts["label"]
            image = cv2.imread(
                str(package_root / facts["relative_path"]), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError("blind image decode failed")
            label_means[label] = image[..., ::-1].astype(np.float64).mean(axis=(0, 1))
        remaining = set(label_means)
        velvia = max(
            remaining,
            key=lambda label: label_means[label][0] / (label_means[label][2] + 1.0),
        )
        remaining.remove(velvia)
        ektar = max(
            remaining,
            key=lambda label: label_means[label][2] / (label_means[label][0] + 1.0),
        )
        remaining.remove(ektar)
        portra = remaining.pop()
        assignments.append(
            {
                "round": row["round"],
                "scene_id": row["scene_id"],
                "label_to_stock": {
                    velvia: "fujifilm_velvia_50",
                    portra: "kodak_portra_400",
                    ektar: "kodak_ektar_100",
                },
            }
        )
    observations = {
        "schema": "neuro-film.sf3-a5-three-stock-blind-observations.v1",
        "status": "observations_frozen_mapping_unread",
        "mapping_files_read": False,
        "render_report_read": False,
        "stock_labeled_output_paths_read": False,
        "confirmation_target_pixels_read": False,
        "package_report_sha256": _sha_file(package_root / "report.json"),
        "public_sheet_sha256": _sha_file(sheet_path),
        "assignments": assignments,
    }
    observations_path = root / "outputs/rehearsal/observations.json"
    observations_raw = _write(observations_path, observations)
    reveal_path = root / "outputs/rehearsal/reveal.json"
    _write(
        reveal_path,
        {
            "schema": "neuro-film.sf3-a5-three-stock-mapping-reveal.v1",
            "status": "mapping_revealed_after_observations_commit",
            "observations_sha256": _sha(observations_raw),
            "private_mapping_sha256": _sha_file(package_root / "private_mapping.json"),
            "observations_commit": "1" * 40,
        },
    )
    return observations_path, reveal_path


def run_rehearsal(contract_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    contract_raw = contract_path.read_bytes()
    contract = json.loads(contract_raw)
    with tempfile.TemporaryDirectory(prefix="nf-sf3-a0a5-") as name:
        root = Path(name)
        paths = _copy_bound_parents(contract, root)
        _derive_contracts(paths)
        packet_path, packet = _fill_receipts(paths, root)
        ledger_path, _manifest_path, binding = _materialize_packet(
            paths, root, packet_path, packet
        )
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        lane_order = tuple(reversed(STOCKS)) if reverse else STOCKS
        lane_results = {
            stock: _stock_lane(stock, root=root, paths=paths, ledger=ledger)
            for stock in lane_order
        }
        render_reports = [lane_results[stock][1] for stock in STOCKS]
        render_roots = [lane_results[stock][2] for stock in STOCKS]
        reviews = [lane_results[stock][3] for stock in STOCKS]
        package_root = root / "outputs/rehearsal/a5"
        package = build_package(
            paths["blind_contract"],
            root=root,
            render_report_path=render_reports,
            render_root=render_roots,
            severe_review_path=reviews,
            secret="sf3-a0a5-structural-rehearsal-v1",
            output_dir=package_root,
        )
        observations_path, reveal_path = _blind_observations(root, package_root)
        adjudication = adjudicate_package(
            paths["blind_contract"],
            package_report_path=package_root / "report.json",
            public_sheet_path=package_root / "public_sheet.json",
            private_mapping_path=package_root / "private_mapping.json",
            observations_path=observations_path,
            reveal_path=reveal_path,
        )
        core = {
            "schema": "neuro-film.sf3-a0a5-synthetic-pipeline-rehearsal-result.v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": _sha(contract_raw),
            "parent_hashes_exact": True,
            "receipt_binding_automatic_pass": binding["automatic_pass"],
            "acquisition_rows": binding["evidence_scan_tasks"],
            "stocks": {
                stock: {
                    "a2_automatic_pass": json.loads(lane_results[stock][0].read_text())[
                        "automatic_pass"
                    ],
                    "a4_automatic_pass": json.loads(lane_results[stock][1].read_text())[
                        "automatic_pass"
                    ],
                    "confirmation_outputs": len(
                        json.loads(lane_results[stock][1].read_text())["outputs"]
                    ),
                }
                for stock in STOCKS
            },
            "blind_package_automatic_pass": package["automatic_pass"],
            "blind_adjudication_automatic_pass": adjudication["automatic_pass"],
            "blind_overall_assignment_accuracy": adjudication["measurements"][
                "overall_assignment_accuracy"
            ],
            "network_reads": 0,
            "persistent_pixel_or_report_writes": 0,
            "temporary_tree_removed_after_return": True,
            "automatic_pass": all(
                json.loads(lane_results[stock][0].read_text())["automatic_pass"]
                and json.loads(lane_results[stock][1].read_text())["automatic_pass"]
                for stock in STOCKS
            )
            and package["automatic_pass"]
            and adjudication["automatic_pass"],
            "decision": contract["decision_if_pass"],
            "claim_ceiling": contract["claim_ceiling"],
        }
        core["stable_evidence_id"] = _sha(_canonical(core))
        return core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run_rehearsal(args.contract, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
