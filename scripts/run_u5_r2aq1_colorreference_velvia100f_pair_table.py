#!/usr/bin/env python
"""Build and audit the frozen AQ1 ColorReference paired measurement table."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from src.roll2film.colorreference_pair_table import (  # noqa: E402
    parse_reference_table,
    repeated_set_metrics,
    sample_source_patches,
)


CONFIG_SHA256 = "3843dc8b81980efd7b5c4ddf179e383c0f4fdf59539365f04ff511696f02aea1"
REPORT_SCHEMA = "neuro-film.u5.r2aq1.colorreference-pair-table.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AQ1 requires a clean tracked worktree")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ1 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2aq1-colorreference-velvia100f-pair-table-v1"
        or config["table_contract"]["expected_rows"] != 8640
        or config["fit_allowed"]
        or config["training_allowed"]
        or config["render_allowed"]
        or config["operator_claim_allowed"]
    ):
        raise ValueError("AQ1 frozen contract mismatch")
    return config


def _member_key(name: str) -> tuple[int, int]:
    match = re.search(
        r"testscan(?P<set>\d+)[_-](?P<slide>[1-5])(?:\D|$)",
        name,
        re.I,
    )
    if match is None:
        raise ValueError("AQ1 cannot derive member key")
    return int(match.group("set")), int(match.group("slide"))


def _archive_tables(
    payload: bytes,
) -> dict[tuple[int, int], Any]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
        ]
    tables = {}
    for name in names:
        key = _member_key(name)
        if key in tables:
            raise ValueError("AQ1 duplicate member key")
        tables[key] = parse_reference_table(payload, name)
    return tables


def _sample_ids(config: dict[str, Any]) -> list[str]:
    geometry = config["source_patch_geometry"]
    main = geometry["main_grid"]
    sample_ids = [
        f"{row}{column}"
        for row in main["row_labels"]
        for column in main["column_labels"]
    ]
    sample_ids.extend(geometry["grayscale"]["sample_ids"])
    if len(sample_ids) != 288 or len(set(sample_ids)) != 288:
        raise ValueError("AQ1 sample ID contract mismatch")
    return sample_ids


def _load_parent_file_hashes(config: dict[str, Any]) -> dict[str, str]:
    report_path = ROOT / config["parent"]["report"]
    payload = report_path.read_bytes()
    if _sha256(payload) != config["parent"]["report_sha256"]:
        raise ValueError("AQ1 parent report hash mismatch")
    report = json.loads(payload)
    if not report["automatic_pass"]:
        raise ValueError("AQ1 parent did not pass")
    return {
        row["filename"]: row["sha256"]
        for row in report["passes"][0]["files"]
    }


def _load_exact(
    path: Path, expected_hashes: dict[str, str]
) -> bytes:
    payload = path.read_bytes()
    expected = expected_hashes.get(path.name)
    if expected is None or _sha256(payload) != expected:
        raise ValueError(f"AQ1 input hash mismatch for {path.name}")
    return payload


def _float_tuple(row: tuple[str, ...], start: int, end: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in row[start:end])
    if not all(np.isfinite(values)):
        raise ValueError("AQ1 non-finite measurement")
    return values


def build_audit(config: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    input_root = ROOT / config["input_root"]
    hashes = _load_parent_file_hashes(config)
    ids = _sample_ids(config)
    geometry = config["source_patch_geometry"]
    main = geometry["main_grid"]
    gray = geometry["grayscale"]
    source_codes = []
    source_ranges = []
    source_file_records = []
    for slide in config["slide_indices"]:
        path = input_root / f"slidescale{slide}.tif"
        payload = _load_exact(path, hashes)
        codes, ranges = sample_source_patches(
            payload,
            main_x=main["column_centers_x"],
            main_y=main["row_centers_y"],
            gray_x=gray["centers_x"],
            gray_y=gray["center_y"],
            radius=geometry["neighborhood_radius"],
        )
        source_codes.append(codes)
        source_ranges.append(ranges)
        source_file_records.append(
            {
                "slide_index": slide,
                "filename": path.name,
                "sha256": _sha256(payload),
                "unique_rgb_count": int(
                    np.unique(codes.astype(np.uint8), axis=0).shape[0]
                ),
                "minimum_code": int(codes.min()),
                "maximum_code": int(codes.max()),
                "maximum_within_patch_channel_range_codes": int(
                    ranges.max()
                ),
            }
        )
    source = np.stack(source_codes, axis=0)
    ranges = np.stack(source_ranges, axis=0)

    it8_tables = {}
    spectral_tables = {}
    for test_set in config["test_set_ids"]:
        it8_name = f"testscan{test_set}.zip"
        spectral_name = f"testscan{test_set}cgt.zip"
        it8_tables.update(
            _archive_tables(
                _load_exact(input_root / it8_name, hashes)
            )
        )
        spectral_tables.update(
            _archive_tables(
                _load_exact(input_root / spectral_name, hashes)
            )
        )

    labs = np.empty((6, 5, 288, 3), dtype=np.float64)
    table_lines = [
        "test_set\tslide_index\tsample_id\tsource_r\tsource_g\tsource_b"
        "\txyz_x\txyz_y\txyz_z\tlab_l\tlab_a\tlab_b"
        "\tspectral_pct_380_to_780"
    ]
    maximum_base_difference = 0.0
    wavelength_contract = config["measurement_integrity_gates"][
        "spectral_wavelengths_exact"
    ]
    spectral_min = float("inf")
    spectral_max = float("-inf")
    xyz_min = float("inf")
    xyz_max = float("-inf")
    for set_offset, test_set in enumerate(config["test_set_ids"]):
        for slide_offset, slide in enumerate(config["slide_indices"]):
            key = (test_set, slide)
            it8 = it8_tables[key]
            spectral = spectral_tables[key]
            if (
                it8.fields[:9] != spectral.fields[:9]
                or len(it8.rows) != 288
                or len(spectral.rows) != 288
            ):
                raise ValueError("AQ1 IT8/CGATS structure mismatch")
            for sample_offset, sample_id in enumerate(ids):
                it8_row = it8.rows[sample_offset]
                spectral_row = spectral.rows[sample_offset]
                if it8_row[0] != sample_id or spectral_row[0] != sample_id:
                    raise ValueError("AQ1 sample order mismatch")
                it8_base = _float_tuple(it8_row, 1, 9)
                spectral_base = _float_tuple(spectral_row, 1, 9)
                maximum_base_difference = max(
                    maximum_base_difference,
                    max(
                        abs(a - b)
                        for a, b in zip(it8_base, spectral_base)
                    ),
                )
                wavelengths = [
                    int(float(spectral_row[index]))
                    for index in range(9, 91, 2)
                ]
                spectral_pct = [
                    float(spectral_row[index])
                    for index in range(10, 91, 2)
                ]
                if wavelengths != wavelength_contract:
                    raise ValueError("AQ1 spectral wavelength mismatch")
                if not np.all(np.isfinite(spectral_pct)):
                    raise ValueError("AQ1 non-finite spectrum")
                spectral_min = min(spectral_min, min(spectral_pct))
                spectral_max = max(spectral_max, max(spectral_pct))
                xyz = it8_base[:3]
                lab = it8_base[3:6]
                xyz_min = min(xyz_min, min(xyz))
                xyz_max = max(xyz_max, max(xyz))
                labs[set_offset, slide_offset, sample_offset] = lab
                rgb = source[slide_offset, sample_offset] / 255.0
                table_lines.append(
                    "\t".join(
                        [
                            str(test_set),
                            str(slide),
                            sample_id,
                            *(format(value, ".17g") for value in rgb),
                            *(format(value, ".17g") for value in xyz),
                            *(format(value, ".17g") for value in lab),
                            ";".join(
                                format(value, ".17g")
                                for value in spectral_pct
                            ),
                        ]
                    )
                )
    table_bytes = ("\n".join(table_lines) + "\n").encode("utf-8")
    metrics = repeated_set_metrics(labs)
    integrity = config["measurement_integrity_gates"]
    repeat_gates = config["repeated_set_identifiability_gates"]
    checks = [
        {
            "name": "expected_table_rows",
            "passed": len(table_lines) - 1
            == config["table_contract"]["expected_rows"],
        },
        {
            "name": "source_patch_uniformity",
            "passed": int(ranges.max())
            <= geometry["maximum_within_patch_channel_range_codes"],
        },
        {
            "name": "source_unique_rgb_support",
            "passed": all(
                row["unique_rgb_count"]
                >= config["table_contract"][
                    "minimum_unique_source_rgb_per_slide"
                ]
                for row in source_file_records
            ),
        },
        {
            "name": "source_code_range",
            "passed": all(
                row["minimum_code"]
                <= config["table_contract"]["source_minimum_code_max"]
                and row["maximum_code"]
                >= config["table_contract"]["source_maximum_code_min"]
                for row in source_file_records
            ),
        },
        {
            "name": "it8_cgats_base_fields",
            "passed": maximum_base_difference
            <= integrity["it8_cgats_base_fields_max_abs_difference"],
        },
        {
            "name": "spectral_value_range",
            "passed": spectral_min >= integrity["spectral_pct_minimum"]
            and spectral_max <= integrity["spectral_pct_maximum"],
        },
        {
            "name": "target_xyz_lab_ranges",
            "passed": bool(
                xyz_min >= integrity["xyz_minimum"]
                and xyz_max <= integrity["xyz_maximum"]
                and labs[..., 0].min() >= integrity["lab_l_minimum"]
                and labs[..., 0].max() <= integrity["lab_l_maximum"]
            ),
        },
        {
            "name": "median_repeated_set_radius",
            "passed": metrics["median_patch_set_radius_deltae76"]
            <= repeat_gates["median_patch_set_radius_deltae76_max"],
        },
        {
            "name": "p95_repeated_set_radius",
            "passed": metrics["p95_patch_set_radius_deltae76"]
            <= repeat_gates["p95_patch_set_radius_deltae76_max"],
        },
        {
            "name": "repeated_set_large_radius_fraction",
            "passed": metrics["fraction_patch_set_radius_above_10"]
            <= repeat_gates["fraction_patch_set_radius_above_10_max"],
        },
        {
            "name": "slide_distance_structure",
            "passed": metrics[
                "median_slide_distance_structure_spearman"
            ]
            >= repeat_gates[
                "median_slide_distance_structure_spearman_min"
            ],
        },
        {
            "name": "between_set_centroid_variance",
            "passed": metrics["between_set_centroid_variance_fraction"]
            <= repeat_gates["between_set_centroid_variance_fraction_max"],
        },
        {
            "name": "leave_one_set_consensus",
            "passed": metrics[
                "leave_one_set_consensus_rmse_deltae76"
            ]
            <= repeat_gates["leave_one_set_consensus_rmse_deltae76_max"],
        },
    ]
    integrity_pass = all(
        bool(check["passed"]) for check in checks[:7]
    )
    repeated_set_pass = all(
        bool(check["passed"]) for check in checks[7:]
    )
    passed = integrity_pass and repeated_set_pass
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "parent_report_sha256": config["parent"]["report_sha256"],
        "source_files": source_file_records,
        "table": {
            "row_count": len(table_lines) - 1,
            "sha256": _sha256(table_bytes),
            "format": "canonical_utf8_tsv_lf",
            "source_rgb_semantics": config["table_contract"][
                "source_profile_interpretation"
            ],
        },
        "measurement_integrity": {
            "maximum_it8_cgats_base_field_difference": maximum_base_difference,
            "spectral_pct_minimum": spectral_min,
            "spectral_pct_maximum": spectral_max,
            "xyz_minimum": xyz_min,
            "xyz_maximum": xyz_max,
        },
        "repeated_set_metrics": metrics,
        "automatic_checks": checks,
        "automatic_pass": passed,
        "decision": (
            config["decision_branches"]["all_gates_pass"]
            if passed
            else (
                config["decision_branches"]["geometry_or_integrity_fail"]
                if not integrity_pass
                else config["decision_branches"][
                    "repeated_set_variability_fail"
                ]
            )
        ),
        "fit_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, table_bytes


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, CONFIG_SHA256)
    report, table_bytes = build_audit(config)
    _atomic_write(output_root / "pair_table.tsv", table_bytes)
    report_bytes = _canonical_json(report)
    _atomic_write(output_root / "report.json", report_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": _sha256(report_bytes),
                "table_sha256": report["table"]["sha256"],
                "metrics": report["repeated_set_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aq1_colorreference_velvia100f_pair_table_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256", default=CONFIG_SHA256
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq1_colorreference_velvia100f_pair_table_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
