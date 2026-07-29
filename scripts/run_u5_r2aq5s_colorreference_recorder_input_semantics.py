#!/usr/bin/env python
"""Audit whether ColorReference recorder RGB can map to a known photo space."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image, TiffTags

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)


CONFIG_SHA256 = "390cb4f96e9bc94752d6bdee8edc8cbf4d61e3b28fc0e1b3ec3ba66d43fd2d7b"
EXPERIMENT_ID = "u5.r2aq5s-colorreference-recorder-input-semantics-v1"
REPORT_SCHEMA = "neuro-film.u5.r2aq5s.recorder-input-semantics.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ5S config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or len(config["source_files"]) != 5
        or config["fit_allowed"]
        or config["training_allowed"]
        or config["render_allowed"]
        or config["production_integration_allowed"]
        or config["identification_gate"]["guess_srgb_or_scene_linear_allowed"]
        or config["identification_gate"][
            "parameter_inference_from_developed_slide_measurements_allowed"
        ]
    ):
        raise ValueError("AQ5S frozen contract mismatch")
    return config


def _tag_name(tag: int) -> str:
    return str(TiffTags.lookup(tag).name)


def audit_tiff(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    observed_sha = _sha256(payload)
    if observed_sha != expected_sha256:
        raise ValueError(f"AQ5S source hash mismatch: {path}")
    with Image.open(path) as image:
        tags = {_tag_name(tag): value for tag, value in image.tag_v2.items()}
        facts = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": observed_sha,
            "mode": image.mode,
            "size": list(image.size),
            "bits_per_sample": list(tags.get("BitsPerSample", [])),
            "samples_per_pixel": int(tags.get("SamplesPerPixel", 0)),
            "photometric_interpretation": int(
                tags.get("PhotometricInterpretation", -1)
            ),
            "software": str(tags.get("Software", "")),
            "icc_profile_present": "icc_profile" in image.info
            or "ICCProfile" in tags,
            "white_point_present": "WhitePoint" in tags,
            "primary_chromaticities_present": "PrimaryChromaticities"
            in tags,
            "transfer_function_present": "TransferFunction" in tags,
        }
    return facts


def run_audit(config_path: Path, expected_sha256: str) -> dict[str, Any]:
    config = load_config(config_path, expected_sha256)
    required = config["required_tiff_facts"]
    rows = [
        audit_tiff(ROOT / row["path"], row["sha256"])
        for row in config["source_files"]
    ]
    fact_keys = (
        "mode",
        "size",
        "bits_per_sample",
        "samples_per_pixel",
        "photometric_interpretation",
        "icc_profile_present",
        "white_point_present",
        "primary_chromaticities_present",
        "transfer_function_present",
    )
    tiff_facts_match = all(
        all(row[key] == required[key] for key in fact_keys) for row in rows
    )
    source = config["official_source"]
    identified_items = {
        "official_or_independent_transform": False,
        "recorder_primaries_or_spectral_response": source[
            "recorder_primaries_or_spectral_sensitivities"
        ]
        != "unknown",
        "recorder_transfer_function": source["recorder_transfer_function"]
        != "unknown",
        "embedded_or_sidecar_profile": any(
            row["icc_profile_present"]
            or row["white_point_present"]
            or row["primary_chromaticities_present"]
            or row["transfer_function_present"]
            for row in rows
        ),
    }
    evidence_count = sum(identified_items.values())
    threshold = config["identification_gate"][
        "minimum_required_evidence_items"
    ]
    identified = tiff_facts_match and evidence_count >= threshold
    decision = (
        config["decision_branches"]["identified"]
        if identified
        else config["decision_branches"]["unidentified"]
    )
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": CONFIG_SHA256,
        "software_commit": _git_commit(),
        "official_source": source,
        "source_tiffs": rows,
        "all_required_tiff_facts_match": tiff_facts_match,
        "identification_evidence": identified_items,
        "identification_evidence_count": evidence_count,
        "minimum_required_evidence_items": threshold,
        "recorder_input_semantics_identified": identified,
        "decision": decision,
        "interpretation": (
            "The official source explicitly defines device RGB and excludes "
            "sRGB; the exact TIFFs contain no ICC profile, white point, "
            "primaries or transfer function. Developed-slide measurements "
            "cannot identify the missing input encoding without circularly "
            "absorbing film response. Photo-domain mapping remains "
            "unidentified."
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aq5s_colorreference_recorder_input_semantics_v1.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq5s_colorreference_recorder_input_semantics_v1/report.json",
    )
    args = parser.parse_args()
    report = run_audit(args.config, args.expected_config_sha256)
    _atomic_write(args.output, _canonical_json(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
