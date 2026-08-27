from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.audit_p286_wildrelight_paired_hdr_source_feasibility import (
    _manifest_facts,
    _source_facts,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p286_wildrelight_paired_hdr_source_feasibility_v1.json"


def test_p286_exact_revision_and_zero_payload_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    dataset = config["official_dataset"]
    assert dataset["id"] == "Lez/wildrelight"
    assert dataset["revision"] == "90ab579145da9f3ea998d706c8defd0d4b87641f"
    assert dataset["revision"] in config["sources"]["readme"]["url"]
    assert "candidate 3" in config["claim_ceiling"]
    assert "EXR/DNG" in config["stop_rule"]


def test_p286_source_parser_accepts_rights_and_capture_roles() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    api = {
        "id": config["official_dataset"]["id"],
        "sha": config["official_dataset"]["revision"],
        "private": False,
        "gated": False,
        "disabled": False,
    }
    readme = (
        b"---\nlicense: cc-by-4.0\n---\n# WildRelight 2605.11696\n"
        b"Creative Commons Attribution 4.0 International (CC BY 4.0)"
    )
    role = {
        "time": "time0",
        "shooting_time": "2025:01:01 12:00:00",
        "iso": 100,
        "source_files": ["a.dng", "b.dng"],
        "exposure_times": [0.01, 0.1],
        "ref_shutter": 0.01,
    }
    meta = {
        "scene": config["expected"]["metadata_witness_scene"],
        "ref_shutter": 0.01,
        "hdr_method": "fixture",
        "photos": [{**role, "time": f"time{i}"} for i in range(3)],
        "envmaps": [
            {key: value for key, value in {**role, "time": f"time{i}"}.items() if key != "source_files"}
            for i in range(3)
        ],
        "alignment_method": "fixture",
    }
    facts = _source_facts(
        config,
        json.dumps(api).encode(),
        readme,
        json.dumps(meta).encode(),
    )
    assert facts["official_identity"]
    assert facts["commercial_compatible_dataset_rights"]
    assert facts["metadata_roles"]
    assert facts["raw_dng_names_present_in_metadata"]


def test_p286_manifest_parser_keeps_dng_names_separate_from_payloads() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    local = copy.deepcopy(config)
    items = [
        {"path": "small-aligned", "type": "directory", "size": 0, "oid": "d"},
        {"path": "small-aligned/s", "type": "directory", "size": 0, "oid": "d"},
        {
            "path": "small-aligned/s/photo/time0_hdr.exr",
            "type": "file",
            "size": 10,
            "oid": "a",
            "lfs": {"sha256": "a" * 64},
        },
        {
            "path": "small-aligned/s/envmap/time0_envmap.exr",
            "type": "file",
            "size": 11,
            "oid": "b",
            "lfs": {"sha256": "b" * 64},
        },
        {
            "path": "small-aligned/s/meta.json",
            "type": "file",
            "size": 12,
            "oid": "c",
            "lfs": None,
        },
    ]
    lines = [
        "small-aligned/s/envmap/time0_envmap.exr|11|b|" + "b" * 64,
        "small-aligned/s/meta.json|12|c|",
        "small-aligned/s/photo/time0_hdr.exr|10|a|" + "a" * 64,
    ]
    manifest = ("\n".join(lines) + "\n").encode()
    local["expected"].update(
        {
            "entry_count": 5,
            "file_count": 3,
            "directory_count": 2,
            "total_file_bytes": 33,
            "lfs_file_count": 2,
            "manifest_bytes": len(manifest),
            "manifest_sha256": __import__("hashlib").sha256(manifest).hexdigest(),
            "variants": ["small-aligned"],
            "scene_count": 1,
            "photo_exr_count": 1,
            "envmap_exr_count": 1,
            "auxiliary_envmap_exr_count": 0,
            "metadata_count": 1,
        }
    )
    facts = _manifest_facts(local, items)
    assert facts["exact_inventory"]
    assert facts["roles_complete"]
    assert facts["dng_count"] == 0
