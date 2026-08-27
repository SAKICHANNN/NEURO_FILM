from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p266_dmeb_multiview_hdr_source_readiness import _extract_facts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p266_dmeb_multiview_hdr_source_readiness_v1.json"


def _responses(config: dict[str, object]) -> dict[str, dict[str, bytes]]:
    repo = config["official_repository"]
    expected = config["expected"]
    paths = [
        "LICENSE",
        "CITATION.cff",
        "dataset/DATASET_CARD.md",
        "code/checkpoints/README.md",
    ] + [f"code/item_{index}.py" for index in range(expected["tree_entry_count"] - 4)]
    return {
        "commit": {
            "body": json.dumps(
                {"sha": repo["commit"], "tree": {"sha": repo["tree"]}}
            ).encode()
        },
        "tree": {
            "body": json.dumps(
                {"sha": repo["tree"], "tree": [{"path": path} for path in paths]}
            ).encode()
        },
        "citation": {"body": (f"title: {expected['citation_title']}\n").encode()},
        "license": {
            "body": b"1. DATASETS Creative Commons Attribution-NonCommercial 4.0 non-commercial research. 2. CODE (code/) MIT License Permission is hereby granted sell copies"
        },
        "dataset_card": {
            "body": b"TODO FINAL [N] [N] synchronized, calibrated depth Multi-view LDR inputs HDR ground truth Intrinsics / extrinsics Exposure / gain metadata Valid masks Robot test holdout Robot train/val Sessions without pseudo-GT are excluded"
        },
        "benchmark": {
            "body": b"varying exposures linear HDR radiance reference-camera field of view"
        },
        "code_readme": {"body": b"reference inference code"},
        "checkpoint": {"body": b"[DRIVE-LINK] [MD5]"},
        "project": {
            "body": (
                f"{expected['project_title']} "
                "https://drive.google.com/drive/folders/fixtureOne"
            ).encode()
        },
    }


def test_p266_sources_are_exact_commit_pinned_official_endpoints() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repo = config["official_repository"]
    assert repo["commit"] == "559e52f4c4bc02a1285b5491fcacb5cd4c5844eb"
    assert repo["tree"] == "8e81f34d70fbfd0aa18484b9077c314d1787c50d"
    for key in ("citation", "license", "dataset_card", "checkpoint", "project"):
        assert repo["commit"] in config["sources"][key]["url"]


def test_p266_parser_separates_new_observation_from_release_readiness() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    facts = _extract_facts(config, _responses(config))
    assert facts["official_identity"]
    assert facts["new_physical_observation"]
    assert facts["code_commercial_compatible"]
    assert facts["dataset_noncommercial_terms_present"]
    assert not facts["dataset_commercial_compatible"]
    assert not facts["public_exact_counts"]
    assert not facts["public_exact_inventory"]
    assert not facts["exact_reference_checkpoint"]
    assert facts["group_isolation"]


def test_p266_contract_forbids_payload_or_candidate_promotion() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_READY_FOR_METADATA_ONLY_EXECUTION"
    assert "Google Drive" in config["stop_rule"]
    assert "candidate 3" in config["claim_ceiling"]
    assert config["gates"]["require_two_reports_exact"]
