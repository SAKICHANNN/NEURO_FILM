from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p304_rawhdr_paired_raw_source_readiness import _extract_facts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p304_rawhdr_paired_raw_source_readiness_v1.json"


def test_p304_sources_are_exact_commit_pinned_official_objects() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repo = config["official_repository"]
    assert repo["commit"] == "c49e8b2d6ae83ee156aeb957a4c328d1df68e9bd"
    assert repo["tree"] == "eacc5b8c8b3a287cb5a954360608adc880e6467a"
    assert f"/git/commits/{repo['commit']}" in config["sources"]["commit"]["url"]
    assert f"/git/trees/{repo['tree']}?recursive=1" in config["sources"]["tree"]["url"]
    assert f"/{repo['commit']}/README.md" in config["sources"]["readme"]["url"]


def test_p304_parser_separates_code_and_dataset_rights() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    readme = b"""
    # RawHDR: High Dynamic Range Image Reconstruction from a Single Raw Image
    Training data /data1/HDR/MAT_train/ and test data /data/HDR/MAT_test/.
    We capture a real paired Raw-to-HDR dataset. Canon 5D Mark IV.
    -3EV, 0EV, and +3EV. 0EV Raw images are served as input images.
    Ground truth images are fused by HDR merging method.
    324 pairs of Raw/HDR images.
    Dataset [OneDrive](https://1drv.ms/f/example) [Baidu](https://pan.baidu.com/s/x).
    """
    commit = json.dumps(
        {
            "sha": config["official_repository"]["commit"],
            "tree": {"sha": config["official_repository"]["tree"]},
        }
    ).encode()
    tree = json.dumps(
        {
            "sha": config["official_repository"]["tree"],
            "tree": [{"path": f"src/{index}.py"} for index in range(14)],
        }
    ).encode()
    responses = {
        "commit": {"body": commit},
        "tree": {"body": tree},
        "readme": {"body": readme},
        "license": {"body": b"MIT License\n"},
    }
    facts = _extract_facts(config, responses)
    assert facts["official_identity"]
    assert facts["materially_distinct_physical_observation"]
    assert facts["anonymous_payload_locator"]
    assert facts["code_license_is_mit"]
    assert not facts["commercial_compatible_data_rights"]
    assert not facts["public_exact_inventory"]
    assert not facts["group_isolation"]


def test_p304_contract_forbids_payload_and_candidate_promotion() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_READY_FOR_METADATA_ONLY_EXECUTION"
    assert "candidate 3" in config["claim_ceiling"]
    assert "dataset" in config["stop_rule"].casefold()
    assert config["gates"]["require_two_reports_exact"]
