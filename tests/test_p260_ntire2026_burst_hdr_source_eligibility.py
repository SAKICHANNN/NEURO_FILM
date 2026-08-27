from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p260_ntire2026_burst_hdr_source_eligibility import (
    _extract_facts,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p260_ntire2026_burst_hdr_source_eligibility_v1.json"


def test_p260_sources_are_commit_pinned_official_endpoints() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repo = config["official_repository"]
    assert repo["commit"] == "1ac0212b73b3dbdabb262bde36b51aad7abee971"
    assert repo["tree"] == "54c975f90ca19a06b243ccf710b79535f8b056f2"
    assert f"/{repo['commit']}/README.md" in config["sources"]["readme"]["url"]
    assert f"/git/trees/{repo['tree']}?recursive=1" in config["sources"]["tree"]["url"]
    assert "openaccess.thecvf.com" in config["sources"]["cvf"]["url"]


def test_p260_fact_parser_separates_observation_from_rights() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    readme = b"""
    CVPR 2026 New Trends in Image Restoration and Enhancement NTIRE workshop.
    Efficient Burst HDR and Restoration.
    Training, validation, and test datasets. 300 scenes. 200 scenes. 20 scenes. 20 scenes.
    Each scene consists of nine input RAW frames Scene-xxx-in-0.tif through Scene-xxx-in-8.tif
    and Scene-xxx-gt.tif. The reference frame is aligned with GT image and uses short exposure time.
    Other inputs use middle exposure time and high exposure time.
    All training datasets and their images must not be shared with others or used for other purposes.
    """
    commit = json.dumps(
        {
            "sha": config["official_repository"]["commit"],
            "tree": {"sha": config["official_repository"]["tree"]},
        }
    ).encode()
    tree_entries = [{"path": f"src/{index}.py"} for index in range(48)]
    tree = json.dumps(
        {"sha": config["official_repository"]["tree"], "tree": tree_entries}
    ).encode()
    cvf = (
        f"<html>{config['expected']['paper_title']} "
        f"{config['expected']['code_url']}</html>"
    ).encode()
    responses = {
        "commit": {"body": commit},
        "tree": {"body": tree},
        "readme": {"body": readme},
        "cvf": {"body": cvf},
    }
    facts = _extract_facts(config, responses)
    assert facts["official_identity"]
    assert facts["new_physical_observation"]
    assert facts["group_isolation"]
    assert facts["training_use_restriction_present"]
    assert not facts["commercial_compatible_data_rights"]
    assert not facts["commercial_compatible_code_rights"]
    assert not facts["anonymous_public_payload"]
    assert not facts["public_exact_inventory"]


def test_p260_contract_forbids_payload_or_candidate_promotion() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_READY_FOR_METADATA_ONLY_EXECUTION"
    assert "dataset" in config["stop_rule"].casefold()
    assert "candidate 3" in config["claim_ceiling"]
    assert config["gates"]["require_two_reports_exact"]
