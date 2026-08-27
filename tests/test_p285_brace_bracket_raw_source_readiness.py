from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p285_brace_bracket_raw_source_readiness import _extract_facts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p285_brace_bracket_raw_source_readiness_v1.json"


def _responses(config: dict[str, object]) -> dict[str, dict[str, bytes]]:
    repo = config["official_repository"]
    expected = config["expected"]
    paths = expected["tree_paths"]
    return {
        "commit": {
            "body": json.dumps(
                {
                    "sha": repo["commit"],
                    "tree": {"sha": repo["tree"]},
                    "parents": [{"sha": repo["parent"]}],
                }
            ).encode()
        },
        "tree": {
            "body": json.dumps(
                {
                    "sha": repo["tree"],
                    "truncated": False,
                    "tree": [
                        {
                            "path": path,
                            "type": "blob" if "." in Path(path).name else "tree",
                        }
                        for path in paths
                    ],
                }
            ).encode()
        },
        "readme": {
            "body": (
                f"# {expected['paper_title']}\n"
                f"arXiv {expected['arxiv_id']}\n"
                "A bracketed RAW dataset built with an automated multi-exposure capture tool. "
                "This motivates a multi-frame bracketed RAW restoration paradigm.\n"
                "## Installation\nTODO\n"
                "## Download Pretrained Models and Datasets\nTODO\n"
                "## Inference\nTODO\n"
                "[Dataset](https://github.com/ZZH-qwq/BRACE)\n"
            ).encode()
        },
    }


def test_p285_sources_are_exact_commit_pinned_official_endpoints() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repo = config["official_repository"]
    assert repo["commit"] == "c17763bdab166da11cff2eb42d7f6df51d0105a2"
    assert repo["tree"] == "63b5906181d31e3141db8585cc8bb5d4f5dbfccf"
    assert repo["commit"] in config["sources"]["readme"]["url"]
    assert repo["tree"] in config["sources"]["tree"]["url"]


def test_p285_parser_separates_observation_from_release_readiness() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    facts = _extract_facts(config, _responses(config))
    assert facts["official_identity"]
    assert facts["new_physical_observation"]
    assert facts["readme_todo_count"] == 3
    assert not facts["code_commercial_compatible"]
    assert not facts["dataset_commercial_compatible"]
    assert not facts["executable_release"]
    assert not facts["exact_public_archive_inventory"]
    assert not facts["exact_reference_checkpoint"]
    assert not facts["group_isolation"]


def test_p285_contract_forbids_payload_or_candidate_promotion() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_READY_FOR_METADATA_ONLY_EXECUTION"
    assert "candidate-3" in config["stop_rule"]
    assert "candidate 3" in config["claim_ceiling"]
    assert config["gates"]["require_two_reports_exact"]
