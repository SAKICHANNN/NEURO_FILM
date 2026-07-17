from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.real_film.commons_union_connectivity import audit_union


def _config() -> dict:
    config = json.loads(
        Path("configs/real_film_commons_union_connectivity_v1.json").read_text(encoding="utf-8")
    )
    config["inputs"] = [{"snapshot": "x", "source_config": "y"}]
    config["connectivity_gate"].update(
        {
            "minimum_strict_rows_per_eligible_stock": 5,
            "minimum_unique_authors_per_eligible_stock": 5,
            "maximum_largest_author_share_per_eligible_stock": 0.4,
            "minimum_connected_stocks": 3,
            "minimum_shared_authors_total": 5,
            "minimum_shared_authors_per_graph_edge": 2,
            "minimum_shared_authors_per_stock_in_component": 2,
            "minimum_component_edges": 3,
            "minimum_component_cycle_rank": 1,
            "maximum_largest_shared_author_edge_share": 0.4,
        }
    )
    return config


def _source_config(stocks: list[str]) -> dict:
    return {
        "categories": [
            {"film_stock_id": stock, "category": stock, "label_scope": "exact_stock_community_category"}
            for stock in stocks
        ],
        "license_policy": {"permissive_candidate_licenses": ["CC BY 4.0"]},
        "metadata_flags": {"non_scene_title_patterns": ["film strip"]},
    }


def _row(stock: str, index: int, author: str) -> dict:
    return {
        "page_id": int(f"{ord(stock)}{index}"),
        "title": f"File:{stock}-{index}.jpg",
        "file_page_url": f"https://commons.test/{stock}/{index}",
        "original_url": f"https://upload.test/{stock}/{index}.jpg",
        "derivative_1600_url": f"https://upload.test/{stock}/thumb/{index}.jpg",
        "api_sha1_base36": f"sha-{stock}-{index}",
        "author_raw_html": author,
        "license_short_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "usage_terms": "",
    }


def _passing_input() -> tuple[dict, dict]:
    stocks = ["a", "b", "c"]
    pair_authors = {
        "a": ["ab-1", "ab-2", "ac-1", "ac-2", "unique-a"],
        "b": ["ab-1", "ab-2", "bc-1", "bc-2", "unique-b"],
        "c": ["ac-1", "ac-2", "bc-1", "bc-2", "unique-c"],
    }
    snapshot = {
        "categories": [
            {
                "film_stock_id": stock,
                "category": stock,
                "label_scope": "exact_stock_community_category",
                "files": [_row(stock, index, author) for index, author in enumerate(pair_authors[stock])],
            }
            for stock in stocks
        ]
    }
    return snapshot, _source_config(stocks)


def test_tracked_contract_is_offline_and_fail_closed() -> None:
    config = json.loads(
        Path("configs/real_film_commons_union_connectivity_v1.json").read_text(encoding="utf-8")
    )
    assert config["network_access_allowed"] is False
    assert config["image_payload_download_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["training_allowed"] is False
    assert config["latent_mode_study_allowed"] is False
    assert config["author_identity"]["verified_aliases"] == {}
    assert config["connectivity_gate"]["minimum_shared_authors_per_graph_edge"] == 2


def test_redundant_three_stock_cycle_passes_repeat_identically() -> None:
    inputs = [_passing_input()]
    result = audit_union(inputs, _config())
    assert result == audit_union(inputs, _config())
    report, decision = result
    assert decision["decision"] == "open_bounded_commons_live_label_rights_preflight"
    assert decision["selected_component"]["cycle_rank"] == 1
    assert len(report["retained_edges"]) == 3


def test_single_author_edges_are_reported_but_not_retained() -> None:
    snapshot, source = _passing_input()
    for category in snapshot["categories"]:
        for row in category["files"]:
            if row["author_raw_html"].endswith("-2"):
                row["author_raw_html"] = f"{row['author_raw_html']}-{category['film_stock_id']}"
    report, decision = audit_union([(snapshot, source)], _config())
    assert report["raw_edges"]
    assert report["retained_edges"] == []
    assert decision["decision"] == "insufficient_shared_author_connectivity"


def test_apparent_reversed_names_are_not_merged() -> None:
    snapshot, source = _passing_input()
    snapshot = deepcopy(snapshot)
    snapshot["categories"][0]["files"][0]["author_raw_html"] = "Svetlov Artem"
    snapshot["categories"][1]["files"][0]["author_raw_html"] = "Artem Svetlov"
    report, _ = audit_union([(snapshot, source)], _config())
    authors = set(report["stock_results"]["a"]["author_counts"])
    assert "svetlov artem" in authors
    assert "artem svetlov" not in authors


def test_cross_stock_identity_overlap_fails_before_graph_pass() -> None:
    snapshot, source = _passing_input()
    snapshot = deepcopy(snapshot)
    snapshot["categories"][1]["files"][0]["page_id"] = snapshot["categories"][0]["files"][0]["page_id"]
    _, decision = audit_union([(snapshot, source)], _config())
    assert decision["decision"] == "cross_stock_identity_overlap"
    assert decision["cross_stock_identity_overlaps"]["page_id"]


def test_input_contract_error_has_highest_priority() -> None:
    _, decision = audit_union([_passing_input()], _config(), input_contract_errors=["hash-mismatch"])
    assert decision["decision"] == "input_contract_mismatch"


def test_duplicate_identity_within_stock_is_contract_mismatch() -> None:
    snapshot, source = _passing_input()
    snapshot = deepcopy(snapshot)
    snapshot["categories"][0]["files"][1]["api_sha1_base36"] = snapshot["categories"][0]["files"][0]["api_sha1_base36"]
    _, decision = audit_union([(snapshot, source)], _config())
    assert decision["decision"] == "input_contract_mismatch"
    assert any("duplicate-api_sha1_base36" in error for error in decision["input_contract_errors"])
