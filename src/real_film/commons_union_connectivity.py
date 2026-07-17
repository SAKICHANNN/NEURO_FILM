"""Offline connectivity audit for the frozen SF2.3 Commons metadata union."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from src.real_film.commons_stock_source import normalize_author


class CommonsUnionConnectivityError(ValueError):
    """Raised when the frozen SF2.3 contract is structurally invalid."""


_CLOSED_FLAGS = (
    "network_access_allowed",
    "image_payload_download_allowed",
    "photo_page_access_allowed",
    "operator_fitting_allowed",
    "training_allowed",
    "latent_mode_study_allowed",
)


def _validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config.get("inputs"), list) or not config["inputs"]:
        raise CommonsUnionConnectivityError("at least one immutable input is required")
    for flag in _CLOSED_FLAGS:
        if config.get(flag) is not False:
            raise CommonsUnionConnectivityError(f"{flag} must remain false")
    identity = config.get("author_identity", {})
    if identity.get("unverified_alias_merging_allowed") is not False:
        raise CommonsUnionConnectivityError("unverified author alias merging must remain false")
    if identity.get("string_reversal_alias_inference_allowed") is not False:
        raise CommonsUnionConnectivityError("string-reversal alias inference must remain false")
    aliases = identity.get("verified_aliases")
    if not isinstance(aliases, Mapping):
        raise CommonsUnionConnectivityError("verified_aliases must be a mapping")
    eligibility = config.get("eligibility", {})
    for flag in (
        "use_each_source_config_permissive_candidate_licenses",
        "use_each_source_config_non_scene_title_patterns",
        "require_nonempty_normalized_author",
        "require_file_page_original_and_derivative_urls",
        "require_derivative_url_distinct_from_original",
        "require_page_id_and_api_sha1",
        "require_license_url_or_public_domain_usage_terms",
        "bot_uploader_never_substitutes_for_author",
    ):
        if eligibility.get(flag) is not True:
            raise CommonsUnionConnectivityError(f"eligibility.{flag} must remain true")
    gates = config.get("connectivity_gate", {})
    for flag in (
        "cross_stock_page_id_overlap_must_be_zero",
        "cross_stock_file_page_url_overlap_must_be_zero",
        "cross_stock_original_url_overlap_must_be_zero",
        "cross_stock_api_sha1_overlap_must_be_zero",
    ):
        if gates.get(flag) is not True:
            raise CommonsUnionConnectivityError(f"{flag} must fail closed")
    if config.get("allowed_decisions") != [
        "open_bounded_commons_live_label_rights_preflight",
        "insufficient_shared_author_connectivity",
        "cross_stock_identity_overlap",
        "input_contract_mismatch",
    ]:
        raise CommonsUnionConnectivityError("allowed decision order does not match the frozen contract")


def _canonical_author(raw: str, aliases: Mapping[str, Any]) -> str:
    normalized = normalize_author(raw)
    target = aliases.get(normalized, normalized)
    if not isinstance(target, str):
        raise CommonsUnionConnectivityError("verified author alias targets must be strings")
    return normalize_author(target)


def _connected_components(nodes: Sequence[str], edges: Mapping[tuple[str, str], Sequence[str]]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[list[str]] = []
    unseen = set(nodes)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component))
        unseen -= component
        components.append(sorted(component))
    return components


def _overlaps(index: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    return [
        {"identity": identity, "stocks": sorted(stocks)}
        for identity, stocks in sorted(index.items())
        if identity and len(stocks) > 1
    ]


def audit_union(
    inputs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    config: Mapping[str, Any],
    *,
    input_contract_errors: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit already-frozen Commons snapshots without network or pixel access."""
    _validate_config(config)
    aliases = config["author_identity"]["verified_aliases"]
    required_scope = str(config["eligibility"]["required_label_scope"])
    strict_by_stock: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors = list(input_contract_errors)
    seen_stocks: set[str] = set()

    for snapshot_index, (snapshot, source_config) in enumerate(inputs):
        categories = snapshot.get("categories")
        configured_categories = source_config.get("categories")
        if not isinstance(categories, list) or not isinstance(configured_categories, list):
            errors.append(f"input-{snapshot_index}:missing-categories")
            continue
        expected = {str(row["film_stock_id"]): row for row in configured_categories}
        permissive = set(source_config["license_policy"]["permissive_candidate_licenses"])
        patterns = [
            re.compile(str(value), re.IGNORECASE)
            for value in source_config["metadata_flags"]["non_scene_title_patterns"]
        ]
        observed_ids: set[str] = set()
        for category in categories:
            stock_id = str(category.get("film_stock_id") or "")
            expected_row = expected.get(stock_id)
            if expected_row is None or stock_id in seen_stocks or stock_id in observed_ids:
                errors.append(f"input-{snapshot_index}:{stock_id}:unexpected-or-duplicate-stock")
                continue
            observed_ids.add(stock_id)
            seen_stocks.add(stock_id)
            if category.get("category") != expected_row.get("category"):
                errors.append(f"input-{snapshot_index}:{stock_id}:category-mismatch")
                continue
            if expected_row.get("label_scope") != required_scope or category.get("label_scope") != required_scope:
                continue
            strict_by_stock.setdefault(stock_id, [])
            files = category.get("files")
            if not isinstance(files, list):
                errors.append(f"input-{snapshot_index}:{stock_id}:missing-files")
                continue
            for row in files:
                if not isinstance(row, Mapping):
                    errors.append(f"input-{snapshot_index}:{stock_id}:non-object-row")
                    continue
                author = _canonical_author(str(row.get("author_raw_html") or ""), aliases)
                title = str(row.get("title") or "")
                page_url = str(row.get("file_page_url") or "")
                original_url = str(row.get("original_url") or "")
                derivative_url = str(row.get("derivative_1600_url") or "")
                licence = str(row.get("license_short_name") or "")
                licence_evidence = bool(str(row.get("license_url") or "")) or (
                    licence == "Public domain" and bool(str(row.get("usage_terms") or ""))
                )
                eligible = (
                    licence in permissive
                    and bool(author)
                    and bool(page_url and original_url and derivative_url)
                    and derivative_url != original_url
                    and bool(str(row.get("page_id") or ""))
                    and bool(str(row.get("api_sha1_base36") or ""))
                    and licence_evidence
                    and not any(pattern.search(title) for pattern in patterns)
                )
                if not eligible:
                    continue
                strict_by_stock[stock_id].append(
                    {
                        "author": author,
                        "page_id": str(row.get("page_id") or ""),
                        "file_page_url": page_url,
                        "original_url": original_url,
                        "api_sha1_base36": str(row.get("api_sha1_base36") or ""),
                    }
                )
        for missing in sorted(set(expected) - observed_ids):
            errors.append(f"input-{snapshot_index}:{missing}:configured-stock-missing")

    for stock_id, rows in strict_by_stock.items():
        for field in ("page_id", "file_page_url", "original_url", "api_sha1_base36"):
            counts = Counter(row[field] for row in rows)
            duplicates = sorted(identity for identity, count in counts.items() if count > 1)
            if duplicates:
                errors.append(f"{stock_id}:duplicate-{field}:{len(duplicates)}")

    gates = config["connectivity_gate"]
    stock_results: dict[str, dict[str, Any]] = {}
    authors_by_stock: dict[str, set[str]] = {}
    for stock_id in sorted(strict_by_stock):
        rows = strict_by_stock[stock_id]
        author_counts = Counter(row["author"] for row in rows)
        largest_share = max(author_counts.values()) / len(rows) if rows else 0.0
        eligible = (
            len(rows) >= int(gates["minimum_strict_rows_per_eligible_stock"])
            and len(author_counts) >= int(gates["minimum_unique_authors_per_eligible_stock"])
            and largest_share <= float(gates["maximum_largest_author_share_per_eligible_stock"])
        )
        authors_by_stock[stock_id] = set(author_counts)
        stock_results[stock_id] = {
            "strict_rows": len(rows),
            "unique_authors": len(author_counts),
            "largest_author": author_counts.most_common(1)[0][0] if author_counts else None,
            "largest_author_share": largest_share,
            "eligible": eligible,
            "author_counts": dict(sorted(author_counts.items())),
        }

    eligible_stocks = sorted(stock for stock, row in stock_results.items() if row["eligible"])
    raw_edges: dict[tuple[str, str], list[str]] = {}
    retained_edges: dict[tuple[str, str], list[str]] = {}
    for left, right in combinations(eligible_stocks, 2):
        shared = sorted(authors_by_stock[left] & authors_by_stock[right])
        if shared:
            raw_edges[(left, right)] = shared
        if len(shared) >= int(gates["minimum_shared_authors_per_graph_edge"]):
            retained_edges[(left, right)] = shared

    component_rows: list[dict[str, Any]] = []
    connected_nodes = sorted({stock for edge in retained_edges for stock in edge})
    for component in _connected_components(connected_nodes, retained_edges) if connected_nodes else []:
        component_set = set(component)
        component_edges = {
            edge: authors for edge, authors in retained_edges.items() if set(edge) <= component_set
        }
        shared_authors = sorted({author for authors in component_edges.values() for author in authors})
        authors_per_stock = {
            stock: sorted(
                {
                    author
                    for edge, authors in component_edges.items()
                    if stock in edge
                    for author in authors
                }
            )
            for stock in component
        }
        author_edge_counts = Counter(author for authors in component_edges.values() for author in authors)
        edge_count = len(component_edges)
        largest_edge_share = max(author_edge_counts.values()) / edge_count if edge_count else 0.0
        cycle_rank = edge_count - len(component) + 1
        passes = (
            len(component) >= int(gates["minimum_connected_stocks"])
            and len(shared_authors) >= int(gates["minimum_shared_authors_total"])
            and all(
                len(authors) >= int(gates["minimum_shared_authors_per_stock_in_component"])
                for authors in authors_per_stock.values()
            )
            and edge_count >= int(gates["minimum_component_edges"])
            and cycle_rank >= int(gates["minimum_component_cycle_rank"])
            and largest_edge_share <= float(gates["maximum_largest_shared_author_edge_share"])
        )
        component_rows.append(
            {
                "stocks": component,
                "edges": edge_count,
                "cycle_rank": cycle_rank,
                "shared_authors": shared_authors,
                "shared_authors_per_stock": authors_per_stock,
                "author_edge_counts": dict(sorted(author_edge_counts.items())),
                "largest_shared_author_edge_share": largest_edge_share,
                "passes": passes,
            }
        )

    identity_fields = ("page_id", "file_page_url", "original_url", "api_sha1_base36")
    overlap_results: dict[str, list[dict[str, Any]]] = {}
    for field in identity_fields:
        index: dict[str, set[str]] = defaultdict(set)
        for stock_id, rows in strict_by_stock.items():
            for row in rows:
                index[row[field]].add(stock_id)
        overlap_results[field] = _overlaps(index)
    has_overlap = any(overlap_results.values())
    passing = sorted(
        (row for row in component_rows if row["passes"]),
        key=lambda row: (-len(row["stocks"]), -len(row["shared_authors"]), -row["edges"], row["stocks"]),
    )
    if errors:
        decision_name = "input_contract_mismatch"
    elif has_overlap:
        decision_name = "cross_stock_identity_overlap"
    elif passing:
        decision_name = "open_bounded_commons_live_label_rights_preflight"
    else:
        decision_name = "insufficient_shared_author_connectivity"
    if decision_name not in config["allowed_decisions"]:
        raise CommonsUnionConnectivityError("decision is outside the frozen branch set")

    boundary = {
        "network_access_allowed": False,
        "photo_page_access_allowed": False,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    decision = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "decision": decision_name,
        "selected_component": passing[0] if passing else None,
        "input_contract_errors": sorted(errors),
        "cross_stock_identity_overlaps": overlap_results,
        **boundary,
    }
    report = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "strict_rows": sum(len(rows) for rows in strict_by_stock.values()),
        "stocks_with_strict_rows": len(strict_by_stock),
        "unique_authors": len({author for authors in authors_by_stock.values() for author in authors}),
        "eligible_stocks": eligible_stocks,
        "stock_results": stock_results,
        "raw_edges": [
            {"stocks": list(edge), "authors": authors, "author_count": len(authors)}
            for edge, authors in sorted(raw_edges.items())
        ],
        "retained_edges": [
            {"stocks": list(edge), "authors": authors, "author_count": len(authors)}
            for edge, authors in sorted(retained_edges.items())
        ],
        "components": component_rows,
        "summary": decision,
        **boundary,
    }
    return report, decision
