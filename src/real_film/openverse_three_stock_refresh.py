"""Fresh-only audit for the bounded SF3.A2 Openverse metadata refresh."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.real_film.openverse_stock_source import (
    OpenverseStockSourceError,
    audit_snapshot,
)

_IDENTITY_FIELDS = (
    "foreign_landing_url",
    "creator_url",
    "source",
    "provider",
    "license",
    "license_version",
)


def _read_bound_json(root: Path, binding: Mapping[str, Any]) -> Mapping[str, Any]:
    path = root / str(binding["path"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != str(binding["sha256"]):
        raise OpenverseStockSourceError(f"bound input hash mismatch: {binding['path']}")
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise OpenverseStockSourceError(
            f"bound input is not an object: {binding['path']}"
        )
    return value


def _prior_identities(snapshot: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    landings: set[str] = set()
    for category in snapshot.get("categories", []):
        for row in category.get("rows", []):
            row_id = str(row.get("id") or "")
            landing = str(row.get("foreign_landing_url") or "")
            if row_id:
                ids.add(row_id)
            if landing:
                landings.add(landing)
    return ids, landings


def _deduplicate_rows(
    rows: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int, list[str]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_count = 0
    conflicts: list[str] = []
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id:
            conflicts.append("missing-id")
            continue
        previous = by_id.get(row_id)
        if previous is None:
            by_id[row_id] = row
            continue
        duplicate_count += 1
        if any(
            str(previous.get(field) or "") != str(row.get(field) or "")
            for field in _IDENTITY_FIELDS
        ):
            conflicts.append(f"conflicting-duplicate:{row_id}")
    return list(by_id.values()), duplicate_count, sorted(set(conflicts))


def audit_fresh_snapshot(
    snapshot: Mapping[str, Any], config: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exclude all prior Openverse identities, then run the frozen graph audit."""
    prior = config["prior_inputs"]
    previous_snapshot = _read_bound_json(root, prior["openverse_snapshot"])
    _read_bound_json(root, prior["yfcc_triangle_report"])
    _read_bound_json(root, prior["commons_connectivity_report"])
    excluded_ids, excluded_landings = _prior_identities(previous_snapshot)

    cleaned = deepcopy(dict(snapshot))
    categories = cleaned.get("categories")
    if not isinstance(categories, list):
        raise OpenverseStockSourceError("snapshot categories are missing")
    duplicate_count = 0
    excluded_count = 0
    conflicts: list[str] = []
    fresh_counts: dict[str, int] = {}
    for category in categories:
        rows = category.get("rows")
        if not isinstance(rows, list):
            raise OpenverseStockSourceError("snapshot rows are missing")
        unique, duplicates, row_conflicts = _deduplicate_rows(rows)
        duplicate_count += duplicates
        conflicts.extend(
            f"{category.get('film_stock_id')}:{value}" for value in row_conflicts
        )
        fresh = [
            row
            for row in unique
            if str(row.get("id") or "") not in excluded_ids
            and str(row.get("foreign_landing_url") or "") not in excluded_landings
        ]
        excluded_count += len(unique) - len(fresh)
        category["rows"] = fresh
        fresh_counts[str(category.get("film_stock_id"))] = len(fresh)

    report, decision = audit_snapshot(cleaned, config)
    if conflicts:
        decision["decision"] = "query_contract_mismatch"
        decision["contract_errors"] = sorted(
            set(decision["contract_errors"] + conflicts)
        )
        report["summary"] = decision
    refresh = {
        "prior_openverse_id_count": len(excluded_ids),
        "prior_openverse_landing_url_count": len(excluded_landings),
        "repeated_page_rows_deduplicated": duplicate_count,
        "prior_rows_excluded": excluded_count,
        "fresh_rows_by_stock": dict(sorted(fresh_counts.items())),
        "duplicate_identity_conflicts": sorted(set(conflicts)),
    }
    report["freshness"] = refresh
    decision["freshness"] = refresh
    report["summary"] = decision
    return report, decision
