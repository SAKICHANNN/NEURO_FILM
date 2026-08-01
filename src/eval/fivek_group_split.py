"""Target-blind group split for large FiveK architecture controls."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence


class FiveKGroupSplitError(ValueError):
    """Raised when a leakage-safe split cannot satisfy the frozen support."""


def _group_order(group: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).hexdigest()
    return digest, group


def assign_group_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    id_field: str,
    seed: int,
    target_confirmation_rows: int,
    minimum_confirmation_rows: int,
    maximum_confirmation_rows: int,
    minimum_development_groups: int,
) -> dict[str, Any]:
    """Choose a deterministic whole-group subset nearest a target row count.

    The dynamic program uses only row identity and the declared nuisance group;
    pixels, targets, operator fits and evaluation outcomes are never inspected.
    """

    if (
        not rows
        or minimum_confirmation_rows <= 0
        or minimum_confirmation_rows > target_confirmation_rows
        or target_confirmation_rows > maximum_confirmation_rows
    ):
        raise FiveKGroupSplitError("invalid split contract")
    identifiers = [str(row[id_field]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise FiveKGroupSplitError("duplicate row identity")

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group = str(row[group_field]).strip()
        if not group:
            raise FiveKGroupSplitError("empty group identity")
        grouped[group].append(str(row[id_field]))
    if len(grouped) < minimum_development_groups + 1:
        raise FiveKGroupSplitError("insufficient independent groups")

    ordered_groups = sorted(grouped, key=lambda value: _group_order(value, seed))
    subsets: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered_groups:
        size = len(grouped[group])
        additions: dict[int, tuple[str, ...]] = {}
        for count, selected in subsets.items():
            updated_count = count + size
            if updated_count > maximum_confirmation_rows:
                continue
            candidate = selected + (group,)
            previous = subsets.get(updated_count) or additions.get(updated_count)
            if previous is None or candidate < previous:
                additions[updated_count] = candidate
        for count, selected in additions.items():
            previous = subsets.get(count)
            if previous is None or selected < previous:
                subsets[count] = selected

    candidates = [
        (abs(count - target_confirmation_rows), count, selected)
        for count, selected in subsets.items()
        if minimum_confirmation_rows <= count <= maximum_confirmation_rows
        and len(grouped) - len(selected) >= minimum_development_groups
    ]
    if not candidates:
        raise FiveKGroupSplitError("no whole-group split satisfies support")
    _, confirmation_count, confirmation_tuple = min(candidates)
    confirmation_groups = set(confirmation_tuple)

    assigned = []
    for row in sorted(rows, key=lambda value: str(value[id_field])):
        item = dict(row)
        item["split"] = (
            "confirmation"
            if str(row[group_field]) in confirmation_groups
            else "development"
        )
        assigned.append(item)
    development_groups = sorted(set(grouped) - confirmation_groups)
    return {
        "rows": assigned,
        "development_rows": len(rows) - confirmation_count,
        "confirmation_rows": confirmation_count,
        "development_groups": development_groups,
        "confirmation_groups": sorted(confirmation_groups),
        "selection_used_target_or_pixels": False,
    }


__all__ = [
    "FiveKGroupSplitError",
    "assign_group_split",
]
