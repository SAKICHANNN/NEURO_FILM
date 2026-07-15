"""Conservative creator/sequence leakage groups for the FSA/OWI archive."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.real_film.fsa_owi import FsaOwiAcquisitionError
from src.real_film.fsa_owi_pilot import UNKNOWN_CREATOR


LOC_NUMERIC_RE = re.compile(r"^fsac\.([0-9a-z]*?)(\d+)$", re.IGNORECASE)


def parse_loc_numeric(identifier: object) -> tuple[str, int]:
    match = LOC_NUMERIC_RE.fullmatch(str(identifier))
    if not match:
        raise FsaOwiAcquisitionError(f"unsupported LOC identifier: {identifier}")
    return match.group(1).lower(), int(match.group(2))


def build_sequence_guard_groups(
    records: Sequence[Mapping[str, Any]], *, maximum_adjacent_gap: int
) -> list[dict[str, Any]]:
    """Merge nearby LOC ids inside each creator as a no-leakage guard group."""
    if maximum_adjacent_gap < 0:
        raise FsaOwiAcquisitionError("maximum adjacent gap must be non-negative")
    by_creator: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        creator = str(record.get("creator_group") or UNKNOWN_CREATOR)
        by_creator[creator].append(record)
    output: list[dict[str, Any]] = []
    for creator, members in sorted(by_creator.items()):
        parsed = []
        for record in members:
            prefix, number = parse_loc_numeric(record["loc_fsac_id"])
            parsed.append((prefix, number, record))
        parsed.sort(key=lambda value: (value[0], value[1], str(value[2]["loc_fsac_id"])))
        runs: list[list[tuple[str, int, Mapping[str, Any]]]] = []
        current: list[tuple[str, int, Mapping[str, Any]]] = []
        previous_prefix: str | None = None
        previous_number: int | None = None
        for item in parsed:
            prefix, number, _ = item
            if (
                current
                and (
                    prefix != previous_prefix
                    or previous_number is None
                    or number - previous_number > maximum_adjacent_gap
                )
            ):
                runs.append(current)
                current = []
            current.append(item)
            previous_prefix, previous_number = prefix, number
        if current:
            runs.append(current)
        for run_index, run in enumerate(runs):
            prefix = run[0][0]
            start, end = run[0][1], run[-1][1]
            group_id = f"{creator}|{prefix}{start:05d}-{prefix}{end:05d}|g{run_index:02d}"
            for _, _, source in run:
                row = dict(source)
                row["evaluation_creator_group"] = creator
                row["sequence_guard_group"] = group_id
                row["sequence_guard_size"] = len(run)
                row["learning_eligible"] = creator != UNKNOWN_CREATOR
                output.append(row)
    output.sort(key=lambda row: str(row["loc_fsac_id"]))
    if len(output) != len(records):
        raise FsaOwiAcquisitionError("grouping lost records")
    return output


def evaluate_group_gate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    known = [row for row in rows if bool(row["learning_eligible"])]
    creator_counts = Counter(str(row["evaluation_creator_group"]) for row in known)
    sequence_counts = Counter(str(row["sequence_guard_group"]) for row in known)
    largest = max(sequence_counts.values(), default=0)
    coverage = len(known) / len(rows) if rows else 0.0
    largest_fraction = largest / len(known) if known else 1.0
    checks = {
        "minimum_known_creator_records": len(known)
        >= int(gates["minimum_known_creator_records"]),
        "minimum_creators_with_eight_records": sum(
            count >= 8 for count in creator_counts.values()
        )
        >= int(gates["minimum_creators_with_eight_records"]),
        "minimum_sequence_groups_with_three_records": sum(
            count >= 3 for count in sequence_counts.values()
        )
        >= int(gates["minimum_sequence_groups_with_three_records"]),
        "maximum_largest_sequence_group_fraction": largest_fraction
        <= float(gates["maximum_largest_sequence_group_fraction"]),
        "minimum_known_creator_coverage": coverage
        >= float(gates["minimum_known_creator_coverage"]),
        "unique_loc_identifiers": len(rows)
        == len({str(row["loc_fsac_id"]) for row in rows}),
    }
    passed = bool(rows and all(checks.values()))
    return {
        "passed": passed,
        "decision": "grouped_phase_c_allowed" if passed else "grouping_ineligible",
        "checks": checks,
        "records": len(rows),
        "known_creator_records": len(known),
        "unknown_creator_records": len(rows) - len(known),
        "known_creator_coverage": coverage,
        "creator_counts": dict(sorted(creator_counts.items())),
        "creators_with_eight_records": sum(count >= 8 for count in creator_counts.values()),
        "sequence_group_count": len(sequence_counts),
        "sequence_groups_with_three_records": sum(
            count >= 3 for count in sequence_counts.values()
        ),
        "largest_sequence_group_records": largest,
        "largest_sequence_group_fraction": largest_fraction,
        "sequence_group_sizes": dict(sorted(sequence_counts.items())),
    }
