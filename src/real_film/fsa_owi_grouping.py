"""Conservative creator/sequence leakage groups for the FSA/OWI archive."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.real_film.fsa_owi import FsaOwiAcquisitionError
from src.real_film.fsa_owi_pilot import UNKNOWN_CREATOR


LOC_NUMERIC_RE = re.compile(r"^fsac\.([0-9a-z]*?)(\d+)$", re.IGNORECASE)

# Caption-level locations supported by the official FSA/OWI assignment tables.
# Abbreviations are deliberately punctuation-bearing to avoid matching normal words.
LOCATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "AL": ("alabama", "ala."), "AZ": ("arizona", "ariz."),
    "CA": ("california", "calif."), "CO": ("colorado", "colo."),
    "CT": ("connecticut", "conn."),
    "DC": ("washington, d.c.", "washington, d.c", "washington, dc"),
    "FL": ("florida", "fla."), "GA": ("georgia", " ga."),
    "IA": ("iowa",), "ID": ("idaho",), "IL": ("illinois", "ill."),
    "IN": ("indiana", "ind."), "KS": ("kansas", "kan.", "kans."),
    "KY": ("kentucky", " ky."), "LA": ("louisiana", " la."),
    "MA": ("massachusetts", "mass."), "ME": ("maine", " me."),
    "MI": ("michigan", "mich."), "MN": ("minnesota", "minn."),
    "MO": ("missouri", " mo."), "MS": ("mississippi", "miss."),
    "MT": ("montana", "mont."), "NC": ("north carolina", "n.c."),
    "NE": ("nebraska", "neb."), "NH": ("new hampshire", "n.h."),
    "NJ": ("new jersey", "n.j."), "NM": ("new mexico", "n.m."),
    "NY": ("new york", "n.y."), "OH": ("ohio",),
    "OK": ("oklahoma", "okla."), "OR": ("oregon", " ore."),
    "PA": ("pennsylvania", " pa."), "PR": ("puerto rico",),
    "SC": ("south carolina", "s.c."), "TN": ("tennessee", "tenn."),
    "TX": ("texas", " tex."), "UT": ("utah",),
    "VA": ("virginia", " va.", "skyline drive"),
    "VI": ("virgin islands", "st. croix", "christiansted", "frederiksted"),
    "VT": ("vermont", " vt."), "WA": ("washington state", " wash."),
    "WI": ("wisconsin", " wis."), "WV": ("west virginia", "w. va.", "w.va."),
    "WY": ("wyoming", "wyo."),
}


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


def extract_assignment_location(description: object) -> str | None:
    """Extract conservative state/territory evidence from the archive caption."""
    text = f" {str(description or '').lower()} "
    hits = sorted(
        code
        for code, patterns in LOCATION_PATTERNS.items()
        if any(pattern in text for pattern in patterns)
    )
    return "+".join(hits) if hits else None


def build_location_guard_groups(
    sequence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Use official assignment-table locations, falling back to sequence guards."""
    output: list[dict[str, Any]] = []
    counts = Counter()
    prepared: list[tuple[str, dict[str, Any]]] = []
    for source in sequence_rows:
        row = dict(source)
        location = extract_assignment_location(row.get("description"))
        row["assignment_location_code"] = location
        if location:
            group = f"{row['evaluation_creator_group']}|location:{location}"
        else:
            group = f"{row['evaluation_creator_group']}|fallback:{row['sequence_guard_group']}"
        row["location_guard_group"] = group
        prepared.append((group, row))
        if bool(row["learning_eligible"]):
            counts[group] += 1
    for group, row in prepared:
        row["location_guard_size"] = counts[group] if bool(row["learning_eligible"]) else 0
        output.append(row)
    output.sort(key=lambda row: str(row["loc_fsac_id"]))
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


def evaluate_location_recovery_gate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    recovery = config["location_recovery"]
    known = [row for row in rows if bool(row["learning_eligible"])]
    located = [row for row in known if row.get("assignment_location_code")]
    counts = Counter(str(row["location_guard_group"]) for row in known)
    largest = max(counts.values(), default=0)
    coverage = len(located) / len(known) if known else 0.0
    largest_fraction = largest / len(known) if known else 1.0
    checks = {
        "minimum_location_coverage": coverage
        >= float(recovery["minimum_location_coverage"]),
        "minimum_location_guard_groups_with_three_records": sum(
            count >= 3 for count in counts.values()
        )
        >= int(recovery["minimum_location_guard_groups_with_three_records"]),
        "maximum_largest_location_guard_fraction": largest_fraction
        <= float(recovery["maximum_largest_location_guard_fraction"]),
    }
    passed = bool(known and all(checks.values()))
    return {
        "passed": passed,
        "decision": "grouped_phase_c_allowed" if passed else "location_grouping_ineligible",
        "checks": checks,
        "known_creator_records": len(known),
        "location_recovered_records": len(located),
        "location_coverage": coverage,
        "location_guard_group_count": len(counts),
        "location_guard_groups_with_three_records": sum(count >= 3 for count in counts.values()),
        "largest_location_guard_records": largest,
        "largest_location_guard_fraction": largest_fraction,
        "location_guard_sizes": dict(sorted(counts.items())),
    }
