"""Deterministic blind-audit protocol for FilmCase style and artifact evidence.

This module anonymizes candidate labels per sample/round and aggregates only
recorded reviews. It is intentionally not an automatic aesthetics model and
must never be described as population preference.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable


AUDIT_SCHEMA_VERSION = 1
SEVERE_CHOICES = frozenset({"no", "yes", "uncertain"})
SCORE_MIN, SCORE_MAX = 1, 5


class VisionAuditError(ValueError):
    """Raised for invalid blind-audit input or a malformed review."""


@dataclass(frozen=True)
class BlindAuditPlan:
    """Public blinded sheet plus private reversible mapping for one audit run."""

    sheet: tuple[dict[str, Any], ...]
    mapping: tuple[dict[str, Any], ...]


def _token(seed: int, round_index: int, sample_id: str) -> int:
    raw = f"{seed}:{round_index}:{sample_id}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def build_blind_audit(sample_ids: Iterable[str], candidate_ids: Iterable[str], *, seed: int, rounds: int = 3) -> BlindAuditPlan:
    """Create per-sample candidate permutations without exposing candidate IDs.

    The returned sheet is safe to hand to an auditor. The mapping must remain a
    private ignored artifact until reviews are finalized.
    """

    samples = [str(value) for value in sample_ids]
    candidates = [str(value) for value in candidate_ids]
    if not samples or len(samples) != len(set(samples)):
        raise VisionAuditError("sample_ids must be non-empty and unique")
    if len(candidates) < 2 or len(candidates) != len(set(candidates)) or len(candidates) > 26:
        raise VisionAuditError("candidate_ids must contain 2..26 unique values")
    if rounds != 3:
        raise VisionAuditError("FilmCase protocol requires exactly three blind rounds")
    labels = [chr(ord("A") + index) for index in range(len(candidates))]
    sheet: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        for sample_id in samples:
            shuffled = candidates.copy()
            random.Random(_token(seed, round_index, sample_id)).shuffle(shuffled)
            pair = {label: candidate for label, candidate in zip(labels, shuffled, strict=True)}
            sheet.append({"round": round_index, "sample_id": sample_id, "labels": labels})
            mapping.append({"round": round_index, "sample_id": sample_id, "label_to_candidate": pair})
    return BlindAuditPlan(tuple(sheet), tuple(mapping))


def aggregate_reviews(plan: BlindAuditPlan, reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Unblind complete review records and apply the pre-registered decision rule."""

    mapping = {(row["round"], row["sample_id"]): row["label_to_candidate"] for row in plan.mapping}
    expected = {(row["round"], row["sample_id"], label) for row in plan.mapping for label in row["label_to_candidate"]}
    seen: set[tuple[int, str, str]] = set()
    entries: list[dict[str, Any]] = []
    for review in reviews:
        try:
            key = (int(review["round"]), str(review["sample_id"]), str(review["label"]))
            severe = review["severe"]
            style = int(review["style_strength"])
            appeal = int(review["appeal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VisionAuditError(f"malformed review: {review!r}") from exc
        if key not in expected or key in seen:
            raise VisionAuditError(f"unknown or duplicate review key: {key}")
        if severe not in SEVERE_CHOICES or not SCORE_MIN <= style <= SCORE_MAX or not SCORE_MIN <= appeal <= SCORE_MAX:
            raise VisionAuditError(f"invalid review values: {key}")
        seen.add(key)
        candidate_id = mapping[(key[0], key[1])][key[2]]
        entries.append({**review, "candidate_id": candidate_id})
    missing = expected - seen
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["candidate_id"], str(entry["sample_id"]))].append(entry)
    cases: list[dict[str, Any]] = []
    for mapping_row in plan.mapping:
        sample_id = mapping_row["sample_id"]
        for candidate_id in mapping_row["label_to_candidate"].values():
            key = (candidate_id, sample_id)
            if any(case["candidate_id"] == candidate_id and case["sample_id"] == sample_id for case in cases):
                continue
            votes = grouped.get(key, [])
            severe_yes = sum(vote["severe"] == "yes" for vote in votes)
            uncertain = sum(vote["severe"] == "uncertain" for vote in votes)
            if len(votes) != 3 or uncertain or severe_yes == 1:
                decision = "needs_original_resolution_adjudication"
            elif severe_yes >= 2:
                decision = "veto"
            else:
                decision = "no_severe_in_blind_pass"
            cases.append({"candidate_id": candidate_id, "sample_id": sample_id, "review_count": len(votes), "severe_yes": severe_yes, "uncertain": uncertain, "decision": decision, "style_median": median([vote["style_strength"] for vote in votes]) if votes else None, "appeal_median": median([vote["appeal"] for vote in votes]) if votes else None})
    candidate_summary: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted({row["candidate_id"] for row in cases}):
        rows = [row for row in cases if row["candidate_id"] == candidate_id]
        candidate_summary[candidate_id] = {
            "sample_count": len(rows),
            "veto_count": sum(row["decision"] == "veto" for row in rows),
            "needs_adjudication_count": sum(row["decision"] == "needs_original_resolution_adjudication" for row in rows),
            "style_median_of_samples": median([row["style_median"] for row in rows if row["style_median"] is not None]) if rows else None,
            "appeal_median_of_samples": median([row["appeal_median"] for row in rows if row["appeal_median"] is not None]) if rows else None,
            "blind_pass_complete": not missing and all(row["decision"] == "no_severe_in_blind_pass" for row in rows),
        }
    return {"schema_version": AUDIT_SCHEMA_VERSION, "reviewed_count": len(entries), "expected_review_count": len(expected), "missing_review_count": len(missing), "cases": cases, "candidates": candidate_summary, "claim_boundary": "autonomous visual evidence only; not population preference"}
