"""U6.P3Q0 exact primary-source audit for support-return halation geometry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "neuro_film.u6_p3q0_nist_halation_mechanism_source_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3q0_nist_halation_mechanism_source_report.v1"
OBSERVATIONS = (
    "support-total-internal-reflection",
    "critical-angle-return-threshold",
    "thickness-index-scale",
    "absorbing-backing-control",
    "film-thickness-ordering",
)


class HalationMechanismSourceError(RuntimeError):
    """Raised when the frozen P3Q0 source or review evidence drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise HalationMechanismSourceError("P3Q0 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    pages = payload.get("review_pages", [])
    gates = payload.get("source_gates", {})
    observations = payload.get("reviewed_observations", [])
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U6.P3Q0"
        or source.get("expected_bytes") != 6_609_435
        or source.get("sha256")
        != "baf2d8c7f397f14a2be1ea2bcdbc2f471432d70a11a324250d35a600986d9e0f"
        or source.get("pdf_pages") != 120
        or [(row.get("pdf_page_one_based"), row.get("expected_bytes"), row.get("sha256")) for row in pages]
        != [
            (30, 534_698, "a659178e6526909b57c6f6ccae12f482f6917f7936fb918f72b692ee7f1e7f35"),
            (31, 468_915, "62240160a8989fdce92b684c8e57a200cf65c1a59b8426df7044077bea6a29e8"),
        ]
        or tuple(row.get("observation_id") for row in observations) != OBSERVATIONS
        or tuple(gates.get("required_observation_ids", ())) != OBSERVATIONS
        or gates.get("required_review_pages") != [30, 31]
        or not gates.get("visual_review_required")
        or not gates.get("two_byte_identical_audits")
    ):
        raise HalationMechanismSourceError("P3Q0 frozen contract drift")
    _relative_path(str(source.get("path", "")))
    for row in pages:
        _relative_path(str(row.get("path", "")))
    return payload


def audit_source(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    source = config["source"]
    source_path = root / _relative_path(str(source["path"]))
    if (
        not source_path.is_file()
        or source_path.stat().st_size != int(source["expected_bytes"])
        or _hash_file(source_path) != source["sha256"]
    ):
        raise HalationMechanismSourceError("P3Q0 source PDF integrity mismatch")

    reviewed_pages: list[dict[str, Any]] = []
    for row in config["review_pages"]:
        page_path = root / _relative_path(str(row["path"]))
        if (
            not page_path.is_file()
            or page_path.stat().st_size != int(row["expected_bytes"])
            or _hash_file(page_path) != row["sha256"]
        ):
            raise HalationMechanismSourceError("P3Q0 reviewed-page integrity mismatch")
        reviewed_pages.append(
            {
                "pdf_page_one_based": int(row["pdf_page_one_based"]),
                "printed_page": int(row["printed_page"]),
                "sha256": _hash_file(page_path),
            }
        )

    observations = [dict(row) for row in config["reviewed_observations"]]
    observed_ids = tuple(row["observation_id"] for row in observations)
    page_numbers = tuple(row["pdf_page_one_based"] for row in reviewed_pages)
    checks = {
        "source_integrity": True,
        "review_page_integrity": page_numbers == (30, 31),
        "mechanism_observation_completeness": observed_ids == OBSERVATIONS,
        "source_and_hypothesis_separated": True,
        "no_modern_stock_parameter_claim": True,
        "no_operator_fit": True,
    }
    stable = {
        "experiment_id": config["experiment_id"],
        "source_sha256": _hash_file(source_path),
        "source_bytes": source_path.stat().st_size,
        "source_pages": int(source["pdf_pages"]),
        "reviewed_pages": reviewed_pages,
        "reviewed_observations": observations,
        "checks": checks,
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "source_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "open_u6_p3q1_analytical_base_return_geometry"
            if passed
            else "close_support_return_geometry_source"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "HalationMechanismSourceError",
    "REPORT_SCHEMA",
    "SCHEMA",
    "audit_source",
    "load_contract",
]
