"""Deterministic source admission audit for latest paired-retouch research."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


class LatestPairedRetouchSourceAuditError(ValueError):
    """Raised when a frozen source identity or expected source fact drifts."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _require(text: str, needle: str, label: str) -> None:
    if needle.casefold() not in text.casefold():
        raise LatestPairedRetouchSourceAuditError(f"missing frozen source fact: {label}")


def _gate_row(
    *,
    candidate_id: str,
    publication_year: int,
    paired_public: bool,
    commercial_data_rights: bool,
    grouping_facts: bool,
    new_observation: bool,
    source_facts: Mapping[str, Any],
) -> dict[str, Any]:
    gates = {
        "publication_year": publication_year >= 2025,
        "exact_official_identity": True,
        "public_paired_pixels_or_explicit_recipe": paired_public,
        "commercial_data_rights": commercial_data_rights,
        "scene_or_file_disjoint_grouping_facts": grouping_facts,
        "new_identifying_observation": new_observation,
    }
    return {
        "candidate_id": candidate_id,
        "publication_year": publication_year,
        "source_facts": dict(source_facts),
        "gates": gates,
        "admitted": all(gates.values()),
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
    }


def analyze_latest_paired_retouch_sources(
    *,
    config: Mapping[str, Any],
    pixtalk_readme: str,
    retouchiq_paper_text: str,
    retouchiq_supplement_text: str,
    instant_readme: str,
    instant_license: str,
    instant_dataset_loader: str,
    instant_paper_text: str,
    instant_supplement_text: str,
    instant_repository_tree: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate frozen source facts without reading pixels or model weights."""

    if config.get("schema") != "kmcfm.p225-latest-paired-retouch-source-audit.v1":
        raise LatestPairedRetouchSourceAuditError("unexpected P225 config schema")

    _require(pixtalk_readme, "[ ] Dataset request form", "PixTalk dataset unavailable")
    _require(pixtalk_readme, "patented worldwide", "PixTalk patent statement")
    _require(
        pixtalk_readme,
        "commercial applications, please contact us",
        "PixTalk commercial contact requirement",
    )

    _require(retouchiq_supplement_text, "user editing histories", "RetouchIQ histories")
    _require(retouchiq_supplement_text, "190k examples", "RetouchIQ example count")
    _require(
        retouchiq_supplement_text,
        "before-editing image",
        "RetouchIQ paired observation",
    )
    retouchiq_combined = retouchiq_paper_text + "\n" + retouchiq_supplement_text

    _require(instant_readme, "Apache-2.0", "InstantRetouch code license")
    _require(instant_license, "Apache License", "InstantRetouch license blob")
    _require(instant_dataset_loader, 'entry["input"]', "InstantRetouch input loader")
    _require(instant_dataset_loader, 'entry["output"]', "InstantRetouch output loader")
    _require(
        instant_paper_text,
        "500 real-",
        "InstantRetouch benchmark size",
    )
    _require(
        instant_paper_text,
        "Lightroom community",
        "InstantRetouch benchmark provenance",
    )
    _require(
        instant_supplement_text,
        "curated from the Adobe Lightroom community",
        "InstantRetouch benchmark provenance supplement",
    )

    instant_data_payloads = tuple(
        path
        for path in instant_repository_tree
        if path.casefold().endswith((".json", ".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        and not path.startswith(("configs/", "latex/"))
    )
    if instant_data_payloads:
        raise LatestPairedRetouchSourceAuditError(
            "unexpected InstantRetouch dataset payloads appeared in frozen repository"
        )

    rows = [
        _gate_row(
            candidate_id="pixtalk_iccv2025",
            publication_year=2025,
            paired_public=False,
            commercial_data_rights=False,
            grouping_facts=False,
            new_observation=True,
            source_facts={
                "dataset_release_state": "repository checklist still marks dataset request form incomplete",
                "commercial_state": "patent statement requires contact for commercial applications",
                "pixel_or_recipe_reads": 0,
            },
        ),
        _gate_row(
            candidate_id="retouchiq_cvpr2026",
            publication_year=2026,
            paired_public=False,
            commercial_data_rights=False,
            grouping_facts=False,
            new_observation=True,
            source_facts={
                "described_observation": "Lightroom user editing histories with before image and parameterized operations",
                "described_examples": 190000,
                "official_dataset_locator_count": sum(
                    token.casefold() in retouchiq_combined.casefold()
                    for token in ("huggingface.co/datasets", "dataset download", "download the dataset")
                ),
                "data_license_statement_count": retouchiq_combined.casefold().count(
                    "dataset license"
                ),
                "pixel_or_recipe_reads": 0,
            },
        ),
        _gate_row(
            candidate_id="instantretouch_cvpr2026",
            publication_year=2026,
            paired_public=False,
            commercial_data_rights=False,
            grouping_facts=False,
            new_observation=True,
            source_facts={
                "described_observation": "500 real before-after pairs from the Adobe Lightroom community",
                "repository_license_scope": "Apache-2.0 repository code",
                "repository_dataset_payload_count": len(instant_data_payloads),
                "repository_loader_only": True,
                "benchmark_data_rights_statement_present": "dataset license"
                in (instant_paper_text + instant_supplement_text).casefold(),
                "pixel_or_recipe_reads": 0,
            },
        ),
    ]
    rows.sort(key=lambda row: row["candidate_id"])
    admitted = [row["candidate_id"] for row in rows if row["admitted"]]
    decision = (
        "PASS_SOURCE_READY_FOR_SEPARATE_PIXEL_LOCK"
        if admitted
        else "FAIL_CLOSED_NO_LATEST_RIGHTS_READY_PAIRED_SOURCE"
    )
    scientific = {
        "schema": "kmcfm.p225-latest-paired-retouch-source-audit-result.v1",
        "experiment_id": config["experiment_id"],
        "information_boundary": config["information_boundary"],
        "candidate_count": len(rows),
        "candidates": rows,
        "admitted_candidates": admitted,
        "decision": decision,
        "claim_ceiling": (
            "metadata-only source admission result; no pixels, recipes, model weights, "
            "training, algorithm, package, schema, capability, stock, or product claim"
        ),
    }
    scientific["stable_identity"] = "sha256:" + sha256_bytes(
        canonical_json_bytes(scientific)
    )
    return scientific
