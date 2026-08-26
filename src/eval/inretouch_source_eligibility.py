"""Pure analysis for the P237 InRetouch RTD metadata-only source gate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from typing import Any


class InRetouchEligibilityError(ValueError):
    """Raised when an official-source response violates the frozen audit shape."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contains_all(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return all(needle.lower() in lowered for needle in needles)


def analyze_inretouch_source_eligibility(
    *,
    config: dict[str, Any],
    github_repository: dict[str, Any],
    github_head: str,
    readme_bytes: bytes,
    license_bytes: bytes,
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return a deterministic gate decision from exact public metadata."""

    if len(github_head) != 40:
        raise InRetouchEligibilityError("GitHub HEAD is not a full commit identity")
    if github_repository.get("full_name") != "omarAlezaby/InRetouch":
        raise InRetouchEligibilityError("unexpected GitHub repository identity")
    if dataset_metadata.get("id") != "omaralezaby/Retouch_Transfer_Dataset":
        raise InRetouchEligibilityError("unexpected Hugging Face dataset identity")
    siblings = dataset_metadata.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise InRetouchEligibilityError("dataset file descriptors are absent")

    readme = readme_bytes.decode("utf-8", errors="replace")
    license_text = license_bytes.decode("utf-8", errors="replace")
    paths = [row.get("rfilename") for row in siblings]
    if not all(isinstance(path, str) and path for path in paths):
        raise InRetouchEligibilityError("dataset contains an invalid file descriptor")

    top_levels = Counter(path.split("/", 1)[0] for path in paths)
    benchmark_partitions = {
        prefix: sum(path.startswith(prefix) for path in paths)
        for prefix in (
            "Train/natural/",
            "Train/Presets/",
            "Validation/natural/",
            "Validation/Presets/",
            "Benchmark/Test/natural/",
            "Benchmark/Test/Presets/",
        )
    }
    grouping_explicit = (
        _contains_all(readme, ("retouch transfer dataset", "presets", "569 images"))
        and benchmark_partitions["Train/natural/"] > 0
        and benchmark_partitions["Train/Presets/"] > 0
        and benchmark_partitions["Validation/natural/"] > 0
        and benchmark_partitions["Validation/Presets/"] > 0
        and benchmark_partitions["Benchmark/Test/natural/"] > 0
        and benchmark_partitions["Benchmark/Test/Presets/"] > 0
    )

    descriptor_size_complete = all(
        isinstance(row.get("size"), int) and row["size"] >= 0 for row in siblings
    )
    descriptor_identity_complete = all(
        (
            isinstance(row.get("blobId"), str)
            and len(row["blobId"]) == 40
        )
        or (
            isinstance(row.get("lfs"), dict)
            and isinstance(row["lfs"].get("sha256"), str)
            and len(row["lfs"]["sha256"]) == 64
        )
        for row in siblings
    )
    tags = sorted(
        tag for tag in dataset_metadata.get("tags", []) if isinstance(tag, str)
    )
    noncommercial = (
        "license:cc-by-nc-sa-4.0" in tags
        or "cc by-nc-sa 4.0" in license_text.lower()
        or "academic research use only" in license_text.lower()
    )
    gated = dataset_metadata.get("gated") not in (False, None)

    gate_results = {
        "public_payload_enumerable_without_authentication": len(siblings) > 0,
        "commercial_model_and_product_use_permitted": not noncommercial,
        "scene_and_preset_grouping_explicit": grouping_explicit,
        "authoritative_file_sizes_and_content_hashes_present": (
            descriptor_size_complete and descriptor_identity_complete
        ),
        "no_contact_sharing_or_additional_terms_acceptance_required": not gated,
    }
    report: dict[str, Any] = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": (
            config["decision"]["pass"]
            if all(gate_results.values())
            else config["decision"]["fail"]
        ),
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "information_boundary": {
            "authenticated_requests": 0,
            "dataset_payload_bytes": 0,
            "image_or_preset_bytes": 0,
            "model_or_checkpoint_bytes": 0,
            "pixel_decodes": 0,
            "training_or_inference": 0,
        },
        "official_facts": {
            "benchmark_partition_file_counts": benchmark_partitions,
            "dataset_descriptor_count": len(siblings),
            "dataset_gated": dataset_metadata.get("gated"),
            "dataset_license_tags": [tag for tag in tags if tag.startswith("license:")],
            "dataset_sha": dataset_metadata.get("sha"),
            "descriptor_blob_id_count": sum(
                isinstance(row.get("blobId"), str) for row in siblings
            ),
            "descriptor_lfs_sha256_count": sum(
                isinstance(row.get("lfs"), dict)
                and isinstance(row["lfs"].get("sha256"), str)
                for row in siblings
            ),
            "descriptor_size_count": sum(
                isinstance(row.get("size"), int) for row in siblings
            ),
            "descriptor_total_bytes": sum(
                row.get("size", 0) for row in siblings if isinstance(row.get("size"), int)
            ),
            "github_default_branch": github_repository.get("default_branch"),
            "github_head": github_head,
            "github_license_spdx": (github_repository.get("license") or {}).get(
                "spdx_id"
            ),
            "license_sha256": _sha256(license_bytes),
            "readme_sha256": _sha256(readme_bytes),
            "top_level_file_counts": dict(sorted(top_levels.items())),
        },
        "schema": "neuro-film.p237-inretouch-rtd-source-eligibility-result.v1",
    }
    report["stable_evidence_id"] = _sha256(canonical_json_bytes(report))
    return report
