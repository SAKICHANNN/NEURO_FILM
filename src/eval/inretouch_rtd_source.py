"""Deterministic public-metadata audit for the gated INRetouch RTD source.

The audit consumes only a previously retained Hugging Face repository API
snapshot plus the local FiveK freeze manifest. It never performs network
access and never opens RTD image payloads.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import re
from typing import Any


_PARTITIONS = (
    "Benchmark/Test_References",
    "Benchmark/Test",
    "Validation",
    "Train",
)
_NATURAL_PATTERN = re.compile(
    r"^(Train|Validation|Benchmark/Test|Benchmark/Test_References)"
    r"/natural/(.+)\.jpg$"
)
_PRESET_PATTERN = re.compile(
    r"^(Train|Validation|Benchmark/Test|Benchmark/Test_References)"
    r"/Presets/(Preset_\d+)/(.+)\.jpg$"
)


def _partition(path: str) -> str | None:
    for value in _PARTITIONS:
        if path.startswith(f"{value}/"):
            return value
    return None


def _finite_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def audit_inretouch_public_topology(
    repository_api: Mapping[str, Any],
    local_source_names: Iterable[str],
) -> dict[str, Any]:
    """Audit RTD path topology and exact local FiveK identity overlap."""

    revision = _finite_string(repository_api.get("sha"), name="repository sha")
    siblings = repository_api.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise ValueError("repository siblings must be a non-empty list")

    paths = []
    for item in siblings:
        if not isinstance(item, Mapping):
            raise ValueError("repository sibling must be an object")
        paths.append(_finite_string(item.get("rfilename"), name="rfilename"))
    if len(paths) != len(set(paths)):
        raise ValueError("repository paths must be unique")

    partition_counts: Counter[str] = Counter()
    natural_by_partition: dict[str, set[str]] = {
        value: set() for value in _PARTITIONS
    }
    presets_by_partition: dict[str, set[str]] = {
        value: set() for value in _PARTITIONS
    }
    preset_contents_by_partition: dict[str, dict[str, set[str]]] = {
        value: {} for value in _PARTITIONS
    }
    preset_file_counts: Counter[str] = Counter()
    for path in paths:
        partition = _partition(path)
        if partition is not None:
            partition_counts[partition] += 1
        natural_match = _NATURAL_PATTERN.fullmatch(path)
        if natural_match is not None:
            natural_by_partition[natural_match.group(1)].add(
                natural_match.group(2)
            )
            continue
        preset_match = _PRESET_PATTERN.fullmatch(path)
        if preset_match is not None:
            preset_partition, preset_id, content_id = preset_match.groups()
            presets_by_partition[preset_partition].add(preset_id)
            preset_contents_by_partition[preset_partition].setdefault(
                preset_id, set()
            ).add(content_id)
            preset_file_counts[preset_partition] += 1

    natural_union = set().union(*natural_by_partition.values())
    preset_union = set().union(*presets_by_partition.values())
    arithmetic = {}
    for partition in _PARTITIONS:
        natural_count = len(natural_by_partition[partition])
        preset_count = len(presets_by_partition[partition])
        expected = natural_count * (1 + preset_count)
        incomplete_or_extra_presets = sorted(
            preset_id
            for preset_id, content_ids in preset_contents_by_partition[
                partition
            ].items()
            if content_ids != natural_by_partition[partition]
        )
        arithmetic[partition] = {
            "natural_count": natural_count,
            "preset_count": preset_count,
            "preset_file_count": preset_file_counts[partition],
            "expected_partition_file_count": expected,
            "actual_partition_file_count": partition_counts[partition],
            "count_exact": expected == partition_counts[partition],
            "preset_content_matrix_exact": not incomplete_or_extra_presets,
            "incomplete_or_extra_preset_count": len(
                incomplete_or_extra_presets
            ),
            "exact": (
                expected == partition_counts[partition]
                and not incomplete_or_extra_presets
            ),
        }

    local = tuple(
        _finite_string(value, name="local source name")
        for value in local_source_names
    )
    if len(local) != len(set(local)):
        raise ValueError("local source names must be unique")
    local_set = set(local)
    overlap_by_partition = {
        partition: len(local_set.intersection(natural_by_partition[partition]))
        for partition in _PARTITIONS
    }
    overlap_union = local_set.intersection(natural_union)

    return {
        "repository_revision": revision,
        "gated": repository_api.get("gated"),
        "listed_files": len(paths),
        "partition_file_counts": {
            key: partition_counts[key] for key in _PARTITIONS
        },
        "natural_ids_by_partition": {
            key: len(natural_by_partition[key]) for key in _PARTITIONS
        },
        "preset_ids_by_partition": {
            key: len(presets_by_partition[key]) for key in _PARTITIONS
        },
        "natural_content_ids": len(natural_union),
        "preset_recipe_ids": len(preset_union),
        "preset_content_matrix_exact": all(
            item["preset_content_matrix_exact"]
            for item in arithmetic.values()
        ),
        "published_split_arithmetic_exact": all(
            item["exact"] for item in arithmetic.values()
        ),
        "partition_arithmetic": arithmetic,
        "local_source_names": len(local),
        "exact_local_overlap": len(overlap_union),
        "local_overlap_by_partition": overlap_by_partition,
        "pixel_payloads_opened": False,
    }


__all__ = ["audit_inretouch_public_topology"]
