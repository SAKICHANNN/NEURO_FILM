from __future__ import annotations

import pytest

from src.eval.inretouch_rtd_source import (
    audit_inretouch_public_topology,
)


def _repository() -> dict:
    paths = [
        "Train/natural/a.jpg",
        "Train/natural/b.jpg",
        "Train/Presets/Preset_1/a.jpg",
        "Train/Presets/Preset_1/b.jpg",
        "Validation/natural/c.jpg",
        "Validation/Presets/Preset_1/c.jpg",
        "Benchmark/Test/natural/c.jpg",
        "Benchmark/Test/Presets/Preset_2/c.jpg",
        "Benchmark/Test_References/natural/a.jpg",
        "Benchmark/Test_References/natural/b.jpg",
        "Benchmark/Test_References/Presets/Preset_2/a.jpg",
        "Benchmark/Test_References/Presets/Preset_2/b.jpg",
    ]
    return {
        "sha": "fixture-revision",
        "gated": "auto",
        "siblings": [{"rfilename": value} for value in paths],
    }


def test_rtd_topology_audit_reproduces_split_and_overlap() -> None:
    report = audit_inretouch_public_topology(
        _repository(), ["a", "c", "outside"]
    )
    assert report["listed_files"] == 12
    assert report["natural_content_ids"] == 3
    assert report["preset_recipe_ids"] == 2
    assert report["preset_content_matrix_exact"] is True
    assert report["published_split_arithmetic_exact"] is True
    assert report["exact_local_overlap"] == 2
    assert report["local_overlap_by_partition"] == {
        "Benchmark/Test_References": 1,
        "Benchmark/Test": 1,
        "Validation": 1,
        "Train": 1,
    }
    assert report["pixel_payloads_opened"] is False


def test_rtd_topology_rejects_duplicate_paths_or_local_ids() -> None:
    repository = _repository()
    repository["siblings"].append(repository["siblings"][0])
    with pytest.raises(ValueError, match="paths must be unique"):
        audit_inretouch_public_topology(repository, ["a"])
    with pytest.raises(ValueError, match="source names must be unique"):
        audit_inretouch_public_topology(_repository(), ["a", "a"])


def test_rtd_topology_rejects_count_preserving_content_substitution() -> None:
    repository = _repository()
    for item in repository["siblings"]:
        if item["rfilename"] == "Train/Presets/Preset_1/b.jpg":
            item["rfilename"] = "Train/Presets/Preset_1/not-natural.jpg"
            break
    report = audit_inretouch_public_topology(repository, ["a"])
    train = report["partition_arithmetic"]["Train"]
    assert train["count_exact"] is True
    assert train["preset_content_matrix_exact"] is False
    assert train["incomplete_or_extra_preset_count"] == 1
    assert train["exact"] is False
    assert report["preset_content_matrix_exact"] is False
    assert report["published_split_arithmetic_exact"] is False
