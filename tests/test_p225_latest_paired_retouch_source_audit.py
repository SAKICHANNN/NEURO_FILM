from __future__ import annotations

from src.eval.latest_paired_retouch_source_audit import (
    LatestPairedRetouchSourceAuditError,
    analyze_latest_paired_retouch_sources,
)


def _config() -> dict[str, object]:
    return {
        "schema": "kmcfm.p225-latest-paired-retouch-source-audit.v1",
        "experiment_id": "P225_LATEST_PAIRED_RETOUCH_SOURCE_AUDIT_V1",
        "information_boundary": {"pixel_downloads": 0, "training": 0},
    }


def _run(*, pix: str | None = None) -> dict[str, object]:
    return analyze_latest_paired_retouch_sources(
        config=_config(),
        pixtalk_readme=pix
        or "[ ] Dataset request form patented worldwide commercial applications, please contact us",
        retouchiq_paper_text="RetouchIQ paper",
        retouchiq_supplement_text=(
            "Initial Data from user editing histories. Each before-editing image. "
            "After these steps, we obtain a dataset of 190k examples."
        ),
        instant_readme="Apache-2.0 repository",
        instant_license="Apache License Version 2.0",
        instant_dataset_loader='entry["input"] entry["output"]',
        instant_paper_text=(
            "iRetouch has 500 real-\nworld pairs from the Adobe Lightroom community"
        ),
        instant_supplement_text="curated from the Adobe Lightroom community",
        instant_repository_tree=("README.md", "LICENSE", "dataset/dataset_5kreq.py"),
    )


def test_all_latest_sources_fail_rights_ready_admission() -> None:
    report = _run()
    assert report["decision"] == "FAIL_CLOSED_NO_LATEST_RIGHTS_READY_PAIRED_SOURCE"
    assert report["candidate_count"] == 3
    assert report["admitted_candidates"] == []
    assert all(not row["admitted"] for row in report["candidates"])
    assert report["stable_identity"].startswith("sha256:")


def test_result_is_order_stable() -> None:
    assert _run() == _run()


def test_missing_frozen_fact_fails_closed() -> None:
    try:
        _run(pix="patented worldwide commercial applications, please contact us")
    except LatestPairedRetouchSourceAuditError as exc:
        assert "PixTalk dataset unavailable" in str(exc)
    else:
        raise AssertionError("missing source fact did not fail closed")


def test_unexpected_repository_dataset_payload_fails_closed() -> None:
    try:
        analyze_latest_paired_retouch_sources(
            config=_config(),
            pixtalk_readme=(
                "[ ] Dataset request form patented worldwide "
                "commercial applications, please contact us"
            ),
            retouchiq_paper_text="RetouchIQ",
            retouchiq_supplement_text=(
                "user editing histories before-editing image 190k examples"
            ),
            instant_readme="Apache-2.0",
            instant_license="Apache License",
            instant_dataset_loader='entry["input"] entry["output"]',
            instant_paper_text=(
                "500 real-\nworld pairs from the Adobe Lightroom community"
            ),
            instant_supplement_text="curated from the Adobe Lightroom community",
            instant_repository_tree=("dataset/pairs.json",),
        )
    except LatestPairedRetouchSourceAuditError as exc:
        assert "dataset payloads" in str(exc)
    else:
        raise AssertionError("unexpected payload did not fail closed")
