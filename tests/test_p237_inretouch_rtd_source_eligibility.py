from __future__ import annotations

from src.eval.inretouch_source_eligibility import (
    InRetouchEligibilityError,
    analyze_inretouch_source_eligibility,
)


def _config() -> dict[str, object]:
    return {
        "claim_ceiling": "metadata only",
        "decision": {"pass": "PASS_PRIVATE", "fail": "NOT_ELIGIBLE"},
        "experiment_id": "P237",
    }


def _siblings() -> list[dict[str, object]]:
    return [
        {
            "rfilename": f"Benchmark/{split}/{kind}/row.jpg",
            "blobId": "1" * 40,
            "size": 7,
        }
        for split in ("Train", "Test")
        for kind in ("Natural", "Presets")
    ]


def _run(*, gated: object = "auto", license_tag: str = "license:cc-by-nc-sa-4.0"):
    return analyze_inretouch_source_eligibility(
        config=_config(),
        github_repository={
            "default_branch": "main",
            "full_name": "omarAlezaby/InRetouch",
            "license": {"spdx_id": "NOASSERTION"},
        },
        github_head="a" * 40,
        readme_bytes=(
            b"Retouch Transfer Dataset: 569 images and professional presets"
        ),
        license_bytes=b"CC BY-NC-SA 4.0 academic research use only",
        dataset_metadata={
            "gated": gated,
            "id": "omaralezaby/Retouch_Transfer_Dataset",
            "sha": "b" * 40,
            "siblings": _siblings(),
            "tags": [license_tag],
        },
    )


def test_noncommercial_gated_dataset_fails_only_rights_and_gate() -> None:
    report = _run()
    assert report["decision"] == "NOT_ELIGIBLE"
    assert {
        name for name, passed in report["gate_results"].items() if not passed
    } == {
        "commercial_model_and_product_use_permitted",
        "no_contact_sharing_or_additional_terms_acceptance_required",
    }


def test_analysis_is_order_stable() -> None:
    assert _run() == _run()


def test_missing_file_descriptors_fail_closed() -> None:
    try:
        analyze_inretouch_source_eligibility(
            config=_config(),
            github_repository={"full_name": "omarAlezaby/InRetouch"},
            github_head="a" * 40,
            readme_bytes=b"Retouch Transfer Dataset 569 images presets",
            license_bytes=b"license",
            dataset_metadata={
                "id": "omaralezaby/Retouch_Transfer_Dataset", "siblings": []
            },
        )
    except InRetouchEligibilityError as exc:
        assert "file descriptors" in str(exc)
    else:
        raise AssertionError("missing descriptors did not fail closed")
