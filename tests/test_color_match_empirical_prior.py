from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.research import (
    CFSM_EMPIRICAL_ALGORITHM_ID,
    EMPIRICAL_PRIOR_SCHEMA_ID,
    cfsm_candidate_from_json,
    cfsm_candidate_to_json,
    compute_empirical_prior_id,
    fit_cfsm_empirical_candidate,
    load_empirical_neutral_prior,
    validate_cfsm_candidate,
    validate_empirical_prior_payload,
)
from src.preprocess import SourceProfile, WorkingImage


def _payload() -> dict[str, object]:
    records = [
        {
            "id": f"{index:04d}_sample",
            "relative_path": f"outputs/freeze/raw/{index:04d}.tif",
            "sha256": f"{index + 1:064x}",
            "width": 8,
            "height": 8,
            "pixel_count": 64,
            "licence_group": (
                "adobe" if index % 2 == 0 else "adobe_mit"
            ),
        }
        for index in range(4)
    ]
    payload: dict[str, object] = {
        "schema_id": EMPIRICAL_PRIOR_SCHEMA_ID,
        "prior_id": "",
        "dataset": {
            "dataset_id": "fixture-neutral-photos",
            "manifest_relative_path": "outputs/freeze/manifest.csv",
            "manifest_sha256": "1" * 64,
            "summary_relative_path": "outputs/freeze/summary.json",
            "summary_sha256": "2" * 64,
        },
        "rights": {
            "scope": "research-only",
            "product_use_allowed": False,
            "commercial_use_allowed": False,
            "licence_snapshot_sha256": {
                "adobe": "3" * 64,
                "adobe_mit": "4" * 64,
            },
            "file_list_snapshot_sha256": {
                "adobe": "5" * 64,
                "adobe_mit": "6" * 64,
            },
        },
        "ingest": {
            "selected_manifest_column": "raw_default_srgb16",
            "encoded_state": "display-srgb16",
            "working_state": "display-linear-linear-srgb",
            "aggregation": "equal-image-pixel-mixture-v1",
            "expert_or_target_pixels_read": False,
        },
        "statistics": {
            "mean_linear_rgb": [0.31, 0.29, 0.27],
            "population_covariance_linear_rgb": [
                [0.040, 0.012, 0.008],
                [0.012, 0.035, 0.010],
                [0.008, 0.010, 0.030],
            ],
            "source_image_count": 4,
            "source_pixel_count": 256,
        },
        "source_images": records,
        "claim_ceiling": (
            "research-only evidence; no product or commercial claim"
        ),
    }
    payload["prior_id"] = compute_empirical_prior_id(payload)
    return payload


def _working() -> WorkingImage:
    rng = np.random.default_rng(2026072702)
    pixels = np.clip(
        rng.normal(
            loc=np.array([0.36, 0.31, 0.28]),
            scale=np.array([0.18, 0.16, 0.14]),
            size=(32, 32, 3),
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    return WorkingImage(
        pixels=pixels,
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile("fixture", "generated"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("synthetic/empirical-reference.exr"),
    )


def test_empirical_prior_roundtrip_and_identity(tmp_path: Path) -> None:
    payload = _payload()
    prior = validate_empirical_prior_payload(payload)
    assert prior.prior_id == payload["prior_id"]
    assert prior.source_image_count == 4
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_empirical_neutral_prior(path)
    assert loaded.prior_id == prior.prior_id
    assert np.array_equal(loaded.mean, prior.mean)
    assert np.array_equal(loaded.covariance, prior.covariance)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["rights"].__setitem__(
                "commercial_use_allowed", True
            ),
            "research-only rights ceiling",
        ),
        (
            lambda payload: payload["statistics"].__setitem__(
                "source_pixel_count", 255
            ),
            "source counts",
        ),
        (
            lambda payload: payload["source_images"][1].__setitem__(
                "sha256", payload["source_images"][0]["sha256"]
            ),
            "identities must be unique",
        ),
    ],
)
def test_empirical_prior_rejects_tampering(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    payload["prior_id"] = compute_empirical_prior_id(payload)
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_empirical_prior_payload(payload)


def test_empirical_prior_rejects_stale_identity() -> None:
    payload = _payload()
    payload["statistics"]["mean_linear_rgb"][0] = 0.32
    with pytest.raises(ReferenceMatchContractError, match="prior_id"):
        validate_empirical_prior_payload(payload)


def test_empirical_cfsm_candidate_is_deterministic_and_replayable() -> None:
    prior = validate_empirical_prior_payload(_payload())
    kwargs = {
        "prior_mean": prior.mean,
        "prior_covariance": prior.covariance,
        "prior_id": prior.prior_id,
        "source_image_count": prior.source_image_count,
        "source_pixel_count": prior.source_pixel_count,
    }
    first = fit_cfsm_empirical_candidate(_working(), **kwargs)
    second = fit_cfsm_empirical_candidate(_working(), **kwargs)
    assert first.algorithm_id == CFSM_EMPIRICAL_ALGORITHM_ID
    assert first.candidate_id == second.candidate_id
    assert first.diagnostics.canonical_prior_mode.endswith(prior.prior_id)
    replayed = cfsm_candidate_from_json(cfsm_candidate_to_json(first))
    assert replayed.candidate_id == first.candidate_id
    assert np.array_equal(replayed.lut.values, first.lut.values)


def test_empirical_cfsm_candidate_rejects_provenance_tamper() -> None:
    prior = validate_empirical_prior_payload(_payload())
    candidate = fit_cfsm_empirical_candidate(
        _working(),
        prior_mean=prior.mean,
        prior_covariance=prior.covariance,
        prior_id=prior.prior_id,
        source_image_count=prior.source_image_count,
        source_pixel_count=prior.source_pixel_count,
    )
    diagnostics = replace(
        candidate.diagnostics,
        canonical_prior_mode="empirical-neutral-photo-v1:" + "0" * 63,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="empirical canonical-prior diagnostics mismatch",
    ):
        validate_cfsm_candidate(replace(candidate, diagnostics=diagnostics))


@pytest.mark.parametrize(
    "field_value",
    [
        ("prior_id", "g" * 64),
        ("source_image_count", True),
        ("source_pixel_count", 1),
    ],
)
def test_empirical_cfsm_rejects_hostile_metadata(field_value) -> None:
    prior = validate_empirical_prior_payload(_payload())
    kwargs = {
        "prior_mean": prior.mean,
        "prior_covariance": prior.covariance,
        "prior_id": prior.prior_id,
        "source_image_count": prior.source_image_count,
        "source_pixel_count": prior.source_pixel_count,
    }
    kwargs[field_value[0]] = field_value[1]
    with pytest.raises(ReferenceMatchContractError):
        fit_cfsm_empirical_candidate(_working(), **kwargs)
