from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.ultra_hdr_ingress import (
    ULTRAHDR_DECODER_VERSION,
    ULTRAHDR_EXTERNAL_PROFILE_ID,
    prepare_ultrahdr_match_view_v1,
)
from src.color_match.ultrahdr_pq import publish_ultrahdr_match_view_pq_png_v1

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_p89_ultrahdr_absolute_rec2020_pq_png_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p89_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepared():
    source = b"p89-fixture"
    rgba = np.array([[[0.0, 1.0, 2.0, 1.0]]], dtype="<f2")
    return prepare_ultrahdr_match_view_v1(
        source_asset=source,
        expected_source_sha256=__import__("hashlib").sha256(source).hexdigest(),
        decoded_rgba16f=rgba.tobytes(),
        width=1,
        height=1,
        decoder_version=ULTRAHDR_DECODER_VERSION,
        producer_profile_id=ULTRAHDR_EXTERNAL_PROFILE_ID,
    )


def test_p89_contract_is_frozen_and_bindings_exist() -> None:
    config = _module()._load_object(
        ROOT / "configs/p89_ultrahdr_absolute_rec2020_pq_png_v1.json"
    )
    assert config["status"] == "FROZEN_AFTER_P88_BEFORE_P89_PQ_EXECUTION"
    assert config["standards"]["cicp_hex"] == "09100001"
    assert len(config["fixtures"]) == 2
    assert (ROOT / "docs/evidence/P87_ULTRAHDR_ABSOLUTE_REC2020_MATCH_VIEW_RESULT.json").is_file()
    assert (ROOT / "docs/evidence/P88_ULTRAHDR_PINNED_DECODER_CONSUMPTION_RESULT.json").is_file()
    assert (ROOT / "docs/evidence/U1_4G_REC2100_PQ_PNG_RAIL_RESULT.json").is_file()


def test_p89_tracked_evidence_binds_formal_result() -> None:
    evidence = _module()._load_object(
        ROOT / "docs/evidence/P89_ULTRAHDR_ABSOLUTE_REC2020_PQ_PNG_RESULT.json"
    )
    assert evidence["status"] == "PASS_PRIVATE_ULTRAHDR_ABSOLUTE_REC2020_PQ_PNG"
    assert evidence["formal_execution"]["report_sha256"] == (
        "55316c3e9bb554059eb0768c15170787f93b8c75d7062149c7e65bae83bec1c1"
    )
    assert evidence["formal_execution"]["stable_evidence_id"] == (
        "6107d3a7bdc821d0ebb403c213c7288ed16c2c34754fa6308a91e5bb7e75ecd6"
    )
    assert evidence["formal_execution"]["all_gates_pass"]
    assert len(evidence["records"]) == 2


def test_p89_private_bridge_publishes_and_rejects_profile_drift(tmp_path: Path) -> None:
    prepared = _prepared()
    output = tmp_path / "valid.png"
    _, samples = publish_ultrahdr_match_view_pq_png_v1(prepared, output)
    assert output.is_file()
    assert samples.shape == (1, 1, 3)

    wrong_descriptor = replace(prepared.descriptor, profile_id="neuro-film.encoded-srgb-d65.v1")
    wrong_view = replace(prepared.prepared_view, descriptor=wrong_descriptor)
    wrong = replace(prepared, prepared_view=wrong_view)
    rejected = tmp_path / "wrong.png"
    with pytest.raises(ReferenceMatchContractError):
        publish_ultrahdr_match_view_pq_png_v1(wrong, rejected)
    assert not rejected.exists()


def test_p89_scalar_oracle_matches_vectorized_probe() -> None:
    module = _module()
    config = module._load_object(
        ROOT / "configs/p89_ultrahdr_absolute_rec2020_pq_png_v1.json"
    )
    probe = module._probe(config)
    assert probe["endpoint_codes_exact"]
    assert probe["encoded_monotone"]
    assert probe["sample_monotone"]
    assert probe["scalar_oracle_exact"]
