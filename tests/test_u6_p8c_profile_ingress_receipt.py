from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
    load_standalone_profile_artifact_bytes,
    render_working_image,
    serialize_standalone_profile_artifact,
)
from src.preprocess.types import SourceProfile, WorkingImage
from src.eval.physical_profile_ingress import validate_contract


ROOT = Path(__file__).resolve().parents[1]
P8B = ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"
P8C = ROOT / "configs/u6_p8c_profile_ingress_receipt_v1.json"
P8C_DECISION = (
    ROOT / "configs/u6_p8c_profile_ingress_receipt_decision_v1.json"
)


def _artifact() -> dict:
    config = json.loads(P8B.read_text(encoding="utf-8"))
    return compile_standalone_profile_artifact(root=ROOT, config=config)


def _working(
    pixels: np.ndarray,
    *,
    working_space: str = "linear_srgb_d65",
    transfer_state: str = "scene_linear",
) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space=working_space,
        transfer_state=transfer_state,
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "synthetic"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("synthetic.raw"),
    )


def test_canonical_artifact_bytes_roundtrip() -> None:
    artifact = _artifact()
    raw = serialize_standalone_profile_artifact(artifact)
    digest = hashlib.sha256(raw).hexdigest()
    assert load_standalone_profile_artifact_bytes(
        raw, expected_sha256=digest
    ) == artifact
    with pytest.raises(ValueError, match="not canonical"):
        load_standalone_profile_artifact_bytes(
            raw + b" ", expected_sha256=hashlib.sha256(raw + b" ").hexdigest()
        )


def test_duplicate_artifact_key_fails_closed() -> None:
    raw = b'{"schema":"x","schema":"y"}'
    with pytest.raises(ValueError, match="duplicate artifact key"):
        load_standalone_profile_artifact_bytes(
            raw, expected_sha256=hashlib.sha256(raw).hexdigest()
        )


@pytest.mark.parametrize("token", [b"1e400", b'"\\ud800"'])
def test_nonfinite_or_surrogate_json_fails_closed(token: bytes) -> None:
    raw = b'{"value":' + token + b"}"
    with pytest.raises(ValueError):
        load_standalone_profile_artifact_bytes(
            raw, expected_sha256=hashlib.sha256(raw).hexdigest()
        )


def test_scene_linear_working_image_render_is_repeat_exact() -> None:
    artifact = _artifact()
    pixels = np.random.default_rng(2026072907).random(
        (33, 35, 3), dtype=np.float32
    )
    working = _working(pixels)
    first, first_receipt = render_working_image(artifact, working)
    second, second_receipt = render_working_image(artifact, working)
    assert np.array_equal(first, second)
    assert first_receipt == second_receipt
    assert first_receipt["output"]["quantized"] is False
    assert first_receipt["input"]["transfer_state"] == "scene_linear"


@pytest.mark.parametrize(
    ("working_space", "transfer_state"),
    [
        ("linear_rec2020", "scene_linear"),
        ("linear_srgb_d65", "display_linear"),
        ("linear_srgb_d65", "unknown"),
    ],
)
def test_ingress_domain_mismatch_fails_closed(
    working_space: str, transfer_state: str
) -> None:
    with pytest.raises(ValueError):
        render_working_image(
            _artifact(),
            _working(
                np.full((3, 4, 3), 0.25, dtype=np.float32),
                working_space=working_space,
                transfer_state=transfer_state,
            ),
        )


def test_unmapped_scene_headroom_fails_closed() -> None:
    pixels = np.full((3, 4, 3), 0.25, dtype=np.float32)
    pixels[1, 2, 0] = 1.01
    with pytest.raises(ValueError, match="scene-to-relative exposure map"):
        render_working_image(_artifact(), _working(pixels))


def test_p8c_contract_binds_strict_scene_linear_ingress() -> None:
    config = json.loads(P8C.read_text(encoding="utf-8"))
    parent, _ = validate_contract(ROOT, config)
    assert parent["node"] == "U6.P8B"
    assert not config["ingress"]["unmapped_headroom_allowed"]
    assert not config["execution"]["final_quantization_allowed"]


def test_p8c_decision_freezes_canonical_receipt_result() -> None:
    decision = json.loads(P8C_DECISION.read_text(encoding="utf-8"))
    assert decision["software_commit"] == "121f924"
    assert decision["two_run_byte_exact"]
    assert decision["artifact_canonical_roundtrip_exact"]
    assert decision["repeat_output_and_receipt_exact"]
    assert not decision["final_quantization_performed"]
    assert not decision["unmapped_scene_headroom_allowed"]
    assert decision["next_leaf"].startswith("U6.P8D")
