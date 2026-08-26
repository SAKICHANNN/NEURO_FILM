from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_p233_r1cz_local_payload_capsule.py"
CONFIG = ROOT / "configs/p233_r1cz_local_payload_capsule_v1.json"
CAPSULE = ROOT / "tests/fixtures/p233_r1cz_payload_capsule_v1.json"
SPEC = importlib.util.spec_from_file_location("p233_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _kwargs() -> dict[str, object]:
    return {
        "expected_schema": "kmcfm.r1cz-payload-capsule.v1",
        "expected_encoding": "hex-lower-v1",
        "expected_payload_bytes": 4376,
        "expected_payload_sha256": "aa7fe0400e7bef106a9ff75305babab317a4148bdb05c573c50395d6f4228cb9",
        "expected_bundle_id": "sha256:2759d2193861b95d39bb451efb2477c6ae796f5f58b3b81cf6487cbd85c3220f",
    }


def test_capsule_decodes_exact_payload() -> None:
    payload, capsule = module.decode_capsule(CAPSULE.read_bytes(), **_kwargs())
    assert len(payload) == 4376
    assert capsule["encoding"] == "hex-lower-v1"


@pytest.mark.parametrize("suffix", [b" ", b"\r\n"])
def test_noncanonical_capsule_rejects(suffix: bytes) -> None:
    with pytest.raises(module.R1CZPayloadCapsuleError, match="canonical"):
        module.decode_capsule(CAPSULE.read_bytes().rstrip(b"\n") + suffix, **_kwargs())


def test_local_only_replay_matches_p230() -> None:
    result = module.run(CONFIG)
    assert result["status"] == "PASS_PRIVATE_R1CZ_LOCAL_PAYLOAD_CAPSULE"
    assert result["application"]["output_sha256"] == (
        "98a1281ef743dae99f762e7175e808b1d2da4cf4e671cdf5fad7ecd2d8036409"
    )
    assert result["execution"]["producer_repository_reads"] == 0
    assert result["execution"]["paired_build_reads"] == 0
    assert result["decision"]["consumer_mapping"] is False


def test_forward_reverse_control_order_is_exact() -> None:
    assert module.run(CONFIG) == module.run(CONFIG, reverse=True)
