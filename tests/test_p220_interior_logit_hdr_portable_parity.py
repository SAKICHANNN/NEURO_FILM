from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.interior_logit_hdr_native_conformance import (
    build_probes,
    load_fixture,
    python_oracle,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/p220_interior_logit_hdr_portable_parity_v1.json"
FIXTURE = ROOT / "tests/fixtures/p220_r1cg_frozen_payloads_v1.json"


def test_p220_contract_binds_exact_r1cg_result_and_zero_target_reads() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == (
        "frozen_before_consumer_native_implementation_or_execution"
    )
    assert config["bindings"]["producer_report_sha256"] == (
        "1622aa050602e876effd4f9c2e84cfdd0adca10268c8e05aa35fcda00b47b7ff"
    )
    assert config["bindings"]["producer_stable_identity"] == (
        "d124e6b3b6fe66b56720ad54a59c7c9bf3be728953f55e620c46c8712297596c"
    )
    assert config["gates"]["maximum_target_or_application_pixel_reads"] == 0


def test_p220_fixture_contains_all_four_frozen_payloads() -> None:
    payloads = load_fixture(FIXTURE)
    assert [row["effect_id"] for row in payloads] == [
        "amber-compress",
        "blue-dense",
        "teal-lift",
        "warm-open",
    ]
    assert all(
        np.asarray(row["payload"]["source_knots"]).shape == (33, 3) for row in payloads
    )


def test_p220_probe_is_exact_bounded_and_contains_boundaries() -> None:
    payloads = load_fixture(FIXTURE)
    first = build_probes(payloads)
    second = build_probes(payloads)
    assert first.shape == (257, 3)
    assert first.dtype == np.float32
    assert first.tobytes() == second.tobytes()
    assert np.min(first) == 0.0
    assert np.max(first) == 10000.0
    assert np.all(np.isfinite(first))


def test_p220_oracle_preserves_boundaries_and_identity() -> None:
    payload = load_fixture(FIXTURE)[0]["payload"]
    probes = build_probes(load_fixture(FIXTURE))
    output = python_oracle(payload, probes)
    boundary = (probes == 0.0) | (probes == 10000.0)
    strict = ~boundary
    np.testing.assert_array_equal(output[boundary], probes[boundary])
    assert np.all(output[strict] > 0.0)
    assert np.all(output[strict] < 10000.0)
    identity_payload = dict(payload)
    identity_payload["identity"] = True
    np.testing.assert_array_equal(python_oracle(identity_payload, probes), probes)
