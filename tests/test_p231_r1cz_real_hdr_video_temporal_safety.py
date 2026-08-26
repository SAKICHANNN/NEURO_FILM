from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_p231_r1cz_real_hdr_video_temporal_safety.py"
EVIDENCE = ROOT / "docs/evidence/P231_R1CZ_REAL_HDR_VIDEO_TEMPORAL_SAFETY_RESULT.json"
SPEC = importlib.util.spec_from_file_location("p231_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_pq_eotf_exact_anchors_and_monotonicity() -> None:
    codes = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    values = module.pq_eotf_nits(codes)
    assert values[0] == 0.0
    assert values[-1] == pytest.approx(10000.0, abs=1e-9)
    assert np.all(np.diff(values) > 0.0)


@pytest.mark.parametrize("bad", [-0.001, 1.001, np.nan])
def test_pq_eotf_rejects_invalid_codes(bad: float) -> None:
    with pytest.raises(ValueError):
        module.pq_eotf_nits(np.asarray([bad], dtype=np.float64))


def test_temporal_vectors_are_order_symmetric() -> None:
    rng = np.random.default_rng(231)
    frames = rng.uniform(0.0, 1000.0, size=(5, 3, 4, 3)).astype(np.float32)
    forward = module.temporal_vectors(frames)
    reverse = module.temporal_vectors(frames[::-1])
    for left, right in zip(forward, reverse, strict=True):
        assert np.array_equal(np.sort(left), np.sort(right))


def test_summary_identity_is_safe_and_nonmaterial() -> None:
    rng = np.random.default_rng(232)
    source = rng.uniform(1.0, 9999.0, size=(4, 2, 3, 3)).astype(np.float32)
    summary = module.summarize(source, source.copy())
    assert summary["new_boundary_fraction"] == 0.0
    assert summary["median_log_rgb_material_effect"] == 0.0
    assert summary["luminance_temporal_p95_ratio"] == 1.0
    assert summary["chroma_temporal_p95_ratio"] == 1.0
    assert summary["new_luminance_spike_fraction"] == 0.0


def test_formal_evidence_is_infrastructure_invalid_before_apply() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "INFRASTRUCTURE_INVALID_P231_DECODE_DOMAIN_MISMATCH"
    assert evidence["decode"]["sampled_source_exact"] is False
    assert evidence["decode"]["candidate_apply_calls"] == 0
    assert evidence["decision"]["not_a_scientific_result"] is True
    assert evidence["consumer_mapping"] is False
