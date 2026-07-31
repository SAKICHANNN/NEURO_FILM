from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.unpaired_operator_nonidentifiability import (
    UnpairedNonidentifiabilityError,
    apply_twist,
    build_ring_set,
    canonical_json_bytes,
    canonical_sort,
    evaluate,
    load_config,
    run,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk20_unpaired_operator_nonidentifiability_v1.json"


def test_contract_binds_bk19_without_real_data() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["training_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["production_integration_allowed"] is False


def test_parent_mismatch_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["parent"]["required_decision"] = "not-bk19"
    candidate = tmp_path / "config.json"
    candidate.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(UnpairedNonidentifiabilityError, match="parent decision"):
        load_config(ROOT, candidate)


def test_twist_is_invertible_and_set_preserving() -> None:
    config = load_config(ROOT, CONFIG)
    source, shifts = build_ring_set(config)
    count = int(config["witness"]["angles_per_ring"])
    target = apply_twist(source, shifts, angle_count=count)
    restored = apply_twist(target, shifts, angle_count=count, inverse=True)
    assert np.max(np.abs(restored - source)) <= 1e-12
    assert canonical_sort(target).tobytes() == canonical_sort(source).tobytes()
    assert not np.array_equal(target, source)


def test_witness_passes_all_frozen_gates() -> None:
    config = load_config(ROOT, CONFIG)
    report = evaluate(config, config_sha256=sha256_file(CONFIG))
    assert report["automatic_pass"] is True
    assert report["observed_target_sets_byte_exact"] is True
    assert report["true_operator_paired_rgb_rmse"] >= 0.02
    assert report["twist_non_affine_residual_rmse"] >= 0.01


def test_formal_runs_are_byte_exact(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    run(ROOT, CONFIG, first)
    run(ROOT, CONFIG, second)
    assert first.read_bytes() == second.read_bytes()
