from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.canoncgt_self_canonical_inverse import (
    fit_inverse_operator,
    uniform_sample_indexes,
)
from src.eval.spcp_global_logit_affine import (
    LogitAffineOperator,
    apply_operator,
)


def test_uniform_sample_indexes_are_exact_and_unique() -> None:
    first = uniform_sample_indexes(100000, 65536)
    second = uniform_sample_indexes(100000, 65536)
    assert np.array_equal(first, second)
    assert first.size == 65536
    assert np.unique(first).size == first.size


def test_inverse_fit_recovers_known_bounded_grade() -> None:
    rng = np.random.default_rng(17)
    canonical = rng.uniform(0.03, 0.97, size=(128, 128, 3)).astype(np.float32)
    truth = LogitAffineOperator(
        matrix=np.asarray(((1.04, 0.02, 0.0), (0.0, 0.98, 0.01), (0.01, 0.0, 1.03))),
        bias=np.asarray((0.03, -0.02, 0.04)),
        dose=1.0,
    )
    reference = apply_operator(canonical, truth)
    fitted, metrics = fit_inverse_operator(
        canonical, reference, maximum_rows=12000, ridge_alpha=1.0e-4
    )
    assert metrics["holdout_error_p95"] < 1.0e-5
    assert np.max(np.abs(fitted.matrix - truth.matrix)) < 1.0e-4
    assert np.max(np.abs(fitted.bias - truth.bias)) < 1.0e-4


def test_preflight_source_has_zero_application_reads() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/eval/canoncgt_self_canonical_inverse.py"
    ).read_text(encoding="utf-8")
    assert "_load_gold_samples" not in source
    assert '"application_source_reads": 0' in source


def test_runner_requires_explicit_output_and_no_drive_literal() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_u5_r2repid7_canoncgt_self_canonical_inverse.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--output", type=Path, required=True)' in source
    assert "D:\\" not in source
    assert "P:\\" not in source
