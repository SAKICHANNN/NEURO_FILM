from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.canoncgt_reference_condition import (
    CanonCGTReferenceError,
    out_of_range_fraction,
    shortlist_candidates,
)


def test_module_does_not_import_external_model_at_import_time() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "eval"
        / "canoncgt_reference_condition.py"
    ).read_text(encoding="utf-8")
    import_line = (
        "from models.networks.end_to_end_finetuning import CanonCGT_E2E"
    )
    assert import_line in source
    assert source.index(import_line) > source.index("def _load_external_model")


def test_out_of_range_fraction_counts_values_not_pixels() -> None:
    values = np.array([[[0.0, -0.1, 1.1], [0.5, 0.75, 1.0]]])
    assert out_of_range_fraction(values) == pytest.approx(2.0 / 6.0)


def test_out_of_range_fraction_fails_nonfinite() -> None:
    with pytest.raises(CanonCGTReferenceError, match="non-finite"):
        out_of_range_fraction(np.array([np.nan]))


def _summary(bucket: str, residual: float, style: float, passed: bool = True):
    return {
        "provenance_bucket": bucket,
        "automatic_survivor": passed,
        "gold_median_non_basic_residual_delta_e76": residual,
        "gold_median_style_delta_e76": style,
    }


def test_shortlist_keeps_one_per_provenance_bucket() -> None:
    summaries = {
        "r1": _summary("a", 5.0, 12.0),
        "r2": _summary("a", 8.0, 8.0),
        "r3": _summary("b", 7.0, 9.0),
        "r4": _summary("c", 6.0, 10.0),
        "r5": _summary("d", 4.0, 20.0),
    }
    config = {"shortlist": {"maximum_candidates": 3}}
    assert shortlist_candidates(
        summaries,
        config,
        bank_sensitive=True,
    ) == ["r2", "r3", "r4"]


def test_shortlist_closes_when_reference_bank_is_not_sensitive() -> None:
    assert (
        shortlist_candidates(
            {"r1": _summary("a", 8.0, 12.0)},
            {"shortlist": {"maximum_candidates": 3}},
            bank_sensitive=False,
        )
        == []
    )


def test_runner_normalizes_all_user_paths() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_u5_r2f1_canoncgt_reference_condition.py"
    ).read_text(encoding="utf-8")
    assert "args.config = _absolute(args.config)" in source
    assert "_absolute(args.render_dir)" in source
    assert "manifest = _absolute(" in source
    assert "output = _absolute(" in source
