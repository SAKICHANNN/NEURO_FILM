from __future__ import annotations

from pathlib import Path
import sys

import pytest

from src.color_match import (
    ContextInvarianceMetrics,
    PhotographicSafetyMetrics,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dpct_invocation_promotion import (  # noqa: E402
    PROTOCOL,
    _atomic_json,
    _context_batch,
    _load_progress,
    _photo_batch,
)


def test_p44_batch_aggregators_preserve_worst_tails() -> None:
    photo = _photo_batch(
        [
            PhotographicSafetyMetrics(
                passed=True,
                reasons=(),
                neutral_chroma_p95=2.0,
                tone_reversal_fraction=0.0,
                tone_largest_reversal_delta_l=0.0,
                tone_plateau_fraction=0.01,
                new_boundary_fraction=0.02,
                semantic_hue_rotation_p95_degrees=3.0,
            ),
            PhotographicSafetyMetrics(
                passed=False,
                reasons=("new-boundary-fraction",),
                neutral_chroma_p95=4.0,
                tone_reversal_fraction=0.1,
                tone_largest_reversal_delta_l=-2.0,
                tone_plateau_fraction=0.2,
                new_boundary_fraction=0.3,
                semantic_hue_rotation_p95_degrees=5.0,
            ),
        ]
    )
    assert (photo.passed_recipe_count, photo.failed_recipe_count) == (1, 1)
    assert photo.maximum_neutral_chroma_p95 == 4.0
    assert photo.largest_tone_reversal_delta_l == -2.0
    assert photo.maximum_new_boundary_fraction == 0.3

    context = _context_batch(
        [
            ContextInvarianceMetrics(True, (), 0.1, 0.2, 0.3),
            ContextInvarianceMetrics(
                False,
                ("shared-colour-median-drift",),
                2.0,
                3.0,
                4.0,
            ),
        ]
    )
    assert (context.passed_recipe_count, context.failed_recipe_count) == (1, 1)
    assert context.maximum_delta_e76_median == 2.0
    assert context.maximum_delta_e76 == 4.0


def test_p44_progress_is_contract_bound_and_resumable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.json"
    empty = _load_progress(path, "a" * 64)
    assert empty == {
        "protocol": PROTOCOL,
        "contract_id": "a" * 64,
        "known_rows": {},
        "photographic_rows": {},
        "context_rows": {},
    }
    empty["known_rows"]["01->02"] = {"metrics": {"passed": False}}
    _atomic_json(path, empty)
    assert _load_progress(path, "a" * 64) == empty
    with pytest.raises(ValueError, match="another contract"):
        _load_progress(path, "b" * 64)


def test_p44_atomic_json_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    payload = {"z": [3, 2, 1], "a": {"finite": 1.25}}
    _atomic_json(path, payload)
    first = path.read_bytes()
    _atomic_json(path, payload)
    assert path.read_bytes() == first
    assert first.endswith(b"\n")
