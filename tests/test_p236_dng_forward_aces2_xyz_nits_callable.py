from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.audit_p236_dng_forward_aces2_xyz_nits_callable import execute
from src.preprocess.dng_forward_aces2_xyz import render_acescg_to_xyz_d65_nits
from src.preprocess.ocio_aces2_output import OcioAces2RuntimeError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p236_dng_forward_aces2_xyz_nits_callable_v1.json"


def test_p236_forward_reverse_science_is_exact() -> None:
    forward = execute(CONFIG, reverse=False)
    reverse = execute(CONFIG, reverse=True)
    assert forward == reverse
    assert forward["status"] == "FAIL_CLOSED_DNG_FORWARD_ACES2_XYZ_NITS_CALLABLE"
    assert {name for name, passed in forward["gate_results"].items() if not passed} == {
        "neutral_probes"
    }


def test_xyz_bridge_owns_output_and_preserves_input() -> None:
    source = np.ascontiguousarray(
        np.asarray([[0.0, 0.0, 0.0], [0.18, 0.18, 0.18]], dtype=np.float32)
    )
    before = source.copy()
    output = render_acescg_to_xyz_d65_nits(source)
    assert np.array_equal(source, before)
    assert not np.shares_memory(source, output)
    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.isfinite(output).all()


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray([], dtype=np.float32),
        np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    ],
)
def test_xyz_bridge_rejects_invalid_inputs(value: np.ndarray) -> None:
    with pytest.raises(OcioAces2RuntimeError):
        render_acescg_to_xyz_d65_nits(value)
