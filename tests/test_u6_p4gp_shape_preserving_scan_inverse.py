import json
from pathlib import Path

import numpy as np

from src.film_physics.monotone_scan_inverse import ShapePreservingScanInverseV1

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gp_shape_preserving_scan_inverse_v1.json"


def test_p4gp_freezes_distinct_operator_and_fresh_confirmation():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidate = contract["candidate"]
    assert "shape-preserving-cubic" in candidate["operator"]
    assert candidate["scan_domain_endpoints"] == [[0.0, 0.0], [1.0, 1.0]]
    assert candidate["fresh_confirmation_offsets_per_interval"] == [0.25, 0.75]
    assert candidate["fresh_confirmation_levels"] == 64
    assert candidate["chart_pixels_used_for_fit"] is False
    assert candidate["extrapolation_allowed"] is False
    assert candidate["hard_clipping_allowed"] is False


def test_p4gp_shape_preserving_inverse_is_monotone_and_bounded():
    profile = ShapePreservingScanInverseV1(
        ((0.0, 0.2, 0.7, 1.0),) * 3,
        (0.0, 0.25, 0.75, 1.0),
    )
    scan = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    values = profile.apply(np.repeat(scan[:, None], 3, axis=1))
    assert np.all(np.diff(values, axis=0) >= 0.0)
    assert np.min(values) == 0.0
    assert np.max(values) == 1.0
    assert profile.to_payload()["interpolation"] == "fritsch-carlson-pchip"
