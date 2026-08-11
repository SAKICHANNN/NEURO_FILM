from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.benchmark_u5_r2cb53_analytic_y_chromaticity_24mp import generate_fields
from src.eval.analytic_y_chromaticity_streaming import (
    select_analytic_y_chromaticity_candidate_streamed,
)
from src.eval.analytic_y_chromaticity_transport import (
    select_analytic_y_chromaticity_candidate,
)
from src.eval.characteristic_lstar_transport import CharacteristicLstarTransportError
from src.eval.fujifilm_characteristic_photographic import _compiled_curve

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb53_analytic_y_chromaticity_24mp_resources_v1.json"


def test_cb54_streaming_is_output_and_fact_exact(tmp_path: Path) -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    cb11 = json.loads(
        (ROOT / config["parents"]["cb11_contract_path"]).read_text(encoding="utf-8")
    )
    cb6 = json.loads(
        (ROOT / config["parents"]["cb6_contract_path"]).read_text(encoding="utf-8")
    )
    source, target = generate_fields((96, 128, 3), row_chunk=17)
    common = {
        "curve": _compiled_curve(cb6),
        "strength": float(cb11["operator"]["nominal_strength"]),
        "boundary_epsilon": float(cb11["operator"]["boundary_epsilon"]),
        "dose_grid": list(config["operator"]["dose_grid"]),
        "maximum_gradient_ratio": float(
            config["operator"]["maximum_gradient_ratio"]
        ),
        "maximum_lstar_inversion_fraction": float(
            config["operator"]["maximum_lstar_inversion_fraction"]
        ),
        "lstar_order_epsilon": float(config["operator"]["lstar_order_epsilon"]),
    }
    expected = select_analytic_y_chromaticity_candidate(source, target, **common)
    for row_chunk in (1, 17, 128):
        actual = select_analytic_y_chromaticity_candidate_streamed(
            source,
            target,
            row_chunk=row_chunk,
            scratch_root=tmp_path,
            **common,
        )
        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1], expected[1])
        assert np.array_equal(actual[2], expected[2])
        assert actual[3] == expected[3]
    assert list(tmp_path.iterdir()) == []


def test_cb54_rejects_invalid_chunk_without_scratch(tmp_path: Path) -> None:
    source, target = generate_fields((9, 11, 3), row_chunk=3)
    try:
        select_analytic_y_chromaticity_candidate_streamed(
            source,
            target,
            curve=None,  # type: ignore[arg-type]
            strength=0.2,
            boundary_epsilon=1 / 510,
            dose_grid=[1.0, 0.0],
            maximum_gradient_ratio=1.35,
            maximum_lstar_inversion_fraction=0.0,
            lstar_order_epsilon=1e-4,
            row_chunk=0,
            scratch_root=tmp_path,
        )
    except CharacteristicLstarTransportError:
        pass
    else:
        raise AssertionError("invalid row chunk was accepted")
    assert list(tmp_path.iterdir()) == []
