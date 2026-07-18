from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from skimage.color import lab2rgb, rgb2lab

from src.eval.filmstylesafe_r1c import FilmStyleSafeR1CError
from src.eval.filmstylesafe_r1c3 import (
    analyze_conditioned_rows,
    conditioned_scis,
    load_r1c3_contract,
)


FIT = {
    "maximum_sample_pixels": 65536,
    "ridge": 0.0001,
    "huber_delta": 0.04,
    "irls_iterations": 4,
}
SCORE = {
    "smooth_grad_threshold": 0.035,
    "residual_thresholds": [4.0, 8.0, 12.0],
    "min_component_pixels": 4,
    "sparse_threshold": 6.0,
    "island_weight": 1000.0,
    "sparse_weight": 200.0,
    "quantile": 0.99,
}


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:96, 0:128]
    reference = np.stack(
        [
            0.25 + 0.35 * xx / 127.0,
            0.28 + 0.28 * yy / 95.0,
            0.35 + 0.15 * (xx + yy) / 222.0,
        ],
        axis=2,
    ).astype(np.float64)
    lab = rgb2lab(reference)
    styled_lab = lab.copy()
    styled_lab[..., 1] = 4.0 + 1.05 * lab[..., 1] + 0.04 * lab[..., 0]
    styled_lab[..., 2] = -3.0 + 0.92 * lab[..., 2] - 0.02 * lab[..., 0]
    styled = np.clip(lab2rgb(styled_lab), 0.0, 1.0).astype(np.float64)
    anomalous = styled.copy()
    anomalous[34:54, 48:76] = [0.95, 0.05, 0.85]
    return reference, styled, anomalous


@pytest.mark.parametrize("degree", [1, 2])
def test_conditioned_scis_preserves_local_anomaly_over_global_style(degree: int) -> None:
    reference, styled, anomalous = _fixture()
    style_score = conditioned_scis(reference, styled, degree=degree, fit=FIT, score=SCORE)
    anomaly_score = conditioned_scis(
        reference, anomalous, degree=degree, fit=FIT, score=SCORE
    )
    key = "conditioned_affine_v1_score" if degree == 1 else "conditioned_quadratic_v1_score"
    assert anomaly_score[key] > style_score[key] + 20.0


def test_conditioned_scis_is_deterministic_and_nonmutating() -> None:
    reference, _, anomalous = _fixture()
    ref_before = reference.copy()
    candidate_before = anomalous.copy()
    first = conditioned_scis(reference, anomalous, degree=1, fit=FIT, score=SCORE)
    second = conditioned_scis(reference, anomalous, degree=1, fit=FIT, score=SCORE)
    assert first == second
    assert np.array_equal(reference, ref_before)
    assert np.array_equal(anomalous, candidate_before)


def test_conditioned_scis_rejects_invalid_capacity() -> None:
    reference, styled, _ = _fixture()
    with pytest.raises(FilmStyleSafeR1CError, match="degree must be 1 or 2"):
        conditioned_scis(reference, styled, degree=3, fit=FIT, score=SCORE)


def test_analysis_selects_simplest_full_pass() -> None:
    rows = []
    for index, score in enumerate((10.0, 11.0, 12.0)):
        rows.append(
            {
                "member_id": f"positive-{index}",
                "role": "synthetic_failure",
                "proxy_severe": True,
                "hard_negative": False,
                "conditioned_affine_v1_score": score,
                "conditioned_quadratic_v1_score": score + 1.0,
            }
        )
    for index, score in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        rows.append(
            {
                "member_id": f"negative-{index}",
                "role": "external_style_control" if index == 3 else "strength_control",
                "proxy_severe": False,
                "hard_negative": index == 4,
                "conditioned_affine_v1_score": score,
                "conditioned_quadratic_v1_score": score + 0.5,
            }
        )
    result = analyze_conditioned_rows(rows)
    assert result["decision"] == "pass"
    assert result["selected_candidate"]["candidate_id"] == "conditioned_affine_v1"


def test_contract_rejects_training_authorization(tmp_path: Path) -> None:
    contract = {
        "contract_id": "kmcfm.filmstylesafe-r1c3.v1",
        "authorization": {
            "model_training": True,
            "hidden_split_population": False,
            "external_human_recruitment": False,
            "paid_resource": False,
            "production_gate": False,
        },
        "decision": {"style_controls_may_be_excluded": False},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(FilmStyleSafeR1CError, match="model_training false"):
        load_r1c3_contract(path)
