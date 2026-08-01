from __future__ import annotations

import numpy as np

from src.eval.fivek_hard_lut_compatibility import _compatibility_matrix


def test_compatibility_places_all_style_eligible_cases_first() -> None:
    error = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
    style = np.asarray([[0.2, 0.8, 1.1], [0.9, 0.1, 0.8]])
    score = _compatibility_matrix(error, style, 0.7)
    assert score[0, 1] < score[0, 0]
    assert score[0, 2] < score[0, 0]
    assert score[1, 0] < score[1, 1]
    assert score[1, 2] < score[1, 1]
