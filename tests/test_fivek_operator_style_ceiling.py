from __future__ import annotations

import numpy as np

from src.eval.fivek_operator_style_ceiling import diagnose_operator_style_ceiling
from src.roll2film.triangular_logit_transport import TriangularLogitTransport


def test_style_ceiling_diagnoses_an_already_fitted_case_bank() -> None:
    source = np.linspace(0.1, 0.9, 48, dtype=np.float64).reshape(4, 4, 3)
    rows = []
    bank = []
    for index in range(16):
        parameters = np.zeros(14, dtype=np.float64)
        parameters[0] = 0.04 + index * 0.002
        parameters[4] = -0.03 - index * 0.001
        operator = TriangularLogitTransport(parameters)
        pair_id = f"pair-{index:02d}"
        rows.append(
            {
                "pair_id": pair_id,
                "group": f"camera-{index:02d}",
                "source": source,
                "target": operator.apply(source),
            }
        )
        bank.append(
            {"pair_id": pair_id, "parameters": parameters.tolist(), "dose": 1.0}
        )
    result = diagnose_operator_style_ceiling(
        rows=rows,
        oracle_report={"case_bank": bank},
        split_spec={"group_bucket_modulus": 4, "validation_bucket": 0},
        safe_spec={"boundary_epsilon": 1.0 / 65535.0},
        samples_per_image=16,
        gates={
            "minimum_self_fit_median_style_retention": 0.7,
            "minimum_error_oracle_median_style_retention": 0.7,
            "minimum_rows_with_style_eligible_fit_case": 0.9,
            "maximum_style_constrained_oracle_error_ratio": 1.1,
            "maximum_new_boundary_fraction": 0.0005,
        },
    )
    assert result["metrics"]["validation_rows"] > 0
    assert result["metrics"]["self_fit_median_style_retention"] > 0.99
    assert result["metrics"]["maximum_new_boundary_fraction"] == 0.0
    assert result["branch"] in {
        "operator_capacity_adequate",
        "selection_objective_loses_style",
        "style_eligible_case_bank_sparse",
        "style_constrained_oracle_not_viable",
    }
