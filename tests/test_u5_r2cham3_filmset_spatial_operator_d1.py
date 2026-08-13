from __future__ import annotations

import numpy as np

from src.eval.filmset_spatial_operator_d1 import (
    _apply_cells,
    finalize_report,
    load_contract,
)


class _Offset:
    def __init__(self, value: float):
        self.value = value

    def apply(self, values: np.ndarray) -> np.ndarray:
        return values + self.value


def test_contract_and_cell_routing() -> None:
    contract = load_contract(__import__("pathlib").Path("configs/u5_r2cham3_filmset_spatial_operator_d1_v1.json"))
    assert contract["dataset"]["domain"] == "classneg"
    cells = np.tile(np.arange(16), 2)
    values = np.zeros((2, 32, 3), dtype=np.float64)
    operators = tuple(_Offset(float(index)) for index in range(16))
    output = _apply_cells(operators, values, cells)
    wrong = _apply_cells(operators, values, cells, offset=5)
    assert np.array_equal(output[0, :, 0], cells)
    assert np.array_equal(wrong[0, :, 0], (cells + 5) % 16)


def test_stable_report_excludes_no_results_and_preserves_fail_decision() -> None:
    contract = load_contract(
        __import__("pathlib").Path(
            "configs/u5_r2cham3_filmset_spatial_operator_d1_v1.json"
        )
    )
    result = {
        "rows": [],
        "metrics": {"row_count": 0},
        "structure": {},
        "checks": {"wins": False},
        "automatic_pass": False,
        "decision": contract["decision_if_fail"],
    }
    left = finalize_report(
        contract,
        result,
        parent_ids={"p": "x"},
        dataset={"confirmatory_count": 0},
    )
    right = finalize_report(
        contract,
        result,
        parent_ids={"p": "x"},
        dataset={"confirmatory_count": 0},
    )
    assert left == right
    assert left["automatic_pass"] is False
