from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.analytic_print_inverse import _build_operator
from src.eval.fp16_print_lut_compiler import load_contract, run_audit
from src.film_physics.print_lut_profile import compile_fp16_print_lut

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6v_fp16_print_lut_compiler_v1.json"


def _operator():
    contract = load_contract(CONTRACT)
    p6u_path = ROOT / contract["parents"]["p6u_contract"]["path"]
    import json

    p6u = json.loads(p6u_path.read_text(encoding="utf-8"))
    p6t = json.loads(
        (ROOT / p6u["parents"]["p6t_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return _build_operator(ROOT, p6t)


def test_fp16_profile_is_compact_repeatable_and_bounded() -> None:
    operator = _operator()
    first = compile_fp16_print_lut(operator, 17)
    second = compile_fp16_print_lut(operator, 17)
    assert first.storage_bytes == 17**3 * 3 * 2
    assert first.values_sha256 == second.values_sha256
    assert first.descriptor() == second.descriptor()
    density = np.asarray([[0.89, 0.89, 0.89], [1.04, 1.04, 1.04]])
    output = first.apply(density)
    assert output.dtype == np.float32
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0


def test_fp16_profile_rejects_outside_density() -> None:
    profile = compile_fp16_print_lut(_operator(), 17)
    with pytest.raises(ValueError, match="outside"):
        profile.apply(np.zeros((1, 3)))


def test_formal_compiler_audit_obeys_frozen_selection() -> None:
    contract = load_contract(CONTRACT)
    report = run_audit(root=ROOT, contract=contract)
    assert [row["size"] for row in report["candidates"]] == [17, 33]
    assert report["selected_size"] in (None, 17, 33)
    assert all(
        set(row["gate_results"]) == set(contract["automatic_gates"])
        for row in report["candidates"]
    )
