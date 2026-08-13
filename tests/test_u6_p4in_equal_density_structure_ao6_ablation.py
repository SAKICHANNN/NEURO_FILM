from __future__ import annotations

from pathlib import Path

from src.eval.equal_density_structure_ao6_ablation import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4in_equal_density_structure_ao6_ablation_v1.json"


def test_p4in_exact_replay(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    first = evaluate(contract, ROOT, contact_path=tmp_path / "a.png")
    second = evaluate(contract, ROOT, contact_path=tmp_path / "b.png")
    assert first == second
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    assert first["decision"] in {
        contract["decision_if_pass"],
        contract["decision_if_fail"],
    }
