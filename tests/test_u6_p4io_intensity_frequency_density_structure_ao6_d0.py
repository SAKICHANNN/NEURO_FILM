from pathlib import Path

from src.eval.intensity_frequency_density_structure_ao6_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4io_intensity_frequency_density_structure_ao6_d0_v1.json"


def test_p4io_exact_replay(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    first = evaluate(contract, ROOT, contact_path=tmp_path / "a.png")
    second = evaluate(contract, ROOT, contact_path=tmp_path / "b.png")
    assert first == second
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
