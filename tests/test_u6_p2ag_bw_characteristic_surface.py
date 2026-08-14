from pathlib import Path

import numpy as np
import pytest

from src.eval.bw_characteristic_surface import (
    compile_surface,
    load_contract,
    run_audit,
    write_report,
)
from src.film_physics.bw_characteristic_surface import BWCharacteristicSurface

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ag_bw_characteristic_surface_v1.json"


def test_surface_is_strict_and_roundtrips() -> None:
    surface, _ = compile_surface(root=ROOT, contract=load_contract(CONTRACT))
    probe = np.linspace(-3.4, 0.3, 65)
    density = surface.density(9.0, probe)
    assert np.max(np.abs(surface.log_exposure(9.0, density) - probe)) < 1e-12
    assert np.all(np.diff(density) > 0.0)
    assert (
        BWCharacteristicSurface.from_dict(surface.to_dict()).identity()
        == surface.identity()
    )


def test_surface_rejects_domain_mismatch() -> None:
    surface, _ = compile_surface(root=ROOT, contract=load_contract(CONTRACT))
    with pytest.raises(ValueError):
        surface.density(5.9, np.array([-1.0]))
    with pytest.raises(ValueError):
        surface.density(8.0, np.array([-3.5]))
    with pytest.raises(ValueError):
        surface.log_exposure(8.0, np.array([99.0]))


def test_formal_audit_replays_exact(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    assert write_report(first, path_a) == write_report(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()
