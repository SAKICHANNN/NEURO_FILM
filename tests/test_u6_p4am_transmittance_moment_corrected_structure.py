from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from src.eval.physical_transmittance_corrected_structure import TransmittanceCorrectedStructureError, load_contract
from src.film_physics.derivative_conditioned_structure import gamma_mean_transmittance_density_offset
ROOT=Path(__file__).resolve().parents[1]; CONTRACT=ROOT/"configs/u6_p4am_transmittance_moment_corrected_structure_v1.json"
def test_contract_mutation_fails_closed(tmp_path: Path) -> None:
    value=json.loads(CONTRACT.read_text(encoding="utf-8")); value["model"]["fitted_parameters"]=1; path=tmp_path/"c.json"; path.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(TransmittanceCorrectedStructureError,match="contract drift"): load_contract(path)
def test_gamma_offset_preserves_analytic_mean_transmittance() -> None:
    mean=np.array([.2,1.,1.8]); variance=np.array([1e-5,1e-4,5e-5]); offset=gamma_mean_transmittance_density_offset(mean,variance); shape=mean**2/variance; scale=variance/mean
    observed=10**(-offset)*(1+np.log(10.)*scale)**(-shape)
    assert np.all(offset>0); assert np.max(np.abs(observed-10**(-mean)))<1e-12
def test_zero_variance_has_zero_offset() -> None:
    assert np.array_equal(gamma_mean_transmittance_density_offset(np.ones(3),np.zeros(3)),np.zeros(3))
