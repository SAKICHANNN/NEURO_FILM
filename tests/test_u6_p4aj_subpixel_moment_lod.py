from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from src.eval.physical_subpixel_moment_lod import SubpixelMomentLodError, evaluate_subpixel_moment_lod, load_contract
from src.film_physics.subpixel_moment_structure import compile_subpixel_density_moments, gaussian_subpixel_covariance
ROOT=Path(__file__).resolve().parents[1]; CONTRACT=ROOT/"configs/u6_p4aj_subpixel_moment_lod_v1.json"
def test_covariance_and_constant_moment_identity() -> None:
    covariance=gaussian_subpixel_covariance(0.65,2); assert covariance.shape==(4,4); assert np.allclose(covariance,covariance.T); assert np.allclose(np.diag(covariance),1.0)
    mean=np.full((8,12,3),0.5); variance=np.full_like(mean,0.01); result=compile_subpixel_density_moments(mean,variance,pixel_size_factor=2,correlation_sigma_pixels=0.65); assert np.array_equal(result.mean_density,np.full((4,6,3),0.5)); assert np.all(result.variance_density>0)
def test_contract_mutation_fails_closed(tmp_path: Path) -> None:
    value=json.loads(CONTRACT.read_text(encoding="utf-8")); value["gates"]["maximum_cell_variance_ratio"]=2.0; path=tmp_path/"contract.json"; path.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(SubpixelMomentLodError,match="contract drift"): load_contract(path)
def test_frozen_evaluation_is_repeat_exact() -> None:
    contract=load_contract(CONTRACT); first=evaluate_subpixel_moment_lod(contract,ROOT); second=evaluate_subpixel_moment_lod(contract,ROOT); assert first==second; assert first["passed"] is all(first["checks"].values())
