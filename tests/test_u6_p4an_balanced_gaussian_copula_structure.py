from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from src.eval.physical_balanced_gaussian_copula_structure import BalancedGaussianCopulaError,load_contract
from src.film_physics.structure_compiler import balanced_correlated_normal_region
ROOT=Path(__file__).resolve().parents[1];CONTRACT=ROOT/"configs/u6_p4an_balanced_gaussian_copula_structure_v1.json"
def test_contract_drift_fails_closed(tmp_path:Path)->None:
 v=json.loads(CONTRACT.read_text(encoding="utf-8"));v["model"]["block_shape"]=[4,4];p=tmp_path/"c.json";p.write_text(json.dumps(v),encoding="utf-8")
 with pytest.raises(BalancedGaussianCopulaError,match="contract drift"):load_contract(p)
@pytest.mark.parametrize("shape",[(64,96),(65,97)])
def test_balanced_field_is_partition_exact(shape:tuple[int,int])->None:
 full=balanced_correlated_normal_region(shape,origin_yx=(0,0),shape=shape,sigma=.65,seed=7);parts=[]
 for y in range(0,shape[0],17):parts.append(balanced_correlated_normal_region(shape,origin_yx=(y,0),shape=(min(17,shape[0]-y),shape[1]),sigma=.65,seed=7))
 assert np.array_equal(full,np.concatenate(parts,axis=0));assert np.all(np.isfinite(full))
