import ctypes
from pathlib import Path

import numpy as np

from src.eval.native_cloud_full_typed_chain import evaluate
from src.eval.native_msvc import build_msvc_c11_dll
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    sample_density_conditioned_cross_layer_poisson_region,
)
from src.film_physics.native_conditioned_cloud import _CountProfile, _sample_counts

ROOT=Path(__file__).resolve().parents[1]


def test_p4em_native_cloud_full_typed_chain(tmp_path: Path) -> None:
    assert evaluate(ROOT,ROOT/"configs/u6_p4em_native_cloud_full_typed_chain_v1.json",tmp_path)["automatic_pass"] is True


def test_p4em_high_rate_v3_counts_match_python(tmp_path: Path) -> None:
    build=build_msvc_c11_dll(root=ROOT,output_dir=tmp_path,source_relative="native/film_physics/nf_density_conditioned_poisson_u16_v3.c",header_relative="native/film_physics/nf_density_conditioned_poisson_u16_v3.h",basename="nf_count_v3_test")
    library=ctypes.CDLL(build["dll_path"]); function=library.nf_density_conditioned_poisson_u16_sample_region_v3
    function.argtypes=[ctypes.POINTER(_CountProfile),*([ctypes.c_size_t]*6),ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint16),ctypes.c_size_t];function.restype=ctypes.c_int
    profile=CrossLayerPoissonProfile((192.,288.,240.),32.,(32.,16.,24.),(.00125,.001125,.001375),78277)
    scale=np.random.default_rng(17).uniform(0,1,(17,23,3))
    reference=sample_density_conditioned_cross_layer_poisson_region(profile,scale,(17,23),origin_yx=(0,0))
    native_profile=_CountProfile(ctypes.sizeof(_CountProfile),3,(ctypes.c_double*3)(*profile.marginal_rates_cmy),profile.shared_all_rate,(ctypes.c_double*3)(*profile.shared_pair_rates_cm_cy_my),profile.seed,profile.component_seed_stride)
    assert np.array_equal(_sample_counts(library,native_profile,scale,(17,23),0),reference)
