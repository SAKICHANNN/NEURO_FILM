import _ctypes
import ctypes
import subprocess
from pathlib import Path

import numpy as np

from src.eval.native_cloud_chain_performance_v2 import _SpatialProfile
from src.eval.native_cloud_row_chain_abi import _chain_library
from src.film_physics.native_conditioned_cloud import _CountProfile

ROOT=Path(__file__).resolve().parents[1]
LLVM=ROOT/"outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def test_p4er_v2_preserves_v1_transmittance_and_replaces_density(tmp_path: Path) -> None:
    # Link V2 explicitly; V1 remains frozen and independently tested.
    dll=tmp_path/"p4er.dll"
    names=("nf_conditioned_cloud_row_chain_f32_v1.c","nf_conditioned_cloud_row_chain_f32_v2.c","nf_density_conditioned_poisson_u16_v3.c","nf_cloud_spatial_response_f32_v2.c","nf_cloud_attenuation_f32_v1.c","nf_deterministic_log10_f32_v1.c")
    sources=[ROOT/"native/film_physics"/name for name in names]
    result=subprocess.run([str(LLVM),"--target=x86_64-w64-windows-gnu","-std=c11","-O2","-Wall","-Wextra","-Werror","-ffp-model=strict","-shared",*[str(path) for path in sources],"-o",str(dll),"-Wl,--no-insert-timestamp"],capture_output=True,check=False)
    assert result.returncode==0,result.stderr.decode(errors="replace")
    library=_chain_library(dll)
    function=library.nf_conditioned_cloud_row_chain_f32_apply_v2; function.argtypes=library.nf_conditioned_cloud_row_chain_f32_apply_v1.argtypes;function.restype=ctypes.c_int
    full,core,width,origin,halo=73,31,47,61,9;scale=np.asarray(np.arange(full*width*3)%1001/1000.,dtype=np.float64).reshape(full,width,3);expected=np.full((core,width,3),.6,np.float32);gain=np.asarray((.3,.35,.25),np.float32)
    cp=_CountProfile(ctypes.sizeof(_CountProfile),3,(ctypes.c_double*3)(192.,288.,240.),32.,(ctypes.c_double*3)(32.,16.,24.),78277,1009);sp=_SpatialProfile(ctypes.sizeof(_SpatialProfile),2,(ctypes.c_double*3)(1.3,1.7,2.1),(ctypes.c_double*3)(.00125,.001125,.001375),4.)
    size=ctypes.c_size_t;cn=size();dn=size();fn=size();library.nf_conditioned_cloud_row_chain_f32_workspace_v1(core,width,halo,ctypes.byref(cn),ctypes.byref(dn),ctypes.byref(fn));counts=np.empty(cn.value,np.uint16);conv=np.empty(dn.value,np.float64);sd=np.empty(fn.value,np.float32);st=np.empty(fn.value,np.float32);d1=np.empty(fn.value,np.float32);t1=np.empty(fn.value,np.float32);d2=np.empty(fn.value,np.float32);t2=np.empty(fn.value,np.float32);p=ctypes.POINTER(ctypes.c_float)
    args=[ctypes.byref(cp),ctypes.byref(sp),full,width,origin,core,halo,scale.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),scale.size,expected.ctypes.data_as(p),expected.size,gain.ctypes.data_as(p),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),counts.size,conv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),conv.size,sd.ctypes.data_as(p),st.ctypes.data_as(p),st.size]
    try:
        assert library.nf_conditioned_cloud_row_chain_f32_apply_v1(*args,d1.ctypes.data_as(p),t1.ctypes.data_as(p),t1.size)==0
        assert function(*args,d2.ctypes.data_as(p),t2.ctypes.data_as(p),t2.size)==0
    finally:_ctypes.FreeLibrary(library._handle)
    assert np.array_equal(t1,t2)
    assert np.max(np.abs(d2.astype(np.float64)+np.log10(t2.astype(np.float64))))<=5e-7
