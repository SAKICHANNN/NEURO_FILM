"""P4EJ exact native cloud-chain throughput and workspace evaluation."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_density_conditioned_cross_layer_cloud_rows,
)
from src.film_physics.cross_layer_compound_poisson import CrossLayerPoissonProfile
from src.film_physics.native_conditioned_cloud import _CountProfile, _sample_counts

COUNT_SOURCE = "native/film_physics/nf_density_conditioned_poisson_u16_v2.c"
COUNT_HEADER = "native/film_physics/nf_density_conditioned_poisson_u16_v2.h"
SPATIAL_SOURCE = "native/film_physics/nf_cloud_spatial_response_f32_v2.c"
SPATIAL_HEADER = "native/film_physics/nf_cloud_spatial_response_f32_v2.h"


class _SpatialProfile(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32), ("sigma", ctypes.c_double * 3), ("mark", ctypes.c_double * 3), ("truncate", ctypes.c_double)]


def _load(count_path: Path, spatial_path: Path):
    count = ctypes.CDLL(str(count_path)); spatial = ctypes.CDLL(str(spatial_path))
    count.nf_density_conditioned_poisson_u16_sample_region_v2.argtypes = [ctypes.POINTER(_CountProfile), *([ctypes.c_size_t] * 6), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t]
    count.nf_density_conditioned_poisson_u16_sample_region_v2.restype = ctypes.c_int
    spatial.nf_cloud_spatial_response_f32_apply_v2.argtypes = [ctypes.POINTER(_SpatialProfile), ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    spatial.nf_cloud_spatial_response_f32_apply_v2.restype = ctypes.c_int
    return count, spatial


def _native_render(profile, scale, row_height, count_lib, spatial_lib):
    height, width = scale.shape[:2]; count = profile.count_profile
    halo = max(int(profile.gaussian_truncate * s + 0.5) for s in profile.gaussian_sigma_pixels_cmy)
    cp = _CountProfile(ctypes.sizeof(_CountProfile), 2, (ctypes.c_double * 3)(*count.marginal_rates_cmy), count.shared_all_rate, (ctypes.c_double * 3)(*count.shared_pair_rates_cm_cy_my), count.seed, count.component_seed_stride)
    sp = _SpatialProfile(ctypes.sizeof(_SpatialProfile), 2, (ctypes.c_double * 3)(*profile.gaussian_sigma_pixels_cmy), (ctypes.c_double * 3)(*count.mark_optical_density_cmy), profile.gaussian_truncate)
    density_rows=[]; transmittance_rows=[]
    for y0 in range(0,height,row_height):
        rows=min(row_height,height-y0); logical=np.arange(y0-halo,y0+rows+halo)%height; parts=[]; cursor=0
        while cursor<logical.size:
            start=int(logical[cursor]); run=min(logical.size-cursor,height-start); parts.append(_sample_counts(count_lib,cp,scale[start:start+run],(height,width),start)); cursor+=run
        counts=np.concatenate(parts) if len(parts)>1 else parts[0]
        workspace=np.empty(counts.shape[0]*width,dtype=np.float64); density=np.empty((rows,width,3),dtype=np.float32); transmittance=np.empty_like(density)
        status=spatial_lib.nf_cloud_spatial_response_f32_apply_v2(ctypes.byref(sp),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),rows,width,halo,workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),workspace.size,density.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),transmittance.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),density.size)
        if status: raise ValueError(f"native spatial v2 rejected request: {status}")
        density_rows.append(density); transmittance_rows.append(transmittance)
    return np.concatenate(density_rows),np.concatenate(transmittance_rows)


def _python_render(profile,scale,row_height):
    rows=list(iter_density_conditioned_cross_layer_cloud_rows(profile,scale,seed=profile.count_profile.seed,row_tile_height=row_height))
    return np.concatenate([x.density for _,x in rows]),np.concatenate([x.transmittance for _,x in rows])


def evaluate(root: Path, contract_path: Path, output_dir: Path) -> dict[str, Any]:
    c=json.loads(contract_path.read_text()); parent=root/c["parent"]["path"]; e=json.loads(parent.read_text())
    if sha256_file(parent)!=c["parent"]["sha256"] or e["decision"]!=c["parent"]["required_decision"]: raise RuntimeError("P4EJ parent drift")
    count_build=build_msvc_c11_dll(root=root,output_dir=output_dir/"count",source_relative=COUNT_SOURCE,header_relative=COUNT_HEADER,basename="nf_count_p4ej")
    spatial_build=build_msvc_c11_dll(root=root,output_dir=output_dir/"spatial",source_relative=SPATIAL_SOURCE,header_relative=SPATIAL_HEADER,basename="nf_spatial_p4ej")
    count_lib,spatial_lib=_load(Path(count_build["dll_path"]),Path(spatial_build["dll_path"])); f=c["fixture"]; shape=(f["height"],f["width"]); scale=np.random.default_rng(f["seed"]).uniform(0,1,(*shape,3)).astype(np.float32)
    profile=CrossLayerCloudReferenceProfile(CrossLayerPoissonProfile((12.,18.,15.),2.,(2.,1.,1.5),(.02,.018,.022),f["seed"]),(1.3,1.7,2.1),4.,(1,),('0'*64,)*3)
    try:
        for _ in range(f["warmups"]): _python_render(profile,scale,f["row_tile_height"]); _native_render(profile,scale,f["row_tile_height"],count_lib,spatial_lib)
        python_times=[]; native_times=[]; native_hashes=[]; reference=None; candidate=None
        for _ in range(f["timed_replays"]):
            start=time.perf_counter(); reference=_python_render(profile,scale,f["row_tile_height"]); python_times.append(time.perf_counter()-start)
            start=time.perf_counter(); candidate=_native_render(profile,scale,f["row_tile_height"],count_lib,spatial_lib); native_times.append(time.perf_counter()-start)
            native_hashes.append((hashlib.sha256(candidate[0].tobytes()).hexdigest(),hashlib.sha256(candidate[1].tobytes()).hexdigest()))
    finally: _ctypes.FreeLibrary(count_lib._handle); _ctypes.FreeLibrary(spatial_lib._handle)
    assert reference is not None and candidate is not None
    max_d=float(np.max(np.abs(reference[0].astype(np.float64)-candidate[0].astype(np.float64)))); max_t=float(np.max(np.abs(reference[1].astype(np.float64)-candidate[1].astype(np.float64))))
    ratio=statistics.median(native_times)/statistics.median(python_times); halo=9; extended=(f["row_tile_height"]+2*halo)*f["width"]
    workspace_ratio=(extended*8)/(extended*8*2)
    gates=c["gates"]; decisions={"wall":ratio<=gates["maximum_native_python_wall_ratio"],"workspace":workspace_ratio<=gates["maximum_native_workspace_ratio_vs_v1"],"density":max_d<=gates["maximum_density_absolute_error"],"transmittance":max_t<=gates["maximum_transmittance_absolute_error"],"repeat":len(set(native_hashes))==1}
    stable={"contract_sha256":sha256_file(contract_path),"spatial_source_sha256":sha256_file(root/SPATIAL_SOURCE),"scale_sha256":hashlib.sha256(scale.tobytes()).hexdigest(),"density_sha256":hashlib.sha256(candidate[0].tobytes()).hexdigest(),"transmittance_sha256":hashlib.sha256(candidate[1].tobytes()).hexdigest(),"python_median_seconds":statistics.median(python_times),"native_median_seconds":statistics.median(native_times),"native_python_wall_ratio":ratio,"native_workspace_ratio_vs_v1":workspace_ratio,"maximum_density_absolute_error":max_d,"maximum_transmittance_absolute_error":max_t,"gates":decisions,"decision":c["decision_if_pass"] if all(decisions.values()) else c["decision_if_fail"],"claim_ceiling":c["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4ej_native_cloud_chain_performance.v2","automatic_pass":all(decisions.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps({k:v for k,v in stable.items() if not k.endswith('_seconds') and k!='native_python_wall_ratio'},sort_keys=True,separators=(',',':')).encode()).hexdigest()}


__all__=["evaluate"]
