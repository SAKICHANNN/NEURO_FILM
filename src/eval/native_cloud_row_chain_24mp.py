"""P4ES 24MP throughput and caller-scratch audit for row-chain V2."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from src.eval.native_cloud_chain_performance_v2 import _SpatialProfile
from src.eval.native_msvc import sha256_file
from src.film_physics.native_conditioned_cloud import _CountProfile


def _build(root: Path, clang: Path, output: Path) -> Path:
    output.mkdir(parents=True,exist_ok=True);dll=output/"p4es.dll";names=("nf_conditioned_cloud_row_chain_f32_v1.c","nf_conditioned_cloud_row_chain_f32_v2.c","nf_density_conditioned_poisson_u16_v3.c","nf_cloud_spatial_response_f32_v2.c","nf_cloud_attenuation_f32_v1.c","nf_deterministic_log10_f32_v1.c");sources=[root/"native/film_physics"/name for name in names]
    result=subprocess.run([str(clang),"--target=x86_64-w64-windows-gnu","-std=c11","-O2","-Wall","-Wextra","-Werror","-ffp-model=strict","-shared",*[str(path) for path in sources],"-o",str(dll),"-Wl,--no-insert-timestamp"],capture_output=True,check=False,timeout=120)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors="replace"))
    return dll


def evaluate(root: Path, contract_path: Path, build_dir: Path, clang: Path) -> dict:
    c=json.loads(contract_path.read_text());parent=root/c["parent"]["path"]
    if sha256_file(parent)!=c["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=c["parent"]["required_decision"]:raise RuntimeError("P4ES parent drift")
    f=c["fixture"];height,width,tile,halo=(f[k] for k in ("height","width","row_tile_height","halo"));dll=_build(root,clang,build_dir);lib=ctypes.CDLL(str(dll));size=ctypes.c_size_t;p=ctypes.POINTER(ctypes.c_float);function=lib.nf_conditioned_cloud_row_chain_f32_apply_v2
    from src.eval.native_cloud_row_chain_abi import _chain_library
    configured=_chain_library(dll);function=configured.nf_conditioned_cloud_row_chain_f32_apply_v2;function.argtypes=configured.nf_conditioned_cloud_row_chain_f32_apply_v1.argtypes;function.restype=ctypes.c_int
    scale=np.empty((height,width,3),np.float64)
    for y in range(height):
        x=np.arange(width,dtype=np.uint64);base=(x*37+np.uint64(y)*101+11)%1001;scale[y,:,0]=base/1000.;scale[y,:,1]=((base+211)%1001)/1000.;scale[y,:,2]=((base+419)%1001)/1000.
    cp=_CountProfile(ctypes.sizeof(_CountProfile),3,(ctypes.c_double*3)(192.,288.,240.),32.,(ctypes.c_double*3)(32.,16.,24.),f["seed"],1009);sp=_SpatialProfile(ctypes.sizeof(_SpatialProfile),2,(ctypes.c_double*3)(1.3,1.7,2.1),(ctypes.c_double*3)(.00125,.001125,.001375),4.);gain=np.asarray((.3,.35,.25),np.float32)
    cn=size();dn=size();fn=size();configured.nf_conditioned_cloud_row_chain_f32_workspace_v1(tile,width,halo,ctypes.byref(cn),ctypes.byref(dn),ctypes.byref(fn));counts=np.empty(cn.value,np.uint16);conv=np.empty(dn.value,np.float64);sd=np.empty(fn.value,np.float32);st=np.empty(fn.value,np.float32);expected=np.empty(fn.value,np.float32);density=np.empty(fn.value,np.float32);trans=np.empty(fn.value,np.float32);scratch=counts.nbytes+conv.nbytes+sd.nbytes+st.nbytes
    replays=[]
    try:
        for _ in range(2):
            digest=hashlib.sha256();boundary=0;start=time.perf_counter()
            for y in range(0,height,tile):
                rows=min(tile,height-y);n=rows*width*3;expected[:n]=.2+(np.arange(n,dtype=np.uint32)%601).astype(np.float32)/1000.
                status=function(ctypes.byref(cp),ctypes.byref(sp),height,width,y,rows,halo,scale.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),scale.size,expected.ctypes.data_as(p),n,gain.ctypes.data_as(p),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),counts.size,conv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),conv.size,sd.ctypes.data_as(p),st.ctypes.data_as(p),st.size,density.ctypes.data_as(p),trans.ctypes.data_as(p),n)
                if status:raise RuntimeError(f"P4ES row failed: {status}")
                digest.update(trans[:n].tobytes());boundary+=int(np.count_nonzero((trans[:n]<=0)|(trans[:n]>=1)))
            replays.append({"seconds":time.perf_counter()-start,"sha256":digest.hexdigest(),"boundary_values":boundary})
    finally:_ctypes.FreeLibrary(configured._handle)
    g=c["gates"];gates={"wall":max(r["seconds"] for r in replays)<=g["maximum_wall_seconds"],"scratch":scratch<=g["maximum_caller_scratch_bytes"],"repeat":len({r["sha256"] for r in replays})==1,"boundary":all(r["boundary_values"]==0 for r in replays)}
    stable={"contract_sha256":sha256_file(contract_path),"replays":replays,"caller_scratch_bytes":scratch,"scale_bytes":scale.nbytes,"gates":gates,"decision":c["decision_if_pass"] if all(gates.values()) else c["decision_if_fail"],"claim_ceiling":c["claim_ceiling"]}
    identity={**stable,"replays":[{k:v for k,v in row.items() if k!="seconds"} for row in replays]}
    return {"schema":"neuro_film.u6_p4es_native_cloud_row_chain_24mp.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


__all__=["evaluate"]
