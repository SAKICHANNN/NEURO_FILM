"""P4ET 24MP bounded scale-window evaluation."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.eval.native_cloud_chain_performance_v2 import _SpatialProfile
from src.eval.native_cloud_row_chain_24mp import _build
from src.eval.native_cloud_row_chain_abi import _chain_library
from src.eval.native_msvc import sha256_file
from src.film_physics.native_conditioned_cloud import _CountProfile


def evaluate(root: Path, contract_path: Path, build_dir: Path, clang: Path, *, developed: bool=False) -> dict:
    c=json.loads(contract_path.read_text());parent=root/c["parent"]["path"]
    if sha256_file(parent)!=c["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=c["parent"]["required_decision"]:raise RuntimeError("P4ET parent drift")
    f=c["fixture"];height,width,tile,halo=(f[k] for k in ("height","width","row_tile_height","halo"));dll=_build(root,clang,build_dir);lib=_chain_library(dll);function=getattr(lib,"nf_conditioned_cloud_row_chain_f32_apply_developed_window_v4" if developed else "nf_conditioned_cloud_row_chain_f32_apply_window_v3");function.restype=ctypes.c_int
    size=ctypes.c_size_t;p=ctypes.POINTER(ctypes.c_float);dp=ctypes.POINTER(ctypes.c_double);cn=size();dn=size();fn=size();lib.nf_conditioned_cloud_row_chain_f32_workspace_v1(tile,width,halo,ctypes.byref(cn),ctypes.byref(dn),ctypes.byref(fn));counts=np.empty(cn.value,np.uint16);conv=np.empty(dn.value,np.float64);sd=np.empty(fn.value,np.float32);st=np.empty(fn.value,np.float32);expected=np.empty(fn.value,np.float32);density=np.empty(fn.value,np.float32);trans=np.empty(fn.value,np.float32);scale=np.empty((tile+2*halo,width,3),np.float32 if developed else np.float64);scale_workspace=np.empty(cn.value,np.float64);capacity=np.asarray(f.get("capacity_cmy",[1.,1.,1.]),np.float64)
    cp=_CountProfile(ctypes.sizeof(_CountProfile),3,(ctypes.c_double*3)(192.,288.,240.),32.,(ctypes.c_double*3)(32.,16.,24.),f["seed"],1009);sp=_SpatialProfile(ctypes.sizeof(_SpatialProfile),2,(ctypes.c_double*3)(1.3,1.7,2.1),(ctypes.c_double*3)(.00125,.001125,.001375),4.);gain=np.asarray((.3,.35,.25),np.float32)
    replays=[]
    try:
        for _ in range(2):
            digest=hashlib.sha256();start=time.perf_counter()
            for y in range(0,height,tile):
                rows=min(tile,height-y);extended=rows+2*halo;logical=np.arange(y-halo,y+rows+halo)%height
                x=np.arange(width,dtype=np.uint64)
                for local,gy in enumerate(logical):
                    base=(x*37+np.uint64(gy)*101+11)%1001
                    if developed:
                        scale[local,:,0]=(base.astype(np.float32)/np.float32(1000.0)).astype(np.float32);scale[local,:,1]=(((base+211)%1001).astype(np.float32)/np.float32(1000.0)).astype(np.float32);scale[local,:,2]=(((base+419)%1001).astype(np.float32)/np.float32(1000.0)).astype(np.float32)
                    else:
                        scale[local,:,0]=base/1000.;scale[local,:,1]=((base+211)%1001)/1000.;scale[local,:,2]=((base+419)%1001)/1000.
                n=rows*width*3;expected[:n]=.2+(np.arange(n,dtype=np.uint32)%601).astype(np.float32)/1000.
                first=int(logical[0]);common=[ctypes.byref(cp),ctypes.byref(sp),height,width,first,rows,halo]
                if developed:
                    status=function(*common,scale.ctypes.data_as(p),extended*width*3,capacity.ctypes.data_as(dp),scale_workspace.ctypes.data_as(dp),scale_workspace.size,expected.ctypes.data_as(p),n,gain.ctypes.data_as(p),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),counts.size,conv.ctypes.data_as(dp),conv.size,sd.ctypes.data_as(p),st.ctypes.data_as(p),st.size,density.ctypes.data_as(p),trans.ctypes.data_as(p),n)
                else:
                    status=function(*common,scale.ctypes.data_as(dp),extended*width*3,expected.ctypes.data_as(p),n,gain.ctypes.data_as(p),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),counts.size,conv.ctypes.data_as(dp),conv.size,sd.ctypes.data_as(p),st.ctypes.data_as(p),st.size,density.ctypes.data_as(p),trans.ctypes.data_as(p),n)
                if status:raise RuntimeError(f"P4ET row failed: {status}")
                digest.update(trans[:n].tobytes())
            replays.append({"seconds":time.perf_counter()-start,"sha256":digest.hexdigest()})
    finally:_ctypes.FreeLibrary(lib._handle)
    parent_payload=json.loads(parent.read_text());target=parent_payload["output_sha256"];maximum_parent=max(max(run["seconds"]) for run in parent_payload["formal_runs"])
    scale_bytes=scale.nbytes;g=c["gates"];ratio=max(row["seconds"] for row in replays)/maximum_parent;limit=g.get("maximum_developed_window_bytes",g.get("maximum_scale_window_bytes"));wall=g.get("maximum_wall_ratio_vs_p4et",g.get("maximum_wall_ratio_vs_p4es"));gates={"output":all(row["sha256"]==target for row in replays),"scale":scale_bytes<=limit,"wall":ratio<=wall,"repeat":len({row["sha256"] for row in replays})==1}
    prior_bytes=parent_payload["results"].get("full_scale_bytes",parent_payload["results"].get("scale_window_bytes"));stable={"contract_sha256":sha256_file(contract_path),"replays":replays,"scale_window_bytes":scale_bytes,"prior_scale_bytes":prior_bytes,"maximum_wall_ratio_vs_parent":ratio,"gates":gates,"decision":c["decision_if_pass"] if all(gates.values()) else c["decision_if_fail"],"claim_ceiling":c["claim_ceiling"]};identity={**stable,"replays":[{"sha256":row["sha256"]} for row in replays],"maximum_wall_ratio_vs_parent":None}
    return {"schema":"neuro_film.u6_p4eu_native_cloud_developed_window_v4.v1" if developed else "neuro_film.u6_p4et_native_cloud_scale_window_v3.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


__all__=["evaluate"]
