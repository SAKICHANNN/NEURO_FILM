"""P4FE exact float32 Gaussian row-window conformance."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from src.eval.native_cloud_spatial_partition import _build
from src.eval.native_msvc import sha256_file
from src.film_physics.native_abi_layouts import NativeGaussianProfileV1
from src.film_physics.native_standard_runtime import _load_gaussian, _pointer


def _profile(contract:dict)->NativeGaussianProfileV1:
    result=NativeGaussianProfileV1();result.struct_size=ctypes.sizeof(result);result.abi_version=1
    result.source_component_sha256=b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    for index,value in enumerate(contract["fixture"]["sigma_pixels_rgb"]):result.sigma_pixels_rgb[index]=value
    result.truncate=contract["fixture"]["truncate"]
    return result


def _configure(path:Path):
    lib=_load_gaussian(path);fp=ctypes.POINTER(ctypes.c_float);size=ctypes.c_size_t
    lib.nf_gaussian_f32_required_halo_v1.argtypes=[ctypes.POINTER(NativeGaussianProfileV1),ctypes.POINTER(ctypes.c_uint32)]
    lib.nf_gaussian_f32_required_halo_v1.restype=ctypes.c_int
    lib.nf_gaussian_row_window_f32_apply_v1.argtypes=[ctypes.POINTER(NativeGaussianProfileV1),size,size,size,size,fp,size,size,fp,size,fp,size,fp,size]
    lib.nf_gaussian_row_window_f32_apply_v1.restype=ctypes.c_int
    return lib


def _execute(lib,contract:dict)->dict:
    fixture=contract["fixture"];height,width=fixture["height"],fixture["width"];profile=_profile(contract)
    y,x,c=np.meshgrid(np.arange(height),np.arange(width),np.arange(3),indexing="ij")
    source=np.ascontiguousarray(((y*17+x*31+c*101+3)%997)/996.,np.float32)
    workspace=np.empty_like(source);reference=np.empty_like(source)
    if lib.nf_gaussian_f32_apply_v1(ctypes.byref(profile),_pointer(source),height,width,_pointer(workspace),workspace.size,_pointer(reference))!=0:raise RuntimeError("P4FE full Gaussian failed")
    halo=ctypes.c_uint32();assert lib.nf_gaussian_f32_required_halo_v1(ctypes.byref(profile),ctypes.byref(halo))==0
    def partition(rows):
        output=[];maximum=0
        for start,end in rows:
            input_start=max(0,start-halo.value);input_end=min(height,end+halo.value)
            window=np.ascontiguousarray(source[input_start:input_end]);work=np.empty_like(window);rendered=np.empty_like(window);core=np.full((end-start,width,3),-77,np.float32)
            status=lib.nf_gaussian_row_window_f32_apply_v1(ctypes.byref(profile),height,width,input_start,window.shape[0],_pointer(window),start,end-start,
                _pointer(work),work.size,_pointer(rendered),rendered.size,_pointer(core),core.size)
            if status:raise RuntimeError(f"P4FE window failed: {status}")
            output.append(core);maximum=max(maximum,window.nbytes+work.nbytes+rendered.nbytes+core.nbytes)
        return np.concatenate(output),maximum
    primary,primary_bytes=partition(fixture["partitions"]);alternate,alternate_bytes=partition(fixture["alternate_partitions"])
    start,end=fixture["partitions"][1];input_start=start-halo.value+1;input_end=end+halo.value
    window=np.ascontiguousarray(source[input_start:input_end]);work=np.empty_like(window);rendered=np.empty_like(window);core=np.full((end-start,width,3),-77,np.float32)
    bad=lib.nf_gaussian_row_window_f32_apply_v1(ctypes.byref(profile),height,width,input_start,window.shape[0],_pointer(window),start,end-start,
        _pointer(work),work.size,_pointer(rendered),rendered.size,_pointer(core),core.size)
    return {"scan":reference,"primary":bool(np.array_equal(reference,primary)),"alternate":bool(np.array_equal(reference,alternate)),
        "atomic":bool(bad!=0 and np.all(core==-77)),"halo":halo.value,"maximum_window_bytes":max(primary_bytes,alternate_bytes),
        "full_workspace_bytes":source.nbytes+workspace.nbytes+reference.nbytes}


def evaluate(root:Path,contract_path:Path,output:Path,llvm:Path)->dict:
    contract=json.loads(contract_path.read_text());parent=root/contract["parent"]["path"]
    if sha256_file(parent)!=contract["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=contract["parent"]["required_decision"]:raise RuntimeError("P4FE parent drift")
    builds={"msvc":_build(root,output/"msvc",None),"llvm":_build(root,output/"llvm",llvm)};rows={}
    for name,path in builds.items():
        lib=_configure(path)
        try:first=_execute(lib,contract);repeat=_execute(lib,contract)
        finally:_ctypes.FreeLibrary(lib._handle)
        rows[name]={"sha256":hashlib.sha256(first["scan"].tobytes()).hexdigest(),"primary":first["primary"],"alternate":first["alternate"],
            "atomic":first["atomic"],"halo":first["halo"],"maximum_window_bytes":first["maximum_window_bytes"],"full_workspace_bytes":first["full_workspace_bytes"],
            "repeat":bool(np.array_equal(first["scan"],repeat["scan"]))}
    gates={"primary":all(r["primary"] for r in rows.values()),"alternate":all(r["alternate"] for r in rows.values()),"compiler":rows["msvc"]["sha256"]==rows["llvm"]["sha256"],
        "repeat":all(r["repeat"] for r in rows.values()),"atomic":all(r["atomic"] for r in rows.values()),"workspace":all(r["maximum_window_bytes"]<r["full_workspace_bytes"] for r in rows.values())}
    stable={"contract_sha256":sha256_file(contract_path),"source_sha256":sha256_file(root/"native/film_physics/nf_gaussian_row_window_f32_v1.c"),
        "header_sha256":sha256_file(root/"native/film_physics/nf_gaussian_row_window_f32_v1.h"),"results":rows,"gates":gates,
        "decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4fe_gaussian_row_window.v1","automatic_pass":all(gates.values()),"stable":stable,
        "stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


__all__=["evaluate"]
