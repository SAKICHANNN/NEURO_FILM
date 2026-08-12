"""P4EQ deterministic binary32 negative-log10 conformance."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.eval.native_cloud_row_chain_android_runtime import evaluate as evaluate_android
from src.eval.native_msvc import build_msvc_c11_dll, sha256_file

SOURCE="native/film_physics/nf_deterministic_log10_f32_v1.c"
HEADER="native/film_physics/nf_deterministic_log10_f32_v1.h"


def _llvm(root: Path, clang: Path, output: Path) -> Path:
    output.mkdir(parents=True,exist_ok=True); dll=output/"nf_deterministic_log10_llvm.dll"
    result=subprocess.run([str(clang),"--target=x86_64-w64-windows-gnu","-std=c11","-O2","-Wall","-Wextra","-Werror","-ffp-model=strict","-shared",str(root/SOURCE),"-o",str(dll),"-Wl,--no-insert-timestamp"],capture_output=True,timeout=120,check=False)
    if result.returncode or not dll.is_file():raise RuntimeError((result.stdout+result.stderr).decode(errors="replace"))
    return dll


def _apply(path: Path, values: np.ndarray) -> tuple[np.ndarray,bool]:
    library=ctypes.CDLL(str(path)); pointer=ctypes.POINTER(ctypes.c_float); function=library.nf_deterministic_neg_log10_f32_apply_v1; function.argtypes=[pointer,ctypes.c_size_t,pointer];function.restype=ctypes.c_int
    output=np.full(values.shape,np.float32(-77)); bad=values.copy();bad[-1]=np.nan;bad_output=np.full(values.shape,np.float32(-77))
    try:
        status=function(values.ctypes.data_as(pointer),values.size,output.ctypes.data_as(pointer));bad_status=function(bad.ctypes.data_as(pointer),bad.size,bad_output.ctypes.data_as(pointer))
    finally:_ctypes.FreeLibrary(library._handle)
    if status:return output,False
    return output,bool(bad_status!=0 and np.all(bad_output==-77))


def evaluate(root: Path, contract_path: Path, build_dir: Path, llvm: Path, ndk: Path, sdk: Path, avd: Path, port: int=5586) -> dict:
    c=json.loads(contract_path.read_text());parent=root/c["parent"]["path"]
    if sha256_file(parent)!=c["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=c["parent"]["required_decision"]:raise RuntimeError("P4EQ parent drift")
    f=c["fixture"];rng=np.random.default_rng(f["seed"]);values=rng.uniform(f["minimum_transmittance"],f["maximum_transmittance"],f["random_values"]).astype(np.float32);values=np.concatenate((np.array([np.float32(f["minimum_transmittance"]),np.float32(f["maximum_transmittance"])],dtype=np.float32),values))
    msvc=build_msvc_c11_dll(root=root,output_dir=build_dir/"msvc",source_relative=SOURCE,header_relative=HEADER,basename="nf_deterministic_log10_msvc"); llvm_dll=_llvm(root,llvm,build_dir/"llvm")
    outputs={};atomic={}
    for name,path in (("msvc",Path(msvc["dll_path"])),("llvm",llvm_dll)):outputs[name],atomic[name]=_apply(path,values)
    reference=-np.log10(values.astype(np.float64));maximum=max(float(np.max(np.abs(value.astype(np.float64)-reference))) for value in outputs.values())
    android=evaluate_android(root,contract_path,ndk,llvm,sdk,avd,build_dir/"android-runtime",port)
    gates={"error":maximum<=c["gates"]["maximum_density_absolute_error"],"windows":np.array_equal(outputs["msvc"],outputs["llvm"]),"atomic":all(atomic.values()),"android":android["automatic_pass"]}
    stable={"contract_sha256":sha256_file(contract_path),"source_sha256":sha256_file(root/SOURCE),"input_sha256":hashlib.sha256(values.tobytes()).hexdigest(),"output_sha256":hashlib.sha256(outputs["msvc"].tobytes()).hexdigest(),"maximum_density_absolute_error":maximum,"android_stable_evidence_id":android["stable_evidence_id"],"gates":gates,"decision":c["decision_if_pass"] if all(gates.values()) else c["decision_if_fail"],"claim_ceiling":c["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4eq_deterministic_log10f.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


__all__=["evaluate"]
