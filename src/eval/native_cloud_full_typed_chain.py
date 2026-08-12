"""P4EM sensitometry to native cloud to scanner typed-chain audit."""

from __future__ import annotations

import _ctypes
import hashlib
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.eval.native_cloud_chain_performance_v2 import (
    SPATIAL_HEADER,
    SPATIAL_SOURCE,
    _load,
    _native_render,
)
from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.eval.physical_scanner_profile import _profile as scanner_profile
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.sensitometry_primitive import build_operator
from src.eval.typed_sensitometry_cloud_chain import _exposure
from src.film_physics.cross_layer_cloud_attenuation_runtime import (
    compile_cloud_attenuation_profile,
    iter_compiled_cloud_attenuation_rows,
)
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import optical_density_capacity_cmy
from src.film_physics.native_cloud_attenuation import (
    apply_native_cloud_attenuation,
    load_native_cloud_attenuation,
)
from src.film_physics.scanner import apply_scanner_profile


def _render(iterator):
    return np.concatenate([result.transmittance for _, result in iterator])


def evaluate(root: Path, contract_path: Path, build_dir: Path) -> dict:
    contract=json.loads(contract_path.read_text()); parent=root/contract["parent"]["path"]; evidence=json.loads(parent.read_text())
    if sha256_file(parent)!=contract["parent"]["sha256"] or evidence["decision"]!=contract["parent"]["required_decision"]: raise RuntimeError("P4EM parent drift")
    f=contract["fixture"]; shape=(f["height"],f["width"]); exposure=_exposure(shape,f["pixel_pitch_um"]); operator=build_operator(json.loads((root/"configs/u2_2a_sensitometry_primitive_v1.json").read_text())); developed=operator.apply(exposure.values)
    reference=CrossLayerCloudReferenceProfile.from_payload(evaluate_capacity(root,root/"configs/u6_p4di_sensitometry_cloud_capacity_v2.json")["compiled_profile"])
    gain=tuple(json.loads((root/"docs/evidence/U6_P4DW_NPS_PRESERVING_CLOUD_RESIDUAL_ATTENUATION_V6_RESULT.json").read_text())["compiled_channel_gain"])
    profile=compile_cloud_attenuation_profile(reference,aperture_factor=4,base_rate_multiplier=16.0,channel_residual_gain=gain); capacity=np.asarray(optical_density_capacity_cmy(profile.base_profile)); scale=developed/capacity
    count_build=build_msvc_c11_dll(root=root,output_dir=build_dir/"count",source_relative="native/film_physics/nf_density_conditioned_poisson_u16_v3.c",header_relative="native/film_physics/nf_density_conditioned_poisson_u16_v3.h",basename="nf_count_p4em"); spatial_build=build_msvc_c11_dll(root=root,output_dir=build_dir/"spatial",source_relative=SPATIAL_SOURCE,header_relative=SPATIAL_HEADER,basename="nf_spatial_p4em"); attenuation_build=build_msvc_c11_dll(root=root,output_dir=build_dir/"attenuation",source_relative="native/film_physics/nf_cloud_attenuation_f32_v1.c",header_relative="native/film_physics/nf_cloud_attenuation_f32_v1.h",basename="nf_attenuation_p4em")
    count_lib,spatial_lib=_load(Path(count_build["dll_path"]),Path(spatial_build["dll_path"]),count_abi=3); attenuation=load_native_cloud_attenuation(Path(attenuation_build["dll_path"])); native_rows={}; native_times=[]
    try:
        start=time.perf_counter(); python_t=_render(iter_compiled_cloud_attenuation_rows(profile,developed,seed=f["seed"],row_tile_height=f["partition_heights"][0])); python_seconds=time.perf_counter()-start
        for height in f["partition_heights"]:
            seeded_base=replace(profile.base_profile,count_profile=replace(profile.base_profile.count_profile,seed=f["seed"])); start=time.perf_counter(); native_density,_=_native_render(seeded_base,scale,height,count_lib,spatial_lib,count_abi=3); base_t=np.power(10.0,-native_density.astype(np.float64)).astype(np.float32); expected=np.power(10.0,-developed).astype(np.float32); _,native_t=apply_native_cloud_attenuation(attenuation,expected,base_t,gain); native_times.append(time.perf_counter()-start); native_rows[str(height)]=native_t
    finally:
        _ctypes.FreeLibrary(count_lib._handle);_ctypes.FreeLibrary(spatial_lib._handle);_ctypes.FreeLibrary(attenuation._handle)
    first=native_rows[str(f["partition_heights"][0])]; scanner=scanner_profile(json.loads((root/"configs/u6_p6a_scanner_profile_boundary_v1.json").read_text())["profiles"]["scanner_a"]); python_final=apply_scanner_profile(python_t.astype(np.float64),scanner,pixel_pitch_um=f["pixel_pitch_um"]); native_final=apply_scanner_profile(first.astype(np.float64),scanner,pixel_pitch_um=f["pixel_pitch_um"]); diff=native_final-python_final; maximum=float(np.max(np.abs(diff))); rmse=float(np.sqrt(np.mean(np.square(diff)))); partition=all(np.array_equal(value,first) for value in native_rows.values()); ratio=statistics.median(native_times)/python_seconds; mean=apply_scanner_profile(np.power(10.,-developed),scanner,pixel_pitch_um=f["pixel_pitch_um"]); stage=float(np.sqrt(np.mean(np.square(python_final-mean)))); boundary=float(np.mean(((native_final<=0)|(native_final>=1))&~((python_final<=0)|(python_final>=1))))
    g=contract["gates"]; gates={"maximum":maximum<=g["maximum_final_scan_absolute_error"],"rmse":rmse<=g["maximum_final_scan_rmse"],"wall":ratio<=g["maximum_native_python_cloud_wall_ratio"],"partition":partition,"stage":stage>0,"boundary":boundary==0}
    stable={"contract_sha256":sha256_file(contract_path),"maximum_final_scan_absolute_error":maximum,"final_scan_rmse":rmse,"native_python_cloud_wall_ratio":ratio,"cloud_stage_rms":stage,"new_boundary_fraction":boundary,"python_final_sha256":hashlib.sha256(python_final.astype('<f8').tobytes()).hexdigest(),"native_final_sha256":hashlib.sha256(native_final.astype('<f8').tobytes()).hexdigest(),"gates":gates,"decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    identity={k:v for k,v in stable.items() if k!="native_python_cloud_wall_ratio"}
    return {"schema":"neuro_film.u6_p4em_native_cloud_full_typed_chain.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


__all__=["evaluate"]
