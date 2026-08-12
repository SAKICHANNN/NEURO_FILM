"""P4EN single-call native conditioned-cloud row-chain conformance."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.eval.native_cloud_chain_performance_v2 import (
    SPATIAL_HEADER,
    SPATIAL_SOURCE,
    _load,
    _native_render,
    _SpatialProfile,
)
from src.eval.native_cloud_full_typed_chain import _exposure
from src.eval.native_msvc import build_msvc_c11_dll, find_msvc_installation, sha256_file
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.cross_layer_cloud_attenuation_runtime import (
    compile_cloud_attenuation_profile,
)
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import optical_density_capacity_cmy
from src.film_physics.native_cloud_attenuation import (
    apply_native_cloud_attenuation,
    load_native_cloud_attenuation,
)
from src.film_physics.native_conditioned_cloud import _CountProfile

SOURCES = [
    "native/film_physics/nf_conditioned_cloud_row_chain_f32_v1.c",
    "native/film_physics/nf_density_conditioned_poisson_u16_v3.c",
    "native/film_physics/nf_cloud_spatial_response_f32_v2.c",
    "native/film_physics/nf_cloud_attenuation_f32_v1.c",
]
HEADER = "native/film_physics/nf_conditioned_cloud_row_chain_f32_v1.h"


def _build(root: Path, out: Path, llvm: Path | None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    dll = (out / ("nf_cloud_row_chain_llvm.dll" if llvm else "nf_cloud_row_chain_msvc.dll")).resolve()
    sources = [str((root / item).resolve()) for item in SOURCES]
    if llvm:
        command = [str(llvm), "--target=x86_64-w64-windows-gnu", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", "-shared", *sources, "-o", str(dll), "-Wl,--no-insert-timestamp"]
        completed = subprocess.run(command, capture_output=True, timeout=120, check=False)
    else:
        installation = find_msvc_installation()
        vcvars = installation / "Common7/Tools/VsDevCmd.bat"
        batch = out / "build.bat"
        quoted = " ".join(f'"{item}"' for item in sources)
        batch.write_text("@echo off\r\n" f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n' "if errorlevel 1 exit /b %errorlevel%\r\n" f'cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD {quoted} /link /Brepro /OUT:"{dll}"\r\n', encoding="ascii", newline="")
        completed = subprocess.run(["cmd.exe", "/d", "/c", str(batch)], cwd=out, capture_output=True, timeout=120, check=False)
    if completed.returncode or not dll.is_file():
        raise RuntimeError((completed.stdout + completed.stderr).decode(errors="replace"))
    return dll


def _chain_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    size = ctypes.c_size_t
    fptr = ctypes.POINTER(ctypes.c_float)
    library.nf_conditioned_cloud_row_chain_f32_workspace_v1.argtypes = [size, size, size, ctypes.POINTER(size), ctypes.POINTER(size), ctypes.POINTER(size)]
    library.nf_conditioned_cloud_row_chain_f32_workspace_v1.restype = ctypes.c_int
    library.nf_conditioned_cloud_row_chain_f32_apply_v1.argtypes = [ctypes.POINTER(_CountProfile), ctypes.POINTER(_SpatialProfile), size, size, size, size, size, ctypes.POINTER(ctypes.c_double), size, fptr, size, fptr, ctypes.POINTER(ctypes.c_uint16), size, ctypes.POINTER(ctypes.c_double), size, fptr, fptr, size, fptr, fptr, size]
    library.nf_conditioned_cloud_row_chain_f32_apply_v1.restype = ctypes.c_int
    return library


def _apply(library: ctypes.CDLL, cp: _CountProfile, sp: _SpatialProfile, scale: np.ndarray, expected: np.ndarray, gain: np.ndarray, origin: int, rows: int, halo: int, *, corrupt: bool = False):
    width = scale.shape[1]
    counts_n = ctypes.c_size_t(); doubles_n = ctypes.c_size_t(); core_n = ctypes.c_size_t()
    assert library.nf_conditioned_cloud_row_chain_f32_workspace_v1(rows, width, halo, ctypes.byref(counts_n), ctypes.byref(doubles_n), ctypes.byref(core_n)) == 0
    counts = np.empty(counts_n.value, np.uint16); conv = np.empty(doubles_n.value, np.float64)
    sd = np.empty(core_n.value, np.float32); st = np.empty(core_n.value, np.float32)
    density = np.full(core_n.value, np.float32(-77)); output = np.full(core_n.value, np.float32(-77))
    actual_scale = scale.copy()
    if corrupt: actual_scale[origin, 0, 0] = np.nan
    status = library.nf_conditioned_cloud_row_chain_f32_apply_v1(ctypes.byref(cp), ctypes.byref(sp), scale.shape[0], width, origin, rows, halo, actual_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), actual_scale.size, expected.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), expected.size, gain.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), counts.size, conv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), conv.size, sd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), st.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), st.size, density.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size)
    return status, density.reshape(rows, width, 3), output.reshape(rows, width, 3), counts_n.value * 2 + doubles_n.value * 8, core_n.value * 8


def evaluate(root: Path, contract_path: Path, build_dir: Path, llvm: Path) -> dict:
    contract = json.loads(contract_path.read_text()); parent = root / contract["parent"]["path"]
    if sha256_file(parent) != contract["parent"]["sha256"] or json.loads(parent.read_text())["decision"] != contract["parent"]["required_decision"]: raise RuntimeError("P4EN parent drift")
    f = contract["fixture"]; shape = (f["height"], f["width"]); developed = build_operator(json.loads((root / "configs/u2_2a_sensitometry_primitive_v1.json").read_text())).apply(_exposure(shape, f["pixel_pitch_um"]).values)
    reference = CrossLayerCloudReferenceProfile.from_payload(evaluate_capacity(root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json")["compiled_profile"])
    gain_tuple = tuple(json.loads((root / "docs/evidence/U6_P4DW_NPS_PRESERVING_CLOUD_RESIDUAL_ATTENUATION_V6_RESULT.json").read_text())["compiled_channel_gain"])
    profile = compile_cloud_attenuation_profile(reference, aperture_factor=4, base_rate_multiplier=16.0, channel_residual_gain=gain_tuple)
    scale = np.ascontiguousarray(developed / np.asarray(optical_density_capacity_cmy(profile.base_profile)), dtype=np.float64)
    expected = np.ascontiguousarray(np.power(10.0, -developed).astype(np.float32)); gain = np.ascontiguousarray(gain_tuple, np.float32)
    seeded = replace(profile.base_profile, count_profile=replace(profile.base_profile.count_profile, seed=f["seed"])); count = seeded.count_profile
    cp = _CountProfile(ctypes.sizeof(_CountProfile), 3, (ctypes.c_double * 3)(*count.marginal_rates_cmy), count.shared_all_rate, (ctypes.c_double * 3)(*count.shared_pair_rates_cm_cy_my), count.seed, count.component_seed_stride)
    sp = _SpatialProfile(ctypes.sizeof(_SpatialProfile), 2, (ctypes.c_double * 3)(*seeded.gaussian_sigma_pixels_cmy), (ctypes.c_double * 3)(*count.mark_optical_density_cmy), seeded.gaussian_truncate)
    halo = max(int(seeded.gaussian_truncate * value + .5) for value in seeded.gaussian_sigma_pixels_cmy)
    # Existing separately loaded kernels are the exact orchestration reference.
    count_build = build_msvc_c11_dll(root=root, output_dir=build_dir / "reference_count", source_relative="native/film_physics/nf_density_conditioned_poisson_u16_v3.c", header_relative="native/film_physics/nf_density_conditioned_poisson_u16_v3.h", basename="nf_count_p4en_reference")
    spatial_build = build_msvc_c11_dll(root=root, output_dir=build_dir / "reference_spatial", source_relative=SPATIAL_SOURCE, header_relative=SPATIAL_HEADER, basename="nf_spatial_p4en_reference")
    attenuation_build = build_msvc_c11_dll(root=root, output_dir=build_dir / "reference_attenuation", source_relative="native/film_physics/nf_cloud_attenuation_f32_v1.c", header_relative="native/film_physics/nf_cloud_attenuation_f32_v1.h", basename="nf_attenuation_p4en_reference")
    count_lib, spatial_lib = _load(Path(count_build["dll_path"]), Path(spatial_build["dll_path"]), count_abi=3)
    attenuation_lib = load_native_cloud_attenuation(Path(attenuation_build["dll_path"]))
    try:
        reference_density, _ = _native_render(seeded, scale, f["partition_heights"][0], count_lib, spatial_lib, count_abi=3)
        reference_base = np.power(10.0, -reference_density.astype(np.float64)).astype(np.float32)
        _, exact_reference = apply_native_cloud_attenuation(attenuation_lib, expected, reference_base, gain_tuple)
    finally:
        _ctypes.FreeLibrary(count_lib._handle); _ctypes.FreeLibrary(spatial_lib._handle); _ctypes.FreeLibrary(attenuation_lib._handle)
    builds = {"msvc": _build(root, build_dir / "msvc", None), "llvm": _build(root, build_dir / "llvm", llvm)}
    outputs = {}; results = {}
    for name, dll in builds.items():
        library = _chain_library(dll)
        try:
            partitions = {}; first = None; workspace_max = 0; stage_scratch_max = 0
            for tile in f["partition_heights"]:
                parts=[]
                for y in range(0, shape[0], tile):
                    rows=min(tile,shape[0]-y); status,_,value,workspace,stage_scratch=_apply(library,cp,sp,scale,expected[y:y+rows],gain,y,rows,halo); assert status==0; parts.append(value); workspace_max=max(workspace_max,workspace); stage_scratch_max=max(stage_scratch_max,stage_scratch)
                joined=np.concatenate(parts); partitions[str(tile)]=joined; first=joined if first is None else first
            bad_status,bad_density,bad_output,_,_=_apply(library,cp,sp,scale,expected[:f["partition_heights"][0]],gain,0,f["partition_heights"][0],halo,corrupt=True)
        finally: _ctypes.FreeLibrary(library._handle)
        outputs[name]=first
        results[name]={"output_sha256":hashlib.sha256(first.tobytes()).hexdigest(),"reference_exact":bool(np.array_equal(first,exact_reference)),"partition_exact":all(np.array_equal(first,value) for value in partitions.values()),"failure_atomic":bool(bad_status!=0 and np.all(bad_density==-77) and np.all(bad_output==-77)),"maximum_workspace_bytes":workspace_max,"maximum_core_stage_scratch_bytes":stage_scratch_max,"workspace_bytes_per_extended_pixel":workspace_max/max((max(f["partition_heights"])+2*halo)*shape[1],1)}
    gates={"reference":all(v["reference_exact"] for v in results.values()),"partition":all(v["partition_exact"] for v in results.values()),"cross_compiler":np.array_equal(outputs["msvc"],outputs["llvm"]),"atomic":all(v["failure_atomic"] for v in results.values()),"workspace":all(v["workspace_bytes_per_extended_pixel"]<=contract["gates"]["maximum_workspace_bytes_per_extended_pixel"] for v in results.values())}
    stable={"contract_sha256":sha256_file(contract_path),"source_sha256":sha256_file(root/SOURCES[0]),"header_sha256":sha256_file(root/HEADER),"scale_sha256":hashlib.sha256(scale.tobytes()).hexdigest(),"results":results,"gates":gates,"decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4en_native_cloud_row_chain_abi.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


__all__ = ["evaluate"]
