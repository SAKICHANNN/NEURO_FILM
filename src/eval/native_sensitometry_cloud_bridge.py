"""P4EY exact double-sensitometry to bounded stochastic-cloud bridge."""

from __future__ import annotations

import _ctypes
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.eval.native_cloud_chain_performance_v2 import _SpatialProfile
from src.eval.native_cloud_row_chain_abi import _chain_library
from src.eval.native_msvc import find_msvc_installation, sha256_file
from src.eval.native_sensitometry_f64 import _profile
from src.film_physics.native_abi_layouts import NativePrintProfileV1
from src.film_physics.native_conditioned_cloud import _CountProfile

SOURCES = tuple("native/film_physics/" + name for name in (
    "nf_sensitometry_cloud_bridge_f32_v1.c",
    "nf_physical_domains_f32_v1.c",
    "nf_conditioned_cloud_row_chain_f32_v1.c",
    "nf_conditioned_cloud_row_chain_f32_v2.c",
    "nf_density_conditioned_poisson_u16_v3.c",
    "nf_cloud_spatial_response_f32_v2.c",
    "nf_cloud_attenuation_f32_v1.c",
    "nf_deterministic_log10_f32_v1.c",
))


def _build(root: Path, output: Path, llvm: Path | None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    dll = output / ("p4ey-llvm.dll" if llvm else "p4ey-msvc.dll")
    sources = [str((root / item).resolve()) for item in SOURCES]
    if llvm:
        command = [str(llvm), "--target=x86_64-w64-windows-gnu", "-std=c11", "-O2",
                   "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", "-shared",
                   *sources, "-o", str(dll), "-Wl,--no-insert-timestamp"]
        result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    else:
        vcvars = find_msvc_installation() / "Common7/Tools/VsDevCmd.bat"
        batch = output / "build.bat"
        quoted = " ".join(f'"{item}"' for item in sources)
        batch.write_text(
            "@echo off\r\n" + f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n' +
            "if errorlevel 1 exit /b %errorlevel%\r\n" +
            f'cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD {quoted} '
            f'/link /Brepro /OUT:"{dll}"\r\n', encoding="ascii", newline="")
        result = subprocess.run(["cmd.exe", "/d", "/c", str(batch)], cwd=output,
                                capture_output=True, check=False, timeout=120)
    if result.returncode or not dll.is_file():
        raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
    return dll


def _configure(path: Path) -> ctypes.CDLL:
    lib = _chain_library(path); size = ctypes.c_size_t
    fp = ctypes.POINTER(ctypes.c_float); dp = ctypes.POINTER(ctypes.c_double)
    lib.nf_physical_sensitometry_f64_apply_v2.argtypes = [
        ctypes.POINTER(NativePrintProfileV1), fp, size, dp]
    lib.nf_physical_sensitometry_f64_apply_v2.restype = ctypes.c_int
    lib.nf_conditioned_cloud_row_chain_f32_apply_window_v3.restype = ctypes.c_int
    lib.nf_sensitometry_cloud_bridge_f32_apply_window_v1.argtypes = [
        ctypes.POINTER(NativePrintProfileV1), ctypes.POINTER(_CountProfile),
        ctypes.POINTER(_SpatialProfile), size, size, size, size, size, fp, size,
        dp, dp, size, fp, size, fp, ctypes.POINTER(ctypes.c_uint16), size,
        dp, size, fp, fp, size, fp, fp, size]
    lib.nf_sensitometry_cloud_bridge_f32_apply_window_v1.restype = ctypes.c_int
    return lib


def _execute(lib: ctypes.CDLL, contract: dict, *, corrupt: bool = False) -> dict:
    f = contract["fixture"]; full, core, width, origin, halo = (
        f[k] for k in ("full_height", "core_height", "width", "origin_y", "halo"))
    first = (origin + full - halo) % full; extended = core + 2 * halo
    logical = np.arange(first, first + extended) % full
    y, x, c = np.meshgrid(logical, np.arange(width), np.arange(3), indexing="ij")
    scene = np.ascontiguousarray(((y * 101 + x * 37 + c * 211 + 11) % 1001) / 1000., np.float32)
    if corrupt: scene.reshape(-1)[-1] = np.nan
    profile = _profile({
        "identity": "neuro-film.p4ey-synthetic-three-knot-density-profile.v1",
        "reference_linear": .18, "black_offset": 1 / 65536,
        "x_knots": [-4.0717306, -1., .7446556],
        "y_knots": [.05, .65, 1.25], "derivatives": [.1953296, .25, .3439074]})
    cp = _CountProfile(ctypes.sizeof(_CountProfile), 3,
        (ctypes.c_double * 3)(192., 288., 240.), 32.,
        (ctypes.c_double * 3)(32., 16., 24.), f["seed"], 1009)
    sp = _SpatialProfile(ctypes.sizeof(_SpatialProfile), 2,
        (ctypes.c_double * 3)(1.3, 1.7, 2.1),
        (ctypes.c_double * 3)(.00125, .001125, .001375), 4.)
    capacity = np.asarray(contract["density_capacity_cmy"], np.float64)
    expected = np.full((core, width, 3), .6, np.float32)
    gain = np.asarray((.3, .35, .25), np.float32)
    size = ctypes.c_size_t; cn=size(); dn=size(); fn=size()
    assert lib.nf_conditioned_cloud_row_chain_f32_workspace_v1(core,width,halo,ctypes.byref(cn),ctypes.byref(dn),ctypes.byref(fn)) == 0
    counts=np.empty(cn.value,np.uint16); conv=np.empty(dn.value,np.float64)
    sd=np.empty(fn.value,np.float32); st=np.empty(fn.value,np.float32)
    scale=np.empty(cn.value,np.float64); manual_scale=np.empty(cn.value,np.float64)
    composed_d=np.full(fn.value,-77,np.float32); composed_t=np.full(fn.value,-77,np.float32)
    manual_d=np.full(fn.value,-77,np.float32); manual_t=np.full(fn.value,-77,np.float32)
    fp=ctypes.POINTER(ctypes.c_float); dp=ctypes.POINTER(ctypes.c_double)
    common=[ctypes.byref(cp),ctypes.byref(sp),full,width,first,core,halo]
    status=lib.nf_sensitometry_cloud_bridge_f32_apply_window_v1(
        ctypes.byref(profile),*common,scene.ctypes.data_as(fp),scene.size,
        capacity.ctypes.data_as(dp),scale.ctypes.data_as(dp),scale.size,
        expected.ctypes.data_as(fp),expected.size,gain.ctypes.data_as(fp),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),counts.size,
        conv.ctypes.data_as(dp),conv.size,sd.ctypes.data_as(fp),st.ctypes.data_as(fp),st.size,
        composed_d.ctypes.data_as(fp),composed_t.ctypes.data_as(fp),composed_t.size)
    if corrupt:
        return {"atomic": status != 0 and np.all(composed_d == -77) and np.all(composed_t == -77)}
    if status: raise RuntimeError(f"P4EY composed bridge failed: {status}")
    if lib.nf_physical_sensitometry_f64_apply_v2(
        ctypes.byref(profile),scene.ctypes.data_as(fp),scene.size//3,
        manual_scale.ctypes.data_as(dp)) != 0: raise RuntimeError("manual sensitometry failed")
    manual_scale.reshape(-1,3)[:] /= capacity
    window=lib.nf_conditioned_cloud_row_chain_f32_apply_window_v3
    manual_status=window(*common,manual_scale.ctypes.data_as(dp),manual_scale.size,
        expected.ctypes.data_as(fp),expected.size,gain.ctypes.data_as(fp),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),counts.size,
        conv.ctypes.data_as(dp),conv.size,sd.ctypes.data_as(fp),st.ctypes.data_as(fp),st.size,
        manual_d.ctypes.data_as(fp),manual_t.ctypes.data_as(fp),manual_t.size)
    if manual_status: raise RuntimeError(f"manual window failed: {manual_status}")
    return {"density":composed_d,"transmittance":composed_t,
            "manual_exact":np.array_equal(composed_d,manual_d) and np.array_equal(composed_t,manual_t),
            "scale_exact":np.array_equal(scale,manual_scale),
            "boundary":int(np.count_nonzero((composed_t<=0)|(composed_t>=1)))}


def evaluate(root: Path, contract_path: Path, output: Path, llvm: Path) -> dict:
    contract=json.loads(contract_path.read_text());parent=root/contract["parent"]["path"]
    if sha256_file(parent)!=contract["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=contract["parent"]["required_decision"]: raise RuntimeError("P4EY parent drift")
    builds={"msvc":_build(root,output/"msvc",None),"llvm":_build(root,output/"llvm",llvm)};rows={}
    for name,path in builds.items():
        lib=_configure(path)
        try:
            first=_execute(lib,contract);repeat=_execute(lib,contract);bad=_execute(lib,contract,corrupt=True)
        finally:_ctypes.FreeLibrary(lib._handle)
        rows[name]={"density_sha256":hashlib.sha256(first["density"].tobytes()).hexdigest(),
            "transmittance_sha256":hashlib.sha256(first["transmittance"].tobytes()).hexdigest(),
            "manual_exact":bool(first["manual_exact"] and first["scale_exact"]),
            "repeat":bool(np.array_equal(first["density"],repeat["density"]) and np.array_equal(first["transmittance"],repeat["transmittance"])),
            "atomic":bool(bad["atomic"]),"boundary":first["boundary"]}
    gates={"manual":all(r["manual_exact"] for r in rows.values()),"compiler":rows["msvc"]["density_sha256"]==rows["llvm"]["density_sha256"] and rows["msvc"]["transmittance_sha256"]==rows["llvm"]["transmittance_sha256"],"repeat":all(r["repeat"] for r in rows.values()),"atomic":all(r["atomic"] for r in rows.values()),"boundary":all(r["boundary"]==0 for r in rows.values())}
    stable={"contract_sha256":sha256_file(contract_path),"source_sha256":sha256_file(root/SOURCES[0]),"header_sha256":sha256_file(root/"native/film_physics/nf_sensitometry_cloud_bridge_f32_v1.h"),"results":rows,"gates":gates,"decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4ey_native_sensitometry_cloud_bridge.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


__all__=["evaluate"]
