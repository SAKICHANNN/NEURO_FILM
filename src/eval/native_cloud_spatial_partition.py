"""P4FB exact row partitioning for cloud density plus spatial response."""

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
from src.film_physics.native_abi_layouts import (
    NativeAdjacencyProfileV1,
    NativeGaussianProfileV1,
    NativePrintProfileV1,
)
from src.film_physics.native_conditioned_cloud import _CountProfile

SOURCES = tuple(
    "native/film_physics/" + name
    for name in (
        "nf_cloud_post_spatial_f32_v1.c",
        "nf_sensitometry_cloud_bridge_f32_v1.c",
        "nf_physical_domains_f32_v1.c",
        "nf_gaussian_rgb_f32_v1.c",
        "nf_bounded_adjacency_f32_v1.c",
        "nf_conditioned_cloud_row_chain_f32_v1.c",
        "nf_conditioned_cloud_row_chain_f32_v2.c",
        "nf_density_conditioned_poisson_u16_v3.c",
        "nf_cloud_spatial_response_f32_v2.c",
        "nf_cloud_attenuation_f32_v1.c",
        "nf_deterministic_log10_f32_v1.c",
    )
)
_SHA = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _build(root: Path, output: Path, llvm: Path | None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    dll = output / ("p4fb-llvm.dll" if llvm else "p4fb-msvc.dll")
    sources = [str((root / item).resolve()) for item in SOURCES]
    if llvm:
        result = subprocess.run(
            [str(llvm), "--target=x86_64-w64-windows-gnu", "-std=c11", "-O2",
             "-Wall", "-Wextra", "-Werror", "-ffp-model=strict", "-shared",
             *sources, "-o", str(dll), "-Wl,--no-insert-timestamp"],
            capture_output=True, check=False, timeout=120,
        )
    else:
        vcvars = find_msvc_installation() / "Common7/Tools/VsDevCmd.bat"
        batch = output / "build.bat"
        quoted = " ".join(f'"{item}"' for item in sources)
        batch.write_text(
            "@echo off\r\n"
            + f'call "{vcvars}" -no_logo -arch=x64 -host_arch=x64 >nul\r\n'
            + "if errorlevel 1 exit /b %errorlevel%\r\n"
            + f'cl.exe /nologo /std:c11 /O2 /fp:strict /W4 /WX /LD {quoted} '
              f'/link /Brepro /OUT:"{dll}"\r\n',
            encoding="ascii", newline="",
        )
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", str(batch)], cwd=output,
            capture_output=True, check=False, timeout=120,
        )
    if result.returncode or not dll.is_file():
        raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
    return dll


def _configure(path: Path) -> ctypes.CDLL:
    lib = _chain_library(path)
    size = ctypes.c_size_t
    fp = ctypes.POINTER(ctypes.c_float)
    dp = ctypes.POINTER(ctypes.c_double)
    gp = ctypes.POINTER(NativeGaussianProfileV1)
    lib.nf_sensitometry_cloud_bridge_f32_apply_window_v1.argtypes = [
        ctypes.POINTER(NativePrintProfileV1), ctypes.POINTER(_CountProfile),
        ctypes.POINTER(_SpatialProfile), size, size, size, size, size, fp, size,
        dp, dp, size, fp, size, fp, ctypes.POINTER(ctypes.c_uint16), size,
        dp, size, fp, fp, size, fp, fp, size,
    ]
    lib.nf_sensitometry_cloud_bridge_f32_apply_window_v1.restype = ctypes.c_int
    lib.nf_cloud_post_spatial_f32_required_halo_v1.argtypes = [gp, gp, gp, ctypes.POINTER(ctypes.c_uint32)]
    lib.nf_cloud_post_spatial_f32_required_halo_v1.restype = ctypes.c_int
    lib.nf_cloud_post_spatial_f32_apply_core_v1.argtypes = [
        ctypes.POINTER(NativePrintProfileV1), gp,
        ctypes.POINTER(NativeAdjacencyProfileV1), gp, gp, fp,
        size, size, size, size, size, size, fp, fp, fp, fp, fp, fp, size, fp, size,
    ]
    lib.nf_cloud_post_spatial_f32_apply_core_v1.restype = ctypes.c_int
    return lib


def _gaussian(sigmas: list[float], truncate: float) -> NativeGaussianProfileV1:
    result = NativeGaussianProfileV1()
    result.struct_size = ctypes.sizeof(result); result.abi_version = 1
    result.source_component_sha256 = _SHA
    for index, value in enumerate(sigmas):
        result.sigma_pixels_rgb[index] = value
    result.truncate = truncate
    return result


def _profiles(contract: dict):
    spatial = contract["spatial_profiles"]
    domains = _profile({
        "identity": "neuro-film.p4fb-synthetic-three-knot-density-profile.v1",
        "reference_linear": .18, "black_offset": 1 / 65536,
        "x_knots": [-4.0717306, -1., .7446556],
        "y_knots": [.05, .65, 1.25],
        "derivatives": [.1953296, .25, .3439074],
    })
    adjacency = NativeAdjacencyProfileV1()
    adjacency.struct_size = ctypes.sizeof(adjacency); adjacency.abi_version = 1
    adjacency.source_component_sha256 = _SHA
    adjacency.maximum_absolute_transmittance_delta = spatial["maximum_absolute_transmittance_delta"]
    adjacency.maximum_absolute_density_delta = spatial["maximum_absolute_density_delta"]
    for index, value in enumerate(spatial["development_adjacency_gain_rgb"]):
        adjacency.gain_rgb[index] = value
        adjacency.black_reference_density[index] = .05
        adjacency.white_reference_density[index] = 1.25
    truncate = spatial["gaussian_truncate"]
    return (
        domains,
        _gaussian(spatial["development_adjacency_sigma_pixels_rgb"], truncate),
        adjacency,
        _gaussian(spatial["dye_diffusion_sigma_pixels_rgb"], truncate),
        _gaussian(spatial["scanner_mtf_sigma_pixels_rgb"], truncate),
    )


def _execute(lib: ctypes.CDLL, contract: dict) -> dict:
    fixture = contract["fixture"]
    full, width, cloud_halo = fixture["full_height"], fixture["width"], fixture["cloud_halo"]
    domains, adjacency_blur, adjacency, diffusion, scanner = _profiles(contract)
    cp = _CountProfile(ctypes.sizeof(_CountProfile), 3,
        (ctypes.c_double * 3)(192., 288., 240.), 32.,
        (ctypes.c_double * 3)(32., 16., 24.), fixture["seed"], 1009)
    sp = _SpatialProfile(ctypes.sizeof(_SpatialProfile), 2,
        (ctypes.c_double * 3)(1.3, 1.7, 2.1),
        (ctypes.c_double * 3)(.00125, .001125, .001375), 4.)
    fp = ctypes.POINTER(ctypes.c_float); dp = ctypes.POINTER(ctypes.c_double)
    capacity = np.asarray(contract["density_capacity_cmy"], np.float64)
    gain = np.asarray((.3, .35, .25), np.float32)
    post_halo = ctypes.c_uint32()
    assert lib.nf_cloud_post_spatial_f32_required_halo_v1(
        ctypes.byref(adjacency_blur), ctypes.byref(diffusion), ctypes.byref(scanner),
        ctypes.byref(post_halo)) == 0

    def cloud_rows(start: int, height: int) -> np.ndarray:
        first = (start + full - cloud_halo) % full
        logical = np.arange(first, first + height + 2 * cloud_halo) % full
        y, x, c = np.meshgrid(logical, np.arange(width), np.arange(3), indexing="ij")
        scene = np.ascontiguousarray(((y * 101 + x * 37 + c * 211 + 11) % 1001) / 1000., np.float32)
        size = ctypes.c_size_t; cn=size(); dn=size(); fn=size()
        assert lib.nf_conditioned_cloud_row_chain_f32_workspace_v1(
            height,width,cloud_halo,ctypes.byref(cn),ctypes.byref(dn),ctypes.byref(fn)) == 0
        expected=np.full((height,width,3),.6,np.float32)
        counts=np.empty(cn.value,np.uint16); conv=np.empty(dn.value,np.float64)
        scale=np.empty(cn.value,np.float64); sd=np.empty(fn.value,np.float32); st=np.empty(fn.value,np.float32)
        density=np.empty(fn.value,np.float32); trans=np.empty(fn.value,np.float32)
        status=lib.nf_sensitometry_cloud_bridge_f32_apply_window_v1(
            ctypes.byref(domains),ctypes.byref(cp),ctypes.byref(sp),full,width,first,
            height,cloud_halo,scene.ctypes.data_as(fp),scene.size,
            capacity.ctypes.data_as(dp),scale.ctypes.data_as(dp),scale.size,
            expected.ctypes.data_as(fp),expected.size,gain.ctypes.data_as(fp),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),counts.size,
            conv.ctypes.data_as(dp),conv.size,sd.ctypes.data_as(fp),st.ctypes.data_as(fp),st.size,
            density.ctypes.data_as(fp),trans.ctypes.data_as(fp),density.size)
        if status: raise RuntimeError(f"P4FB cloud failed: {status}")
        return density.reshape(height,width,3)

    def post(cloud: np.ndarray, logical_start: int, offset: int, height: int, *, bad_offset: int | None = None):
        flat=np.ascontiguousarray(cloud.reshape(-1),np.float32); work=[np.empty_like(flat) for _ in range(6)]
        output=np.full(height*width*3,-77,np.float32); use_offset=offset if bad_offset is None else bad_offset
        status=lib.nf_cloud_post_spatial_f32_apply_core_v1(
            ctypes.byref(domains),ctypes.byref(adjacency_blur),ctypes.byref(adjacency),
            ctypes.byref(diffusion),ctypes.byref(scanner),flat.ctypes.data_as(fp),cloud.shape[0],width,
            full,logical_start,use_offset,height,*[item.ctypes.data_as(fp) for item in work],flat.size,
            output.ctypes.data_as(fp),output.size)
        return status,output.reshape(height,width,3)

    full_cloud=cloud_rows(0,full); status,reference=post(full_cloud,0,0,full)
    if status: raise RuntimeError(f"P4FB full post failed: {status}")
    def partition(rows):
        outputs=[]
        for start,end in rows:
            ext_start=max(0,start-post_halo.value);ext_end=min(full,end+post_halo.value)
            cloud=cloud_rows(ext_start,ext_end-ext_start)
            status,output=post(cloud,start,start-ext_start,end-start)
            if status: raise RuntimeError(f"P4FB partition post failed: {status}")
            outputs.append(output)
        return np.concatenate(outputs,axis=0)
    primary=partition(fixture["partitions"]);alternate=partition(fixture["alternate_partitions"])
    middle_start,middle_end=fixture["partitions"][1]
    ext_start=middle_start-post_halo.value;ext_end=middle_end+post_halo.value
    bad_cloud=cloud_rows(ext_start,ext_end-ext_start)
    bad_status,bad_output=post(bad_cloud,middle_start,post_halo.value,middle_end-middle_start,bad_offset=post_halo.value-1)
    return {
        "scan":reference,
        "primary_exact":bool(np.array_equal(reference,primary)),
        "alternate_exact":bool(np.array_equal(reference,alternate)),
        "failure_atomic":bool(bad_status!=0 and np.all(bad_output==-77)),
        "post_halo":post_halo.value,
    }


def render_physical_partition(
    lib: ctypes.CDLL,
    contract: dict,
    scene_full: np.ndarray,
    start: int,
    height: int,
) -> np.ndarray:
    """Run the exact P4EY/P4FB physical provider for one logical core."""

    fixture=contract["fixture"];full,width,cloud_halo=fixture["full_height"],fixture["width"],fixture["cloud_halo"]
    if scene_full.shape!=(full,width,3) or start<0 or height<=0 or start+height>full:
        raise ValueError("P4FB physical provider geometry drift")
    domains,adjacency_blur,adjacency,diffusion,scanner=_profiles(contract)
    cp=_CountProfile(ctypes.sizeof(_CountProfile),3,(ctypes.c_double*3)(192.,288.,240.),32.,
        (ctypes.c_double*3)(32.,16.,24.),fixture["seed"],1009)
    sp=_SpatialProfile(ctypes.sizeof(_SpatialProfile),2,(ctypes.c_double*3)(1.3,1.7,2.1),
        (ctypes.c_double*3)(.00125,.001125,.001375),4.)
    fp=ctypes.POINTER(ctypes.c_float);dp=ctypes.POINTER(ctypes.c_double)
    capacity=np.asarray(contract["density_capacity_cmy"],np.float64);gain=np.asarray((.3,.35,.25),np.float32)
    post_halo=ctypes.c_uint32()
    if lib.nf_cloud_post_spatial_f32_required_halo_v1(ctypes.byref(adjacency_blur),ctypes.byref(diffusion),ctypes.byref(scanner),ctypes.byref(post_halo))!=0:
        raise RuntimeError("P4FB post halo failed")
    ext_start=max(0,start-post_halo.value);ext_end=min(full,start+height+post_halo.value);ext_height=ext_end-ext_start
    first=(ext_start+full-cloud_halo)%full;logical=np.arange(first,first+ext_height+2*cloud_halo)%full
    scene=np.ascontiguousarray(scene_full[logical],np.float32)
    size=ctypes.c_size_t;cn=size();dn=size();fn=size()
    if lib.nf_conditioned_cloud_row_chain_f32_workspace_v1(ext_height,width,cloud_halo,ctypes.byref(cn),ctypes.byref(dn),ctypes.byref(fn))!=0:
        raise RuntimeError("P4FB provider workspace failed")
    expected=np.full((ext_height,width,3),.6,np.float32);counts=np.empty(cn.value,np.uint16)
    conv=np.empty(dn.value,np.float64);scale=np.empty(cn.value,np.float64)
    sd=np.empty(fn.value,np.float32);st=np.empty(fn.value,np.float32);density=np.empty(fn.value,np.float32);trans=np.empty(fn.value,np.float32)
    status=lib.nf_sensitometry_cloud_bridge_f32_apply_window_v1(ctypes.byref(domains),ctypes.byref(cp),ctypes.byref(sp),full,width,first,
        ext_height,cloud_halo,scene.ctypes.data_as(fp),scene.size,capacity.ctypes.data_as(dp),scale.ctypes.data_as(dp),scale.size,
        expected.ctypes.data_as(fp),expected.size,gain.ctypes.data_as(fp),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),counts.size,
        conv.ctypes.data_as(dp),conv.size,sd.ctypes.data_as(fp),st.ctypes.data_as(fp),st.size,density.ctypes.data_as(fp),trans.ctypes.data_as(fp),density.size)
    if status: raise RuntimeError(f"P4FB physical provider cloud failed: {status}")
    cloud=density.reshape(ext_height,width,3);flat=np.ascontiguousarray(cloud.reshape(-1),np.float32)
    work=[np.empty_like(flat) for _ in range(6)];output=np.empty(height*width*3,np.float32)
    status=lib.nf_cloud_post_spatial_f32_apply_core_v1(ctypes.byref(domains),ctypes.byref(adjacency_blur),ctypes.byref(adjacency),
        ctypes.byref(diffusion),ctypes.byref(scanner),flat.ctypes.data_as(fp),ext_height,width,full,start,start-ext_start,height,
        *[item.ctypes.data_as(fp) for item in work],flat.size,output.ctypes.data_as(fp),output.size)
    if status: raise RuntimeError(f"P4FB physical provider post failed: {status}")
    return output.reshape(height,width,3)


def evaluate(root: Path, contract_path: Path, output: Path, llvm: Path) -> dict:
    contract=json.loads(contract_path.read_text());parent=root/contract["parent"]["path"]
    if sha256_file(parent)!=contract["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=contract["parent"]["required_decision"]:
        raise RuntimeError("P4FB parent drift")
    builds={"msvc":_build(root,output/"msvc",None),"llvm":_build(root,output/"llvm",llvm)};rows={}
    for name,path in builds.items():
        lib=_configure(path)
        try:first=_execute(lib,contract);repeat=_execute(lib,contract)
        finally:_ctypes.FreeLibrary(lib._handle)
        rows[name]={"scan_sha256":hashlib.sha256(first["scan"].tobytes()).hexdigest(),
            "primary_exact":first["primary_exact"],"alternate_exact":first["alternate_exact"],
            "failure_atomic":first["failure_atomic"],"post_halo":first["post_halo"],
            "repeat":bool(np.array_equal(first["scan"],repeat["scan"]))}
    gates={"primary":all(r["primary_exact"] for r in rows.values()),"alternate":all(r["alternate_exact"] for r in rows.values()),
        "compiler":rows["msvc"]["scan_sha256"]==rows["llvm"]["scan_sha256"],"repeat":all(r["repeat"] for r in rows.values()),
        "atomic":all(r["failure_atomic"] for r in rows.values()),"halo":all(r["post_halo"]>0 for r in rows.values())}
    stable={"contract_sha256":sha256_file(contract_path),"source_sha256":sha256_file(root/SOURCES[0]),
        "header_sha256":sha256_file(root/"native/film_physics/nf_cloud_post_spatial_f32_v1.h"),"results":rows,"gates":gates,
        "decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4fb_native_cloud_spatial_partition.v1","automatic_pass":all(gates.values()),"stable":stable,
        "stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


__all__=["evaluate","render_physical_partition"]
