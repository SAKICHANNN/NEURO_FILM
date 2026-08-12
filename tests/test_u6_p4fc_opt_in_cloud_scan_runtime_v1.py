from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.native_msvc import sha256_file
from src.eval.physical_native_composed_display_conformance import _build_all
from src.film_physics.native_abi_layouts import NativeGaussianProfileV1
from src.film_physics.native_cloud_scan_runtime import NativeCloudScanRuntime
from src.film_physics.native_standard_runtime import _load_gaussian, _pointer

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"configs/u6_p4fc_opt_in_cloud_scan_runtime_v1.json"


def _profile() -> NativeGaussianProfileV1:
    result=NativeGaussianProfileV1();result.struct_size=ctypes.sizeof(result);result.abi_version=1
    result.source_component_sha256=b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    result.sigma_pixels_rgb[:]=(1.1,.9,.7);result.truncate=3.
    return result


def _source(height:int,width:int)->np.ndarray:
    y,x,c=np.meshgrid(np.arange(height),np.arange(width),np.arange(3),indexing="ij")
    return np.ascontiguousarray(((y*17+x*31+c*101+3)%997)/996.,np.float32)


def test_p4fc_runtime_boundary(tmp_path:Path)->None:
    contract=json.loads(CONTRACT.read_text());parent=ROOT/contract["parent"]["path"]
    assert sha256_file(parent)==contract["parent"]["sha256"]
    builds=_build_all(root=ROOT,output_dir=tmp_path/"build")
    dll=Path(builds["gaussian"]["dll_path"]);profile=_profile();lib=_load_gaussian(dll)
    source=_source(contract["fixture"]["height"],contract["fixture"]["width"])
    workspace=np.empty_like(source);reference=np.empty_like(source)
    assert lib.nf_gaussian_f32_apply_v1(ctypes.byref(profile),_pointer(source),source.shape[0],source.shape[1],_pointer(workspace),workspace.size,_pointer(reference))==0
    reference=np.ascontiguousarray(reference*.5,dtype=np.float32)
    component=hashlib.sha256((ROOT/"native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()).hexdigest()
    def provider(forward:np.ndarray,y0:int,height:int)->np.ndarray:
        return np.ascontiguousarray(forward[y0:y0+height]*.5,dtype=np.float32)
    hashes=[]
    for tile_rows in (contract["fixture"]["tile_rows"],contract["fixture"]["alternate_tile_rows"]):
        runtime=NativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=profile,
            physical_rows=provider,physical_component_sha256=component,tile_rows=tile_rows)
        sink_rows=[]
        def sink(y0:int,y1:int,rows:np.ndarray)->None:
            sink_rows.append((y0,y1,rows.copy()))
        receipt=runtime.render_to_sink(source,output_sink=sink)
        assembled=np.concatenate([row[2] for row in sink_rows],axis=0)
        assert np.array_equal(assembled,reference)
        assert [row[0] for row in sink_rows]==sorted(row[0] for row in sink_rows)
        assert receipt["physical_component_sha256"]==component
        hashes.append(receipt["output_sha256"])
    assert len(set(hashes))==1
    before=source.copy()
    runtime=NativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=profile,
        physical_rows=provider,physical_component_sha256=component,tile_rows=11)
    with pytest.raises(RuntimeError,match="sink failure"):
        runtime.render_to_sink(source,output_sink=lambda y0,y1,rows:(_ for _ in ()).throw(RuntimeError("sink failure")))
    assert np.array_equal(source,before) and source.flags.writeable
