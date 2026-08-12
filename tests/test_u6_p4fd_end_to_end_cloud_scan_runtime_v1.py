from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure,
    render_physical_partition,
)
from src.eval.native_msvc import sha256_file
from src.film_physics.native_cloud_scan_runtime import NativeCloudScanRuntime
from src.film_physics.native_standard_runtime import _pointer
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile, _source

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"configs/u6_p4fd_end_to_end_cloud_scan_runtime_v1.json"
P4FB=ROOT/"configs/u6_p4fb_native_cloud_spatial_partition_v1.json"


def test_p4fd_end_to_end_runtime(tmp_path:Path)->None:
    contract=json.loads(CONTRACT.read_text());parent=ROOT/contract["parent"]["path"]
    assert sha256_file(parent)==contract["parent"]["sha256"]
    p4fb=json.loads(P4FB.read_text());dll=_build(ROOT,tmp_path/"build",None);lib=_configure(dll)
    profile=_profile();source=_source(contract["fixture"]["height"],contract["fixture"]["width"])
    workspace=np.empty_like(source);forward=np.empty_like(source)
    assert lib.nf_gaussian_f32_apply_v1(ctypes.byref(profile),_pointer(source),source.shape[0],source.shape[1],_pointer(workspace),workspace.size,_pointer(forward))==0
    reference=render_physical_partition(lib,p4fb,forward,0,source.shape[0])
    component=hashlib.sha256((ROOT/"native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()).hexdigest()
    def provider(values:np.ndarray,y0:int,height:int)->np.ndarray:
        return render_physical_partition(lib,p4fb,values,y0,height)
    outputs=[];before=source.copy()
    for tile_rows in (contract["fixture"]["tile_rows"],contract["fixture"]["alternate_tile_rows"]):
        runtime=NativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=profile,
            physical_rows=provider,physical_component_sha256=component,tile_rows=tile_rows)
        for _ in range(2):
            rows=[];receipt=runtime.render_to_sink(source,output_sink=lambda y0,y1,value,target=rows:target.append(value.copy()))
            result=np.concatenate(rows,axis=0);assert np.array_equal(result,reference)
            assert np.all(np.isfinite(result)) and np.all((result>=0)&(result<=1))
            outputs.append(receipt["output_sha256"])
    assert len(set(outputs))==1
    assert np.array_equal(source,before) and source.flags.writeable
    runtime=NativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=profile,
        physical_rows=provider,physical_component_sha256=component,tile_rows=11)
    with pytest.raises(RuntimeError,match="injected sink"):
        runtime.render_to_sink(source,output_sink=lambda y0,y1,value:(_ for _ in ()).throw(RuntimeError("injected sink")))
    assert np.array_equal(source,before) and source.flags.writeable
