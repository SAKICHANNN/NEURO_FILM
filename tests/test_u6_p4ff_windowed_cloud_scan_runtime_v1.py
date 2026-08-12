from __future__ import annotations

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
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile, _source

ROOT=Path(__file__).resolve().parents[1];CONTRACT=ROOT/"configs/u6_p4ff_windowed_cloud_scan_runtime_v1.json";P4FB=ROOT/"configs/u6_p4fb_native_cloud_spatial_partition_v1.json"


def test_p4ff_windowed_runtime(tmp_path:Path)->None:
    contract=json.loads(CONTRACT.read_text());parent=ROOT/contract["parent"]["path"]
    assert sha256_file(parent)==contract["parent"]["sha256"]
    p4fb=json.loads(P4FB.read_text());dll=_build(ROOT,tmp_path/"build",None);lib=_configure(dll);profile=_profile()
    source=_source(contract["fixture"]["height"],contract["fixture"]["width"]);before=source.copy()
    component=hashlib.sha256((ROOT/"native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()).hexdigest()
    def provider(forward_rows,y0:int,height:int)->np.ndarray:return render_physical_partition(lib,p4fb,forward_rows,y0,height)
    hashes=[]
    for tile_rows in (contract["fixture"]["tile_rows"],contract["fixture"]["alternate_tile_rows"]):
        runtime=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=profile,physical_rows=provider,
            physical_component_sha256=component,tile_rows=tile_rows)
        for _ in range(2):
            rows=[];receipt=runtime.render_to_sink(source,output_sink=lambda y0,y1,value,target=rows:target.append(value.copy()))
            result=np.concatenate(rows);assert np.all(np.isfinite(result)) and np.all((result>=0)&(result<=1))
            assert receipt["output_sha256"]==contract["fixture"]["p4fd_reference_sha256"]
            assert not receipt["full_forward_frame_retained"]
            assert receipt["maximum_forward_workspace_values"]>0
            hashes.append(receipt["output_sha256"])
    assert len(set(hashes))==1 and np.array_equal(source,before) and source.flags.writeable
    runtime=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=profile,physical_rows=provider,
        physical_component_sha256=component,tile_rows=11)
    with pytest.raises(RuntimeError,match="injected sink"):
        runtime.render_to_sink(source,output_sink=lambda y0,y1,value:(_ for _ in ()).throw(RuntimeError("injected sink")))
    assert np.array_equal(source,before) and source.flags.writeable
