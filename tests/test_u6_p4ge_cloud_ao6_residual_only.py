from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.evaluate_u6_p4gc_cloud_standard_compatibility import scene,spread
from src.eval.native_cloud_residual_display import render_cloud_with_ao6_residual
from src.eval.native_cloud_spatial_partition import _build,_configure,render_physical_partition
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime
ROOT=Path(__file__).resolve().parents[1];P4FB=ROOT/'configs/u6_p4fb_native_cloud_spatial_partition_v1.json'
def test_residual_only_cloud_output_is_exact_nonzero_and_noncollapsed(tmp_path:Path):
 h,w=64,96;s=scene(8011,h,w);sha=hashlib.sha256(memoryview(s).cast('B')).hexdigest();standard=_runtime(tmp_path);p=json.loads(P4FB.read_text());p['fixture']['full_height']=h;p['fixture']['width']=w;dll=_build(ROOT,tmp_path/'native',None);lib=_configure(dll);component=hashlib.sha256((ROOT/'native/film_physics/nf_cloud_post_spatial_f32_v1.c').read_bytes()).hexdigest()
 def provider(fr,y0,n):return render_physical_partition(lib,p,fr,y0,n)
 outputs=[]
 for tr in (11,17):
  cloud=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=_profile(),physical_rows=provider,physical_component_sha256=component,tile_rows=tr);parts=[];r=render_cloud_with_ao6_residual(standard,cloud,height=h,width=w,source_rows=lambda st,n:np.ascontiguousarray(s[st:st+n]),expected_input_sha256=sha,output_sink=lambda a,b,x:parts.append(x.copy()));outputs.append(np.concatenate(parts));assert r['source_passes']==3;assert r['residual_linear_rms']>.0001
 assert np.array_equal(outputs[0],outputs[1]);assert spread(outputs[0])[0]>.01

def test_p4ge_formal_result_retains_residual_only_composition():
 r=json.loads((ROOT/'docs/evidence/U6_P4GE_CLOUD_AO6_RESIDUAL_ONLY_RESULT.json').read_text());assert r['decision']=='retain-cloud-ao6-residual-only';assert r['pass_rate']==1.;assert r['distinct_scene_outputs'];assert all(x['passed'] and x['residual_linear_rms']>.0001 and .75<=x['range_ratio']<=1.5 and x['new_boundary_fraction']==0 and x['source_passes']==3 for x in r['rows'])
