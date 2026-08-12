#!/usr/bin/env python3
"""Formal profile-bound physical-context display ablation."""
from __future__ import annotations
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.evaluate_u6_p4gc_cloud_standard_compatibility import scene,spread
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.native_cloud_spatial_partition import _build,_configure,render_physical_partition
from src.eval.native_profile_bound_cloud_display import render_profile_bound_cloud_display
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime
P4FB=ROOT/'configs/u6_p4fb_native_cloud_spatial_partition_v1.json'
def evaluate(contract:Path)->dict:
 f=json.loads(contract.read_text());fs=f['formal_scenes'];h,w=fs['height'],fs['width'];g=f['gates'];rows=[]
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
  t=Path(td);standard=_runtime(t);p=json.loads(P4FB.read_text());p['fixture']['full_height']=h;p['fixture']['width']=w;dll=_build(ROOT,t/'native',None);lib=_configure(dll);component=hashlib.sha256((ROOT/'native/film_physics/nf_cloud_post_spatial_f32_v1.c').read_bytes()).hexdigest()
  def provider(fr,y0,n):return render_physical_partition(lib,p,fr,y0,n)
  for seed in fs['seeds']:
   source=scene(seed,h,w);sha=hashlib.sha256(memoryview(source).cast('B')).hexdigest();source_context=standard._build_source_context(source);ao6=standard._apply_display(np.ascontiguousarray(linear_srgb_to_encoded(standard._apply_gauge(source).astype(np.float64)),np.float32),context=source_context);outputs=[];receipts=[];source_negative=[]
   for tr in (17,29):
    cloud=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=_profile(),physical_rows=provider,physical_component_sha256=component,tile_rows=tr)
    for _ in range(2):
     parts=[];receipts.append(render_profile_bound_cloud_display(standard,cloud,height=h,width=w,source_rows=lambda st,n:np.ascontiguousarray(source[st:st+n]),expected_input_sha256=sha,output_sink=lambda a,b,x:parts.append(x.copy())));outputs.append(np.concatenate(parts))
    scans=[];cloud.render_to_sink(source,output_sink=lambda a,b,x:scans.append(x.copy()));scan=np.concatenate(scans);source_negative.append(standard._apply_display(np.ascontiguousarray(linear_srgb_to_encoded(standard._apply_gauge(scan).astype(np.float64)),np.float32),context=source_context))
   exact=all(np.array_equal(outputs[0],x) for x in outputs[1:]) and len({r['output_sha256'] for r in receipts})==1;ar,astd,ab=spread(ao6);cr,cstd,cb=spread(outputs[0]);nr,nstd,_=spread(source_negative[0]);rr=cr/ar;sr=cstd/astd;newb=max(0.,cb-ab);passed=exact and rr>=g['minimum_combined_to_ao6_p99_p01_range_ratio'] and sr>=g['minimum_combined_to_ao6_std_ratio'] and newb<=g['maximum_new_boundary_fraction'] and cr>nr and cstd>nstd
   rows.append({'seed':seed,'source_sha256':sha,'physical_output_sha256':receipts[0]['physical_output_sha256'],'output_sha256':receipts[0]['output_sha256'],'partition_repeat_exact':exact,'source_context_negative_range':nr,'source_context_negative_std':nstd,'ao6_p99_p01_range':ar,'combined_p99_p01_range':cr,'range_ratio':rr,'ao6_std':astd,'combined_std':cstd,'std_ratio':sr,'new_boundary_fraction':newb,'source_passes':receipts[0]['source_passes'],'passed':passed})
 distinct=len({r['output_sha256'] for r in rows})==len(rows);rate=sum(r['passed'] for r in rows)/len(rows);passed=rate>=g['minimum_scene_pass_rate'] and distinct;core={'schema':f['schema'].replace('-contract','-result'),'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'rows':rows,'pass_rate':rate,'distinct_scene_outputs':distinct,'decision':'retain-profile-bound-cloud-display' if passed else 'close-profile-bound-cloud-display','claim_ceiling':f['claim_ceiling']};return {**core,'scientific_identity':hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();r=evaluate(a.contract);raw=json.dumps(r,sort_keys=True,separators=(',',':'));a.output.write_text(raw) if a.output else print(raw)
if __name__=='__main__':main()
