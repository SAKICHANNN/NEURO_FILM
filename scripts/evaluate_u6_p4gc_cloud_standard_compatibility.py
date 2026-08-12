#!/usr/bin/env python3
"""Formal compatibility ablation for cloud physics and frozen Standard display."""
from __future__ import annotations
import argparse, hashlib, json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.native_cloud_spatial_partition import _build,_configure,render_physical_partition
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime
P4FB=ROOT/'configs/u6_p4fb_native_cloud_spatial_partition_v1.json'

def scene(seed:int,h:int,w:int)->np.ndarray:
 rng=np.random.default_rng(seed);y,x=np.meshgrid(np.linspace(0,1,h,dtype=np.float32),np.linspace(0,1,w,dtype=np.float32),indexing='ij')
 phase=rng.uniform(-np.pi,np.pi,6);freq=rng.integers(1,8,6)
 values=[]
 for c in range(3):
  v=.12+.43*x+.29*y+.12*np.sin((freq[c]*x+freq[c+3]*y)*np.pi*2+phase[c])+.04*rng.standard_normal((h,w),dtype=np.float32)
  values.append(np.clip(v,.02,.98))
 return np.ascontiguousarray(np.stack(values,axis=-1),np.float32)

def spread(v:np.ndarray)->tuple[float,float,float]:
 lum=np.asarray(v,np.float64).mean(axis=2);return float(np.percentile(lum,99)-np.percentile(lum,1)),float(np.std(lum)),float(np.mean((v<=0)|(v>=1)))

def evaluate(contract:Path)->dict:
 frozen=json.loads(contract.read_text());fs=frozen['formal_scenes'];h,w=fs['height'],fs['width'];g=frozen['gates'];rows=[]
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
  tmp=Path(td);standard=_runtime(tmp);p=json.loads(P4FB.read_text());p['fixture']['full_height']=h;p['fixture']['width']=w;dll=_build(ROOT,tmp/'native',None);lib=_configure(dll);component=hashlib.sha256((ROOT/'native/film_physics/nf_cloud_post_spatial_f32_v1.c').read_bytes()).hexdigest()
  def provider(fr,y0,n):return render_physical_partition(lib,p,fr,y0,n)
  for seed in fs['seeds']:
   source=scene(seed,h,w);source_sha=hashlib.sha256(memoryview(source).cast('B')).hexdigest();context=standard._build_source_context(source);ao6=standard._apply_display(np.ascontiguousarray(linear_srgb_to_encoded(standard._apply_gauge(source).astype(np.float64)),np.float32),context=context)
   outputs=[];scan_hashes=[]
   for tile_rows in (17,29):
    cloud=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=_profile(),physical_rows=provider,physical_component_sha256=component,tile_rows=tile_rows)
    for _ in range(2):
     parts=[];receipt=cloud.render_to_sink(source,output_sink=lambda a,b,v:parts.append(v.copy()));scan=np.concatenate(parts);combined=standard._apply_display(np.ascontiguousarray(linear_srgb_to_encoded(standard._apply_gauge(scan).astype(np.float64)),np.float32),context=context);outputs.append(combined);scan_hashes.append(receipt['output_sha256'])
   partition_exact=all(np.array_equal(outputs[0],v) for v in outputs[1:]) and len(set(scan_hashes))==1
   physics=np.ascontiguousarray(linear_srgb_to_encoded(standard._apply_gauge(scan).astype(np.float64)),np.float32)
   scan_range,scan_std,scan_boundary=spread(scan);physics_range,physics_std,physics_boundary=spread(physics)
   ar,astd,ab=spread(ao6);cr,cstd,cb=spread(outputs[0]);rr=cr/ar if ar else 0.;sr=cstd/astd if astd else 0.;newb=max(0.,cb-ab)
   passed=partition_exact and rr>=g['minimum_combined_to_ao6_p99_p01_range_ratio'] and sr>=g['minimum_combined_to_ao6_std_ratio'] and newb<=g['maximum_new_boundary_fraction']
   rows.append({'seed':seed,'source_sha256':source_sha,'scan_sha256':scan_hashes[0],'combined_sha256':hashlib.sha256(memoryview(np.ascontiguousarray(outputs[0])).cast('B')).hexdigest(),'partition_repeat_exact':partition_exact,'scan_p99_p01_range':scan_range,'scan_std':scan_std,'scan_boundary_fraction':scan_boundary,'physics_only_p99_p01_range':physics_range,'physics_only_std':physics_std,'physics_only_boundary_fraction':physics_boundary,'ao6_p99_p01_range':ar,'combined_p99_p01_range':cr,'range_ratio':rr,'ao6_std':astd,'combined_std':cstd,'std_ratio':sr,'new_boundary_fraction':newb,'passed':passed})
 pass_rate=sum(r['passed'] for r in rows)/len(rows);decision='retain-exact-composition' if pass_rate>=g['minimum_scene_pass_rate'] else 'close-exact-composition-profile-incompatible'
 core={'schema':frozen['schema'].replace('-contract','-result'),'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'rows':rows,'pass_rate':pass_rate,'decision':decision,'claim_ceiling':frozen['claim_ceiling']}
 return {**core,'scientific_identity':hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();result=evaluate(a.contract);raw=json.dumps(result,sort_keys=True,separators=(',',':'));a.output.write_text(raw) if a.output else print(raw)
if __name__=='__main__':main()
