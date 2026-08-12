#!/usr/bin/env python3
"""Formal synthetic chart veto for the retained cloud residual chain."""
from __future__ import annotations
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.cross_layer_cloud_chart_artifact import _chart,_metrics
from src.eval.native_cloud_residual_display import render_cloud_with_ao6_residual
from src.eval.native_cloud_spatial_partition import _build,_configure,render_physical_partition
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime
P4FB=ROOT/'configs/u6_p4fb_native_cloud_spatial_partition_v1.json'
def _visual(path:Path,source:np.ndarray,physics:np.ndarray,residual:np.ndarray)->None:
 panels=[]
 for label,value in [('source',linear_srgb_to_encoded(source)),('physics-only',physics),('physics + residual',residual)]:
  image=Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8),'RGB');draw=ImageDraw.Draw(image);draw.rectangle((0,0,230,30),fill=(0,0,0));draw.text((8,8),label,fill=(255,255,255));panels.append(image)
 result=Image.new('RGB',(source.shape[1],source.shape[0]*3));
 for i,image in enumerate(panels):result.paste(image,(0,i*source.shape[0]))
 path.parent.mkdir(parents=True,exist_ok=True);result.save(path)
def evaluate(contract:Path,visual:Path|None=None)->dict:
 f=json.loads(contract.read_text());fx=f['fixture'];h,w=fx['height'],fx['width'];source64,patches=_chart((h,w));source=np.ascontiguousarray(source64,np.float32);sha=hashlib.sha256(memoryview(source).cast('B')).hexdigest();p=json.loads(P4FB.read_text());p['fixture']['full_height']=h;p['fixture']['width']=w;outputs=[];physics=[];receipts=[]
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
  t=Path(td);standard=_runtime(t);dll=_build(ROOT,t/'native',None);lib=_configure(dll);component=hashlib.sha256((ROOT/'native/film_physics/nf_cloud_post_spatial_f32_v1.c').read_bytes()).hexdigest()
  def provider(fr,y0,n):return render_physical_partition(lib,p,fr,y0,n)
  for tr in fx['tile_rows']:
   cloud=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=_profile(),physical_rows=provider,physical_component_sha256=component,tile_rows=tr);scans=[];cloud.render_to_sink(source,output_sink=lambda a,b,x:scans.append(x.copy()));scan=np.concatenate(scans);physics.append(np.ascontiguousarray(linear_srgb_to_encoded(standard._apply_gauge(scan).astype(np.float64)),np.float32))
   for _ in range(2):
    parts=[];receipts.append(render_cloud_with_ao6_residual(standard,cloud,height=h,width=w,source_rows=lambda st,n:np.ascontiguousarray(source[st:st+n]),expected_input_sha256=sha,output_sink=lambda a,b,x:parts.append(x.copy())));outputs.append(np.concatenate(parts))
 exact=all(np.array_equal(outputs[0],x) for x in outputs[1:]) and np.array_equal(physics[0],physics[1]);pm=_metrics(physics[0],patches,f);rm=_metrics(outputs[0],patches,f);g=f['gates'];checks={'neutral_chroma':rm['neutral_chroma_p99']<=g['maximum_neutral_chroma_p99'],'patch_chroma':rm['maximum_neutral_patch_mean_chroma']<=g['maximum_neutral_patch_mean_chroma'],'overshoot':rm['edge_overshoot']<=g['maximum_edge_overshoot'],'undershoot':rm['edge_undershoot']<=g['maximum_edge_undershoot'],'boundary':rm['new_boundary_fraction']<=g['maximum_new_boundary_fraction'],'residual_neutral_increment':rm['neutral_chroma_p99']-pm['neutral_chroma_p99']<=g['maximum_residual_neutral_chroma_increase'],'residual_overshoot_increment':rm['edge_overshoot']-pm['edge_overshoot']<=g['maximum_residual_edge_overshoot_increase'],'residual_undershoot_increment':rm['edge_undershoot']-pm['edge_undershoot']<=g['maximum_residual_edge_undershoot_increase'],'partition_repeat_exact':exact}
 if visual is not None:_visual(visual,source,physics[0],outputs[0])
 stable={'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'source_sha256':sha,'physics_output_sha256':hashlib.sha256(memoryview(physics[0]).cast('B')).hexdigest(),'residual_output_sha256':receipts[0]['output_sha256'],'physics':pm,'residual':rm,'residual_linear_rms':receipts[0]['residual_linear_rms'],'gates':checks,'decision':f['decision_if_pass'] if all(checks.values()) else f['decision_if_fail'],'claim_ceiling':f['claim_ceiling']};return {'schema':f['schema'].replace('-contract','-result'),'automatic_pass':all(checks.values()),'stable':stable,'stable_evidence_id':hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--visual',type=Path);a=p.parse_args();r=evaluate(a.contract,a.visual);raw=json.dumps(r,sort_keys=True,separators=(',',':'));a.output.write_text(raw) if a.output else print(raw)
if __name__=='__main__':main()
