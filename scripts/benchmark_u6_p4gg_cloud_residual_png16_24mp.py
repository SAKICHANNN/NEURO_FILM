#!/usr/bin/env python3
"""Measure retained cloud residual-only verified PNG16 at 24MP."""
from __future__ import annotations
import _ctypes,argparse,hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.benchmark_u6_p4fr_native_standard_replayable_24mp import _input_sha,_runtime,_source_rows
from scripts.benchmark_u6_p4gb_cloud_standard_png16_24mp import monitor
from src.eval.native_cloud_spatial_partition import _build,_configure,render_physical_partition
from src.film_physics.native_cloud_residual_staging import stage_cloud_with_ao6_residual
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from src.film_physics.native_standard_output import commit_verified_native_standard_png16,verify_native_standard_png16
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile
P4FB=ROOT/'configs/u6_p4fb_native_cloud_spatial_partition_v1.json'
def worker(*,contract:Path,output:Path,transaction:Path)->None:
 contract=contract.resolve();output=output.resolve();transaction=transaction.resolve();f=json.loads(contract.read_text());s=f['scenario'];h,w=int(s['height']),int(s['width']);tr=int(s['tile_rows']);transaction.mkdir(parents=True,exist_ok=False);dll=_build(ROOT,(transaction/'native').resolve(),None);lib=_configure(dll);p=json.loads(P4FB.read_text());p['fixture']['full_height']=h;p['fixture']['width']=w;component=hashlib.sha256((ROOT/'native/film_physics/nf_cloud_post_spatial_f32_v1.c').read_bytes()).hexdigest()
 def provider(fr,y0,n):return render_physical_partition(lib,p,fr,y0,n)
 cloud=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=_profile(),physical_rows=provider,physical_component_sha256=component,tile_rows=tr);standard=_runtime();sha=_input_sha(h,w,tr);raw=transaction/'render.f32';rr=transaction/'render.raw.json';png=transaction/'render.png';pr=transaction/'render.png.json';started=time.perf_counter();cloud_handle=cloud._gaussian._handle;provider_handle=lib._handle
 try:
  staged=stage_cloud_with_ao6_residual(standard,cloud,height=h,width=w,source_rows=lambda st,n:_source_rows(st,n,h,w),expected_input_sha256=sha,output_path=raw,report_path=rr);payload=json.loads(rr.read_text());committed=commit_verified_native_standard_png16(staging_report_path=rr,expected_staging_report_sha256=staged['report_sha256'],expected_staging_run_id=staged['run_id'],output_path=png,report_path=pr);verified=verify_native_standard_png16(report_path=pr,expected_report_sha256=committed['report_sha256'],expected_delivery_id=committed['delivery_id']);result={'input_sha256':sha,'raw_output_sha256':staged['output_sha256'],'residual_receipt_schema':payload['working_image_receipt']['schema'],'source_passes':payload['working_image_receipt']['source_passes'],'png_output_sha256':committed['output_sha256'],'png_verification_id':verified['verification_id'],'png_bytes':png.stat().st_size,'wall_seconds':time.perf_counter()-started,'restart_verified':True}
 finally:
  del cloud,lib;_ctypes.FreeLibrary(cloud_handle);_ctypes.FreeLibrary(provider_handle);shutil.rmtree(transaction,ignore_errors=False)
 result['cleanup_pass']=not transaction.exists();output.write_text(json.dumps(result,sort_keys=True,separators=(',',':')))
def benchmark(contract:Path,out:Path)->dict:
 f=json.loads(contract.read_text());s,g=f['scenario'],f['gates'];out.mkdir(parents=True,exist_ok=True);runs=[]
 for i in range(int(s['runs'])):
  o=(out/f'worker-{i+1}.json').resolve();t=(out/f'transaction-{i+1}').resolve();runs.append(monitor([sys.executable,str(Path(__file__).resolve()),'--worker','--contract',str(contract.resolve()),'--output',str(o),'--transaction',str(t)],o,float(s['timeout_seconds']),float(s['sample_interval_seconds'])))
 peaks=[x['peak_process_tree_rss_bytes'] for x in runs];walls=[x['worker']['wall_seconds'] for x in runs];checks={'identity':len({(x['worker']['raw_output_sha256'],x['worker']['png_output_sha256']) for x in runs})==1,'peak':max(peaks)<=g['maximum_peak_process_tree_rss_bytes'],'wall':max(walls)<=g['maximum_worker_wall_seconds'],'repeat':max(peaks)/min(peaks)<=g['maximum_peak_repeat_ratio'],'verified':all(x['worker']['restart_verified'] for x in runs),'cleanup':all(x['worker']['cleanup_pass'] for x in runs),'mechanism':all(x['worker']['residual_receipt_schema']=='neuro_film.cloud_ao6_residual_receipt.v1' and x['worker']['source_passes']==3 for x in runs)};stable={'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'runs':runs,'metrics':{'peak_process_tree_rss_bytes':peaks,'worker_wall_seconds':walls,'peak_repeat_ratio':max(peaks)/min(peaks)},'gates':checks,'decision':f['decision_if_pass'] if all(checks.values()) else f['decision_if_fail'],'claim_ceiling':f['claim_ceiling']};return {'schema':f['schema'].replace('-contract','-result'),'automatic_pass':all(checks.values()),'stable':stable,'stable_evidence_id':hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--transaction',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args();worker(contract=a.contract,output=a.output,transaction=a.transaction) if a.worker else print(json.dumps(benchmark(a.contract,a.output_dir),sort_keys=True))
if __name__=='__main__':main()
