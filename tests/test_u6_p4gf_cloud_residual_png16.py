from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.evaluate_u6_p4gc_cloud_standard_compatibility import scene
from src.eval.native_cloud_spatial_partition import _build,_configure,render_physical_partition
from src.film_physics.native_cloud_residual_staging import stage_cloud_with_ao6_residual
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from src.film_physics.native_standard_output import commit_verified_native_standard_png16,verify_native_standard_png16
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime
ROOT=Path(__file__).resolve().parents[1];P4FB=ROOT/'configs/u6_p4fb_native_cloud_spatial_partition_v1.json'
def test_retained_residual_composition_reaches_verified_png16(tmp_path:Path):
 h,w=64,96;s=scene(8011,h,w);sha=hashlib.sha256(memoryview(s).cast('B')).hexdigest();standard=_runtime(tmp_path);p=json.loads(P4FB.read_text());p['fixture']['full_height']=h;p['fixture']['width']=w;dll=_build(ROOT,tmp_path/'native',None);lib=_configure(dll);component=hashlib.sha256((ROOT/'native/film_physics/nf_cloud_post_spatial_f32_v1.c').read_bytes()).hexdigest()
 def provider(fr,y0,n):return render_physical_partition(lib,p,fr,y0,n)
 cloud=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=_profile(),physical_rows=provider,physical_component_sha256=component,tile_rows=11);staged=stage_cloud_with_ao6_residual(standard,cloud,height=h,width=w,source_rows=lambda st,n:np.ascontiguousarray(s[st:st+n]),expected_input_sha256=sha,output_path=tmp_path/'residual.f32',report_path=tmp_path/'residual.raw.json');committed=commit_verified_native_standard_png16(staging_report_path=Path(staged['report_path']),expected_staging_report_sha256=staged['report_sha256'],expected_staging_run_id=staged['run_id'],output_path=tmp_path/'residual.png',report_path=tmp_path/'residual.png.json');verified=verify_native_standard_png16(report_path=Path(committed['report_path']),expected_report_sha256=committed['report_sha256'],expected_delivery_id=committed['delivery_id']);payload=json.loads((tmp_path/'residual.raw.json').read_text());assert verified['output_sha256']==committed['output_sha256'];assert payload['working_image_receipt']['schema']=='neuro_film.cloud_ao6_residual_receipt.v1';assert payload['working_image_receipt']['source_passes']==3
