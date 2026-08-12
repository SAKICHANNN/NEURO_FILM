"""Profile-bound two-pass cloud scan to frozen AO6 display composition."""
from __future__ import annotations
import ctypes,hashlib
from collections.abc import Callable
import numpy as np
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.film_physics.native_ao6_base_profile import NativeAo6BaseContextF32V1
from src.film_physics.native_ao6_context_profile import NativeAo6ContextStateF32V1
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from src.film_physics.native_standard_runtime import NativeStandardRuntime,NativeStandardRuntimeError,OutputSink,_pointer
SourceRows=Callable[[int,int],np.ndarray]

def render_profile_bound_cloud_display(standard:NativeStandardRuntime,cloud:WindowedNativeCloudScanRuntime,*,height:int,width:int,source_rows:SourceRows,expected_input_sha256:str,output_sink:OutputSink)->dict:
 """Build AO6 context from exact physical output, then replay and display."""
 if height<=0 or width<=0 or not callable(source_rows) or not callable(output_sink) or len(expected_input_sha256)!=64:raise ValueError('invalid profile-bound cloud display source')
 state=NativeAo6ContextStateF32V1()
 if standard._context.nf_ao6_context_f32_init_v1(ctypes.byref(state))!=0:raise NativeStandardRuntimeError('AO6 context init failed')
 def update(_y0:int,_y1:int,scan:np.ndarray)->None:
  gauged=standard._apply_gauge(scan);encoded=np.ascontiguousarray(linear_srgb_to_encoded(gauged.astype(np.float64)),np.float32);scratch=np.empty_like(encoded)
  status=standard._context.nf_ao6_context_f32_update_v2(ctypes.byref(standard._base_profile),ctypes.byref(state),_pointer(encoded),encoded.shape[0]*encoded.shape[1],_pointer(scratch))
  if status!=0:raise NativeStandardRuntimeError(f'AO6 physical context update failed: {status}')
 context_receipt=cloud.render_rows_to_sink(height=height,width=width,source_rows=source_rows,expected_input_sha256=expected_input_sha256,output_sink=update)
 context=NativeAo6BaseContextF32V1()
 if standard._context.nf_ao6_context_f32_finalize_v1(ctypes.byref(state),ctypes.byref(context))!=0:raise NativeStandardRuntimeError('AO6 physical context finalize failed')
 digest=hashlib.sha256();consumed=0
 def display(y0:int,y1:int,scan:np.ndarray)->None:
  nonlocal consumed
  if y0!=consumed:raise NativeStandardRuntimeError('profile-bound cloud output order drift')
  gauged=standard._apply_gauge(scan);encoded=np.ascontiguousarray(linear_srgb_to_encoded(gauged.astype(np.float64)),np.float32);rows=np.ascontiguousarray(standard._apply_display(encoded,context=context),np.float32)
  if not np.all(np.isfinite(rows)) or np.any(rows<0) or np.any(rows>1):raise NativeStandardRuntimeError('profile-bound cloud output invalid')
  rows.flags.writeable=False;output_sink(y0,y1,rows);digest.update(memoryview(rows).cast('B'));consumed=y1
 output_receipt=cloud.render_rows_to_sink(height=height,width=width,source_rows=source_rows,expected_input_sha256=expected_input_sha256,output_sink=display)
 if consumed!=height:raise NativeStandardRuntimeError('profile-bound cloud output incomplete')
 if context_receipt['output_sha256']!=output_receipt['output_sha256']:raise NativeStandardRuntimeError('profile-bound cloud replay drift')
 return {'input_sha256':expected_input_sha256,'physical_output_sha256':output_receipt['output_sha256'],'output_sha256':digest.hexdigest(),'shape':[height,width,3],'source_passes':context_receipt['source_passes']+output_receipt['source_passes'],'context_domain':'cloud-scan-neutral-gauge-srgb-oetf','full_source_frame_retained':False,'cloud_receipt':output_receipt,'standard_package_sha256':standard.package_sha256,'standard_artifact_sha256':standard.artifact_sha256,'production_default_changed':False,'claim_ceiling':'synthetic-profile-bound-cloud-display-not-calibrated-not-promoted'}

__all__=['render_profile_bound_cloud_display']
