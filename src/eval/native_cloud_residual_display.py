"""Correct-domain cloud scan plus frozen AO6 residual-only display."""
from __future__ import annotations
import ctypes,hashlib
from collections.abc import Callable
import numpy as np
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.film_physics.native_ao6_residual_profile import NativeAo6ResidualProfileF32V1
from src.film_physics.native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from src.film_physics.native_standard_runtime import NativeStandardRuntime,NativeStandardRuntimeError,OutputSink,_pointer
SourceRows=Callable[[int,int],np.ndarray]

def _apply_residual(standard:NativeStandardRuntime,linear:np.ndarray)->np.ndarray:
 source=np.ascontiguousarray(linear,np.float32);output=np.empty_like(source);fn=standard._display.nf_ao6_residual_f32_apply_scratch_v2
 fn.argtypes=[ctypes.POINTER(NativeAo6ResidualProfileF32V1),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)];fn.restype=ctypes.c_int
 status=fn(ctypes.byref(standard._residual_profile),_pointer(source),source.shape[0]*source.shape[1],_pointer(output),None,None)
 if status!=0:raise NativeStandardRuntimeError(f'AO6 residual-only failed: {status}')
 return output

def render_cloud_with_ao6_residual(standard:NativeStandardRuntime,cloud:WindowedNativeCloudScanRuntime,*,height:int,width:int,source_rows:SourceRows,expected_input_sha256:str,output_sink:OutputSink)->dict:
 if height<=0 or width<=0 or not callable(source_rows) or not callable(output_sink) or len(expected_input_sha256)!=64:raise ValueError('invalid cloud residual source')
 digest=hashlib.sha256();consumed=0;residual_energy=0.;values=0
 def consume(y0:int,y1:int,scan:np.ndarray)->None:
  nonlocal consumed,residual_energy,values
  if y0!=consumed:raise NativeStandardRuntimeError('cloud residual output order drift')
  gauged=standard._apply_gauge(scan);residual=_apply_residual(standard,gauged);delta=residual.astype(np.float64)-gauged.astype(np.float64);residual_energy+=float(np.sum(delta*delta));values+=delta.size
  encoded=np.ascontiguousarray(linear_srgb_to_encoded(residual.astype(np.float64)),np.float32)
  if not np.all(np.isfinite(encoded)) or np.any(encoded<0) or np.any(encoded>1):raise NativeStandardRuntimeError('cloud residual output invalid')
  encoded.flags.writeable=False;output_sink(y0,y1,encoded);digest.update(memoryview(encoded).cast('B'));consumed=y1
 receipt=cloud.render_rows_to_sink(height=height,width=width,source_rows=source_rows,expected_input_sha256=expected_input_sha256,output_sink=consume)
 if consumed!=height:raise NativeStandardRuntimeError('cloud residual output incomplete')
 return {'input_sha256':expected_input_sha256,'physical_output_sha256':receipt['output_sha256'],'output_sha256':digest.hexdigest(),'shape':[height,width,3],'source_passes':receipt['source_passes'],'residual_linear_rms':float(np.sqrt(residual_energy/values)),'full_source_frame_retained':False,'physical_order':['scene-linear-relative-exposure','forward-scatter','sensitometry','developed-dye-cloud-density','bounded-development-adjacency','dye-diffusion','density-interpretation','scanner-mtf','scan-linear','neutral-gauge','ao6-linear-residual-only','srgb-oetf'],'cloud_receipt':receipt,'standard_package_sha256':standard.package_sha256,'standard_artifact_sha256':standard.artifact_sha256,'production_default_changed':False,'claim_ceiling':'synthetic-cloud-residual-only-not-calibrated-not-promoted'}

__all__=['render_cloud_with_ao6_residual']
