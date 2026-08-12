"""Create-only staging for the retained cloud residual-only composition."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from src.eval.native_cloud_residual_display import render_cloud_with_ao6_residual
from .native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from .native_cloud_standard_staging import _stage_cloud_display_rows
from .native_standard_runtime import NativeStandardRuntime

def stage_cloud_with_ao6_residual(
    standard:NativeStandardRuntime,cloud:WindowedNativeCloudScanRuntime,*,height:int,width:int,source_rows:Any,expected_input_sha256:str,output_path:Path,report_path:Path
)->dict[str,Any]:
    return _stage_cloud_display_rows(
        standard,cloud,renderer=render_cloud_with_ao6_residual,
        receipt_schema="neuro_film.cloud_ao6_residual_receipt.v1",
        height=height,width=width,source_rows=source_rows,
        expected_input_sha256=expected_input_sha256,
        output_path=output_path,report_path=report_path,
    )

__all__=["stage_cloud_with_ao6_residual"]
