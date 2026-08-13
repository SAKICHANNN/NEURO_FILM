import json
from pathlib import Path

import numpy as np

from src.eval.opponent_diffusion_photographic_confirmation import (
    SigmoidCharacteristicScannerRuntime,
    _runtime,
)
from src.film_physics.compact_log_scanner_compiler import CompactLogScannerCompiler

ROOT = Path(__file__).resolve().parents[1]


def test_sigmoid_runtime_executes_complete_structured_chain() -> None:
    contract = json.loads(
        (ROOT / "configs/u6_p4ic_characteristic_ingress_structure_development_v1.json").read_text(
            encoding="utf-8"
        )
    )
    profile = json.loads(
        (
            ROOT
            / "outputs/experiments/u6_p4hk_bounded_photographic_profile_bundle_v1/run_a/profile.json"
        ).read_text(encoding="utf-8")
    )
    base = _runtime(profile, contract["candidate"])
    row = json.loads(
        (ROOT / "configs/u6_p4if_negative_scanner_inverse_d0_v2.json").read_text(
            encoding="utf-8"
        )
    )["compiler"]
    runtime = SigmoidCharacteristicScannerRuntime(
        components=base.components,
        correlation=base.correlation,
        candidate=base.candidate,
        native_toolchains={"backend": "python-p4ij-runtime-test"},
        compiler=CompactLogScannerCompiler(
            row["compiler_id"],
            tuple(tuple(values) for values in row["matrix_density_to_log10_rgb"]),
            tuple(row["bias_log10_rgb"]),
        ),
    )
    y, x = np.mgrid[0:48, 0:64].astype(np.float32)
    source = np.stack(
        (
            0.1 + 0.7 * x / 63.0,
            0.1 + 0.7 * y / 47.0,
            np.full_like(x, 0.45),
        ),
        axis=-1,
    )
    output, diagnostics = runtime.apply_source(
        np.ascontiguousarray(source), seeds=(320260812, 320261821, 320262830)
    )
    assert output.shape == source.shape
    assert np.all(np.isfinite(output))
    assert np.all(output >= -4e-6) and np.all(output <= 1.0 + 4e-6)
    assert diagnostics["sigmoid_curve_fit_rmse_maximum"] <= 0.04
    assert diagnostics["bounded_residual_rms"] >= 1e-4
