from pathlib import Path

import numpy as np
import pytest

from src.eval.presampling_reference_convergence import (
    evaluate_presampling_reference_convergence,
    load_contract,
)
from src.film_physics.circular_scanner_aperture import (
    apply_circular_aperture,
    compile_circular_aperture_kernel,
)
from src.film_physics.presampling_reference_convergence import (
    apply_circular_aperture_fft,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6ae_presampling_reference_convergence_v1.json"
PARENT = (
    ROOT / "outputs/experiments/u6_p6ac_presampling_dye_cloud_scan_v1/report_run1.json"
)
FORMAL_REPORT = (
    ROOT
    / "outputs/experiments/u6_p6ae_presampling_reference_convergence_v1/report_run1.json"
)


def test_fft_matches_spatial_convolution():
    rng = np.random.default_rng(9)
    image = rng.random((31, 29, 3))
    kernel = compile_circular_aperture_kernel(
        aperture_diameter_um=12.5,
        pixel_pitch_um=1.0,
        subpixels_per_axis=64,
    )
    assert (
        np.max(
            np.abs(
                apply_circular_aperture_fft(image, kernel)
                - apply_circular_aperture(image, kernel)
            )
        )
        < 1e-12
    )


@pytest.mark.skipif(not PARENT.is_file(), reason="P6AC report unavailable")
def test_frozen_evaluator_repeats():
    config = load_contract(CONTRACT)
    first = evaluate_presampling_reference_convergence(config, ROOT)
    second = evaluate_presampling_reference_convergence(config, ROOT)
    assert first == second
    assert first["metrics"]["repeat_error"] == 0


@pytest.mark.skipif(not FORMAL_REPORT.is_file(), reason="P6AE report unavailable")
def test_formal_result_is_stably_bound():
    import hashlib
    import json

    payload = FORMAL_REPORT.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "ad1155d40f9a718b7596fc091d246067ae2195eb02610fcbec6a4735238a66ef"
    )
    assert report["stable_evidence_id"] == (
        "6fe3d8828dc243cc4e66263a5e959b9451d4a6593003b1be85b4244b6985c24a"
    )
    assert report["automatic_pass"] is False
    assert report["decision"] == "revoke_16x_presampling_reference"
