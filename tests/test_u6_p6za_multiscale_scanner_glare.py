from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.multiscale_scanner_glare import load_contract, run_audit
from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    ScannerGlareDomainError,
    apply_scanner_glare,
    compile_scanner_glare_kernel,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6za_multiscale_scanner_glare_v1.json"


def test_formal_multiscale_scanner_glare_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["context_center_lifts"] == sorted(
        report["measurements"]["context_center_lifts"]
    )


def test_constant_field_is_preserved() -> None:
    profile = MultiscaleScannerGlareProfile(
        components=(ScannerGlareComponent(1.0, 2.0),),
        flare_fraction=0.1,
        truncate_sigma=4.0,
    )
    kernel = compile_scanner_glare_kernel(profile, kernel_size=17)
    source = np.full((37, 41, 3), 0.42, dtype=np.float64)
    assert np.allclose(
        apply_scanner_glare(source, kernel, flare_fraction=profile.flare_fraction),
        source,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "component_values",
    [(), ((0.4, 2.0),), ((1.0, 0.0),)],
)
def test_profile_rejects_invalid_components(
    component_values: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises(ScannerGlareDomainError):
        MultiscaleScannerGlareProfile(
            components=tuple(
                ScannerGlareComponent(weight, sigma)
                for weight, sigma in component_values
            ),
            flare_fraction=0.1,
            truncate_sigma=4.0,
        )
