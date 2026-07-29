from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_scanner_characterization_bundle import (
    compile_measured_scanner_characterization_bundle,
    write_bundle,
    write_report,
)
from src.film_physics.scanner import ScannerProfile
from src.film_physics.scanner_characterization import (
    MeasuredScannerCharacterization,
    apply_measured_scanner_characterization,
    measured_scanner_bundle_from_payload,
    select_measured_scanner_characterization,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "u6_p6j_measured_scanner_characterization_bundle_v1.json"
)


def _entry() -> MeasuredScannerCharacterization:
    return MeasuredScannerCharacterization(
        scanner="scanner-a",
        software="software-a",
        target_set=1,
        archive_sha256="a" * 64,
        common_power=1.5,
        matrix=((0.7, 0.1, 0.0), (0.0, 0.8, 0.1), (0.1, 0.0, 0.6)),
        fit_patch_count=1320,
        fit_support_sha256="b" * 64,
    )


def test_characterization_applies_exact_explicit_operator() -> None:
    source = np.asarray(
        [[[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]]], dtype=np.float32
    )
    expected = np.power(source.astype(np.float64), 1.5) @ np.asarray(
        _entry().matrix
    ).T
    np.testing.assert_array_equal(
        apply_measured_scanner_characterization(source, _entry()), expected
    )


@pytest.mark.parametrize(
    "values,error",
    [
        (np.ones((2, 3), dtype=np.int16), TypeError),
        (np.ones((2, 4), dtype=np.float64), ValueError),
        (np.asarray([[0.0, np.nan, 0.0]]), ValueError),
        (np.asarray([[0.0, 1.01, 0.0]]), ValueError),
    ],
)
def test_characterization_fails_closed(values: np.ndarray, error: type) -> None:
    with pytest.raises(error):
        apply_measured_scanner_characterization(values, _entry())


@pytest.fixture(scope="module")
def compiled() -> tuple:
    return compile_measured_scanner_characterization_bundle(ROOT, CONTRACT)


def test_compiled_bundle_passes_and_is_not_forward_profile(compiled: tuple) -> None:
    bundle, report = compiled
    assert report["automatic_pass"]
    assert len(bundle.entries) == 13
    assert all(entry.fit_patch_count == 1320 for entry in bundle.entries)
    assert not isinstance(bundle.entries[0], ScannerProfile)
    assert report["maximum_apply_vs_compiler_reference_abs_error"] == 0.0


def test_bundle_roundtrip_and_identity_selection(compiled: tuple) -> None:
    bundle, _ = compiled
    restored = measured_scanner_bundle_from_payload(bundle.to_payload())
    assert restored.bundle_id == bundle.bundle_id
    entry = bundle.entries[0]
    selected = select_measured_scanner_characterization(
        restored,
        scanner=entry.scanner,
        software=entry.software,
        target_set=entry.target_set,
        archive_sha256=entry.archive_sha256,
    )
    assert selected.entry_id == entry.entry_id
    with pytest.raises(ValueError, match="identity mismatch"):
        select_measured_scanner_characterization(
            restored,
            scanner=entry.scanner,
            software=entry.software,
            target_set=entry.target_set + 1,
            archive_sha256=entry.archive_sha256,
        )


def test_bundle_and_report_repeat_byte_exact(
    compiled: tuple, tmp_path: Path
) -> None:
    bundle, report = compiled
    second_bundle, second_report = (
        compile_measured_scanner_characterization_bundle(ROOT, CONTRACT)
    )
    assert bundle == second_bundle
    assert report == second_report
    assert write_bundle(bundle, tmp_path / "a.bundle.json") == write_bundle(
        second_bundle, tmp_path / "b.bundle.json"
    )
    assert (tmp_path / "a.bundle.json").read_bytes() == (
        tmp_path / "b.bundle.json"
    ).read_bytes()
    assert write_report(report, tmp_path / "a.report.json") == write_report(
        second_report, tmp_path / "b.report.json"
    )


def test_parent_hash_drift_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["parents"]["p6i_report_sha256"] = "0" * 64
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="p6i_report_path hash mismatch"):
        compile_measured_scanner_characterization_bundle(ROOT, path)
