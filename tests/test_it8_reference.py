from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.run_sf2_9r_colorreference_it8_audit import audit_archive
from src.real_film.it8_reference import parse_cgats_spectral, parse_it8


def test_parse_it8_reference() -> None:
    payload = b"""IT8.7/1
MATERIAL "Example Film"
NUMBER_OF_FIELDS 4
BEGIN_DATA_FORMAT
SAMPLE_ID LAB_L LAB_A LAB_B
END_DATA_FORMAT
NUMBER_OF_SETS 2
BEGIN_DATA
A1 50 1 2
A2 60 3 4
END_DATA
"""
    parsed = parse_it8(payload)
    assert parsed.header["MATERIAL"] == "Example Film"
    assert parsed.sample_ids == ("A1", "A2")
    assert parsed.rows[1]["LAB_B"] == "4"


def test_parse_wrapped_cgats_spectral_reference() -> None:
    payload = b"""IT8.7/1
MATERIAL "Example Film"
NUMBER_OF_FIELDS 13
BEGIN_DATA_FORMAT
SAMPLE_ID XYZ_X XYZ_Y XYZ_Z LAB_L LAB_A LAB_B LAB_C LAB_H SPECTRAL_NM SPECTRAL_PCT SPECTRAL_NM SPECTRAL_PCT
END_DATA_FORMAT
NUMBER_OF_SETS 2
BEGIN_DATA
A1 1 2 3 40 5 6 7 8 380 0.1
390 0.2
A2 2 3 4 50 6 7 8 9 380 0.3 390 0.4
END_DATA
"""
    parsed = parse_cgats_spectral(payload)
    assert parsed.sample_ids == ("A1", "A2")
    assert parsed.lab[0] == (40.0, 5.0, 6.0)
    assert parsed.wavelengths_nm == (380, 390)
    assert parsed.spectra_pct[1] == (0.3, 0.4)


def test_parse_rejects_duplicate_sample_ids() -> None:
    payload = b"""IT8.7/1
NUMBER_OF_FIELDS 2
BEGIN_DATA_FORMAT
SAMPLE_ID LAB_L
END_DATA_FORMAT
NUMBER_OF_SETS 2
BEGIN_DATA
A1 50
A1 60
END_DATA
"""
    with pytest.raises(ValueError, match="duplicate"):
        parse_it8(payload)


def test_audit_archive_checks_crc_and_matching_tables(tmp_path: Path) -> None:
    it8 = b"""IT8.7/1
MATERIAL "Example Film"
NUMBER_OF_FIELDS 4
BEGIN_DATA_FORMAT
SAMPLE_ID LAB_L LAB_A MEAN_DE
END_DATA_FORMAT
NUMBER_OF_SETS 2
BEGIN_DATA
A1 40 5 0.2
A2 50 6 0.4
END_DATA
"""
    spectral = b"""IT8.7/1
MATERIAL "Example Film"
NUMBER_OF_FIELDS 13
BEGIN_DATA_FORMAT
SAMPLE_ID XYZ_X XYZ_Y XYZ_Z LAB_L LAB_A LAB_B LAB_C LAB_H SPECTRAL_NM SPECTRAL_PCT SPECTRAL_NM SPECTRAL_PCT
END_DATA_FORMAT
NUMBER_OF_SETS 2
BEGIN_DATA
A1 1 2 3 40 5 6 7 8 380 0.1 390 0.2
A2 2 3 4 50 6 7 8 9 380 0.3 390 0.4
END_DATA
"""
    path = tmp_path / "reference.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("reference.it8", it8)
        archive.writestr("extras/reference.cgt", spectral)
    record = audit_archive(
        path,
        {
            "path": path.name,
            "declared_family": "fixture",
            "exact_stock_id": "unknown",
        },
    )
    assert record["archive_crc_clean"] is True
    assert record["sample_count"] == 2
    assert record["spectral_wavelengths_nm"] == [380, 390]
    assert record["mean_batch_delta_e76_median"] == pytest.approx(0.3)


def test_audit_archive_accepts_exact_legacy_charge_txt_only(
    tmp_path: Path,
) -> None:
    it8 = b"""IT8.7/1
NUMBER_OF_FIELDS 2
BEGIN_DATA_FORMAT
SAMPLE_ID MEAN_DE
END_DATA_FORMAT
NUMBER_OF_SETS 1
BEGIN_DATA
A1 0.2
END_DATA
"""
    spectral = b"""IT8.7/1
NUMBER_OF_FIELDS 11
BEGIN_DATA_FORMAT
SAMPLE_ID XYZ_X XYZ_Y XYZ_Z LAB_L LAB_A LAB_B LAB_C LAB_H SPECTRAL_NM SPECTRAL_PCT
END_DATA_FORMAT
NUMBER_OF_SETS 1
BEGIN_DATA
A1 1 2 3 40 5 6 7 8 380 0.1
END_DATA
"""
    path = tmp_path / "E040227.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("E040227/E040227.txt", it8)
        archive.writestr("E040227/Readme.txt", b"not measurement data")
        archive.writestr("E040227/EXTRAS/E040227.cgt", spectral)
    record = audit_archive(
        path,
        {
            "path": path.name,
            "declared_family": "fixture",
            "exact_stock_id": "unknown",
        },
    )
    assert record["it8_member"] == "E040227/E040227.txt"
