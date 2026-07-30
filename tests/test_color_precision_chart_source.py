from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.color_precision_chart_source import (
    ColorPrecisionChartSourceError,
    inspect_archive,
    parse_pdfimages_list,
    parse_pdf_text_pages,
    summarize_connectivity,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT / "configs/u5_r2bd0_color_precision_chart_source_v1.json"
    ).read_text(encoding="utf-8")
)


def _archive_bytes(*names: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, name in enumerate(names):
            archive.writestr(name, f"PDF-{index}".encode())
    return stream.getvalue()


def test_contract_keeps_rights_and_training_closed() -> None:
    validate_contract(CONFIG)
    assert CONFIG["source"]["rights_status"].startswith("unknown_")
    assert "operator_fitting" in CONFIG["forbidden_actions"]
    assert "training" in CONFIG["forbidden_actions"]
    assert CONFIG["acquisition"]["other_download_categories_allowed"] is False


def test_archive_inventory_is_exact_and_nonrendering(tmp_path: Path) -> None:
    path = tmp_path / "Charts-PDFs.zip"
    path.write_bytes(_archive_bytes("Charts/A.pdf", "Charts/B.pdf"))
    report = inspect_archive(path, CONFIG)
    assert report["member_count"] == 2
    assert report["file_count"] == 2
    assert report["extension_counts"] == {".pdf": 2}
    assert report["zip_crc_pass"] is True
    assert report["pixel_rendering_performed"] is False
    assert report["operator_fitting_allowed"] is False
    assert report["training_allowed"] is False
    assert report["connectivity_decision"] == "pending_member_semantic_audit"


@pytest.mark.parametrize("member", ["../escape.pdf", "/absolute.pdf"])
def test_archive_rejects_unsafe_paths(tmp_path: Path, member: str) -> None:
    path = tmp_path / "unsafe.zip"
    path.write_bytes(_archive_bytes(member))
    with pytest.raises(ColorPrecisionChartSourceError, match="unsafe"):
        inspect_archive(path, CONFIG)


def test_archive_rejects_bounds_and_contract_drift(tmp_path: Path) -> None:
    path = tmp_path / "bounded.zip"
    path.write_bytes(_archive_bytes("Charts/A.pdf", "Charts/B.pdf"))
    config = deepcopy(CONFIG)
    config["acquisition"]["maximum_files_after_extraction"] = 1
    with pytest.raises(ColorPrecisionChartSourceError, match="member count"):
        inspect_archive(path, config)

    config = deepcopy(CONFIG)
    config["source"]["rights_status"] = "cleared"
    with pytest.raises(ColorPrecisionChartSourceError, match="rights"):
        validate_contract(config)


def test_pdf_text_parser_and_connectivity_pass_without_opening_fit() -> None:
    template = """
Portra 160
EV -3, -1, 0, +1, +2, +3, +4, tungsten illuminant
{scanner} Scan
\f
Portra 400 Pushed +1
EV -3, -1, 0, +1, +2, +3, +4, tungsten illuminant
{scanner} Scan
\f
Ektachrome
EV -3, -2, -1, 0, +1, +2, +3, tungsten illuminant
{scanner} Scan
\f
Lighting Diagram
"""
    frontier = parse_pdf_text_pages(
        template.format(scanner="Frontier"), scanner_id="frontier"
    )
    noritsu = parse_pdf_text_pages(
        template.format(scanner="Noritsu"), scanner_id="noritsu"
    )
    config = deepcopy(CONFIG)
    config["data_readiness_gates"][
        "minimum_rows_in_largest_connected_component"
    ] = 6
    report = summarize_connectivity(frontier + noritsu, config)
    assert report["connectivity_passed"] is True
    assert report["metrics"]["distinct_named_stock_count"] == 3
    assert report["metrics"]["stock_process_variant_count"] == 3
    assert report["metrics"]["cross_scanner_variant_count"] == 3
    assert report["operator_fitting_allowed"] is False
    assert report["training_allowed"] is False
    assert report["latent_mode_clustering_allowed"] is False


def test_pdf_text_parser_rejects_unknown_stock_and_scanner_drift() -> None:
    text = "Mystery 400\nEV -1, 0, +1, +2, +3, +4, +5\nFrontier Scan\n"
    with pytest.raises(ColorPrecisionChartSourceError, match="unknown stock"):
        parse_pdf_text_pages(text, scanner_id="frontier")

    text = "Portra 160\nEV -1, 0, +1, +2, +3, +4, +5\nNoritsu Scan\n"
    with pytest.raises(ColorPrecisionChartSourceError, match="scanner"):
        parse_pdf_text_pages(text, scanner_id="frontier")


def test_pdfimages_parser_exposes_duplicate_and_missing_objects() -> None:
    listing = """
page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio
1 0 image 3112 2082 icc 3 8 image yes 9 0 778 778 15M 80%
1 1 image 3112 2082 icc 3 8 image yes 10 0 778 778 15M 80%
1 2 image 3112 2082 icc 3 8 image yes 10 0 778 778 15M 80%
1 3 image 2100 2100 icc 3 8 image yes 12 0 1419 1419 880K 7%
1 4 smask 2100 2100 gray 1 8 image no 12 0 1419 1419 45K 1%
2 5 image 2768 1842 icc 3 8 image yes 20 0 692 692 9M 60%
"""
    rows = parse_pdfimages_list(listing, scanner_id="noritsu")
    assert rows == [
        {
            "scanner_id": "noritsu",
            "page_index": 1,
            "photo_placement_count": 3,
            "unique_photo_object_count": 2,
            "duplicate_photo_placement_count": 1,
        },
        {
            "scanner_id": "noritsu",
            "page_index": 2,
            "photo_placement_count": 1,
            "unique_photo_object_count": 1,
            "duplicate_photo_placement_count": 0,
        },
    ]


def test_frozen_decision_keeps_exposure_and_fit_closed() -> None:
    decision = json.loads(
        (
            ROOT
            / "configs/u5_r2bd0_color_precision_chart_source_decision_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["archive"]["bytes"] == 2_410_038_342
    assert decision["connectivity"]["named_stocks"] == 9
    assert decision["connectivity"]["cross_scanner_variants"] == 12
    assert decision["layout_diagnostic"]["exposure_assignment_allowed"] is False
    assert decision["rights"]["operator_fitting_allowed"] is False
    assert decision["rights"]["training_allowed"] is False
    assert decision["rights"]["latent_mode_clustering_allowed"] is False
