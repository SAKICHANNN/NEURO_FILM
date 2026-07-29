from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts.run_u5_r2aq1_colorreference_velvia100f_pair_table import (
    CONFIG_SHA256,
    load_config,
)
from src.roll2film.colorreference_pair_table import (
    parse_reference_table,
    repeated_set_metrics,
    sample_source_patches,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq1_colorreference_velvia100f_pair_table_v1.json"
)


def _archive(name: str, payload: bytes) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, payload)
    return stream.getvalue()


def test_contract_is_frozen_and_fit_closed() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["table_contract"]["expected_rows"] == 8640
    assert config["test_set_ids"] == [1, 2, 3, 4, 5, 9]
    assert config["fit_allowed"] is False
    assert config["operator_claim_allowed"] is False
    assert config["rights"]["redistribution_allowed"] is False


def test_wrapped_reference_rows_are_reconstructed() -> None:
    text = b"\n".join(
        [
            b"CGATS.5",
            b'NUMBER_OF_FIELDS "5"',
            b"BEGIN_DATA_FORMAT",
            b"SAMPLE_ID LAB_L LAB_A LAB_B SPECTRAL_NM",
            b"END_DATA_FORMAT",
            b'NUMBER_OF_SETS "2"',
            b"BEGIN_DATA",
            b"A1 50 1",
            b"2 380 A2",
            b"60 3 4",
            b"390",
            b"END_DATA",
        ]
    )
    table = parse_reference_table(_archive("testscan1-1.txt", text), "testscan1-1.txt")
    assert table.fields == (
        "SAMPLE_ID",
        "LAB_L",
        "LAB_A",
        "LAB_B",
        "SPECTRAL_NM",
    )
    assert table.rows == (
        ("A1", "50", "1", "2", "380"),
        ("A2", "60", "3", "4", "390"),
    )


def test_source_sampling_is_ordered_and_reports_range() -> None:
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[1:4, 1:4] = (10, 20, 30)
    image[5:8, 5:8] = (40, 50, 60)
    image[5, 5, 0] = 42
    stream = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(stream, format="TIFF")
    medians, ranges = sample_source_patches(
        stream.getvalue(),
        main_x=[2],
        main_y=[2],
        gray_x=[6],
        gray_y=6,
        radius=1,
    )
    np.testing.assert_array_equal(
        medians, np.array([[10, 20, 30], [40, 50, 60]])
    )
    assert ranges[0].max() == 0
    assert ranges[1, 0] == 2


def test_repeated_set_metrics_separate_consensus_and_set_shift() -> None:
    base = np.array(
        [
            [
                [10.0, 0.0, 0.0],
                [20.0, 2.0, 1.0],
                [26.0, -3.0, 4.0],
            ],
            [
                [30.0, -1.0, 3.0],
                [40.0, 4.0, -2.0],
                [48.0, 1.0, 6.0],
            ],
        ]
    )
    offsets = np.array([-0.2, -0.1, 0.0, 0.0, 0.1, 0.2])
    labs = np.stack(
        [base + np.array([value, 0.0, 0.0]) for value in offsets],
        axis=0,
    )
    metrics = repeated_set_metrics(labs)
    assert metrics["observation_count"] == 36
    assert metrics["median_patch_set_radius_deltae76"] == pytest.approx(0.1)
    assert metrics["p95_patch_set_radius_deltae76"] < 0.21
    assert metrics["median_slide_distance_structure_spearman"] == pytest.approx(1.0)
    assert metrics["between_set_centroid_variance_fraction"] < 0.001


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["fit_allowed"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
