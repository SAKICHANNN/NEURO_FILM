import csv
import hashlib

import pytest
from PIL import Image

from scripts.audit_ai_course_correction import (
    gallery_role,
    recovery_matrix,
    verify_report,
)


@pytest.mark.parametrize(
    "path",
    [
        "eval/neural_film_lut_v2/scheme_b_nilut_distilled_v1_s2p0/portra_400/after/11_portra_400_nilut.png",
        "roll2film/ct5_v1/fullres/cases/cinema/lab_mean_std/train_y-jg-fuji-x-h2s-1148.png/best_basic_wb_contrast_saturation.png",
        "ai_vcg_reference_development_v3/02_prior_nlut.png",
        "ai_unknown/02_learned.png",
    ],
)
def test_names_or_controls_cannot_acquire_learning_eligibility(path):
    assert not gallery_role(path)["learned_example"]


def test_selected_success_example_does_not_promote_failed_family():
    role = gallery_role("ai_vcg_reference_development_v3/02_learned.png")
    assert role["learned_example"]
    assert not role["promotion"]
    assert not role["independent_assessment"]


def test_training_example_remains_training():
    assert (
        gallery_role("ai_photo_distribution_pilot_v1/04_lut.png")["data_role"]
        == "training"
    )


def test_changed_report_is_rejected(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"original negative")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_report(tmp_path, path.name, expected)["verified"]
    path.write_bytes(b"rewritten success")
    with pytest.raises(ValueError, match="Report drift"):
        verify_report(tmp_path, path.name, expected)


def test_reference_family_is_not_filtered_as_reference_image(tmp_path):
    paths = [
        "ai_vcg_reference_development_v3/00_identity.png",
        "ai_vcg_reference_development_v3/00_learned.png",
        "ai_clip_lut_pilot_v1/00_original.png",
        "ai_clip_lut_pilot_v1/00_lut.png",
    ]
    images = {}
    for index, path in enumerate(paths):
        file = tmp_path / "outputs" / path
        file.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red" if index % 2 else "blue").save(file)
        images[index] = {"id": index, "path": path}
    matrix = recovery_matrix(tmp_path, images)
    assert len(matrix["groups"]) == 1
    assert len(matrix["groups"][0]["outputs"]) == 2
    assert matrix["unresolved_originals"] == []


def test_manifest_recovers_original_missing_from_catalog(tmp_path):
    folder = tmp_path / "outputs/ai_recovery_20260907/neural_s800/ektar_100"
    after = folder / "after/01_ektar_100_neural_lut.png"
    before = (
        tmp_path
        / "outputs/color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/inputs/source.jpg"
    )
    after.parent.mkdir(parents=True)
    before.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(before)
    Image.new("RGB", (8, 8), "red").save(after)
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["before", "after"])
        writer.writeheader()
        writer.writerow({"before": str(before), "after": str(after)})
    images = {1: {"id": 1, "path": after.relative_to(tmp_path / "outputs").as_posix()}}
    matrix = recovery_matrix(tmp_path, images)
    assert matrix["unresolved_originals"] == []
    assert len(matrix["groups"]) == 1
    assert matrix["groups"][0]["outputs"][0]["original_link"].endswith("manifest.csv")
