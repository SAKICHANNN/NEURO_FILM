import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gm_capacity_matched_cloud_chart_v1.json"


def test_p4gm_freezes_materially_repaired_chart_candidate():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidate = contract["candidate"]
    assert candidate["cloud_profile"].startswith("exact-p4di")
    assert candidate["display_interpretation"].endswith("no-second-neutral-gauge")
    assert candidate["look"] == "existing-ao6-linear-residual-only"
    assert candidate["hard_clipping_allowed"] is False
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert contract["gates"]["maximum_neutral_chroma_p99"] == 0.09
    assert contract["gates"]["maximum_neutral_patch_mean_chroma"] == 0.012


def test_p4gm_numeric_pass_does_not_override_visual_severe_veto():
    evidence = json.loads(
        (
            ROOT
            / "docs/evidence/U6_P4GM_CAPACITY_MATCHED_CLOUD_CHART_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["numeric_automatic_pass"] is True
    assert all(evidence["numeric_gates"].values())
    assert evidence["visual_severe_artifact_pass"] is False
    assert evidence["automatic_pass"] is False
    assert evidence["physics_neutral_patch_encoded_u8_medians"][0] == 217
    assert evidence["source_neutral_patch_encoded_u8_medians"][0] == 97
    assert evidence["decision"] == "close_capacity_matched_cloud_chart_v1"
