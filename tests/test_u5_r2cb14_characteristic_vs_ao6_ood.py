from __future__ import annotations

from pathlib import Path

from src.eval.characteristic_vs_ao6_ood import _validate, load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_cb14_validates_disjoint_ood_population() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb14_characteristic_vs_ao6_ood_v1.json"
    )
    _, _, _, rows, _ = _validate(contract, ROOT)
    assert len(rows) == 12
    assert len({row["make"] for row in rows}) == 12
    assert {row["id"] for row in rows}.isdisjoint(
        {
            "epson_r_d1x",
            "motorola_moto_g_7_play",
            "lg_lg_h850",
            "autel_robotics_xb015",
            "hasselblad_lunar",
            "huawei_eml_l29",
            "apple_iphone_12_pro",
            "blackmagic_pocket_cinema_camera_4k",
            "leica_c_typ_112",
            "olympus_e_450",
            "xiaomi_m2010j19cg",
            "leaf_aptus_22",
        }
    )


def test_cb14_retains_original_blind_gates() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb14_characteristic_vs_ao6_ood_v1.json"
    )
    assert contract["protocol"]["minimum_candidate_round_wins"] == 2
    assert contract["protocol"]["minimum_candidate_aggregate_choices"] == 22
    assert contract["protocol"]["minimum_candidate_source_majorities"] == 7
    assert contract["protocol"]["maximum_new_hard_boundary_fraction"] == 0.0
