from src.eval.three_way_look_visual import SUPPORTED_EXPERIMENT_IDS, blind_orders


def test_ap3v_frozen_comparisons_are_supported() -> None:
    assert {
        "u5.r2ap3v-b0-ektachrome-residual-t10-c25-v1",
        "u5.r2ap3v-b0-ektachrome-residual-t20-c50-v1",
    } <= SUPPORTED_EXPERIMENT_IDS


def test_ap3v_blind_orders_are_distinct_and_deterministic() -> None:
    roles = ["b0_fixed_base", "ao6_velvia_residual", "ap3_ektachrome_t10_c25"]
    first = blind_orders(20260730, roles, 3)
    assert first == blind_orders(20260730, roles, 3)
    assert len({tuple(order) for order in first}) == 3
