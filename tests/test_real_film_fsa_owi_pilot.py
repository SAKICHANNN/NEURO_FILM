from __future__ import annotations

from PIL import Image

from src.real_film.fsa_owi_pilot import dhash64, hamming64, select_creator_balanced


def test_creator_balanced_selection_is_stable_and_round_robin() -> None:
    rows = [
        {"loc_fsac_id": f"fsac.1a{i:05d}", "creator_group": group}
        for group in ("a", "b", None)
        for i in range(5)
    ]
    first = select_creator_balanced(rows, count=9, seed="fixed")
    second = select_creator_balanced(rows, count=9, seed="fixed")
    assert [row["loc_fsac_id"] for row in first] == [row["loc_fsac_id"] for row in second]
    counts = {group: sum(row["pilot_creator_group"] == group for row in first) for group in ("a", "b", "__unknown_creator__")}
    assert counts == {"a": 3, "b": 3, "__unknown_creator__": 3}


def test_dhash_and_hamming_are_deterministic() -> None:
    black = Image.new("RGB", (32, 32), "black")
    white = Image.new("RGB", (32, 32), "white")
    assert dhash64(black) == dhash64(black.copy())
    assert hamming64(dhash64(black), dhash64(white)) == 0
