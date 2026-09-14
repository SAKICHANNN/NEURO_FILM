import itertools
import random

import pytest

from src.eval.fable_role_assignment import solve_role_assignment


def exhaustive(rows, cameras, masks, counts, required):
    rows = sorted(rows, key=lambda r: r["candidate_rank"])
    ids = [r["identity"] for r in rows]
    cam = {r["identity"]: r["camera"] for r in rows}
    ordered_cameras = sorted(set(cam.values()), key=cameras.__getitem__)
    for held_cameras in itertools.combinations(ordered_cameras, counts["held_cameras"]):
        trainable = [i for i in ids if cam[i] not in held_cameras]
        for fit in itertools.combinations(trainable, counts["fit"]):
            represented = {cam[i] for i in fit}
            eligible = [i for i in trainable if i not in fit and cam[i] in represented]
            for val in itertools.combinations(eligible, counts["validation"]):
                for rep in itertools.combinations([i for i in eligible if i not in val],
                                                  counts["represented_assessment"]):
                    pools = [list(itertools.combinations([i for i in ids if cam[i] == c],
                                  counts["assessment_per_held_camera"])) for c in held_cameras]
                    for held_groups in itertools.product(*pools):
                        taken = set(fit + val + rep + sum(held_groups, ()))
                        for queries in itertools.combinations([i for i in ids if i not in taken],
                                                               counts["queries"]):
                            mask = 0
                            for i in queries:
                                mask |= masks[i]
                            if mask & required == required:
                                return {"held_cameras": list(held_cameras), "fit": list(fit),
                                        "validation": list(val), "represented_assessment": list(rep),
                                        "held_assessment": dict(zip(held_cameras, map(list, held_groups))),
                                        "queries": list(queries)}
    return None


def fixture_rows(camera_sequence):
    return [{"identity": str(i), "candidate_rank": f"{i:04}", "camera": camera}
            for i, camera in enumerate(camera_sequence)]


COUNTS = {"fit": 2, "validation": 1, "represented_assessment": 1,
          "held_cameras": 1, "assessment_per_held_camera": 1, "queries": 1}


@pytest.mark.parametrize("seed", range(12))
def test_matches_exhaustive_complete_assignments(seed):
    rng = random.Random(seed)
    rows = fixture_rows([rng.choice("ABC") for _ in range(8)])
    masks = {r["identity"]: rng.randrange(16) for r in rows}
    cameras = {"A": "1", "B": "2", "C": "3"}
    counts = dict(COUNTS, queries=2)
    expected = exhaustive(rows, cameras, masks, counts, 15)
    assert solve_role_assignment(rows[::-1], cameras, masks, counts, block_size=3) == expected


def test_earliest_fit_must_be_reserved_for_only_query():
    rows = fixture_rows("ABBBBBBB")
    masks = {r["identity"]: 15 if r["identity"] == "1" else 0 for r in rows}
    result = solve_role_assignment(rows, {"A": "0", "B": "1"}, masks, COUNTS)
    assert result["held_cameras"] == ["A"]
    assert result["fit"] == ["2", "3"]
    assert result["queries"] == ["1"]


def test_earliest_camera_may_make_fit_representation_impossible():
    rows = fixture_rows("AAAABCDE")
    masks = {r["identity"]: 15 for r in rows}
    cameras = {c: c for c in "ABCDE"}
    result = solve_role_assignment(rows, cameras, masks, COUNTS, block_size=2)
    assert result == exhaustive(rows, cameras, masks, COUNTS, 15)
    assert result["held_cameras"] == ["B"]


def test_queries_can_use_held_camera_and_cover_multiple_bits():
    rows = fixture_rows("AABBBBBB")
    masks = {r["identity"]: 15 if r["identity"] == "0" else 0 for r in rows}
    result = solve_role_assignment(rows, {"A": "0", "B": "1"}, masks, COUNTS)
    assert result["held_assessment"] == {"A": ["1"]}
    assert result["queries"] == ["0"]


def test_unknown_masks_rejected_and_impossible_masks_infeasible():
    rows = fixture_rows("AABBBBBB")
    cameras = {"A": "0", "B": "1"}
    with pytest.raises(ValueError, match="explicitly bound"):
        solve_role_assignment(rows, cameras, {}, COUNTS)
    with pytest.raises(ValueError, match="explicitly bound"):
        solve_role_assignment(rows, cameras, {r["identity"]: None for r in rows}, COUNTS)
    assert solve_role_assignment(rows, cameras, {r["identity"]: 0 for r in rows}, COUNTS) is None


def test_optimistic_solution_validated_by_its_queries_is_exact():
    rows = fixture_rows("AABBBBBB")
    cameras = {"A": "0", "B": "1"}
    upper = {r["identity"]: 15 for r in rows}
    optimistic = solve_role_assignment(rows, cameras, upper, COUNTS)
    actual = {r["identity"]: 15 if r["identity"] in optimistic["queries"] else 0 for r in rows}
    assert solve_role_assignment(rows, cameras, actual, COUNTS) == optimistic
