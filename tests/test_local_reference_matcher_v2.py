import torch

from scripts.calibrate_local_reference_matcher_v2 import reciprocal_third_scores


def test_leave_one_file_out_third_order_excludes_self_file():
    angles = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    features = torch.stack([angles.cos(), angles.sin()], 1)
    files = torch.arange(5)
    scores, rows = reciprocal_third_scores(features, files)
    distances = (1 - features @ features.T).clamp_min(0)
    distances.fill_diagonal_(float("inf"))
    expected = distances.sort(1).values[:, 2]
    assert torch.equal(scores, expected)
    assert len(rows) == 5 and all(row["finite_third_scores"] == 1 for row in rows)


def test_reciprocity_uses_left_out_query_pool_and_distinct_files():
    features = torch.tensor([[1., 0.]] * 5 + [[1., 0.]] * 4)
    files = torch.tensor([0] * 5 + [1, 2, 3, 4])
    scores, rows = reciprocal_third_scores(features, files, reciprocal_k=3)
    assert rows[0]["finite_third_scores"] == 3
    assert int(torch.isfinite(scores[:5]).sum()) == 3
    assert all(row["finite_third_scores"] == 1 for row in rows[1:])
    insufficient_files = torch.tensor([0] * 5 + [1] * 4)
    invalid, _ = reciprocal_third_scores(features, insufficient_files)
    assert not bool(torch.isfinite(invalid).any())


def test_file_relabeling_preserves_calibration_distribution():
    torch.manual_seed(12)
    features = torch.nn.functional.normalize(torch.rand(24, 9), dim=1)
    files = torch.arange(6).repeat_interleave(4)
    before, _ = reciprocal_third_scores(features, files)
    after, _ = reciprocal_third_scores(features, 5 - files)
    assert torch.equal(before.sort().values, after.sort().values)
