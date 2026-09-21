import json
from pathlib import Path

import torch

from src.color_match.research.local_reference_field_v1 import (
    apply_field, bounded_nodes, centers_for, descriptors, interpolate_nodes,
    load_descriptor, match_references, objective, projected_quantiles,
    reference_threshold, spline_basis, support_field, target_permutation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/local_reference_field_v1.json").read_text())


def test_operator_identity_endpoints_color_change_and_gradient():
    torch.manual_seed(21)
    x = torch.rand(1, 3, 13, 17, dtype=torch.float64)
    x[:, :, 0] = 0
    x[:, :, 1] = 1
    assert torch.equal(apply_field(x, torch.zeros(1, 12, 13, 17, dtype=x.dtype)), x)
    raw = torch.randn(1, 12, 4, 4, dtype=x.dtype, requires_grad=True)
    nodes = bounded_nodes(raw, CONFIG)
    field = interpolate_nodes(nodes, (13, 17), (13, 17)).double()
    y = apply_field(x, field)
    assert torch.equal(y[:, :, :2], x[:, :, :2])
    assert bool(((y >= 0) & (y <= 1) & torch.isfinite(y)).all())
    assert float((y - x).abs().mean()) > 0.02
    y.square().mean().backward()
    assert bool(torch.isfinite(raw.grad).all())
    matrix = field[:, 3:].reshape(1, 3, 3, 13, 17)
    assert float(matrix.abs().sum(2).max()) <= CONFIG["matrix_row_l1_bound"] + 1e-6
    assert float(field[:, :3].abs().max()) <= CONFIG["bias_bound"] + 1e-6


def test_frozen_field_logit_jacobian_not_input_conditioning_claim():
    torch.manual_seed(3)
    nodes = bounded_nodes(torch.randn(1, 12, 1, 1, dtype=torch.float64), CONFIG)
    b = nodes[0, :3, 0, 0]
    a = nodes[0, 3:, 0, 0].reshape(3, 3)
    z = torch.tensor([-0.8, 0.5, 1.3], dtype=torch.float64, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda q: q + b + a @ (2 * q.sigmoid() - 1), z)
    residual = jac - torch.eye(3, dtype=jac.dtype)
    assert float(residual.abs().sum(1).max()) <= 0.5 + 1e-9
    assert float(torch.linalg.det(jac)) > 0


def test_spline_partition_bounds_and_native_coordinate_consistency():
    basis = spline_basis(torch.linspace(-50, 200, 501), 3, 150)
    assert bool((basis >= 0).all())
    assert torch.allclose(basis.sum(1), torch.ones(501), atol=2e-7)
    constant = torch.full((1, 12, 4, 4), 0.37)
    working = interpolate_nodes(constant, (150, 150), (150, 150))
    native = interpolate_nodes(constant, (300, 300), (150, 150))
    assert torch.allclose(working, torch.full_like(working, 0.37), atol=2e-7)
    assert torch.allclose(native, torch.full_like(native, 0.37), atol=2e-7)
    empty = torch.empty(0, 2)
    support = support_field((100, 80), (50, 40), empty, torch.empty(0), CONFIG)
    assert torch.equal(support, torch.zeros_like(support))


def test_distinct_reference_support_reciprocity_and_shuffled_descriptor_control():
    source = torch.eye(4)
    reference = torch.cat([source for _ in range(4)])
    files = torch.arange(4).repeat_interleave(4)
    target = torch.arange(16, dtype=torch.float32)[:, None, None].expand(16, 13, 5)
    matched = match_references(source, reference, files, target, 0.2, CONFIG)
    assert matched["selected"].tolist() == [0, 1, 2, 3]
    for record in matched["records"]:
        assert len({r["reference_file_index"] for r in record["candidates"]}) == 4
    reversed_match = match_references(source.flip(0), reference, files, target, 0.2, CONFIG)
    assert torch.equal(reversed_match["targets"], matched["targets"].flip(0))
    duplicate_file = match_references(source, reference, torch.zeros_like(files), target, 0.2, CONFIG)
    assert duplicate_file["selected"].numel() == 0
    assert reference_threshold(reference, torch.zeros_like(files), CONFIG) == 0
    centers = torch.tensor([[0., 0.], [1., 0.], [10., 0.], [11., 0.]])
    permutation = target_permutation(centers)
    assert sorted(permutation.tolist()) == [0, 1, 2, 3]
    assert bool((permutation != torch.arange(4)).all())
    assert permutation.tolist() == [2, 3, 0, 1]


def test_objective_gradient_and_local_vs_global_same_color_probe():
    torch.manual_seed(8)
    x = 0.1 + 0.8 * torch.rand(1, 3, 64, 64)
    centers = torch.tensor([[15., 15.], [48., 48.]])
    target = projected_quantiles((x * 0.8 + 0.1), centers, CONFIG).detach()
    raw = torch.zeros(1, 12, 3, 3, requires_grad=True)
    nodes = bounded_nodes(raw, CONFIG)
    y = apply_field(x, interpolate_nodes(nodes, (64, 64), (64, 64)))
    loss, terms = objective(x, y, nodes, centers, target, torch.ones(2), CONFIG)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0
    assert set(terms) == {"style", "coefficient", "spatial", "edge", "shadow"}
    with torch.no_grad():
        raw[0, 0, :, 0] = -1
        raw[0, 0, :, -1] = 1
    field = interpolate_nodes(bounded_nodes(raw, CONFIG), (64, 64), (64, 64))
    palette = torch.full_like(x, 0.5)
    local = apply_field(palette, field)
    assert float((local[..., 0] - local[..., -1]).abs().max()) > 0.1
    global_field = interpolate_nodes(bounded_nodes(raw.mean((-1, -2), keepdim=True), CONFIG), (64, 64), (64, 64))
    global_result = apply_field(palette, global_field)
    assert torch.equal(global_result[..., 0], global_result[..., -1])


def test_pinned_full_vgg_embedded_preprocessing_and_descriptor_smoke():
    torch.set_num_threads(4)
    model = load_descriptor(ROOT, CONFIG, torch.device("cpu"))
    value = torch.tensor([0.2, 0.4, 0.6])[None, :, None, None]
    expected = torch.tensor([0.6 * 255 - 103.939, 0.4 * 255 - 116.779, 0.2 * 255 - 123.680])[None, :, None, None]
    assert torch.allclose(model[0](value), expected, atol=2e-5, rtol=0)
    torch.manual_seed(11)
    x = torch.rand(1, 3, 128, 128)
    centers = centers_for(x, 128)
    feature = descriptors(x, centers, model, CONFIG)
    assert feature.shape == (1, (256 + 512) * 4)
    assert torch.isfinite(feature).all()
    assert torch.allclose(feature.norm(dim=1), torch.ones(1), atol=1e-6)
