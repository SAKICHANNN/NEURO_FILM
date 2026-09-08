import copy
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from src.eval import tst_reference_distillation as net
from src.eval import tst_reference_response as old


def samples():
    rng = np.random.default_rng(48)
    a, b = rng.random((13, 3)), rng.random((11, 3))
    xs = [a, a.copy(), b]
    ys = [x*.7+.1+rng.normal(0, .02, x.shape) for x in xs]
    ps = [rng.random((n, 3)) for n in (9, 15, 12)]
    rs = [p*.8+.05 for p in ps]
    return xs, ys, ps, rs


def direct_loss(u, sources, targets, paired):
    errors = [torch.as_tensor(x-y)+torch.as_tensor(old.lattice_design(x, 3).toarray())@v
              for x, y, v in zip(sources, targets, u, strict=True)]
    if paired:
        first = ((errors[0]+errors[1])/2).square().mean()+2*((errors[0]-errors[1])/2).square().mean()
    else:
        first = (errors[0].square().mean()+errors[1].square().mean())/2
    return (first+errors[2].square().mean())/2


@pytest.mark.parametrize('paired', [False, True])
def test_pixel_quadratic_direct_value_gradient_and_finite_difference(paired):
    torch.set_num_threads(2)
    xs, ys, ps, rs = samples()
    source, target = (xs, ys) if paired else (ps, rs)
    quadratic = net.PixelQuadratic(source, target, [0, 0, 1], 3, paired)
    u = torch.randn(3, 27, 3, dtype=torch.float64, requires_grad=True)*.1
    actual, expected = quadratic(u), direct_loss(u, source, target, paired)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    ga = torch.autograd.grad(actual, u)[0]
    ge = torch.autograd.grad(expected, u)[0]
    torch.testing.assert_close(ga, ge, rtol=1e-11, atol=1e-14)
    direction = torch.randn_like(u)
    eps = 1e-6
    fd = (quadratic(u+eps*direction)-quadratic(u-eps*direction))/(2*eps)
    torch.testing.assert_close(fd, (ga*direction).sum(), rtol=1e-8, atol=1e-10)
    with pytest.raises(AssertionError):
        net.PixelQuadratic(ps, rs, [0, 0, 1], 3, True)


def test_full_three_task_network_gradient_smoothness_and_kd_normalization():
    torch.manual_seed(48)
    xs, ys, ps, rs = samples()
    query = net.PixelQuadratic(xs, ys, [0, 0, 1], 3, True)
    reference = net.PixelQuadratic(ps, rs, [0, 0, 1], 3, False)
    data = {k: torch.randn(3, 4, dtype=torch.float64) for k in ('X', 'Y', 'P', 'R')}
    data.update(sX=torch.randn(3, 2, dtype=torch.float64), sP=torch.randn(3, 2, dtype=torch.float64))
    data['X'][1] = data['X'][0]
    data['sX'][1] = data['sX'][0]
    weights = torch.tensor(net.row_weights([0, 0, 1]), dtype=torch.float64)
    config = {'architecture': {'grid': 3}, 'smoothness': 1e-6, 'kd_weight': 0.}
    for paired in (True, False):
        encoder = net.Encoder(8 if paired else 4, 5, 3).double()
        decoder = net.Decoder(2, 3, 3).double()
        actual, _ = net.objective(encoder, decoder, data, query, reference, weights, config, paired=paired)
        zr = net.paired_code(encoder, data['P'], data['R']) if paired else encoder(data['R'])
        zy = net.paired_code(encoder, data['X'], data['Y']) if paired else encoder(data['Y'])
        ur, uy, up = decoder(data['sX'], zr), decoder(data['sX'], zy), decoder(data['sP'], zr)
        derivative = torch.as_tensor(old.second_derivatives(3).toarray())
        penalties = torch.stack([torch.stack([(derivative@u).square().mean() for u in grid]) for grid in (ur, uy, up)]).mean(0)
        expected = .5*direct_loss(ur, xs, ys, True)+.5*direct_loss(uy, xs, ys, True)+direct_loss(up, ps, rs, False)+1e-6*(weights*penalties).sum()
        parameters = list(encoder.parameters())+list(decoder.parameters())
        ga = torch.autograd.grad(actual, parameters, retain_graph=True)
        ge = torch.autograd.grad(expected, parameters)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-14)
        for a, e in zip(ga, ge, strict=True):
            torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-12)
    zr = torch.ones(3, 16, requires_grad=True)
    teacher = torch.zeros(3, 16, requires_grad=True)
    kd = net.distillation_loss(zr, zr, teacher, teacher, weights)
    assert kd.item() == 16.
    kd.backward()
    assert teacher.grad is None and zr.grad is not None


def test_initialization_frozen_decoder_and_reference_only_swap():
    cfg = {'seed': 20260908, 'architecture': {'feature': 4, 'hidden': 5, 'code': 3, 'source': 2, 'grid': 3, 'epsilon': 1e-12}}
    teacher, decoder, student, di, si = net.initialize(cfg)
    assert all(torch.equal(v, decoder.state_dict()[k]) for k, v in di.items())
    assert all(torch.equal(v, student.state_dict()[k]) for k, v in si.items())
    s0, direct = copy.deepcopy(student), copy.deepcopy(student)
    assert all(torch.equal(v, s0.state_dict()[k]) and torch.equal(v, direct.state_dict()[k]) for k, v in si.items())
    decoder.requires_grad_(False)
    source = torch.randn(1, 2).repeat(2, 1)
    refs = torch.randn(2, 4)
    values = net.after_only(student, decoder, source, refs)
    torch.testing.assert_close(net.after_only(student, decoder, source, refs.flip(0)), values.flip(0), rtol=0, atol=0)
    values.square().sum().backward()
    assert all(p.grad is None for p in decoder.parameters())
    assert any(p.grad is not None for p in student.parameters())
    assert all(torch.equal(v, decoder.state_dict()[k]) for k, v in di.items())
    assert list(inspect.signature(net.after_only).parameters) == ['encoder', 'decoder', 'source_scores', 'reference_features']
    assert sum(p.numel() for p in net.Encoder(7996).parameters()) == 512848
    assert sum(p.numel() for p in net.Encoder().parameters()) == 256976
    assert sum(p.numel() for p in net.Decoder().parameters()) == 122451
    assert torch.isfinite(teacher(torch.zeros(2, 8))).all()


def entry():
    path = Path(__file__).resolve().parents[1]/'scripts/run_tst_reference_distillation.py'
    spec = importlib.util.spec_from_file_location('distillation_entry_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prediction_stage_keys_and_no_privileged_dependency(monkeypatch, tmp_path):
    module = entry()
    rows = [{'group': g, 'target': j} for g in range(24, 32) for j in (1, 2)]
    cfg = {'development_rows': rows, 'fit_rows': [], 'feature_caches': ['forbidden_train', 'xr_only'],
           'old_model': 'normalization', 'source_crops': 'crops', 'output': 'unused', 'preview_strengths': [1., .8]}
    assert len(module.stage_keys(cfg, 'predict')) == 24
    assert all('_P' not in k and '_Y' not in k for k in module.stage_keys(cfg, 'predict'))
    seen = []
    monkeypatch.setattr(module, 'load_features', lambda paths: {'24_X_stats': np.zeros(30)})
    monkeypatch.setattr(module, 'transformed', lambda old, cache, keys: (torch.ones(len(keys), 4), torch.ones(len(keys), 2)))
    monkeypatch.setattr(module, 'load_json', lambda path: {'groups': [{'regions': {}} for _ in range(8)]})
    def load_arm(cfg, arm):
        assert arm in ('S', 'S0', 'D')
        seen.append(arm)
        return net.Encoder(4, 5, 3), net.Decoder(2, 3, 7)
    monkeypatch.setattr(module, 'load_arm', load_arm)
    def render(cfg, directory, image_key, values, name, strengths, legacy, boxes):
        assert image_key.endswith('_X')
        for s in strengths:
            (directory/f'{name}_{s}.tif').write_bytes(b'synthetic')
    monkeypatch.setattr(module, 'render_native', render)
    monkeypatch.setattr(module.frozen, 'digest', lambda path: 'synthetic')
    class Legacy:
        def load_model(self, path):
            return {}

        def save(self, path, value):
            pass
    result = module.predict(cfg, tmp_path, Legacy())
    assert seen == ['S', 'S0', 'D']
    assert result['development_P_Y_decodes'] == 0 and result['teacher_weights_loaded'] is False
    assert len(result['native_outputs']) == 96


def test_cumulative_budget_never_resets():
    module = entry()
    cfg = {'remaining_cpu_seconds': 1789.84375, 'remaining_wall_seconds': 1787.61}
    reports = [{'accounted_cpu_seconds': 90., 'supervisor': {'wall_seconds': 92.}},
               {'accounted_cpu_seconds': 1400., 'supervisor': {'wall_seconds': 1410.}}]
    assert module.remaining_budget(cfg, reports) == (299.84375, 285.6099999999999)
    reports.append({'accounted_cpu_seconds': 300., 'supervisor': {'wall_seconds': 290.}})
    cpu, wall = module.remaining_budget(cfg, reports)
    assert cpu < 0 and wall < 0


def test_protocol_covers_all_required_sources_and_references_without_running():
    module = entry()
    cfg = module.load_json(module.CONFIG)
    required = {k for stage in module.STAGES for k in module.stage_keys(cfg, stage)}
    assert required == set(cfg['images']) and len(required) == 208
    assert len(module.stage_keys(cfg, 'fit_features')) == 86
    assert len(module.stage_keys(cfg, 'diagnostic_features')) == 32
    assert cfg['gates']['development_joint'] == 6
    assert cfg['gates']['fit_Q'] == .15532116321850929
