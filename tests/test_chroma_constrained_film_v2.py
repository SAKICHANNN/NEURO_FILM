import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from src.color_match.research.local_reference_field_v1 import interpolate_nodes
from src.models.chroma_constrained_film_v2 import ChromaField, RadialGamutScale, apply_chroma_field, encoded_to_oklab, polynomial_rgb, render_frames, render_native, rgb_polynomial, tone_curve, train_step
from src.models.semantic_film_adversarial_v1 import PhotometricCritic


ROOT = Path(__file__).resolve().parents[1]
V2 = json.loads((ROOT / "configs/chroma_constrained_film_v2.json").read_text())
CONFIG = {**json.loads((ROOT / "configs/semantic_film_adversarial_v1.json").read_text()), **V2}


def field(values, shape=(1, 1)):
    return torch.tensor(values, dtype=torch.float64)[None, :, None, None].expand(1, 6, *shape).clone()


class ChromaConstrainedFilmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_oklab_reference_values_and_identity_roundtrip(self):
        colors = torch.tensor([[1., 1., 1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64)[:, :, None, None]
        lab = encoded_to_oklab(colors)[:, :, 0, 0]
        expected = torch.tensor([[1., 0., 0.], [.6279554, .2248631, .1258463], [.8664396, -.2338876, .1794985], [.4520137, -.0324570, -.3115281]], dtype=torch.float64)
        torch.testing.assert_close(lab, expected, atol=2e-6, rtol=0)
        torch.manual_seed(3)
        image = torch.rand(2, 3, 19, 23, dtype=torch.float64)
        image[:, :, 0, 0], image[:, :, 0, 1] = 0., 1.
        output, scale = apply_chroma_field(image, field([0.] * 6, image.shape[-2:]).expand(2, -1, -1, -1), CONFIG)
        torch.testing.assert_close(output, image, atol=2e-6, rtol=0)
        self.assertTrue(bool((output[:, :, 0, :2] == image[:, :, 0, :2]).all()))
        self.assertTrue(bool((scale == 1).all()))

    def test_gamut_scale_is_first_exit_on_dense_ray_and_output_in_cube(self):
        torch.manual_seed(5)
        image = torch.rand(1, 3, 24, 24, dtype=torch.float64)
        values = torch.tensor(CONFIG["field_bounds"]["B"], dtype=torch.float64)[None, :, None, None] * (2 * torch.rand(1, 6, 24, 24, dtype=torch.float64) - 1)
        values[:, 2] = torch.tensor(CONFIG["field_bounds"]["B"][2], dtype=torch.float64)
        output, scale = apply_chroma_field(image, values, CONFIG)
        self.assertGreater(float((scale < 1).double().mean()), .2)
        self.assertTrue(bool(((output >= 0) & (output <= 1)).all()))
        lab = encoded_to_oklab(image)
        lightness = tone_curve(lab[:, :1].clamp(0, 1), values[:, :1], values[:, 1:2])
        a, b = lab[:, 1:2], lab[:, 2:3]
        angle = values[:, 3:4]
        chroma = values[:, 2:3].exp() * torch.cat((angle.cos() * a - angle.sin() * b, angle.sin() * a + angle.cos() * b), 1) + 4 * lab[:, :1] * (1 - lab[:, :1]) * values[:, 4:6]
        q = rgb_polynomial(lightness, chroma)
        grid = torch.linspace(0, 1, 20001, dtype=torch.float64)[:, None, None, None, None]
        rgb = q[0][None] + grid * (q[1][None] + grid * (q[2][None] + grid * q[3][None]))
        feasible = ((rgb >= -1e-12) & (rgb <= 1 + 1e-12)).all(2)
        first_bad = torch.where(~feasible, torch.arange(len(grid))[:, None, None, None], len(grid)).amin(0)
        brute = torch.where(first_bad == len(grid), torch.ones_like(scale[:, 0], dtype=torch.float64), (first_bad - 1).clamp_min(0) / (len(grid) - 1))
        torch.testing.assert_close(scale[:, 0], brute, atol=1.1 / (len(grid) - 1), rtol=0)

    def test_implicit_gamut_gradient_matches_finite_difference(self):
        torch.manual_seed(7)
        lightness = (.3 + .4 * torch.rand(1, 1, 3, 3, dtype=torch.float64))
        chroma = .6 * torch.randn(1, 2, 3, 3, dtype=torch.float64)
        q = [value.detach().requires_grad_() for value in rgb_polynomial(lightness, chroma)]
        args = (CONFIG["gamut_iterations"] + 30, 0., 1e-14)
        scale = RadialGamutScale.apply(*q, *args)
        self.assertTrue(bool((scale < 1).all()))
        weights = torch.rand_like(scale)
        gradients = torch.autograd.grad((scale * weights).sum(), q)
        for index in range(4):
            for channel in range(3):
                delta = torch.zeros_like(q[index])
                delta[:, channel] = 1e-7
                plus = RadialGamutScale.apply(*[value.detach() + (delta if i == index else 0) for i, value in enumerate(q)], *args)
                minus = RadialGamutScale.apply(*[value.detach() - (delta if i == index else 0) for i, value in enumerate(q)], *args)
                numeric = ((plus - minus) * weights).sum((1,)) / 2e-7
                torch.testing.assert_close(gradients[index][:, channel], numeric, atol=2e-5, rtol=1e-3)

    def test_clipped_white_black_and_primaries_backward_without_nonregular_root(self):
        colors = torch.tensor([[1., 1., 1.], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.], [.9999, 1., 1.], [1., 1., .998], [.0001, 0., 0.]])
        for dtype in (torch.float32, torch.float64):
            image = colors.T[None, :, :, None].to(dtype).expand(1, 3, 9, 4).contiguous()
            for arm in ("A", "B"):
                for sign in (-1., 1.):
                    values = (sign * torch.tensor(CONFIG["field_bounds"][arm], dtype=dtype))[None, :, None, None].expand(1, 6, 9, 4).clone().requires_grad_()
                    output, scale = apply_chroma_field(image, values, CONFIG)
                    output.sum().backward()
                    self.assertTrue(bool(torch.isfinite(values.grad).all()))
                    self.assertTrue(bool(((output >= 0) & (output <= 1)).all()))
                    self.assertEqual(float(scale[..., 0, :].detach().min()), 1.)
                    torch.testing.assert_close(output[:, :, :2], image[:, :, :2], atol=0, rtol=0)

    def test_float32_near_cube_surface_small_fields_stay_within_roundoff_contract(self):
        torch.manual_seed(23)
        levels = torch.tensor([0., 1 / 255, 2 / 255, .0999, .2368, .5216, .98, 254 / 255, 1.])
        grid = torch.stack(torch.meshgrid(levels, levels, levels, indexing="ij")).reshape(1, 3, 27, 27)
        image = torch.cat((grid, torch.tensor([0., .2368, .5216])[None, :, None, None].expand(1, 3, 27, 27), torch.tensor([.055, .0999, 8.6e-5])[None, :, None, None].expand(1, 3, 27, 27)), 2).float()
        for arm in ("A", "B"):
            bounds = torch.tensor(CONFIG["field_bounds"][arm])[None, :, None, None]
            for strength in (0., 1e-4, 1e-3, 2e-3, 1e-2):
                for _ in range(20):
                    values = strength * bounds * (2 * torch.rand(1, 6, 81, 27) - 1)
                    output, _ = apply_chroma_field(image, values, CONFIG)
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(bool(((output >= 0) & (output <= 1)).all()))

    def test_tone_monotone_endpoints_and_neutral_without_tint(self):
        lightness = torch.linspace(0, 1, 4001, dtype=torch.float64)[None, None, :, None]
        for bias in (-2., 0., 2.):
            for slope in (-1., 1.):
                curve = tone_curve(lightness, torch.tensor(bias, dtype=torch.float64), torch.tensor(slope, dtype=torch.float64))
                self.assertTrue(bool((curve.diff(dim=2) > 0).all()))
                self.assertEqual(float(curve[..., 0, 0]), 0.)
                self.assertEqual(float(curve[..., -1, 0]), 1.)
        gray = torch.linspace(.05, .95, 13, dtype=torch.float64)[None, None, :, None].expand(1, 3, -1, 1)
        bounds = CONFIG["field_bounds"]["B"]
        output, _ = apply_chroma_field(gray, field([bounds[0], -bounds[1], bounds[2], bounds[3], 0., 0.], gray.shape[-2:]), CONFIG)
        torch.testing.assert_close(output[:, 0], output[:, 1], atol=3e-6, rtol=0)
        torch.testing.assert_close(output[:, 1], output[:, 2], atol=3e-6, rtol=0)
        tinted, _ = apply_chroma_field(gray, field([0., 0., 0., 0., bounds[4], 0.], gray.shape[-2:]), CONFIG)
        self.assertGreater(float((tinted[:, 0] - tinted[:, 1]).abs().max()), .02)

    def test_native_tiles_match_frame_render_and_report_gamut(self):
        torch.manual_seed(9)
        image = torch.rand(1, 3, 41, 57)
        nodes = torch.tensor(CONFIG["field_bounds"]["A"])[None, :, None, None] * (2 * torch.rand(1, 6, 5, 6) - 1)
        expected, _ = apply_chroma_field(image, interpolate_nodes(nodes, (41, 57), (41, 57)), CONFIG)
        actual, stats = render_native(image, nodes, {**CONFIG, "native_tile_size": 16})
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
        framed, scales = render_frames(torch.nn.functional.pad(image, (3, 4, 2, 5)), nodes, [(2, 3, 41, 57)], CONFIG)
        torch.testing.assert_close(framed[..., 2:43, 3:60], expected, atol=2e-6, rtol=0)
        self.assertEqual(float(framed[..., :2, :].abs().max()), 0.)
        self.assertTrue(0 <= stats["gamut_scaled_fraction"] <= 1 and 0 < stats["minimum_chroma_retention"] <= 1)

    def test_arm_bounds_identity_init_and_only_chroma_bounds_differ(self):
        config = {**CONFIG, "feature_dim": 8, "hidden_channels": 12, "grid_size": 4}
        a, b = ChromaField(config, "A"), ChromaField(config, "B")
        bound_a, bound_b = a.bounds.flatten(), b.bounds.flatten()
        torch.testing.assert_close(bound_a[:2], bound_b[:2])
        torch.testing.assert_close(bound_b[2:], 3 * bound_a[2:])
        images, masks, boxes, features = torch.rand(2, 3, 56, 56), torch.ones(2, 1, 56, 56), [(0, 0, 56, 56)] * 2, torch.randn(2, 8, 4, 4)
        self.assertEqual(float(a(images, masks, boxes, features).detach().abs().max()), 0.)
        with torch.no_grad():
            b.head.bias.fill_(50.)
        nodes = b(images, masks, boxes, features)
        self.assertTrue(bool((nodes.abs() <= b.bounds + 1e-7).all()))

    def test_adversarial_minitrain_updates_depend_on_target_and_stay_bounded(self):
        config = {**CONFIG, "feature_dim": 8, "hidden_channels": 12, "critic_channels": 8, "grid_size": 4, "clusters": 2, "generator_lr": .003, "critic_lr": .003}
        torch.manual_seed(12)
        source = .2 + .5 * torch.rand(2, 3, 84, 84)
        mask, boxes = torch.ones(2, 1, 84, 84), [(0, 0, 84, 84)] * 2
        features, centers, labels = torch.randn(2, 8, 6, 6), torch.tensor([[42, 42]] * 2), torch.tensor([0, 1])
        targets = [source.clone(), source.clone()]
        targets[0][:, 0] = (targets[0][:, 0] + .23).clamp_max(.98)
        targets[1][:, 2] = (targets[1][:, 2] + .23).clamp_max(.98)
        results = []
        for target in targets:
            torch.manual_seed(19)
            generator, critic = ChromaField(config, "A"), PhotometricCritic(config, True)
            initial = copy.deepcopy(generator.state_dict())
            go = torch.optim.Adam(generator.parameters(), lr=config["generator_lr"], betas=(0., .99))
            co = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"], betas=(0., .99))
            for step in range(24):
                terms = train_step(generator, critic, go, co, source, mask, boxes, features, centers, target, centers, labels, step, config)
                self.assertTrue(all(np.isfinite(value) for value in terms.values()))
            self.assertGreater(terms["generator_gradient_norm"], 0)
            self.assertFalse(torch.equal(generator.head.weight, initial["head.weight"]))
            nodes = generator(source, mask, boxes, features)
            self.assertTrue(bool((nodes.abs() <= generator.bounds + 1e-7).all()))
            results.append(render_frames(source, nodes, boxes, config)[0].detach())
        self.assertGreater(float((results[0] - results[1]).abs().mean()), 1e-4)
        self.assertGreater(float((results[0] - source).abs().mean()), 1e-4)


if __name__ == "__main__":
    unittest.main()
