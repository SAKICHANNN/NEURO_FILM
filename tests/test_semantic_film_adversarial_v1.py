import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.color_match.research.local_reference_field_v1 import apply_field, bounded_nodes, interpolate_nodes
from src.models.semantic_film_adversarial_v1 import PhotometricCritic, SemanticField, balanced_schedule, dino_input, patch_candidates, photometric_patches, render_frames, render_native, token_mask, train_step, valid_features
from src.models.source_preparation_v1 import preview_frame


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/semantic_film_adversarial_v1.json").read_text())


class SemanticFilmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_renderer_exact_reuse_native_tiles_range_and_endpoints(self):
        torch.manual_seed(2)
        image = torch.rand(2, 3, 57, 83)
        image[:, :, 0, :2] = torch.tensor([0., 1.])
        nodes = bounded_nodes(torch.randn(2, 12, 5, 7) * 3, CONFIG)
        expected = apply_field(image, interpolate_nodes(nodes, image.shape[-2:], (224, 196)))
        actual = render_native(image, nodes, 19)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-6)
        self.assertEqual(actual.shape, image.shape)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertGreaterEqual(float(actual.min()), 0)
        self.assertLessEqual(float(actual.max()), 1)
        torch.testing.assert_close(actual[:, :, 0, :2], image[:, :, 0, :2])
        torch.testing.assert_close(render_native(image, torch.zeros_like(nodes), 19), image)

    def test_stride14_valid_sampling_pool_and_padding(self):
        image = torch.rand(1, 3, 100, 200)
        frame, mask, box = preview_frame(image, 224)
        features = torch.full((1, 2, 16, 16), 10000.)
        valid = token_mask(mask)
        for y in range(16):
            for x in range(16):
                if valid[0, 0, y, x]:
                    features[0, :, y, x] = torch.tensor([14 * y + 6.5, 14 * x + 6.5])
        local, pooled = valid_features(features, mask, [box], 4)
        ys, xs = torch.where(valid[0, 0])
        torch.testing.assert_close(pooled[0, :, 0, 0], torch.stack(((ys.float() * 14 + 6.5).mean(), (xs.float() * 14 + 6.5).mean())))
        t, l, h, w = box
        unit = (torch.arange(4) + .5) / 4
        yy = (t + h * unit - .5).clamp(float(ys.min() * 14 + 6.5), float(ys.max() * 14 + 6.5))
        xx = (l + w * unit - .5).clamp(float(xs.min() * 14 + 6.5), float(xs.max() * 14 + 6.5))
        torch.testing.assert_close(local[0, 0], yy[:, None].expand(4, 4))
        torch.testing.assert_close(local[0, 1], xx[None].expand(4, 4))
        candidates = patch_candidates([box], 224, 14, 64)
        for _, _, _, y, x in candidates:
            self.assertEqual(float(mask[..., y - 32:y + 32, x - 32:x + 32].min()), 1)

    def test_color_and_critic_preprocessing_contract(self):
        image = torch.full((2, 3, 84, 84), .25)
        mask = torch.ones(2, 1, 84, 84)
        normalized = dino_input(image, mask)
        mean = torch.tensor([.485, .456, .406])[None, :, None, None]
        std = torch.tensor([.229, .224, .225])[None, :, None, None]
        torch.testing.assert_close(normalized * std + mean, image)
        centers = torch.tensor([[42, 42], [42, 42]])
        patches = photometric_patches(image, centers, CONFIG)
        self.assertEqual(patches.shape, (2, 3, 16, 16))
        torch.testing.assert_close(patches, torch.full_like(patches, .25))
        image.requires_grad_()
        photometric_patches(image, centers, CONFIG).sum().backward()
        self.assertGreater(float(image.grad.abs().sum()), 0)

    def test_explicit_exif_orientation8_without_color_change(self):
        from scripts.run_semantic_film_adversarial_v1 import canonical_film_pixels
        array = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        image = Image.fromarray(array)
        image.getexif()[274] = 8
        oriented, metadata = canonical_film_pixels(image)
        np.testing.assert_array_equal(oriented, np.rot90(array))
        self.assertEqual(metadata["exif_orientation"], 8)
        self.assertEqual(metadata["decoded_shape_hwc"], [6, 4, 3])
        self.assertEqual(metadata["color_action"], "assume_untagged_sRGB")

    def test_cluster_balance_seed_pairing_and_empty_support(self):
        config = {**CONFIG, "steps": 6, "batch_size": 4, "clusters": 3}
        source = np.asarray([[i, 3, 3, 49, 49, k] for i in range(4) for k in range(3)])
        reference = np.asarray([[i, 3, 3, 49, 49, k] for i in range(6) for k in range(3)])
        collections = np.arange(6) % 3
        first, report = balanced_schedule(source, reference, collections, config, 9)
        second, _ = balanced_schedule(source, reference, collections, config, 9)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["source"][..., -1], first["reference"][..., -1])
        np.testing.assert_array_equal(collections[first["reference"][..., 0]], first["collections"])
        self.assertEqual(report["collection_counts"], [8, 8, 8])
        self.assertEqual(report["source_cluster_counts"], report["reference_cluster_counts"])
        broken = reference.copy()
        broken[collections[broken[:, 0]] == 2, -1] = 9
        with self.assertRaisesRegex(ValueError, "NO_SUPPORT collection 2"):
            balanced_schedule(source, broken, collections, config, 9)

    def test_real_adversarial_minitrain_changes_with_target_corpus(self):
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
            generator, critic = SemanticField(config), PhotometricCritic(config, True)
            initial = copy.deepcopy(generator.state_dict())
            go = torch.optim.Adam(generator.parameters(), lr=config["generator_lr"], betas=(0., .99))
            co = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"], betas=(0., .99))
            for step in range(24):
                terms = train_step(generator, critic, go, co, source, mask, boxes, features, centers, target, centers, labels, step, config)
                self.assertTrue(all(np.isfinite(value) for value in terms.values()))
            self.assertGreater(terms["generator_gradient_norm"], 0)
            self.assertTrue(all(not torch.equal(value, initial[key]) for key, value in generator.state_dict().items()))
            original_nodes = generator(source, mask, boxes, features)
            changed_rgb = source.clone()
            changed_rgb[:, 1] *= .7
            self.assertGreater(float((generator(changed_rgb, mask, boxes, features) - original_nodes).abs().max().detach()), 1e-6)
            results.append(render_frames(source, generator(source, mask, boxes, features), boxes).detach())
        self.assertGreater(float((results[0] - results[1]).abs().mean()), .0001)
        self.assertGreater(float((results[0] - source).abs().mean()), .0001)

    def test_critic_equal_initial_state_and_zero_conditioning(self):
        torch.manual_seed(11)
        conditional = PhotometricCritic(CONFIG, True).eval()
        torch.manual_seed(11)
        control = PhotometricCritic(CONFIG, False).eval()
        self.assertEqual(sum(p.numel() for p in conditional.parameters()), sum(p.numel() for p in control.parameters()))
        for key, value in conditional.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[key])
        image = torch.rand(2, 3, 16, 16)
        torch.testing.assert_close(control(image, torch.tensor([0, 1])), control(image, torch.tensor([2, 3])))


if __name__ == "__main__":
    unittest.main()
