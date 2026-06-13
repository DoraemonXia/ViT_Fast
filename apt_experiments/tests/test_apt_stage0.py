import unittest
import os
import sys

import torch

APT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APT_ROOT)
sys.path.insert(0, APT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from Hierarchical_16_32_Learned_APT.train import (
    HierarchicalAPTViT,
    TokenStatsAccumulator,
    build_key_attention_mask,
    compute_patch_entropy,
    denormalize_to_255,
)
from datasets import get_cifar100_loader


class APTUtilityTests(unittest.TestCase):
    def test_entropy_orders_constant_checkerboard_and_noise(self):
        torch.manual_seed(7)
        constant = torch.full((1, 3, 32, 32), 128.0)
        checker = torch.arange(32).view(1, 1, 1, 32)
        checker = ((checker + checker.transpose(-1, -2)) % 2).float() * 255
        checker = checker.expand(1, 3, 32, 32)
        noise = torch.randint(0, 256, (1, 3, 32, 32)).float()

        images = torch.cat([constant, checker, noise], dim=0)
        entropy = compute_patch_entropy(images, patch_sizes=(16,), bins=64)[16]
        means = entropy.mean(dim=(1, 2))

        self.assertLess(means[0].item(), 1e-6)
        self.assertGreater(means[1].item(), means[0].item())
        self.assertGreater(means[2].item(), means[1].item())

    def test_denormalize_round_trip(self):
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        pixels = torch.rand(2, 3, 8, 8)
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        normalized = (pixels - mean_t) / std_t
        restored = denormalize_to_255(normalized, mean, std) / 255.0
        torch.testing.assert_close(restored, pixels)

    def test_attention_mask_shape_and_semantics(self):
        valid = torch.tensor([[True, True, False], [True, False, False]])
        mask = build_key_attention_mask(valid)
        self.assertEqual(tuple(mask.shape), (2, 1, 1, 3))
        torch.testing.assert_close(mask[:, 0, 0], valid)

    def test_token_stats_tracks_real_and_padded_lengths(self):
        stats = TokenStatsAccumulator()
        stats.update(torch.tensor([3, 5]), padded_length=5)
        stats.update(torch.tensor([4]), padded_length=4)
        result = stats.compute()
        self.assertEqual(result.minimum, 3)
        self.assertEqual(result.maximum, 5)
        self.assertAlmostEqual(result.mean, 4.0)
        self.assertAlmostEqual(result.padded_mean, 14 / 3, places=6)


class APTModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(11)
        cls.images = torch.randn(2, 3, 224, 224)

    def _assert_batch_consistency(self, model):
        model.eval()
        with torch.no_grad():
            single = model(self.images[:1])
            mixed = model(self.images)[:1]
        torch.testing.assert_close(single, mixed, atol=1e-5, rtol=1e-5)
        self.assertIsNotNone(model._last_token_counts)

    def test_a4_learned_token_counts_are_bounded(self):
        model = HierarchicalAPTViT(
            num_classes=10,
            thresholds={32: 4.0},
            patch_sizes=(16, 32),
            aggregation="learned",
            pretrained=False,
            backbone_name="vit_tiny_patch16_224",
        )
        self._assert_batch_consistency(model)
        self.assertTrue(
            torch.all(model._last_token_counts >= 49).item()
        )
        self.assertTrue(
            torch.all(model._last_token_counts <= 196).item()
        )

    def test_a3_average_backward_smoke(self):
        model = HierarchicalAPTViT(
            num_classes=10,
            thresholds={32: 4.0},
            patch_sizes=(16, 32),
            aggregation="average",
            pretrained=False,
            backbone_name="vit_tiny_patch16_224",
        )
        logits = model(self.images[:1])
        logits.sum().backward()
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)


class DatasetSplitTests(unittest.TestCase):
    def test_cifar_validation_is_separate_and_deterministic(self):
        first = get_cifar100_loader(
            batch_size=4,
            data_dir="./data",
            num_workers=0,
            return_val=True,
            val_ratio=0.1,
            split_seed=42,
        )
        second = get_cifar100_loader(
            batch_size=4,
            data_dir="./data",
            num_workers=0,
            return_val=True,
            val_ratio=0.1,
            split_seed=42,
        )
        train_a, val_a, test_a, _ = first
        train_b, val_b, test_b, _ = second
        self.assertEqual(len(train_a.dataset), 45000)
        self.assertEqual(len(val_a.dataset), 5000)
        self.assertEqual(len(test_a.dataset), 10000)
        self.assertEqual(train_a.dataset.indices, train_b.dataset.indices)
        self.assertEqual(val_a.dataset.indices, val_b.dataset.indices)
        self.assertEqual(len(test_a.dataset), len(test_b.dataset))


if __name__ == "__main__":
    unittest.main()
