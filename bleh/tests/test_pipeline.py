"""
Automated Unit Tests for Off-Road Terrain Traversability Pipeline.
Tests:
- Fast-SCNN Architecture & Forward Pass
- Combined Loss (CE + Dice) & Backward Gradient Propagation
- Metrics & mIoU Calculation
- Synthetic Dataset Generation
- Costmap & Colored Overlay Generation
"""

import os
import sys
import unittest
import numpy as np
import torch

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import FastSCNN, build_model
from data import (
    CLASS_NAMES,
    COLOR_PALETTE,
    NAV2_COST_MAPPING,
    SyntheticOffRoadDataset,
    mask_to_color,
    mask_to_nav2_costmap,
)
from utils import CombinedLoss, DiceLoss, SegmentationMetrics


class TestOffRoadTraversability(unittest.TestCase):

    def test_model_forward_shape(self):
        """Test Fast-SCNN forward pass output shape."""
        batch_size = 2
        height, width = 256, 512
        num_classes = 4

        model = FastSCNN(in_channels=3, num_classes=num_classes)
        model.eval()

        dummy_img = torch.randn(batch_size, 3, height, width)
        with torch.no_grad():
            output = model(dummy_img)

        self.assertEqual(
            output.shape,
            (batch_size, num_classes, height, width),
            f"Expected output shape {(batch_size, num_classes, height, width)}, got {output.shape}",
        )

    def test_model_parameters(self):
        """Verify Fast-SCNN is lightweight (<1.5M parameters)."""
        model = FastSCNN(num_classes=4)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertLess(
            total_params,
            1_500_000,
            f"Model has {total_params} parameters, expected < 1.5M for edge deployment.",
        )

    def test_combined_loss_backward(self):
        """Verify CE + Dice loss computes scalar and gradients propagate cleanly."""
        model = FastSCNN(num_classes=4)
        criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)

        dummy_img = torch.randn(2, 3, 128, 256)
        dummy_targets = torch.randint(0, 4, (2, 128, 256), dtype=torch.long)

        logits = model(dummy_img)
        loss = criterion(logits, dummy_targets)

        self.assertFalse(torch.isnan(loss).item(), "Loss should not be NaN")
        self.assertGreater(loss.item(), 0.0, "Loss should be strictly positive")

        loss.backward()
        # Verify model gradients exist
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        self.assertTrue(has_grads, "Model parameters should receive gradients")

    def test_metrics_calculation(self):
        """Verify mIoU and pixel accuracy calculation on known inputs."""
        metrics = SegmentationMetrics(num_classes=4, class_names=CLASS_NAMES)

        # Perfect prediction case
        targets = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        preds = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        metrics.update(preds, targets)
        results = metrics.get_results()

        self.assertAlmostEqual(results["pixel_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(results["mIoU"], 1.0, places=4)

    def test_synthetic_dataset_and_mapping(self):
        """Verify synthetic dataset produces valid frames and costmaps."""
        dataset = SyntheticOffRoadDataset(num_samples=2, target_size=(128, 256), is_training=False)
        self.assertEqual(len(dataset), 2)

        image_t, mask_t = dataset[0]
        self.assertEqual(image_t.shape, (3, 128, 256))
        self.assertEqual(mask_t.shape, (128, 256))

        # Check mask values are in [0..3]
        mask_np = mask_t.numpy()
        self.assertTrue(np.all((mask_np >= 0) & (mask_np < 4)))

        # Check costmap mapping
        costmap = mask_to_nav2_costmap(mask_np)
        self.assertEqual(costmap.shape, (128, 256))
        unique_costs = set(np.unique(costmap))
        valid_costs = set(NAV2_COST_MAPPING.values())
        self.assertTrue(unique_costs.issubset(valid_costs))

        # Check color overlay
        color_vis = mask_to_color(mask_np)
        self.assertEqual(color_vis.shape, (128, 256, 3))


if __name__ == "__main__":
    unittest.main()
