"""
Evaluation metrics for semantic segmentation:
- Confusion matrix
- Per-class IoU (Intersection over Union)
- Mean IoU (mIoU)
- Pixel Accuracy (PA)
- Per-class Dice / F1 Score
"""

from typing import Dict, List, Optional
import numpy as np
import torch


class SegmentationMetrics:
    """
    Computes running confusion matrix and segmentation metrics across batches.

    Args:
        num_classes: Number of semantic classes.
        class_names: Optional human-readable names for classes.
        ignore_index: Index to ignore in evaluation.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = [f"Class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self) -> None:
        """Reset running confusion matrix."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update confusion matrix with batch predictions and targets.

        Args:
            preds: Predicted class labels of shape (B, H, W) or logits (B, C, H, W).
            targets: Ground truth class labels of shape (B, H, W).
        """
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        # Filter out ignored indices
        valid = targets_np != self.ignore_index
        preds_valid = preds_np[valid]
        targets_valid = targets_np[valid]

        # Valid range check
        in_range = (
            (preds_valid >= 0)
            & (preds_valid < self.num_classes)
            & (targets_valid >= 0)
            & (targets_valid < self.num_classes)
        )
        preds_valid = preds_valid[in_range]
        targets_valid = targets_valid[in_range]

        # Accumulate 2D histogram
        indices = self.num_classes * targets_valid + preds_valid
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape(
            (self.num_classes, self.num_classes)
        )

    def get_results(self) -> Dict[str, float]:
        """
        Compute evaluation metrics from accumulated confusion matrix.

        Returns:
            Dictionary with mIoU, pixel_acc, and per-class IoU.
        """
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp

        denom = tp + fp + fn
        ious = np.divide(
            tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom != 0
        )

        valid_classes = denom > 0
        miou = (
            np.mean(ious[valid_classes])
            if np.any(valid_classes)
            else 0.0
        )

        total_correct = tp.sum()
        total_pixels = cm.sum()
        pixel_acc = (
            float(total_correct) / float(total_pixels)
            if total_pixels > 0
            else 0.0
        )

        results: Dict[str, float] = {
            "mIoU": float(miou),
            "pixel_accuracy": float(pixel_acc),
        }

        for i, name in enumerate(self.class_names):
            results[f"IoU_{name}"] = float(ious[i])

        return results

    def summary_table(self) -> str:
        """Formatted summary table of metrics."""
        results = self.get_results()
        lines = [
            "=" * 50,
            f"{'Class':<25} | {'IoU (%)':>10}",
            "-" * 50,
        ]
        for name in self.class_names:
            iou = results[f"IoU_{name}"] * 100
            lines.append(f"{name:<25} | {iou:>10.2f}%")
        lines.append("-" * 50)
        lines.append(f"{'Mean IoU (mIoU)':<25} | {results['mIoU'] * 100:>10.2f}%")
        lines.append(
            f"{'Pixel Accuracy':<25} | {results['pixel_accuracy'] * 100:>10.2f}%"
        )
        lines.append("=" * 50)
        return "\n".join(lines)
