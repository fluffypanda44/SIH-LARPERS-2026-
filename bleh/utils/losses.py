"""
Loss functions for off-road terrain semantic segmentation.
Combines Cross-Entropy with Multiclass Dice Loss to handle severe class imbalance
(e.g., expansive traversable ground vs sparse rocks, ditches, or obstacles).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss.

    Args:
        smooth: Smoothing constant to avoid division by zero.
        ignore_index: Class index to ignore (e.g., 255 for void pixels).
        weight: Optional class weight tensor of shape (num_classes,).
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape (B, C, H, W).
            targets: Ground truth class indices of shape (B, H, W).

        Returns:
            Scalar Dice loss tensor.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Create valid mask
        valid_mask = targets != self.ignore_index
        targets_clamped = torch.clamp(targets, 0, num_classes - 1)

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets_clamped, num_classes=num_classes).permute(
            0, 3, 1, 2
        ).float()

        # Apply valid mask
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask_expanded
        targets_one_hot = targets_one_hot * valid_mask_expanded

        # Compute intersection and cardinality per class
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.weight is not None:
            weight = self.weight.to(dice_loss.device)
            return (dice_loss * weight).sum() / (weight.sum() + 1e-8)

        return dice_loss.mean()


class CombinedLoss(nn.Module):
    """
    Hybrid Cross-Entropy + Dice Loss:
        L_total = ce_weight * L_CE + dice_weight * L_Dice

    Args:
        ce_weight: Multiplier for Cross-Entropy loss.
        dice_weight: Multiplier for Dice loss.
        class_weights: Optional class weighting tensor (shape: C) for CE and Dice.
        ignore_index: Index of ignored void/unlabeled pixels.
        label_smoothing: Optional label smoothing epsilon for CE.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.dice_loss = DiceLoss(
            ignore_index=ignore_index,
            weight=class_weights,
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        l_ce = self.ce_loss(logits, targets)
        l_dice = self.dice_loss(logits, targets)
        return self.ce_weight * l_ce + self.dice_weight * l_dice


class TerrainFocalLoss(nn.Module):
    """
    Focal Loss for dense pixel classification on difficult off-road hazards.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return focal_loss[valid_mask].mean()
