"""
Training Script for Off-Road Terrain Traversability Segmentation.

Features:
- Fast-SCNN Architecture
- Combined Cross-Entropy + Dice Loss (optimizing for both large ground areas and sparse obstacles)
- Cosine Annealing Learning Rate Schedule with Warmup
- Real (RUGD/RELLIS-3D/custom) and Synthetic Dataset Support
- mIoU, per-class IoU, and Pixel Accuracy tracking
- Automatic Best Checkpoint Saving
"""

import argparse
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import build_model
from data import OffRoadDataset, SyntheticOffRoadDataset, CLASS_NAMES
from utils import CombinedLoss, SegmentationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Off-Road Traversability Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img-height", type=int, default=512, help="Input image height")
    parser.add_argument("--img-width", type=int, default=1024, help="Input image width")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of semantic classes")
    parser.add_argument("--ce-weight", type=float, default=1.0, help="Cross-entropy loss weight")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--masks-dir", type=str, default=None, help="Directory containing masks")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic dataset for training/testing")
    parser.add_argument("--num-synthetic-samples", type=int, default=120, help="Number of synthetic samples")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(dataloader))
    duration = time.time() - start_time
    return avg_loss, duration


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    metrics: SegmentationMetrics,
    device: torch.device,
) -> Tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    metrics.reset()

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, masks)

    avg_loss = total_loss / max(1, len(dataloader))
    eval_results = metrics.get_results()
    return avg_loss, eval_results


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    target_size = (args.img_height, args.img_width)

    print("=" * 60)
    print("Off-Road Traversability Segmentation - Training Pipeline")
    print("=" * 60)
    print(f"Device:      {device}")
    print(f"Resolution:  {args.img_width}x{args.img_height}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Epochs:      {args.epochs}")
    print(f"Classes:     {CLASS_NAMES}")

    # 1. Dataset Setup
    if args.use_synthetic or (args.images_dir is None or not os.path.exists(args.images_dir)):
        print("\nUsing Synthetic Off-Road Dataset generator (terrain, trails, obstacles, ditches)...")
        full_dataset = SyntheticOffRoadDataset(
            num_samples=args.num_synthetic_samples,
            target_size=target_size,
            is_training=True,
        )
        val_size = max(10, int(0.2 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    else:
        print(f"\nLoading dataset from: {args.images_dir} and {args.masks_dir}")
        full_dataset = OffRoadDataset(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            target_size=target_size,
            is_training=True,
        )
        val_size = max(1, int(0.2 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # 2. Model & Loss Setup
    model = build_model("fast_scnn", num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Fast-SCNN ({total_params / 1e6:.2f}M parameters)")

    # Class weights: increase penalty for negative obstacles and positive obstacles
    class_weights = torch.tensor([0.5, 1.0, 2.0, 2.5], device=device)
    criterion = CombinedLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        class_weights=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    metrics = SegmentationMetrics(
        num_classes=args.num_classes, class_names=CLASS_NAMES
    )

    # 3. Training Loop
    best_miou = 0.0
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    last_model_path = os.path.join(args.save_dir, "last_model.pth")

    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, duration = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, eval_results = evaluate(
            model, val_loader, criterion, metrics, device
        )
        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        miou = eval_results["mIoU"] * 100
        pix_acc = eval_results["pixel_accuracy"] * 100

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Time: {duration:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mIoU: {miou:.2f}% | "
            f"PixAcc: {pix_acc:.2f}% | "
            f"LR: {curr_lr:.6f}"
        )

        # Checkpoint saving
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mIoU": eval_results["mIoU"],
                "args": vars(args),
            },
            last_model_path,
        )

        if eval_results["mIoU"] > best_miou:
            best_miou = eval_results["mIoU"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model checkpoint (mIoU: {best_miou * 100:.2f}%) to {best_model_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation mIoU: {best_miou * 100:.2f}%")
    print(f"Best Model Weights:  {best_model_path}")
    print("=" * 60)
    print(metrics.summary_table())


if __name__ == "__main__":
    main()
