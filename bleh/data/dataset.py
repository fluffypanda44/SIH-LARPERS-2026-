"""
Off-Road Terrain Dataset Loader and Augmentations.
Supports:
- 4-Class Traversability Representation:
    0: Background / Sky (Masked out / Free)
    1: Traversable (Grass, Dirt, Gravel, Packed trail)
    2: Positive Obstacle (Rock, Tree Trunk, Boulder, Dense Vegetation, Obstacle)
    3: Negative Obstacle / Hazard (Ditch, Hole, Steep Drop, Water)
- Off-Road Augmentations (UGV chassis vibration blur, sunlight glare & shadow jitter)
- Real image/mask directory loader (compatible with RUGD, RELLIS-3D, ORFD)
- Synthetic dataset generator for instant pipeline testing and benchmarking
"""

import os
import glob
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance


# 4-class traversability names and visualization colors (RGB)
CLASS_NAMES = [
    "Background_Sky",
    "Traversable",
    "Positive_Obstacle",
    "Negative_Obstacle",
]

COLOR_PALETTE = np.array(
    [
        [70, 70, 70],      # Class 0: Dark Gray (Sky / Background)
        [34, 139, 34],     # Class 1: Forest Green (Traversable Ground)
        [220, 20, 60],     # Class 2: Crimson Red (Positive Obstacle)
        [255, 140, 0],     # Class 3: Dark Orange (Negative Obstacle / Hazard)
    ],
    dtype=np.uint8,
)

# Mapping to Nav2 Costmap values (0-254)
NAV2_COST_MAPPING = {
    0: 0,    # Background/Sky: Cost 0 (Free/Ignored)
    1: 0,    # Traversable: Cost 0 (Free path)
    2: 254,  # Positive Obstacle: Cost 254 (Lethal obstacle)
    3: 250,  # Negative Obstacle: Cost 250 (High-risk hazard)
}


class OffRoadAugmentor:
    """
    Data augmentation suite specifically tailored for rough off-road UGV conditions:
    - UGV chassis vibration (motion blur)
    - Dramatic lighting variations (canopy shadows, direct solar glare)
    - Random scaling, cropping, and horizontal flipping
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 1024),  # (H, W)
        is_training: bool = True,
        motion_blur_prob: float = 0.35,
        color_jitter_prob: float = 0.5,
        horizontal_flip_prob: float = 0.5,
    ):
        self.target_height, self.target_width = target_size
        self.is_training = is_training
        self.motion_blur_prob = motion_blur_prob
        self.color_jitter_prob = color_jitter_prob
        self.horizontal_flip_prob = horizontal_flip_prob

        # ImageNet normalization statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies paired augmentation to image and mask.

        Args:
            image: PIL RGB Image.
            mask: PIL Grayscale Mask (L mode, pixel values 0..num_classes-1).

        Returns:
            Tuple of (image_tensor: [3, H, W] float32, mask_tensor: [H, W] int64).
        """
        if self.is_training:
            # 1. Random Horizontal Flip
            if random.random() < self.horizontal_flip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. UGV Chassis Vibration Simulation (Motion/Box Blur)
            if random.random() < self.motion_blur_prob:
                blur_radius = random.choice([1, 2])
                image = image.filter(ImageFilter.BoxBlur(blur_radius))

            # 3. Sunlight / Canopy Shadow Simulation (Color Jitter)
            if random.random() < self.color_jitter_prob:
                # Brightness
                factor_b = random.uniform(0.7, 1.3)
                image = ImageEnhance.Brightness(image).enhance(factor_b)
                # Contrast
                factor_c = random.uniform(0.7, 1.3)
                image = ImageEnhance.Contrast(image).enhance(factor_c)
                # Color saturation
                factor_s = random.uniform(0.7, 1.3)
                image = ImageEnhance.Color(image).enhance(factor_s)

            # 4. Random Scale and Crop
            scale = random.uniform(0.85, 1.15)
            new_w = int(self.target_width * scale)
            new_h = int(self.target_height * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)

            # Crop or pad to target size
            if new_w >= self.target_width and new_h >= self.target_height:
                crop_x = random.randint(0, new_w - self.target_width)
                crop_y = random.randint(0, new_h - self.target_height)
                image = image.crop(
                    (crop_x, crop_y, crop_x + self.target_width, crop_y + self.target_height)
                )
                mask = mask.crop(
                    (crop_x, crop_y, crop_x + self.target_width, crop_y + self.target_height)
                )
            else:
                image = image.resize((self.target_width, self.target_height), Image.BILINEAR)
                mask = mask.resize((self.target_width, self.target_height), Image.NEAREST)
        else:
            # Deterministic resize for validation/testing
            image = image.resize((self.target_width, self.target_height), Image.BILINEAR)
            mask = mask.resize((self.target_width, self.target_height), Image.NEAREST)

        # Convert image to Normalized Float Tensor [3, H, W]
        img_arr = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)
        img_arr = np.transpose(img_arr, (2, 0, 1))           # (3, H, W)
        img_arr = (img_arr - self.mean) / self.std
        image_tensor = torch.from_numpy(img_arr).float()

        # Convert mask to Long Tensor [H, W]
        mask_arr = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_arr).long()

        return image_tensor, mask_tensor


class OffRoadDataset(Dataset):
    """
    Standard Off-Road Terrain Dataset reading paired images and masks.

    Directory structure:
        root_dir/
            images/
                frame_001.png
                frame_002.png
            masks/
                frame_001.png
                frame_002.png
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        target_size: Tuple[int, int] = (512, 1024),
        is_training: bool = True,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augmentor = OffRoadAugmentor(target_size=target_size, is_training=is_training)

        valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = []
        for ext in valid_extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths.sort()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        basename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(basename)[0]

        mask_candidates = [
            os.path.join(self.masks_dir, f"{name_without_ext}.png"),
            os.path.join(self.masks_dir, f"{name_without_ext}.jpg"),
            os.path.join(self.masks_dir, basename),
        ]
        mask_path = None
        for cand in mask_candidates:
            if os.path.exists(cand):
                mask_path = cand
                break

        image = Image.open(img_path).convert("RGB")
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            # Fallback zero mask if mask file not found
            mask = Image.new("L", image.size, 0)

        return self.augmentor(image, mask)


class SyntheticOffRoadDataset(Dataset):
    """
    Synthetic Off-Road Terrain Dataset Generator.
    Synthesizes diverse outdoor terrain scenes (Sky, Traversable Ground/Trail,
    Positive Obstacles/Boulders/Trees, Negative Obstacles/Ditches).
    Enables immediate pipeline testing, unit tests, and CI/CD validation.
    """

    def __init__(
        self,
        num_samples: int = 100,
        target_size: Tuple[int, int] = (512, 1024),
        is_training: bool = True,
    ):
        self.num_samples = num_samples
        self.target_height, self.target_width = target_size
        self.augmentor = OffRoadAugmentor(target_size=target_size, is_training=is_training)

    def __len__(self) -> int:
        return self.num_samples

    def _generate_synthetic_scene(self) -> Tuple[Image.Image, Image.Image]:
        h, w = self.target_height, self.target_width
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Horizon Line (Top 25% - 40% is Sky)
        horizon_y = random.randint(int(h * 0.25), int(h * 0.40))
        # Sky: blue/grey gradient
        sky_blue = np.array([random.randint(140, 180), random.randint(180, 220), random.randint(220, 255)])
        img[:horizon_y, :] = sky_blue
        mask[:horizon_y, :] = 0  # Class 0: Sky

        # 2. Ground / Terrain (Traversable by default)
        ground_base = np.array([random.randint(60, 90), random.randint(100, 140), random.randint(40, 70)])
        ground_noise = np.random.randint(-15, 15, size=(h - horizon_y, w, 3))
        ground_color = np.clip(ground_base + ground_noise, 0, 255).astype(np.uint8)
        img[horizon_y:, :] = ground_color
        mask[horizon_y:, :] = 1  # Class 1: Traversable

        # 3. Add curved dirt trail
        trail_center = random.randint(int(w * 0.35), int(w * 0.65))
        trail_color = np.array([140, 120, 90], dtype=np.uint8)
        for y in range(horizon_y, h):
            ratio = (y - horizon_y) / (h - horizon_y)
            trail_width = int(20 + 120 * ratio)
            curve_offset = int(40 * np.sin(ratio * 3.14))
            cx = trail_center + curve_offset
            x_start = max(0, cx - trail_width // 2)
            x_end = min(w, cx + trail_width // 2)
            img[y, x_start:x_end] = trail_color + np.random.randint(-10, 10, size=(x_end - x_start, 3), dtype=np.int16).clip(-20, 20).astype(np.uint8)
            mask[y, x_start:x_end] = 1

        # 4. Positive Obstacles (Boulders, Tree Trunks, Dense Bushes)
        num_obstacles = random.randint(3, 8)
        for _ in range(num_obstacles):
            obs_y = random.randint(horizon_y + 20, h - 20)
            obs_x = random.randint(20, w - 20)
            radius = random.randint(15, 45)
            # Create circular/elliptical obstacle
            y_indices, x_indices = np.ogrid[:h, :w]
            dist_from_center = (x_indices - obs_x) ** 2 + (y_indices - obs_y) ** 2
            obs_mask = dist_from_center <= radius**2
            # Dark rock/bark color
            obs_color = [random.randint(40, 70), random.randint(35, 60), random.randint(30, 50)]
            img[obs_mask] = obs_color
            mask[obs_mask] = 2  # Class 2: Positive Obstacle

        # 5. Negative Obstacles / Hazards (Ditches, Steep ravines)
        if random.random() < 0.6:  # 60% chance of ditch
            min_y = min(h - 15, horizon_y + max(5, int(h * 0.1)))
            max_y = max(min_y, h - max(5, int(h * 0.08)))
            ditch_y = random.randint(min_y, max_y)
            ditch_x = random.randint(max(10, int(w * 0.2)), min(w - 10, int(w * 0.8)))
            ditch_w = random.randint(max(10, int(w * 0.08)), max(20, int(w * 0.25)))
            ditch_h = random.randint(max(5, int(h * 0.04)), max(10, int(h * 0.12)))
            y1, y2 = ditch_y, min(h, ditch_y + ditch_h)
            x1, x2 = max(0, ditch_x - ditch_w // 2), min(w, ditch_x + ditch_w // 2)
            # Ditches appear darker or shadow-filled
            img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * 0.35).astype(np.uint8)
            mask[y1:y2, x1:x2] = 3  # Class 3: Negative Obstacle

        return Image.fromarray(img), Image.fromarray(mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self._generate_synthetic_scene()
        return self.augmentor(img, mask)


def mask_to_color(mask_np: np.ndarray) -> np.ndarray:
    """
    Converts 2D integer class mask (H, W) into an RGB colored visualization.

    Args:
        mask_np: NumPy array of shape (H, W) with class indices 0..3.

    Returns:
        RGB NumPy array of shape (H, W, 3) uint8.
    """
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(COLOR_PALETTE):
        color_mask[mask_np == cls_idx] = color
    return color_mask


def mask_to_nav2_costmap(mask_np: np.ndarray) -> np.ndarray:
    """
    Maps 2D class mask (H, W) to Nav2 costmap values (0-254).

    Args:
        mask_np: NumPy array of shape (H, W) with class indices 0..3.

    Returns:
        NumPy array of shape (H, W) uint8 with cost values.
    """
    costmap = np.zeros_like(mask_np, dtype=np.uint8)
    for cls_idx, cost_val in NAV2_COST_MAPPING.items():
        costmap[mask_np == cls_idx] = cost_val
    return costmap
