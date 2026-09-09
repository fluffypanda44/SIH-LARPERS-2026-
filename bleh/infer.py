"""
Inference & Visualization Script for Off-Road Terrain Traversability.

Supports:
- Running on single image, folder of images, or synthetic test generator
- Fast-SCNN PyTorch checkpoint (.pth) or ONNX model (.onnx)
- Outputting:
  1. Alpha-blended semantic overlay (Green = Traversable, Red = Rock/Obstacle, Orange = Ditch)
  2. Metric Nav2 Costmap (0 = Free, 254 = Lethal Obstacle, 250 = Hazard)
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
from PIL import Image
import torch

from models import build_model
from data import (
    CLASS_NAMES,
    COLOR_PALETTE,
    NAV2_COST_MAPPING,
    OffRoadAugmentor,
    SyntheticOffRoadDataset,
    mask_to_color,
    mask_to_nav2_costmap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Off-Road Traversability Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--input-dir", type=str, default=None, help="Path to directory of images")
    parser.add_argument("--use-synthetic", action="store_true", help="Generate and infer on synthetic off-road frame")
    parser.add_argument("--weights", type=str, default=None, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model (.onnx)")
    parser.add_argument("--output-dir", type=str, default="inference_output", help="Directory to save visual outputs")
    parser.add_argument("--img-height", type=int, default=512, help="Input height")
    parser.add_argument("--img-width", type=int, default=1024, help="Input width")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha blend factor for overlay (0.0 to 1.0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class TraversabilityPredictor:
    """Handles preprocessing, neural inference, and postprocessing."""

    def __init__(
        self,
        weights_path: str = None,
        onnx_path: str = None,
        target_size: Tuple[int, int] = (512, 1024),
        num_classes: int = 4,
        device: str = "cpu",
    ):
        self.target_height, self.target_width = target_size
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.use_onnx = onnx_path is not None

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        if self.use_onnx:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"Loaded ONNX model from: {onnx_path}")
        else:
            self.model = build_model("fast_scnn", num_classes=num_classes)
            if weights_path and os.path.exists(weights_path):
                state = torch.load(weights_path, map_location=self.device)
                state_dict = state.get("model_state_dict", state)
                self.model.load_state_dict(state_dict)
                print(f"Loaded PyTorch checkpoint from: {weights_path}")
            else:
                print("Notice: No checkpoint provided, running with initialized weights.")
            self.model.to(self.device)
            self.model.eval()

    def preprocess(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        orig_size = image.size  # (W, H)
        resized_img = image.resize((self.target_width, self.target_height), Image.BILINEAR)
        img_arr = np.array(resized_img, dtype=np.float32) / 255.0  # (H, W, 3)
        img_arr = np.transpose(img_arr, (2, 0, 1))                 # (3, H, W)
        img_tensor = np.expand_dims(img_arr, axis=0)               # (1, 3, H, W)
        normalized = (img_tensor - self.mean) / self.std
        return normalized, orig_size

    def predict(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Runs prediction on a PIL Image.

        Returns:
            mask_np: (H, W) uint8 class prediction mask (0..num_classes-1)
            costmap_np: (H, W) uint8 Nav2 costmap
            latency_ms: Inference time in milliseconds
        """
        input_data, _ = self.preprocess(image)

        t0 = time.perf_counter()
        if self.use_onnx:
            outputs = self.session.run(None, {self.input_name: input_data})
            logits = outputs[0]  # (1, C, H, W)
            pred_mask = np.argmax(logits, axis=1)[0].astype(np.uint8)
        else:
            with torch.no_grad():
                tensor_in = torch.from_numpy(input_data).float().to(self.device)
                logits = self.model(tensor_in)
                pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        costmap = mask_to_nav2_costmap(pred_mask)
        return pred_mask, costmap, latency_ms

    def create_visualization(
        self,
        orig_image: Image.Image,
        pred_mask: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Overlays color-coded semantic prediction on input image."""
        resized_orig = orig_image.resize((self.target_width, self.target_height), Image.BILINEAR)
        color_mask = mask_to_color(pred_mask)
        orig_arr = np.array(resized_orig, dtype=np.float32)
        blended = (1.0 - alpha) * orig_arr + alpha * color_mask.astype(np.float32)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    target_size = (args.img_height, args.img_width)

    predictor = TraversabilityPredictor(
        weights_path=args.weights,
        onnx_path=args.onnx,
        target_size=target_size,
        num_classes=args.num_classes,
        device=args.device,
    )

    images_to_process = []
    if args.use_synthetic or (args.image is None and args.input_dir is None):
        print("Generating synthetic off-road frame for demonstration...")
        syn_dataset = SyntheticOffRoadDataset(num_samples=1, target_size=target_size)
        img, gt_mask = syn_dataset._generate_synthetic_scene()
        images_to_process.append(("synthetic_sample.png", img))
    elif args.image:
        images_to_process.append((os.path.basename(args.image), Image.open(args.image).convert("RGB")))
    elif args.input_dir:
        for fname in os.listdir(args.input_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(args.input_dir, fname)
                images_to_process.append((fname, Image.open(fpath).convert("RGB")))

    print(f"\nProcessing {len(images_to_process)} image(s)...")
    for name, img in images_to_process:
        mask, costmap, latency = predictor.predict(img)
        vis_overlay = predictor.create_visualization(img, mask, alpha=args.alpha)

        base_name = os.path.splitext(name)[0]
        vis_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
        costmap_path = os.path.join(args.output_dir, f"{base_name}_costmap.png")
        mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")

        vis_overlay.save(vis_path)
        Image.fromarray(costmap).save(costmap_path)
        Image.fromarray(mask_to_color(mask)).save(mask_path)

        print(f"[{name}] Latency: {latency:.2f} ms")
        print(f"  -> Saved overlay:  {vis_path}")
        print(f"  -> Saved costmap:  {costmap_path}")
        print(f"  -> Saved mask:     {mask_path}")


if __name__ == "__main__":
    main()
