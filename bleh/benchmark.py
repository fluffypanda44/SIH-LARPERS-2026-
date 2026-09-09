"""
Benchmark Script for Off-Road Terrain Traversability Model.
Profiles:
- Model Parameters & Estimated Model Size (MB)
- Inference Latency (Mean, Median, P95, Min, Max in ms)
- Throughput (Frames Per Second - FPS)
- Supports CPU and CUDA / Jetson Orin
"""

import argparse
import time
import numpy as np
import torch

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Fast-SCNN Latency & FPS")
    parser.add_argument("--img-height", type=int, default=512, help="Input image height")
    parser.add_argument("--img-width", type=int, default=1024, help="Input image width")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of semantic classes")
    parser.add_argument("--warmup-iters", type=int, default=15, help="Number of warmup iterations")
    parser.add_argument("--test-iters", type=int, default=50, help="Number of timed benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint weights")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("Off-Road Traversability Segmentation - Benchmark Suite")
    print("=" * 60)
    print(f"Device:         {device}")
    if device.type == "cuda":
        print(f"Device Name:    {torch.cuda.get_device_name(0)}")
    print(f"Input Shape:    ({args.batch_size}, 3, {args.img_height}, {args.img_width})")
    print(f"Num Classes:    {args.num_classes}")

    # Build model
    model = build_model("fast_scnn", num_classes=args.num_classes)
    if args.weights and torch.os.path.exists(args.weights):
        state = torch.load(args.weights, map_location="cpu")
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from: {args.weights}")
    model.to(device)
    model.eval()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)

    print(f"Total Params:   {total_params:,} ({total_params / 1e6:.3f}M)")
    print(f"Trainable:      {trainable_params:,}")
    print(f"Model Size:     {model_size_mb:.2f} MB (FP32)")

    # Dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.img_height, args.img_width, device=device)

    # Warmup
    print(f"\nWarming up ({args.warmup_iters} iterations)...")
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Benchmark loop
    print(f"Profiling ({args.test_iters} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in range(args.test_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # to milliseconds

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    fps = 1000.0 / mean_lat * args.batch_size

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Mean Latency:    {mean_lat:.2f} ms")
    print(f"Median Latency:  {median_lat:.2f} ms")
    print(f"P95 Latency:     {p95_lat:.2f} ms")
    print(f"Min / Max:       {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"Throughput:      {fps:.2f} FPS")
    print("=" * 60)


if __name__ == "__main__":
    main()
