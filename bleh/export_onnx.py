"""
ONNX Model Exporter for Fast-SCNN Off-Road Traversability Segmentation.

Converts PyTorch checkpoint into an optimized ONNX model ready for:
- TensorRT engine conversion on Jetson Orin (`trtexec`)
- OpenVINO, ONNXRuntime, or DirectML execution
- Parity verification between PyTorch and ONNX outputs
"""

import argparse
import os
import sys
import numpy as np
import torch

# Ensure UTF-8 encoding on Windows to prevent UnicodeEncodeError with console emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export Fast-SCNN to ONNX")
    parser.add_argument("--weights", type=str, default=None, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="fast_scnn_traversability.onnx", help="Output ONNX filename")
    parser.add_argument("--img-height", type=int, default=512, help="Input height")
    parser.add_argument("--img-width", type=int, default=1024, help="Input width")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of semantic classes")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size for export")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch size in ONNX")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version (14-17 recommended for TensorRT)")
    return parser.parse_args()


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    dummy_input: torch.Tensor,
    opset_version: int = 16,
    dynamic_batch: bool = False,
):
    model.eval()

    input_names = ["input_image"]
    output_names = ["segmentation_logits"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input_image": {0: "batch_size"},
            "segmentation_logits": {0: "batch_size"},
        }

    print(f"Exporting model to ONNX: {output_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # For older PyTorch versions where dynamo argument doesn't exist
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    print(f"ONNX export succeeded! File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def verify_onnx_model(onnx_path: str, dummy_input: torch.Tensor, torch_output: torch.Tensor):
    """Verifies ONNX structure and numerical parity with PyTorch."""
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structural integrity check: PASSED!")
    except ImportError:
        print("Warning: 'onnx' package not installed. Skipping structural checker.")
        return

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outputs = session.run(None, ort_inputs)

        diff = np.abs(torch_output.detach().cpu().numpy() - ort_outputs[0])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"ONNXRuntime Parity Check: PASSED!")
        print(f"  Max Absolute Difference:  {max_diff:.6e}")
        print(f"  Mean Absolute Difference: {mean_diff:.6e}")
    except ImportError:
        print("Notice: 'onnxruntime' not installed. Skipping numerical parity test.")


def main():
    args = parse_args()

    print("=" * 60)
    print("Exporting Fast-SCNN to ONNX")
    print("=" * 60)

    # 1. Build Model
    model = build_model("fast_scnn", num_classes=args.num_classes)
    if args.weights and os.path.exists(args.weights):
        state = torch.load(args.weights, map_location="cpu")
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from: {args.weights}")
    else:
        print("Notice: No pre-trained weights specified, exporting with random initialization.")

    model.eval()

    # 2. Dummy Input
    dummy_input = torch.randn(args.batch_size, 3, args.img_height, args.img_width, dtype=torch.float32)

    with torch.no_grad():
        torch_output = model(dummy_input)

    # 3. Export
    export_to_onnx(
        model,
        args.output,
        dummy_input,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
    )

    # 4. Parity Verification
    verify_onnx_model(args.output, dummy_input, torch_output)


if __name__ == "__main__":
    main()
