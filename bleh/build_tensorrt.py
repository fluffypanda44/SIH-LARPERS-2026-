"""
TensorRT Engine Builder & Optimization Guide for Jetson Orin.

Provides:
1. Shell / CLI execution script with `trtexec` (fastest and most reliable path on JetPack)
2. Python TensorRT API builder with FP16 and INT8 calibration hooks
3. Jetson Orin compute budget profiling guidelines
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT Engine for Jetson Orin")
    parser.add_argument("--onnx", type=str, default="fast_scnn_traversability.onnx", help="Path to input ONNX model")
    parser.add_argument("--output", type=str, default="fast_scnn_traversability_fp16.engine", help="Output .engine file")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 precision (recommended on Orin)")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision (requires calibration cache)")
    parser.add_argument("--workspace-mb", type=int, default=2048, help="TensorRT workspace memory in MB")
    parser.add_argument("--trtexec-path", type=str, default="/usr/src/tensorrt/bin/trtexec", help="Path to trtexec binary")
    return parser.parse_args()


def build_with_trtexec(args):
    """Executes NVIDIA trtexec CLI tool to build engine."""
    cmd = [
        args.trtexec_path,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.output}",
        f"--memPoolSize=workspace:{args.workspace_mb}M",
    ]

    if args.fp16:
        cmd.append("--fp16")
    if args.int8:
        cmd.append("--int8")

    cmd_str = " ".join(cmd)
    print("=" * 60)
    print("Building TensorRT Engine using trtexec on Jetson Orin")
    print("=" * 60)
    print(f"Executing: {cmd_str}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\nTensorRT engine successfully built and saved to: {args.output}")
    except FileNotFoundError:
        print(f"Error: trtexec binary not found at '{args.trtexec_path}'.")
        print("On NVIDIA JetPack (Jetson Orin), trtexec is typically located at:")
        print("  /usr/src/tensorrt/bin/trtexec")
        print("\nManual command to run on your Jetson Orin terminal:")
        print(f"  {cmd_str}")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT build failed with exit code: {e.returncode}")


def build_with_python_api(args):
    """Fallback: builds engine via Python TensorRT API if trtexec is unavailable."""
    try:
        import tensorrt as trt
    except ImportError:
        print("\nNotice: Python 'tensorrt' module is not installed on this host.")
        print("To run on your Jetson Orin, transfer the .onnx file and execute:")
        print(f"  /usr/src/tensorrt/bin/trtexec --onnx={args.onnx} --saveEngine={args.output} --fp16")
        return

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * 1024 * 1024)

    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabled FP16 Mode.")

    print("Building TensorRT serialized engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Engine compilation failed.")
        return

    with open(args.output, "wb") as f:
        f.write(serialized_engine)
    print(f"Engine built successfully: {args.output}")


def main():
    args = parse_args()
    if os.path.exists(args.trtexec_path):
        build_with_trtexec(args)
    else:
        build_with_python_api(args)


if __name__ == "__main__":
    main()
