# Off-Road Autonomous Terrain Traversability AI Model (Fast-SCNN)

Real-time semantic segmentation and traversability costmap pipeline designed for off-road Unmanned Ground Vehicles (UGVs) and edge deployment on NVIDIA Jetson Orin.

---

## 1. Overview & Architecture

Standard models like YOLOv8-seg perform *instance* segmentation (detecting countable objects), whereas off-road traversability requires *semantic* segmentation (every pixel is classified into continuous terrain categories).

This package implements **Fast-SCNN (Fast Segmentation Convolutional Neural Network)**:
- **Parameter Count**: ~1.14M parameters (ultra-lightweight, <5 MB)
- **Input Resolution**: $512 \times 1024$ (optimized for wide-angle stereo cameras)
- **Edge Performance**: 25–45+ FPS on Jetson Orin with TensorRT FP16
- **Branches**:
  - **Learning to Downsample (LDS)**: Rapid 8× spatial downsampling preserving shallow spatial edge details.
  - **Global Feature Extractor (GFE)**: Inverted bottleneck residual blocks + Pyramid Pooling Module (PPM) capturing multi-scale context.
  - **Feature Fusion Module (FFM)**: Fuses spatial edge details with deep semantic context.
  - **Classifier Head**: Depthwise separable convolutions outputting per-pixel class logits.

### Traversability Classes

| Class ID | Class Name | Description | Costmap Value | Visualization Color |
| :--- | :--- | :--- | :--- | :--- |
| **0** | `Background_Sky` | Sky, horizon, background (masked out/ignored) | `0` (Free/Ignore) | Dark Gray `[70, 70, 70]` |
| **1** | `Traversable` | Grass, dirt, gravel, packed trail | `0` (Free Path) | Forest Green `[34, 139, 34]` |
| **2** | `Positive_Obstacle` | Rocks, tree trunks, boulders, thick bushes | `254` (Lethal) | Crimson Red `[220, 20, 60]` |
| **3** | `Negative_Obstacle` | Ditches, steep drops, holes, water hazards | `250` (Hazard) | Dark Orange `[255, 140, 0]` |

---

## 2. Directory Structure

```
├── models/
│   ├── __init__.py           # Model factory
│   └── fast_scnn.py          # Fast-SCNN PyTorch architecture
├── data/
│   ├── __init__.py           # Data exports
│   └── dataset.py            # RUGD/RELLIS-3D loader, augmentations & synthetic generator
├── utils/
│   ├── __init__.py
│   ├── losses.py             # Combined Cross-Entropy + Multiclass Dice loss
│   └── metrics.py            # mIoU, per-class IoU, pixel accuracy tracker
├── ros2/
│   ├── __init__.py
│   └── terrain_segmentation_node.py # ROS 2 node subscribing to camera & publishing costmap
├── train.py                  # Full training & validation loop
├── benchmark.py              # Latency, FPS, parameter profiler
├── export_onnx.py            # ONNX exporter & numerical parity checker
├── build_tensorrt.py         # Jetson Orin TensorRT engine build script
├── infer.py                  # Standalone inference & visualizer
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## 3. Off-Road Augmentation & Loss Design

### Domain Gap & Vibration Augmentation
Off-road UGVs experience heavy vibration and dramatic lighting transitions under tree canopies. `data/dataset.py` includes:
- **UGV Vibration Simulation**: Random motion blur simulating camera shake.
- **Lighting & Canopy Jitter**: Solar glare, shadow contrast, and saturation variations.
- **Aspect-Preserving Scaling & Cropping**: Random scaling and cropping to $512 \times 1024$.

### Loss Function
$$\mathcal{L}_{total} = \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{Dice}$$
- **Dice Loss**: Counteracts the massive class imbalance where traversable ground occupies the vast majority of pixels while ditches and rocks are sparse.
- **Class Weights**: Penalizes false negatives on lethal rocks (`weight=2.0`) and hazards/ditches (`weight=2.5`).

---

## 4. Quickstart: Training & Evaluation

### Synthetic Demo (Instant Run without Downloading Datasets)
Train and validate immediately on synthetic off-road frames:
```bash
python train.py --use-synthetic --epochs 5 --batch-size 4
```

### Real-World Training (RUGD / RELLIS-3D / Custom Robot Frames)
```bash
python train.py \
  --images-dir /path/to/dataset/images \
  --masks-dir /path/to/dataset/masks \
  --epochs 30 \
  --batch-size 8 \
  --lr 0.001 \
  --img-height 512 \
  --img-width 1024 \
  --save-dir checkpoints
```

---

## 5. Benchmarking Latency & FPS

Benchmark latency, throughput, and memory usage on your machine or Jetson Orin:
```bash
python benchmark.py --img-height 512 --img-width 1024 --batch-size 1
```

---

## 6. ONNX & TensorRT Deployment on Jetson Orin

### Step 1: Export PyTorch to ONNX
```bash
python export_onnx.py \
  --weights checkpoints/best_model.pth \
  --output fast_scnn_traversability.onnx \
  --img-height 512 \
  --img-width 1024 \
  --opset 16
```

### Step 2: Build TensorRT FP16 Engine on Jetson Orin
Transfer `fast_scnn_traversability.onnx` to the Jetson Orin and compile:
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=fast_scnn_traversability.onnx \
  --saveEngine=fast_scnn_traversability_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:2048M
```
*Or use the provided script:*
```bash
python build_tensorrt.py --onnx fast_scnn_traversability.onnx --fp16
```

---

## 7. Standalone Inference & Costmap Generation

Test inference on an image or video:
```bash
# On a single image
python infer.py --image test_frame.png --weights checkpoints/best_model.pth

# On a synthetic test frame (demonstration)
python infer.py --use-synthetic
```
Outputs generated in `inference_output/`:
1. `*_overlay.png`: Green traversable path, Red obstacles, Orange ditches.
2. `*_costmap.png`: Nav2-compatible cost values (0..254).
3. `*_mask.png`: Raw class indices.

---

## 8. ROS 2 & Nav2 Integration

The node `ros2/terrain_segmentation_node.py` hooks directly into the ROS 2 Nav2 stack:
```bash
# Run node
python ros2/terrain_segmentation_node.py
```

### ROS 2 Interface Topics
- **Input**:
  - `/camera/left/image_raw` (`sensor_msgs/msg/Image`)
- **Outputs**:
  - `/terrain/segmentation_mask` (`sensor_msgs/msg/Image`, mono8)
  - `/terrain/traversability_costmap` (`sensor_msgs/msg/Image`, mono8: cost 0..254)
  - `/terrain/colored_overlay` (`sensor_msgs/msg/Image`, rgb8 for RViz2)

### Nav2 Costmap Configuration (`nav2_params.yaml`)
Feed `/terrain/traversability_costmap` into Nav2's local rolling costmap via a custom costmap layer or spatio-temporal voxel layer.
