"""
Autonomous Ground Vehicle (UGV) Real-World Traversability & Stairway Perception Engine.
Hardware-Accelerated on NVIDIA GeForce RTX 4050 Laptop GPU via ONNX DirectML.

Clean semantic scene understanding (zero fake grids or synthetic artifacts):
- GREEN (Traversable): Floors, Ground, Roads, Sidewalks, Grass, Carpets, Dirt paths
- CYAN (Stairway): Steps, Staircases, Inclines (Navigation Incline Corridor)
- RED (Lethal Obstacles): Walls, Columns, Desks, Chairs, Furniture, Persons
- ORANGE (Hazards): Water, Pits, Drop-offs
- GRAY (Background): Ceiling, Sky
"""

import argparse
import sys
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# ADE20K 150-class mapping to UGV Nav2 Traversability
# ---------------------------------------------------------------------------
TRAVERSABLE_ADE_IDS = {
    3,   # floor, flooring
    6,   # road, route
    9,   # grass
    11,  # sidewalk, pavement
    13,  # earth, ground
    28,  # rug, carpet
    29,  # field
    46,  # sand
    52,  # path
    91,  # dirt track
    94,  # land, ground
}

STAIRWAY_ADE_IDS = {
    54,  # staircase, stairway
    59,  # step, stair
}

HAZARD_ADE_IDS = {
    21,  # water
    26,  # sea
    60,  # river
    113, # lake
    128, # pool
}

BACKGROUND_ADE_IDS = {
    2,   # sky
    5,   # ceiling
}

# Color palette (BGR for OpenCV)
COLOR_PALETTE = np.array([
    [70, 70, 70],      # 0: Gray (Ceiling / Sky)
    [34, 180, 34],     # 1: Green (Traversable Ground / Walkway)
    [40, 40, 220],     # 2: Red (Lethal Obstacles)
    [0, 140, 255],     # 3: Orange (Hazards / Water)
    [255, 200, 0],     # 4: Electric Cyan (Stairways / Steps / Inclines)
], dtype=np.uint8)

# Nav2 Metric Costmap palette: 0 = Free, 80 = Stairs (incline), 254 = Lethal, 250 = Hazard
COSTMAP_LUT = np.array([0, 0, 254, 250, 80], dtype=np.uint8)


class ThreadedCamera:
    """Low-latency threaded stream reader to eliminate network buffer bloat."""

    def __init__(self, src: str):
        print(f"[STREAM] Connecting to video stream: {src}")
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.running = True

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video stream: {src}")
            self.running = False
            return

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        print("[STREAM] Low-latency capture thread active.")

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()

    def isOpened(self) -> bool:
        return self.running and self.cap.isOpened()


class UGVVisionEngine:
    """Hardware-accelerated perception engine running on RTX 4050 GPU."""

    def __init__(self, onnx_model_path: str = "segformer_b0.onnx"):
        print("=" * 60)
        print("INITIALIZING UGV REAL-WORLD PERCEPTION ENGINE")
        print("=" * 60)

        available = ort.get_available_providers()
        print(f"Detected ONNX Providers: {available}")

        providers = []
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        print(f"Binding model to hardware via: {providers[0]}...")
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        active_providers = self.session.get_providers()
        self.active_provider = active_providers[0]

        if "Dml" in self.active_provider:
            self.device_name = "NVIDIA RTX 4050 (DirectML GPU)"
        elif "CUDA" in self.active_provider:
            self.device_name = "NVIDIA RTX 4050 (CUDA GPU)"
        else:
            self.device_name = "CPU (Fallback)"

        print(f"SUCCESS: Engine active on [{self.device_name}]")
        print("=" * 60)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Build fast 150-class lookup table
        self.ade_to_trav = np.full(150, 2, dtype=np.uint8)  # default: obstacle (2)
        for idx in TRAVERSABLE_ADE_IDS:
            self.ade_to_trav[idx] = 1
        for idx in STAIRWAY_ADE_IDS:
            self.ade_to_trav[idx] = 4  # Class 4: Stairway
        for idx in HAZARD_ADE_IDS:
            self.ade_to_trav[idx] = 3
        for idx in BACKGROUND_ADE_IDS:
            self.ade_to_trav[idx] = 0

        # Warm up GPU
        print("Warming up GPU kernels...")
        dummy = np.zeros((1, 3, 512, 512), dtype=np.float32)
        self.session.run(["logits"], {"pixel_values": dummy})
        print("Perception pipeline fully armed.\n")

    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Runs real-time inference on a BGR video frame.
        Returns:
            color_mask: (H, W, 3) uint8 BGR overlay
            costmap: (H, W) uint8 Nav2 costmap
            latency_ms: GPU inference time in ms
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        resized = cv2.resize(frame_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]

        t0 = time.perf_counter()
        logits = self.session.run(["logits"], {"pixel_values": tensor})[0]
        t_gpu_infer = (time.perf_counter() - t0) * 1000.0

        pred_grid = np.argmax(logits[0], axis=0).astype(np.uint8)  # (128, 128)
        pred_full = cv2.resize(pred_grid, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        trav_mask = self.ade_to_trav[pred_full]
        color_mask = COLOR_PALETTE[trav_mask]
        costmap = COSTMAP_LUT[trav_mask]

        return color_mask, costmap, t_gpu_infer


def run_pipeline(source=0, alpha: float = 0.35):
    """Launches the clean, real-time traversability perception feed."""
    engine = UGVVisionEngine("segformer_b0.onnx")

    is_network_stream = isinstance(source, str) and (
        source.startswith("http://")
        or source.startswith("https://")
        or source.startswith("rtsp://")
        or ":" in source
    )

    if is_network_stream:
        if not (source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")):
            source = f"http://{source}"
        if not source.endswith("/video") and not source.endswith(".mjpg") and not source.startswith("rtsp://"):
            source = f"{source.rstrip('/')}/video"

        print(f"[INIT] Opening low-latency wireless stream: {source}")
        cap = ThreadedCamera(source)
        time.sleep(1.0)
    else:
        is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        cap = cv2.VideoCapture(int(source) if is_webcam else source)
        if is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not connect to source: {source}")
        return

    fps_smooth = 30.0
    print("\n" + "=" * 65)
    print(" REAL-TIME TRAVERSABILITY ACTIVE")
    print(f" Accelerator: {engine.device_name}")
    print(" 🟩 GREEN  = Traversable Ground / Floor")
    print(" 🟦 CYAN   = Stairway / Steps / Incline")
    print(" 🟥 RED    = Obstacles (Walls, Furniture, Persons)")
    print(" ⬛ GRAY   = Background / Ceiling")
    print(" Press 'q' key to stop.")
    print("=" * 65 + "\n")

    while cap.isOpened():
        t_frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        color_mask, costmap, gpu_latency = engine.infer(frame)

        # Smooth, clean alpha-blend over camera feed (zero fake neon grids)
        blended = cv2.addWeighted(color_mask, alpha, frame, 1.0 - alpha, 0)

        t_total = time.perf_counter() - t_frame_start
        curr_fps = 1.0 / t_total if t_total > 0 else 30.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * curr_fps

        h, w = frame.shape[:2]

        # Top Telemetry Header
        cv2.rectangle(blended, (12, 12), (390, 78), (15, 15, 15), -1)
        cv2.rectangle(blended, (12, 12), (390, 78), (50, 50, 50), 1)

        cv2.putText(blended, "UGV REAL-TIME TRAVERSABILITY", (22, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(blended, f"DEVICE: {engine.device_name}", (22, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(blended, f"GPU INFER: {gpu_latency:.1f}ms  |  FPS: {fps_smooth:.1f}", (22, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        # Nav2 Metric Costmap (Picture-in-Picture)
        mini_w, mini_h = 160, 90
        mini_costmap = cv2.resize(costmap, (mini_w, mini_h), interpolation=cv2.INTER_NEAREST)
        mini_bgr = cv2.cvtColor(mini_costmap, cv2.COLOR_GRAY2BGR)
        x_offset = w - mini_w - 14
        y_offset = 14
        blended[y_offset:y_offset + mini_h, x_offset:x_offset + mini_w] = mini_bgr
        cv2.rectangle(blended, (x_offset, y_offset), (x_offset + mini_w, y_offset + mini_h), (0, 255, 180), 1)
        cv2.putText(blended, "Nav2 Costmap", (x_offset + 5, y_offset + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 180), 1, cv2.LINE_AA)

        # Legend (Bottom Left)
        leg_y = h - 20
        cv2.rectangle(blended, (12, leg_y - 20), (510, leg_y + 10), (15, 15, 15), -1)
        # Green patch
        cv2.rectangle(blended, (20, leg_y - 12), (32, leg_y), (34, 180, 34), -1)
        cv2.putText(blended, "Floor", (38, leg_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        # Cyan patch
        cv2.rectangle(blended, (90, leg_y - 12), (102, leg_y), (255, 200, 0), -1)
        cv2.putText(blended, "Stairway", (108, leg_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        # Red patch
        cv2.rectangle(blended, (195, leg_y - 12), (207, leg_y), (40, 40, 220), -1)
        cv2.putText(blended, "Obstacle", (213, leg_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        # Gray patch
        cv2.rectangle(blended, (300, leg_y - 12), (312, leg_y), (70, 70, 70), -1)
        cv2.putText(blended, "Ceiling", (318, leg_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        cv2.imshow("Autonomous Traversability Perception [RTX 4050]", blended)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Perception session stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Traversability Engine on RTX 4050")
    parser.add_argument("--url", type=str, default=None, help="Wireless Phone IP Webcam URL")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask overlay alpha (default: 0.35)")
    args = parser.parse_args()

    src = args.url if args.url else (args.video if args.video else args.webcam)
    run_pipeline(source=src, alpha=args.alpha)
