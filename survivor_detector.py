"""
Urban Search & Rescue (USAR) Survivor Identification Engine.
Hardware-Accelerated on GPU via ONNX DirectML / CUDA.

Key Features:
- High-Recall Detection (conf_thresh=0.18, iou_thresh=0.60) catches partially occluded & adjacent victims.
- Anthropometric Distance Estimator: Estimates target distance based on biomechanical proportions (shoulder span ~0.42m).
- 17 Anatomical Keypoints: Classifies posture & entrapment severity heuristic.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from camera_utils import ThreadedCamera, format_stream_url


# ---------------------------------------------------------------------------
# COCO Pose Skeleton Connectivity
# ---------------------------------------------------------------------------
SKELETON_BONES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left Arm & Right Arm
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Left Leg & Right Leg
    (11, 13), (13, 15), (12, 14), (14, 16)
]


@dataclass
class SurvivorTarget:
    target_id: int
    bbox: Tuple[int, int, int, int]       # (x1, y1, x2, y2)
    confidence: float                     # Detection confidence
    keypoints: np.ndarray                 # Shape (17, 3): [x, y, conf]
    visible_joints: int                   # Count of joints with conf > 0.35
    posture: str                          # PRONE_FLAT, PINNED_UPPER, CURLED, UPRIGHT
    entrapment: str                       # SEVERELY_TRAPPED, PARTIALLY_TRAPPED, EXPOSED
    bearing_deg: float                    # Relative angle: -deg (left) to +deg (right)
    est_distance_m: float                 # Estimated metric distance in meters


class SurvivorDetector:
    """High-speed survivor pose detector running on GPU."""

    def __init__(self, onnx_model_path: str = "yolov8n-pose.onnx", conf_thresh: float = 0.18, iou_thresh: float = 0.60):
        print("=" * 65)
        print("INITIALIZING USAR SURVIVOR DETECTION ENGINE (HIGH RECALL)")
        print("=" * 65)

        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Model file not found: {onnx_model_path}")

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = (640, 640)

        available = ort.get_available_providers()
        providers = []
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        print(f"Binding model to: {providers[0]}...")
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.active_provider = self.session.get_providers()[0]

        if "Dml" in self.active_provider:
            self.device_name = "NVIDIA DirectML GPU"
        elif "CUDA" in self.active_provider:
            self.device_name = "NVIDIA CUDA GPU"
        else:
            self.device_name = "CPU (Fallback)"

        print(f"SUCCESS: Engine active on [{self.device_name}]")
        print("=" * 65)

        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        self.session.run(None, {"images": dummy})
        print("Survivor detection engine ready.\n")

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = frame.shape[:2]
        r = min(self.input_size[0] / h, self.input_size[1] / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        pad_w, pad_h = (self.input_size[1] - new_w) / 2, (self.input_size[0] - new_h) / 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        letterboxed = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        blob = letterboxed[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        return blob, r, (left, top)

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[SurvivorTarget], float]:
        """Runs survivor detection with exception shielding against bad frames."""
        orig_h, orig_w = frame.shape[:2]

        try:
            blob, r, (pad_x, pad_y) = self._preprocess(frame)

            # 1. Forward pass on GPU
            t0 = time.perf_counter()
            outputs = self.session.run(None, {"images": blob})
            latency_ms = (time.perf_counter() - t0) * 1000.0

            preds = outputs[0][0].T

            # 2. Filter candidates by human confidence (High Recall)
            scores = preds[:, 4]
            mask = scores >= self.conf_thresh
            valid_preds = preds[mask]

            if len(valid_preds) == 0:
                return frame.copy(), [], latency_ms

            cx = valid_preds[:, 0]
            cy = valid_preds[:, 1]
            bw = valid_preds[:, 2]
            bh = valid_preds[:, 3]

            x1 = (cx - bw / 2 - pad_x) / r
            y1 = (cy - bh / 2 - pad_y) / r
            w_scaled = bw / r
            h_scaled = bh / r

            x1 = np.clip(x1, 0, orig_w - 1)
            y1 = np.clip(y1, 0, orig_h - 1)
            w_scaled = np.clip(w_scaled, 1, orig_w)
            h_scaled = np.clip(h_scaled, 1, orig_h)

            boxes_for_nms = []
            for i in range(len(valid_preds)):
                boxes_for_nms.append([int(x1[i]), int(y1[i]), int(w_scaled[i]), int(h_scaled[i])])

            indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores[mask].tolist(), self.conf_thresh, self.iou_thresh)

            survivors: List[SurvivorTarget] = []
            annotated = frame.copy()

            if len(indices) == 0:
                return annotated, [], latency_ms

            indices = np.array(indices).flatten()

            # Pinhole focal approximation (~80° HFOV)
            focal_est = orig_w * 0.60

            for target_id, idx in enumerate(indices):
                pred_row = valid_preds[idx]
                conf = float(pred_row[4])
                bx, by, bw_i, bh_i = boxes_for_nms[idx]
                x2 = min(bx + bw_i, orig_w - 1)
                y2 = min(by + bh_i, orig_h - 1)

                raw_kpts = pred_row[5:].reshape(17, 3)
                scaled_kpts = np.zeros_like(raw_kpts)
                for k in range(17):
                    kx = (raw_kpts[k, 0] - pad_x) / r
                    ky = (raw_kpts[k, 1] - pad_y) / r
                    kc = raw_kpts[k, 2]
                    scaled_kpts[k] = [np.clip(kx, 0, orig_w - 1), np.clip(ky, 0, orig_h - 1), kc]

                vis_joints = int(np.sum(scaled_kpts[:, 2] > 0.35))
                aspect_ratio = bw_i / max(bh_i, 1)

                # Posture heuristic
                if aspect_ratio >= 1.25:
                    posture = "PRONE_FLAT"
                elif vis_joints <= 6:
                    posture = "PINNED_UPPER"
                elif aspect_ratio < 0.60:
                    posture = "UPRIGHT"
                else:
                    posture = "CURLED"

                # Entrapment severity heuristic
                if vis_joints <= 6:
                    entrapment = "SEVERELY_TRAPPED"
                    tag_color = (0, 0, 255)       # Red alert
                elif vis_joints <= 12:
                    entrapment = "PARTIALLY_TRAPPED"
                    tag_color = (0, 140, 255)     # Orange warning
                else:
                    entrapment = "EXPOSED"
                    tag_color = (0, 255, 120)     # Green

                # Bearing angle relative to camera center
                victim_center_x = (bx + x2) / 2
                bearing_rad = math.atan((victim_center_x - (orig_w / 2)) / focal_est)
                bearing_deg = math.degrees(bearing_rad)

                # Anthropometric Distance Estimation:
                # Based on adult biacromial shoulder span (~0.42m)
                has_shoulders = (scaled_kpts[5, 2] > 0.35 and scaled_kpts[6, 2] > 0.35)
                if has_shoulders:
                    shoulder_px = math.hypot(scaled_kpts[5, 0] - scaled_kpts[6, 0], scaled_kpts[5, 1] - scaled_kpts[6, 1])
                    dist_from_shoulders = (focal_est * 0.42) / max(shoulder_px, 12.0)
                else:
                    dist_from_shoulders = 999.0

                has_ankles = (scaled_kpts[15, 2] > 0.35 or scaled_kpts[16, 2] > 0.35)
                if has_ankles and posture == "UPRIGHT":
                    h_metric = 1.65  # Standing adult estimate
                else:
                    h_metric = 0.85  # Seated / upper body only

                dist_from_box = (focal_est * h_metric) / max(bh_i, 18)

                if has_shoulders and dist_from_shoulders < 12.0:
                    est_distance_m = 0.65 * dist_from_shoulders + 0.35 * dist_from_box
                else:
                    est_distance_m = dist_from_box

                est_distance_m = max(0.8, min(18.0, est_distance_m))

                survivor = SurvivorTarget(
                    target_id=target_id + 1,
                    bbox=(bx, by, x2, y2),
                    confidence=conf,
                    keypoints=scaled_kpts,
                    visible_joints=vis_joints,
                    posture=posture,
                    entrapment=entrapment,
                    bearing_deg=bearing_deg,
                    est_distance_m=est_distance_m
                )
                survivors.append(survivor)
                self._render_target(annotated, survivor, tag_color)

            return annotated, survivors, latency_ms

        except Exception as e:
            print(f"[SURVIVOR WARNING] Frame detection exception: {e}")
            return frame.copy(), [], 0.0

    def _render_target(self, img: np.ndarray, s: SurvivorTarget, color: Tuple[int, int, int]):
        bx, by, x2, y2 = s.bbox

        # Corner Brackets
        line_len = max(15, int(min(x2 - bx, y2 - by) * 0.22))
        thickness = 2
        cv2.line(img, (bx, by), (bx + line_len, by), color, thickness)
        cv2.line(img, (bx, by), (bx, by + line_len), color, thickness)
        cv2.line(img, (x2, by), (x2 - line_len, by), color, thickness)
        cv2.line(img, (x2, by), (x2, by + line_len), color, thickness)
        cv2.line(img, (bx, y2), (bx + line_len, y2), color, thickness)
        cv2.line(img, (bx, y2), (bx, y2 - line_len), color, thickness)
        cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - line_len), color, thickness)

        # Skeletal Bones
        kpts = s.keypoints
        for p1, p2 in SKELETON_BONES:
            if kpts[p1, 2] > 0.35 and kpts[p2, 2] > 0.35:
                pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                cv2.line(img, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        # Keypoint Joint Nodes
        for k in range(17):
            if kpts[k, 2] > 0.35:
                center = (int(kpts[k, 0]), int(kpts[k, 1]))
                cv2.circle(img, center, 4, (0, 200, 255), -1)
                cv2.circle(img, center, 5, (255, 255, 255), 1)

        # Survivor Header Tag
        tag_w, tag_h = 240, 52
        tag_y = max(10, by - tag_h - 4)
        tag_x = min(bx, img.shape[1] - tag_w - 10)

        cv2.rectangle(img, (tag_x, tag_y), (tag_x + tag_w, tag_y + tag_h), (15, 15, 15), -1)
        cv2.rectangle(img, (tag_x, tag_y), (tag_x + tag_w, tag_y + tag_h), color, 1)

        cv2.putText(img, f"SURVIVOR LOCK #{s.target_id} [{s.entrapment}]",
                    (tag_x + 6, tag_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"POSTURE: {s.posture} ({s.visible_joints}/17 Joints)",
                    (tag_x + 6, tag_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)
        bearing_str = f"+{s.bearing_deg:.1f}" if s.bearing_deg >= 0 else f"{s.bearing_deg:.1f}"
        cv2.putText(img, f"BEARING: {bearing_str} deg | EST DIST: ~{s.est_distance_m:.1f}m",
                    (tag_x + 6, tag_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 200), 1, cv2.LINE_AA)


def run_survivor_stream(source=0, conf=0.18):
    """Executes live survivor detection feed."""
    detector = SurvivorDetector("yolov8n-pose.onnx", conf_thresh=conf)

    is_network_stream = isinstance(source, str) and (
        source.startswith("http://")
        or source.startswith("https://")
        or source.startswith("rtsp://")
        or ":" in source
    )

    if is_network_stream:
        stream_url = format_stream_url(str(source))
        cap = ThreadedCamera(stream_url)
        time.sleep(1.0)
    else:
        is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        cap = cv2.VideoCapture(int(source) if is_webcam else source)
        if is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    fps_smooth = 30.0
    print("\n" + "=" * 65)
    print(" USAR SURVIVOR DETECTION LIVE (HIGH RECALL)")
    print(f" Hardware: {detector.device_name}")
    print(" Press 'q' key to stop.")
    print("=" * 65 + "\n")

    while cap.isOpened():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        annotated, survivors, gpu_latency = detector.detect(frame)

        dt = time.perf_counter() - t0
        curr_fps = 1.0 / dt if dt > 0 else 30.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * curr_fps

        h, w = frame.shape[:2]
        cv2.rectangle(annotated, (12, 12), (430, 80), (15, 15, 15), -1)
        cv2.rectangle(annotated, (12, 12), (430, 80), (60, 60, 60), 1)

        cv2.putText(annotated, "USAR DISASTER RESCUE PERCEPTION", (22, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"ACCELERATOR: {detector.device_name}", (22, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"GPU INFER: {gpu_latency:.1f}ms | {fps_smooth:.1f} FPS | VICTIMS: {len(survivors)}", (22, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("USAR Survivor Identification", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Detection session stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USAR Survivor Identification Engine")
    parser.add_argument("--url", type=str, default=None, help="Wireless Phone IP Webcam URL")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--conf", type=float, default=0.18, help="Confidence threshold (default: 0.18)")
    args = parser.parse_args()

    src = args.url if args.url else (args.video if args.video else args.webcam)
    run_survivor_stream(source=src, conf=args.conf)
