"""
Urban Search & Rescue (USAR) Sleek Survivor Identification Engine.
Hardware-Accelerated on NVIDIA GeForce RTX 4050 Laptop GPU via ONNX DirectML.

Optimized for:
- High-recall detection (conf >= 0.18, IoU = 0.60) to track adjacent/occluded survivors.
- Calibrated wide-angle distance estimator (3m - 6m real-world range).
- Sleek minimalist antialiased skeletal bone telemetry with micro-joint nodes.
"""

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort


# COCO Pose Skeleton Connectivity
SKELETON_BONES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]


@dataclass
class SurvivorTarget:
    target_id: int
    bbox: Tuple[int, int, int, int]       # (x1, y1, x2, y2)
    confidence: float
    keypoints: np.ndarray                 # (17, 3) [x, y, conf]
    visible_joints: int
    posture: str                          # PRONE_FLAT, PINNED_UPPER, CURLED, UPRIGHT
    entrapment: str                       # SEVERELY_TRAPPED, PARTIALLY_TRAPPED, EXPOSED
    bearing_deg: float
    est_distance_m: float


class ThreadedCamera:
    """Zero-latency threaded stream reader for wireless phone IP cameras."""

    def __init__(self, src: str):
        print(f"[STREAM] Connecting to low-latency stream: {src}")
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
        print("[STREAM] Stream thread active.")

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


class SurvivorDetector:
    """High-speed survivor pose detector running on RTX 4050 GPU."""

    def __init__(self, onnx_model_path: str = "yolov8n-pose.onnx", conf_thresh: float = 0.18, iou_thresh: float = 0.60):
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

        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.device_name = "NVIDIA RTX 4050 (DirectML)" if "Dml" in self.active_provider else "CPU"

        # Warm up GPU
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        self.session.run(None, {"images": dummy})

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
        orig_h, orig_w = frame.shape[:2]
        blob, r, (pad_x, pad_y) = self._preprocess(frame)

        # 1. Forward pass on RTX 4050 GPU
        t0 = time.perf_counter()
        outputs = self.session.run(None, {"images": blob})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        preds = outputs[0][0].T

        # 2. Filter candidates by human confidence (High recall: >= 0.18)
        scores = preds[:, 4]
        mask = scores >= self.conf_thresh
        valid_preds = preds[mask]

        if len(valid_preds) == 0:
            return frame.copy(), [], latency_ms

        cx = valid_preds[:, 0]
        cy = valid_preds[:, 1]
        bw = valid_preds[:, 2]
        bh = valid_preds[:, 3]

        x1 = np.clip((cx - bw / 2 - pad_x) / r, 0, orig_w - 1)
        y1 = np.clip((cy - bh / 2 - pad_y) / r, 0, orig_h - 1)
        w_scaled = np.clip(bw / r, 1, orig_w)
        h_scaled = np.clip(bh / r, 1, orig_h)

        boxes_for_nms = []
        for i in range(len(valid_preds)):
            boxes_for_nms.append([int(x1[i]), int(y1[i]), int(w_scaled[i]), int(h_scaled[i])])

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores[mask].tolist(), self.conf_thresh, self.iou_thresh)

        survivors: List[SurvivorTarget] = []
        annotated = frame.copy()

        if len(indices) == 0:
            return annotated, [], latency_ms

        indices = np.array(indices).flatten()

        # Wide-angle optical calibration: Nothing Phone 2a has ~82 deg HFOV -> fx ≈ 0.58 * orig_w
        fx = orig_w * 0.58

        for target_id, idx in enumerate(indices):
            pred_row = valid_preds[idx]
            conf = float(pred_row[4])
            bx, by, bw_i, bh_i = boxes_for_nms[idx]
            x2 = min(bx + bw_i, orig_w - 1)
            y2 = min(by + bh_i, orig_h - 1)

            # Keypoints
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
            if aspect_ratio >= 1.20:
                posture = "PRONE_FLAT"
            elif vis_joints <= 6:
                posture = "PINNED_UPPER"
            elif aspect_ratio < 0.65:
                posture = "UPRIGHT"
            else:
                posture = "CURLED"

            if vis_joints <= 6:
                entrapment = "SEVERELY_TRAPPED"
                tag_color = (0, 0, 255)       # Red
            elif vis_joints <= 12:
                entrapment = "PARTIALLY_TRAPPED"
                tag_color = (0, 160, 255)     # Amber
            else:
                entrapment = "EXPOSED"
                tag_color = (0, 255, 120)     # Neon Green

            # Horizontal Bearing Angle
            victim_center_x = (bx + x2) / 2
            bearing_deg = math.degrees(math.atan((victim_center_x - (orig_w / 2)) / fx))

            # Calibrated Anatomical Distance Estimator
            # If standing/ankles visible -> 1.65m; if seated/upper torso -> 0.85m
            ankles_vis = (scaled_kpts[15, 2] > 0.35 or scaled_kpts[16, 2] > 0.35)
            if ankles_vis and posture == "UPRIGHT":
                ref_h = 1.65
            else:
                ref_h = 0.90 if posture != "PRONE_FLAT" else 0.50

            est_distance_m = max(0.8, min(15.0, (fx * ref_h) / max(bh_i, 25)))

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

            # Render Sleek Minimalist Reticle & Bones
            self._render_sleek_survivor(annotated, survivor, tag_color)

        return annotated, survivors, latency_ms

    def _render_sleek_survivor(self, img: np.ndarray, s: SurvivorTarget, color: Tuple[int, int, int]):
        """Renders sleek 1.5px antialiased bone vectors and minimalist floating pill badge."""
        bx, by, x2, y2 = s.bbox
        kpts = s.keypoints

        # Sleek Antialiased Bones (thin 1-2px)
        for p1, p2 in SKELETON_BONES:
            if kpts[p1, 2] > 0.35 and kpts[p2, 2] > 0.35:
                pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                cv2.line(img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(img, pt1, pt2, (0, 240, 255), 1, cv2.LINE_AA)

        # Micro Joint Nodes (3px radius)
        for k in range(17):
            if kpts[k, 2] > 0.35:
                center = (int(kpts[k, 0]), int(kpts[k, 1]))
                cv2.circle(img, center, 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, center, 4, (0, 0, 0), 1, cv2.LINE_AA)

        # Minimalist Corner Ticks (No giant boxes)
        tick_len = max(8, int(min(x2 - bx, y2 - by) * 0.15))
        # Top Left
        cv2.line(img, (bx, by), (bx + tick_len, by), color, 1, cv2.LINE_AA)
        cv2.line(img, (bx, by), (bx, by + tick_len), color, 1, cv2.LINE_AA)
        # Top Right
        cv2.line(img, (x2, by), (x2 - tick_len, by), color, 1, cv2.LINE_AA)
        cv2.line(img, (x2, by), (x2, by + tick_len), color, 1, cv2.LINE_AA)
        # Bottom Left
        cv2.line(img, (bx, y2), (bx + tick_len, y2), color, 1, cv2.LINE_AA)
        cv2.line(img, (bx, y2), (bx, y2 - tick_len), color, 1, cv2.LINE_AA)
        # Bottom Right
        cv2.line(img, (x2, y2), (x2 - tick_len, y2), color, 1, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - tick_len), color, 1, cv2.LINE_AA)

        # Low-Profile Telemetry Pill Badge
        sign = "+" if s.bearing_deg >= 0 else ""
        pill_text = f"#{s.target_id} [{s.posture}] {s.est_distance_m:.1f}m {sign}{s.bearing_deg:.0f}°"
        (tw, th), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)

        pill_x = max(5, min(bx, img.shape[1] - tw - 16))
        pill_y = max(18, by - 6)

        # Frosted Pill Background
        cv2.rectangle(img, (pill_x - 2, pill_y - th - 4), (pill_x + tw + 6, pill_y + 4), (10, 10, 10), -1)
        cv2.rectangle(img, (pill_x - 2, pill_y - th - 4), (pill_x + tw + 6, pill_y + 4), color, 1)
        cv2.putText(img, pill_text, (pill_x + 2, pill_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
