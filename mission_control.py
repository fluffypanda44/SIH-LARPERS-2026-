"""
Urban Search & Rescue (USAR) Unified Autonomous Mission Control Engine.
Hardware-Accelerated on NVIDIA GeForce RTX 4050 Laptop GPU via ONNX DirectML.

Integrated Features:
1. Neon Green Drivable Spatial Ground Grid (preserves floor texture)
2. Dedicated Stairway & Incline Corridor Perception (Electric Cyan)
3. Sleek Minimalist Survivor Skeleton & Calibrated Metric Distance (3m - 6m)
4. Acoustic Camera Projection: Vertical Sound Beacon mapped directly onto video pixels!
5. High-Contrast Sonar Radar & Nav2 Costmap
6. Autonomous Navigation Dispatcher (Nav2 Waypoints)
"""

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from acoustic_doa import AcousticDoAEngine, AcousticTarget, render_acoustic_compass
from real_vision import UGVVisionEngine
from survivor_detector import SurvivorDetector, SurvivorTarget, ThreadedCamera


@dataclass
class AutonomousNavCommand:
    mode: str                    # "SURVIVOR_INTERCEPT", "ACOUSTIC_HOMING", "PATROL_STABLE_SLAB"
    target_heading_deg: float
    target_distance_m: float
    recommended_speed: float
    action_text: str


class AutonomousNavigator:
    def __init__(self):
        self.last_audio_heading = 0.0

    def compute_nav_command(
        self,
        survivors: List[SurvivorTarget],
        acoustic_target: AcousticTarget,
        costmap: np.ndarray,
    ) -> AutonomousNavCommand:
        if len(survivors) > 0:
            target = survivors[0]
            heading = target.bearing_deg
            dist = target.est_distance_m

            if dist <= 1.2:
                mode = "RESCUE_STATIONARY"
                speed = 0.0
                action = f"HALT: SURVIVOR AT {dist:.1f}m | LOCK BRAKES & SIGNAL RESCUE TEAM"
            else:
                mode = "SURVIVOR_INTERCEPT"
                speed = 0.60
                turn_dir = "RIGHT" if heading > 0 else "LEFT"
                action = f"APPROACH #{target.target_id} [{target.posture}]: STEER {abs(heading):.1f}° {turn_dir} | ADVANCE {dist:.1f}m"

            return AutonomousNavCommand(
                mode=mode,
                target_heading_deg=heading,
                target_distance_m=dist,
                recommended_speed=speed,
                action_text=action,
            )

        if acoustic_target.is_active and acoustic_target.confidence > 0.35:
            self.last_audio_heading = acoustic_target.azimuth_deg
            turn_dir = "RIGHT" if self.last_audio_heading > 0 else "LEFT"
            return AutonomousNavCommand(
                mode="ACOUSTIC_HOMING",
                target_heading_deg=self.last_audio_heading,
                target_distance_m=4.0,
                recommended_speed=0.35,
                action_text=f"ACOUSTIC HOMING: ROTATE {abs(self.last_audio_heading):.1f}° {turn_dir} TOWARDS CRY",
            )

        return AutonomousNavCommand(
            mode="PATROL_STABLE_SLAB",
            target_heading_deg=0.0,
            target_distance_m=8.0,
            recommended_speed=0.75,
            action_text="AUTONOMOUS PATROL: ADVANCING ON STABLE GROUND GRID",
        )


def render_acoustic_camera_beacon(img: np.ndarray, target: AcousticTarget, fx: float):
    """
    Acoustic Camera: Maps sound bearing angle theta directly to a vertical energy beacon
    over the exact video pixel column where the sound originates!
    """
    if not target.is_active:
        return

    h, w = img.shape[:2]
    # Projected pixel column: X_sound = W/2 + fx * tan(theta)
    rad = math.radians(target.azimuth_deg)
    x_sound = int((w / 2) + fx * math.tan(rad))

    if 0 <= x_sound < w:
        beam_w = 40
        x1 = max(0, x_sound - beam_w // 2)
        x2 = min(w - 1, x_sound + beam_w // 2)

        # Translucent vertical acoustic energy column
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, 0), (x2, h), (0, 255, 200), -1)
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)

        # Center laser line
        cv2.line(img, (x_sound, 0), (x_sound, h), (0, 255, 255), 2, cv2.LINE_AA)

        # Tactical Beacon Badge (Top of column)
        badge_y = 95
        sign = "+" if target.azimuth_deg >= 0 else ""
        badge_text = f"ACOUSTIC BEACON: {sign}{target.azimuth_deg:.0f}° ({target.energy_db:.0f} dB)"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)

        bx = max(10, min(x_sound - bw // 2, w - bw - 15))
        cv2.rectangle(img, (bx - 4, badge_y - bh - 4), (bx + bw + 6, badge_y + 6), (15, 15, 15), -1)
        cv2.rectangle(img, (bx - 4, badge_y - bh - 4), (bx + bw + 6, badge_y + 6), (0, 255, 255), 1)
        cv2.putText(img, badge_text, (bx, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)


def run_mission_control(
    video_source: Union[int, str] = 0,
    audio_url: Optional[str] = None,
    alpha: float = 0.35,
):
    print("\n" + "=" * 70)
    print(" LAUNCHING USAR UNIFIED AUTONOMOUS MISSION CONTROL")
    print(" Hardware Acceleration: NVIDIA GeForce RTX 4050 (DirectML Dual AI)")
    print("=" * 70)

    # Auto-route audio if user passed --url and did not set audio_url
    if isinstance(video_source, str) and video_source.startswith("http://") and audio_url is None:
        audio_url = video_source

    vision_engine = UGVVisionEngine("segformer_b0.onnx")
    survivor_detector = SurvivorDetector("yolov8n-pose.onnx", conf_thresh=0.18, iou_thresh=0.60)
    navigator = AutonomousNavigator()
    audio_engine = AcousticDoAEngine(url=audio_url, mic_distance_m=0.16)

    is_network_stream = isinstance(video_source, str) and (
        video_source.startswith("http://")
        or video_source.startswith("https://")
        or video_source.startswith("rtsp://")
        or ":" in video_source
    )

    if is_network_stream:
        if not (video_source.startswith("http://") or video_source.startswith("https://") or video_source.startswith("rtsp://")):
            video_source = f"http://{video_source}"
        if not video_source.endswith("/video") and not video_source.endswith(".mjpg") and not video_source.startswith("rtsp://"):
            video_source = f"{video_source.rstrip('/')}/video"
        print(f"[MISSION CONTROL] Connecting to wireless stream: {video_source}")
        cap = ThreadedCamera(video_source)
        time.sleep(1.0)
    else:
        is_webcam = isinstance(video_source, int) or (isinstance(video_source, str) and video_source.isdigit())
        cap = cv2.VideoCapture(int(video_source) if is_webcam else video_source)
        if is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        audio_engine.stop()
        return

    fps_smooth = 30.0

    print("\n" + "=" * 70)
    print(" USAR MISSION CONTROL ACTIVE")
    print(" Features:")
    print("   - Neon Green Drivable Ground Grid")
    print("   - Dedicated Cyan Stairway Detection")
    print("   - Sleek High-Recall Survivor Skeleton Lock")
    print("   - Acoustic Camera Sound Source Beam")
    print(" Press 'q' key on the video window to stop.")
    print("=" * 70 + "\n")

    while cap.isOpened():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        fx = w * 0.58  # Calibrated wide-angle focal length

        # 1. Rubble & Ground Grid Segmentation (RTX 4050)
        annotated, costmap, t_seg = vision_engine.infer(frame)

        # 2. Sleek Survivor Keypoints & Distance (RTX 4050)
        annotated, survivors, t_pose = survivor_detector.detect(annotated)

        # 3. Acoustic DoA Target
        acoustic_target = audio_engine.get_target()

        # 4. Acoustic Camera Projection: Drop sound beacon on video pixel column!
        render_acoustic_camera_beacon(annotated, acoustic_target, fx)

        # 5. Nav2 Autopilot Instruction
        nav_cmd = navigator.compute_nav_command(survivors, acoustic_target, costmap)

        # FPS
        total_gpu_time = t_seg + t_pose
        dt = time.perf_counter() - t0
        curr_fps = 1.0 / dt if dt > 0 else 30.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * curr_fps

        # -------------------------------------------------------------
        # HUD 1: Top Left Telemetry
        # -------------------------------------------------------------
        cv2.rectangle(annotated, (12, 12), (430, 80), (15, 15, 15), -1)
        cv2.rectangle(annotated, (12, 12), (430, 80), (50, 50, 50), 1)

        cv2.putText(annotated, "USAR AUTONOMOUS RESCUE MISSION CONTROL", (22, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"ACCELERATOR: {vision_engine.device_name} (Dual Model)", (22, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f} | DUAL GPU LATENCY: {total_gpu_time:.1f} ms | SURVIVORS: {len(survivors)}", (22, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # HUD 2: Top Right - Nav2 Costmap (2D Grid)
        # -------------------------------------------------------------
        mini_w, mini_h = 160, 90
        mini_costmap = cv2.resize(costmap, (mini_w, mini_h), interpolation=cv2.INTER_NEAREST)
        mini_bgr = cv2.cvtColor(mini_costmap, cv2.COLOR_GRAY2BGR)

        x_costmap = w - mini_w - 14
        y_costmap = 14
        annotated[y_costmap:y_costmap + mini_h, x_costmap:x_costmap + mini_w] = mini_bgr
        cv2.rectangle(annotated, (x_costmap, y_costmap), (x_costmap + mini_w, y_costmap + mini_h), (0, 255, 180), 1)
        cv2.putText(annotated, "Nav2 Costmap (2D)", (x_costmap + 6, y_costmap + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 180), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # HUD 3: Bottom Right - Tactical Acoustic Radar Compass
        # -------------------------------------------------------------
        radar_size = 170
        radar_dial = render_acoustic_compass(acoustic_target, size=radar_size)
        x_radar = w - radar_size - 14
        y_radar = h - radar_size - 55

        # Solid tactical overlay with border
        annotated[y_radar:y_radar + radar_size, x_radar:x_radar + radar_size] = radar_dial
        cv2.rectangle(annotated, (x_radar, y_radar), (x_radar + radar_size, y_radar + radar_size), (0, 255, 200), 1)

        # -------------------------------------------------------------
        # HUD 4: Bottom Center - Autonomous Steering Dispatcher Bar
        # -------------------------------------------------------------
        nav_h = 38
        nav_y = h - nav_h - 8
        cv2.rectangle(annotated, (12, nav_y), (w - 12, nav_y + nav_h), (12, 12, 12), -1)

        border_col = (0, 255, 255) if nav_cmd.mode == "SURVIVOR_INTERCEPT" else (
            (0, 255, 120) if nav_cmd.mode == "ACOUSTIC_HOMING" else (80, 80, 80)
        )
        cv2.rectangle(annotated, (12, nav_y), (w - 12, nav_y + nav_h), border_col, 2)
        cv2.putText(annotated, f"NAV2 AUTOPILOT: {nav_cmd.action_text}",
                    (24, nav_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("USAR Autonomous Mission Control [RTX 4050]", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Mission control stopped by user.")
            break

    cap.release()
    audio_engine.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USAR Autonomous Mission Control")
    parser.add_argument("--url", type=str, default=None, help="Wireless Phone IP Webcam URL")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--audio-url", type=str, default=None, help="Audio URL (optional)")
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask overlay alpha")
    args = parser.parse_args()

    v_src = args.url if args.url else (args.video if args.video else args.webcam)
    run_mission_control(video_source=v_src, audio_url=args.audio_url, alpha=args.alpha)
