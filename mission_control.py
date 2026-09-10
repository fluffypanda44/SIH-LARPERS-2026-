"""
Urban Search & Rescue (USAR) Unified Autonomous Mission Control Engine.
Hardware-Accelerated on NVIDIA GeForce RTX 4050 Laptop GPU via ONNX DirectML.

Clean, authentic disaster perception (zero fake HUDs, zero artificial neon floor grids):
1. Rubble & Stairway Traversability (SegFormer-B0 DirectML: Green=Floor, Cyan=Stairway, Red=Obstacle)
2. Trapped Survivor Keypoint & Posture Detection (YOLOv8-Pose DirectML: High-Recall conf=0.18)
3. Acoustic Voice Direction of Arrival Radar (GCC-PHAT Microphone Array in bottom-right corner)
4. Autonomous Navigation Waypoint & Robot Steering Dispatcher (Nav2 Compatible)

Supports:
- Local Webcams / Integrated HD Cameras
- Wireless Phone IP Webcam Streams (--url http://<IP>:8080)
- Recorded Disaster Video Files (--video <path>)
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

# Modular engines
from acoustic_doa import AcousticDoAEngine, AcousticTarget, render_acoustic_compass
from real_vision import UGVVisionEngine
from survivor_detector import SurvivorDetector, SurvivorTarget, ThreadedCamera


@dataclass
class AutonomousNavCommand:
    mode: str                    # "VISUAL_SURVIVOR_LOCK", "ACOUSTIC_HOMING", "PATROL_STABLE_SLAB"
    target_heading_deg: float    # Desired steering angle (-90° to +90°)
    target_distance_m: float     # Estimated distance
    recommended_speed: float     # Linear speed m/s
    action_text: str             # Human readable mission dispatch command


class AutonomousNavigator:
    """Computes real-time navigation dispatch commands based on multi-modal sensory input."""

    def __init__(self):
        self.last_audio_heading = 0.0

    def compute_nav_command(
        self,
        survivors: List[SurvivorTarget],
        acoustic_target: AcousticTarget,
        costmap: np.ndarray,
    ) -> AutonomousNavCommand:
        # Priority 1: Visual Survivor Lock (Line-of-Sight)
        if len(survivors) > 0:
            target = survivors[0]
            heading = target.bearing_deg
            dist = target.est_distance_m

            if dist <= 1.2:
                mode = "RESCUE_STATIONARY"
                speed = 0.0
                action = f"HALT: SURVIVOR AT {dist:.1f}m | LOCK BRAKES & SIGNAL CREW"
            else:
                mode = "VISUAL_SURVIVOR_LOCK"
                speed = 0.65 if target.entrapment == "EXPOSED" else 0.40
                turn_dir = "RIGHT" if heading > 0 else "LEFT"
                action = f"APPROACH SURVIVOR #{target.target_id} [{target.posture}]: STEER {abs(heading):.1f}° {turn_dir} | ADVANCE {dist:.1f}m"

            return AutonomousNavCommand(
                mode=mode,
                target_heading_deg=heading,
                target_distance_m=dist,
                recommended_speed=speed,
                action_text=action,
            )

        # Priority 2: Acoustic Homing (Trapped sound behind debris / Non-Line-of-Sight)
        if acoustic_target.is_active and acoustic_target.confidence > 0.30:
            self.last_audio_heading = acoustic_target.azimuth_deg
            turn_dir = "RIGHT" if self.last_audio_heading > 0 else "LEFT"
            return AutonomousNavCommand(
                mode="ACOUSTIC_HOMING",
                target_heading_deg=self.last_audio_heading,
                target_distance_m=5.0,
                recommended_speed=0.35,
                action_text=f"ACOUSTIC HOMING: ROTATE {abs(self.last_audio_heading):.1f}° {turn_dir} TOWARDS DISTRESS SOUND",
            )

        # Priority 3: Autonomous Patrol on Safe Rubble / Slabs
        h, w = costmap.shape[:2]
        forward_sector = costmap[int(h * 0.70):, int(w * 0.35):int(w * 0.65)]
        mean_cost = float(np.mean(forward_sector)) if forward_sector.size > 0 else 0.0

        if mean_cost > 150:
            action = "OBSTACLE IN FRONT: SCANNING ALTERNATE CORRIDOR"
            speed = 0.0
            heading = 25.0
        else:
            action = "SECTOR PATROL: FORWARD PATH CLEAR ON STABLE GROUND"
            speed = 0.80
            heading = 0.0

        return AutonomousNavCommand(
            mode="PATROL_STABLE_SLAB",
            target_heading_deg=heading,
            target_distance_m=10.0,
            recommended_speed=speed,
            action_text=action,
        )


def run_mission_control(
    video_source: Union[int, str] = 0,
    audio_url: Optional[str] = None,
    alpha: float = 0.35,
):
    print("\n" + "=" * 70)
    print(" LAUNCHING USAR UNIFIED AUTONOMOUS MISSION CONTROL")
    print(" Hardware Acceleration: NVIDIA GeForce RTX 4050 (DirectML)")
    print(" Clean Perception: Semantic Segmentation + Survivor Tracking + Acoustic Radar")
    print("=" * 70)

    # 1. Initialize Dual AI Engines on RTX 4050
    vision_engine = UGVVisionEngine("segformer_b0.onnx")
    survivor_detector = SurvivorDetector("yolov8n-pose.onnx", conf_thresh=0.18, iou_thresh=0.60)
    navigator = AutonomousNavigator()

    # 2. Initialize Audio DoA Engine
    # If audio_url is not provided, defaults to laptop's built-in stereo array for true left/right DoA!
    audio_engine = AcousticDoAEngine(url=audio_url, mic_distance_m=0.16)

    # 3. Initialize Video Ingestion
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
        print(f"[MISSION CONTROL] Connecting to video stream: {video_source}")
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
    show_traversability = True
    show_survivors = True
    show_radar = True

    print("\n" + "=" * 70)
    print(" USAR MISSION CONTROL ACTIVE")
    print(" Controls:")
    print("   - Press 'q' to exit")
    print("   - Press 't' to toggle Traversability Mask")
    print("   - Press 's' to toggle Survivor Detection")
    print("   - Press 'r' to toggle Acoustic Radar")
    print("=" * 70 + "\n")

    while cap.isOpened():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # STEP 1: Rubble & Stairway Traversability (RTX 4050 GPU)
        # -------------------------------------------------------------
        if show_traversability:
            color_mask, costmap, t_seg = vision_engine.infer(frame)
            # Authentic semi-transparent semantic overlay (zero fake grids)
            blended = cv2.addWeighted(color_mask, alpha, frame, 1.0 - alpha, 0)
        else:
            blended = frame.copy()
            costmap = np.zeros((h, w), dtype=np.uint8)
            t_seg = 0.0

        # -------------------------------------------------------------
        # STEP 2: Survivor Detection & Skeleton Lock (RTX 4050 GPU)
        # -------------------------------------------------------------
        if show_survivors:
            blended, survivors, t_pose = survivor_detector.detect(blended)
        else:
            survivors = []
            t_pose = 0.0

        # -------------------------------------------------------------
        # STEP 3: Acoustic Voice Direction Acquisition (Radar only)
        # -------------------------------------------------------------
        acoustic_target = audio_engine.get_target()

        # -------------------------------------------------------------
        # STEP 4: Autonomous Navigation Decision Loop
        # -------------------------------------------------------------
        nav_cmd = navigator.compute_nav_command(survivors, acoustic_target, costmap)

        # Timing & Framerate
        total_gpu_time = t_seg + t_pose
        dt = time.perf_counter() - t0
        curr_fps = 1.0 / dt if dt > 0 else 30.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * curr_fps

        # -------------------------------------------------------------
        # HUD 1: Top Left Mission Telemetry Header
        # -------------------------------------------------------------
        header_w, header_h = 450, 88
        cv2.rectangle(blended, (12, 12), (12 + header_w, 12 + header_h), (15, 15, 15), -1)
        cv2.rectangle(blended, (12, 12), (12 + header_w, 12 + header_h), (50, 50, 50), 1)

        cv2.putText(blended, "USAR AUTONOMOUS RESCUE MISSION CONTROL", (22, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(blended, "ACCELERATOR: NVIDIA RTX 4050 (DirectML Dual AI)", (22, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(blended, f"PIPELINE: {fps_smooth:.1f} FPS | DUAL GPU LATENCY: {total_gpu_time:.1f} ms", (22, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(blended, f"SURVIVORS: {len(survivors)} | AUDIO: {acoustic_target.status_text} [{acoustic_target.source_info}]", (22, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # HUD 2: Picture-in-Picture 1: Nav2 Metric Costmap (Top Right)
        # -------------------------------------------------------------
        mini_w, mini_h = 160, 90
        mini_costmap = cv2.resize(costmap, (mini_w, mini_h), interpolation=cv2.INTER_NEAREST)
        mini_bgr = cv2.cvtColor(mini_costmap, cv2.COLOR_GRAY2BGR)

        x_costmap = w - mini_w - 14
        y_costmap = 14
        blended[y_costmap:y_costmap + mini_h, x_costmap:x_costmap + mini_w] = mini_bgr
        cv2.rectangle(blended, (x_costmap, y_costmap), (x_costmap + mini_w, y_costmap + mini_h), (0, 255, 180), 1)
        cv2.putText(blended, "Nav2 Costmap (2D)", (x_costmap + 6, y_costmap + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 180), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # HUD 3: Picture-in-Picture 2: Acoustic Sonar Radar (Bottom Right)
        # -------------------------------------------------------------
        if show_radar:
            radar_size = 180
            radar_dial = render_acoustic_compass(acoustic_target, size=radar_size)
            x_radar = w - radar_size - 14
            y_radar = h - radar_size - 60

            # Render solid tactical radar window
            blended[y_radar:y_radar + radar_size, x_radar:x_radar + radar_size] = radar_dial
            cv2.rectangle(blended, (x_radar, y_radar), (x_radar + radar_size, y_radar + radar_size), (0, 255, 200), 1)

        # -------------------------------------------------------------
        # HUD 4: Bottom Center Autonomous Navigation Command Bar
        # -------------------------------------------------------------
        nav_bar_h = 42
        nav_bar_y = h - nav_bar_h - 10
        cv2.rectangle(blended, (12, nav_bar_y), (w - 12, nav_bar_y + nav_bar_h), (12, 12, 12), -1)

        if nav_cmd.mode == "VISUAL_SURVIVOR_LOCK":
            border_col = (0, 255, 255)   # Yellow
            mode_tag = "[SURVIVOR INTERCEPT]"
        elif nav_cmd.mode == "RESCUE_STATIONARY":
            border_col = (0, 0, 255)     # Red
            mode_tag = "[TARGET REACHED]"
        elif nav_cmd.mode == "ACOUSTIC_HOMING":
            border_col = (0, 255, 120)   # Bright green
            mode_tag = "[ACOUSTIC HOMING]"
        else:
            border_col = (100, 100, 100) # Neutral
            mode_tag = "[AUTONOMOUS PATROL]"

        cv2.rectangle(blended, (12, nav_bar_y), (w - 12, nav_bar_y + nav_bar_h), border_col, 2)
        cv2.putText(blended, f"NAV2 AUTOPILOT {mode_tag}: {nav_cmd.action_text}",
                    (24, nav_bar_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("USAR Autonomous Mission Control [NVIDIA RTX 4050]", blended)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Mission control stopped by operator.")
            break
        elif key == ord("t"):
            show_traversability = not show_traversability
        elif key == ord("s"):
            show_survivors = not show_survivors
        elif key == ord("r"):
            show_radar = not show_radar

    cap.release()
    audio_engine.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USAR Autonomous Mission Control")
    parser.add_argument("--url", type=str, default=None, help="Wireless Phone IP Webcam URL (e.g. http://172.21.131.53:8080)")
    parser.add_argument("--webcam", type=int, default=0, help="Local Webcam device ID (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--audio-url", type=str, default=None, help="Network audio URL (optional)")
    parser.add_argument("--alpha", type=float, default=0.35, help="Traversability mask alpha (default: 0.35)")
    args = parser.parse_args()

    v_src = args.url if args.url else (args.video if args.video else args.webcam)
    # Auto-route audio URL to match video URL if user passed --url
    a_src = args.audio_url if args.audio_url else (args.url if args.url else None)

    run_mission_control(video_source=v_src, audio_url=a_src, alpha=args.alpha)
