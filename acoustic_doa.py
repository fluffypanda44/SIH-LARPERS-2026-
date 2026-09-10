"""
Urban Search & Rescue (USAR) Acoustic Direction of Arrival (DoA) Engine.
Localizes trapped victims and distress sounds ("Where is the voice coming from?").

Technique:
- Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
- Real-time time delay of arrival (TDoA) calculation across stereo microphone array.
- Bandpass vocal frequency filtering (300 Hz - 3400 Hz) to reject robot chassis/motor hum.
- Real-time tactical acoustic compass HUD dial.
"""

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import sounddevice as sd


@dataclass
class AcousticTarget:
    is_active: bool            # True if sound above noise threshold
    azimuth_deg: float         # Bearing angle: -deg (Left) to +deg (Right)
    confidence: float          # Correlation peak sharpness (0.0 to 1.0)
    energy_db: float           # Signal energy in dB
    status_text: str           # "VOICE DETECTED", "SEARCHING...", "IDLE"


class AcousticDoAEngine:
    """
    Real-time Sound Source Localization (SSL) using GCC-PHAT.
    Captures stereo microphone stream and continuously computes sound arrival bearing.
    """

    def __init__(
        self,
        samplerate: int = 44100,
        buffer_size: int = 4096,
        mic_distance_m: float = 0.15,
        device_index: Optional[int] = None,
        speed_of_sound: float = 343.0,
    ):
        self.fs = samplerate
        self.buffer_size = buffer_size
        self.d = mic_distance_m
        self.c = speed_of_sound
        self.max_tau = self.d / self.c  # Max physical time delay in seconds

        self.lock = threading.Lock()
        self.audio_buffer = np.zeros((self.buffer_size, 2), dtype=np.float32)
        self.current_target = AcousticTarget(
            is_active=False,
            azimuth_deg=0.0,
            confidence=0.0,
            energy_db=-60.0,
            status_text="INITIALIZING",
        )
        self.smoothed_angle = 0.0
        self.running = True

        # Precompute bandpass filter mask in frequency domain (300Hz - 3400Hz)
        n_fft = 2 * self.buffer_size
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.fs)
        self.freq_mask = (freqs >= 250.0) & (freqs <= 3800.0)

        # Open audio stream
        print("=" * 60)
        print("INITIALIZING ACOUSTIC VOICE LOCALIZATION ENGINE (PHASE 2)")
        print("=" * 60)
        print(f"Sampling Rate: {self.fs} Hz | Buffer Size: {self.buffer_size}")
        print(f"Microphone Baseline (d): {self.d * 100:.1f} cm | Speed of Sound: {self.c} m/s")

        try:
            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=2,
                blocksize=self.buffer_size // 2,
                device=device_index,
                callback=self._audio_callback,
                dtype="float32",
            )
            self.stream.start()
            print("[AUDIO] Hardware stereo stream active.")
        except Exception as e:
            print(f"[ERROR] Failed to start sounddevice stream: {e}")
            self.running = False
            return

        # Start background processor thread
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        print("[AUDIO] GCC-PHAT spatial resolver thread running.")
        print("=" * 60 + "\n")

    def _audio_callback(self, indata, frames, time_info, status):
        """Asynchronously receives audio blocks from sound hardware."""
        if not self.running:
            return
        with self.lock:
            # Shift buffer left and append new chunk
            chunk_len = len(indata)
            self.audio_buffer[:-chunk_len] = self.audio_buffer[chunk_len:]
            self.audio_buffer[-chunk_len:] = indata

    def _gcc_phat(self, s1: np.ndarray, s2: np.ndarray, interp: int = 8) -> Tuple[float, float]:
        """
        Computes Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
        Returns:
            tau: Time delay in seconds
            confidence: Peak sharpness ratio
        """
        n = s1.shape[0] + s2.shape[0]
        S1 = np.fft.rfft(s1, n=n)
        S2 = np.fft.rfft(s2, n=n)

        # Apply vocal bandpass mask
        S1 = S1 * self.freq_mask
        S2 = S2 * self.freq_mask

        # Cross-Power Spectral Density
        R = S1 * np.conj(S2)
        # Phase Transform Normalization (strips amplitude reverberations)
        denom = np.abs(R) + 1e-15
        R_norm = R / denom

        # Inverse FFT with high interpolation for sub-sample precision
        cc = np.fft.irfft(R_norm, n=interp * n)
        max_shift = int(interp * self.fs * self.max_tau)

        # Re-center cross correlation
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

        # Peak detection
        peak_idx = np.argmax(np.abs(cc))
        peak_val = np.abs(cc[peak_idx])
        mean_val = np.mean(np.abs(cc)) + 1e-15
        confidence = float(min(1.0, (peak_val / (mean_val * 4.0))))

        shift = peak_idx - max_shift
        tau = shift / float(interp * self.fs)
        return tau, confidence

    def _process_loop(self):
        """Continuous background loop computing DoA."""
        while self.running:
            with self.lock:
                buf = self.audio_buffer.copy()

            s1 = buf[:, 0]
            s2 = buf[:, 1]

            # Compute audio energy (RMS in dB)
            rms = np.sqrt(np.mean(s1**2 + s2**2) / 2.0) + 1e-12
            db = 20.0 * math.log10(rms)

            # Energy gate (distress cry vs background hiss)
            if db > -42.0:  # Active sound detected
                tau, conf = self._gcc_phat(s1, s2)

                # Solve angle theta: sin(theta) = c * tau / d
                sin_arg = np.clip((self.c * tau) / self.d, -1.0, 1.0)
                # Angle: left is negative, right is positive
                angle_deg = -math.degrees(math.asin(sin_arg))

                # Smooth with exponential moving average (EMA)
                alpha = 0.35 if conf > 0.4 else 0.10
                self.smoothed_angle = (1.0 - alpha) * self.smoothed_angle + alpha * angle_deg

                status = "VOICE / CRY DETECTED" if conf > 0.45 else "ACOUSTIC ACTIVITY"
                is_active = True
            else:
                conf = 0.0
                status = "MONITORING RUBBLE..."
                is_active = False

            with self.lock:
                self.current_target = AcousticTarget(
                    is_active=is_active,
                    azimuth_deg=self.smoothed_angle,
                    confidence=conf,
                    energy_db=db,
                    status_text=status,
                )

            time.sleep(0.03)  # ~30 Hz update rate

    def get_target(self) -> AcousticTarget:
        """Returns the latest acoustic bearing target."""
        with self.lock:
            return self.current_target

    def stop(self):
        """Releases audio stream."""
        self.running = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()


def render_acoustic_compass(target: AcousticTarget, size: int = 420) -> np.ndarray:
    """
    Renders an authentic, tactical acoustic radar compass dial.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2 + 10)
    radius = int(size * 0.38)

    # 1. Dark circular radar background
    cv2.circle(img, center, radius, (20, 20, 20), -1)
    # Range rings
    cv2.circle(img, center, radius, (60, 60, 60), 1)
    cv2.circle(img, center, int(radius * 0.66), (40, 40, 40), 1)
    cv2.circle(img, center, int(radius * 0.33), (40, 40, 40), 1)

    # 2. Polar Azimuth Tick Marks (-90 to +90)
    for ang in range(-90, 91, 15):
        rad = math.radians(ang - 90)
        x_inner = int(center[0] + (radius - 8) * math.cos(rad))
        y_inner = int(center[1] + (radius - 8) * math.sin(rad))
        x_outer = int(center[0] + radius * math.cos(rad))
        y_outer = int(center[1] + radius * math.sin(rad))
        cv2.line(img, (x_inner, y_inner), (x_outer, y_outer), (80, 80, 80), 1)

        # Label major ticks
        if ang in [-90, -45, 0, 45, 90]:
            label = f"{abs(ang)}L" if ang < 0 else (f"{ang}R" if ang > 0 else "0")
            lx = int(center[0] + (radius + 18) * math.cos(rad)) - 10
            ly = int(center[1] + (radius + 18) * math.sin(rad)) + 4
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 140, 140), 1)

    # 3. Center Robot Glyph
    cv2.circle(img, center, 6, (0, 255, 180), -1)
    cv2.line(img, (center[0], center[1] - 12), (center[0], center[1] + 12), (0, 255, 180), 1)
    cv2.line(img, (center[0] - 12, center[1]), (center[0] + 12, center[1]), (0, 255, 180), 1)

    # 4. Acoustic Direction Vector Beam
    beam_color = (0, 255, 120) if target.is_active else (70, 70, 70)
    beam_rad = math.radians(target.azimuth_deg - 90)
    beam_len = radius if target.is_active else int(radius * 0.5)
    bx = int(center[0] + beam_len * math.cos(beam_rad))
    by = int(center[1] + beam_len * math.sin(beam_rad))

    if target.is_active:
        # Glow cone around target direction
        cone_left = math.radians(target.azimuth_deg - 98)
        cone_right = math.radians(target.azimuth_deg - 82)
        pt_l = (int(center[0] + (radius - 5) * math.cos(cone_left)), int(center[1] + (radius - 5) * math.sin(cone_left)))
        pt_r = (int(center[0] + (radius - 5) * math.cos(cone_right)), int(center[1] + (radius - 5) * math.sin(cone_right)))
        cone_pts = np.array([center, pt_l, pt_r], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [cone_pts], (0, 180, 80))
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        # Primary needle
        cv2.line(img, center, (bx, by), beam_color, 3, cv2.LINE_AA)
        cv2.circle(img, (bx, by), 6, (0, 255, 255), -1)
    else:
        cv2.line(img, center, (bx, by), (50, 50, 50), 1, cv2.LINE_AA)

    # 5. Telemetry Banner (Top)
    cv2.rectangle(img, (10, 10), (size - 10, 55), (15, 15, 15), -1)
    cv2.rectangle(img, (10, 10), (size - 10, 55), (50, 50, 50), 1)
    cv2.putText(img, "ACOUSTIC VOICE LOCALIZATION", (20, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    stat_col = (0, 255, 180) if target.is_active else (120, 120, 120)
    cv2.putText(img, f"STATUS: {target.status_text}", (20, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, stat_col, 1, cv2.LINE_AA)

    # 6. Bearing & SNR Readout (Bottom)
    sign = "+" if target.azimuth_deg >= 0 else ""
    bearing_text = f"BEARING: {sign}{target.azimuth_deg:.1f} deg"
    cv2.putText(img, bearing_text, (20, size - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"CONF: {int(target.confidence * 100)}% | PWR: {target.energy_db:.1f} dB",
                (20, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1, cv2.LINE_AA)

    return img


def run_standalone_radar():
    """Runs live interactive acoustic compass radar."""
    engine = AcousticDoAEngine()

    print("\n" + "=" * 65)
    print(" ACOUSTIC RADAR ACTIVE")
    print(" Speak, whistle, or snap your fingers on the left or right of your laptop.")
    print(" Watch the acoustic vector needle point directly at you!")
    print(" Press 'q' key to stop.")
    print("=" * 65 + "\n")

    while engine.running:
        target = engine.get_target()
        radar_img = render_acoustic_compass(target)

        cv2.imshow("USAR Acoustic Radar [Voice Localization]", radar_img)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("Acoustic radar session ended.")
            break

    engine.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USAR Acoustic Voice Direction of Arrival")
    parser.add_argument("--mic-distance", type=float, default=0.15, help="Microphone spacing in meters (default: 0.15)")
    args = parser.parse_args()

    run_standalone_radar()
