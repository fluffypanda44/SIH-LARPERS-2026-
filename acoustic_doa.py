"""
Urban Search & Rescue (USAR) Acoustic Direction of Arrival (DoA) Engine.
Localizes trapped victims and distress sounds ("Where is the voice coming from?").

Supports:
- Local Laptop Stereo Microphones (Realtek Array)
- Wireless Mobile Phone IP Webcam Audio Streams (--url http://<IP>:8080)

Technique:
- Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
- Real-time time delay of arrival (TDoA) calculation across dual microphones.
- Bandpass vocal frequency filtering (300 Hz - 3400 Hz) to reject robot chassis/motor hum.
- Real-time tactical acoustic compass HUD dial.
"""

import argparse
import math
import sys
import threading
import time
import urllib.error
import urllib.request
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
    source_info: str           # "PHONE MIC (STEREO)" or "LAPTOP ARRAY"


class NetworkAudioStreamer:
    """Streams live linear PCM audio from mobile IP Webcam (/audio.wav)."""

    def __init__(self, url: str, buffer_size: int = 4096):
        self.url = self._format_url(url)
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        self.audio_buffer = np.zeros((self.buffer_size, 2), dtype=np.float32)
        self.channels = 1
        self.fs = 44100
        self.running = True
        self.connected = False

        print(f"[NETWORK AUDIO] Connecting to IP Webcam audio: {self.url}")
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()

    def _format_url(self, url: str) -> str:
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"http://{url}"
        url = url.rstrip("/")
        if not url.endswith("/audio.wav") and not url.endswith(".wav"):
            url = f"{url}/audio.wav"
        return url

    def _stream_worker(self):
        while self.running:
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": "USAR-Perception-Bot/1.0"})
                response = urllib.request.urlopen(req, timeout=5)

                # Read 44-byte standard RIFF WAV header
                header = response.read(44)
                if len(header) >= 44 and header[:4] == b"RIFF":
                    self.channels = int.from_bytes(header[22:24], "little")
                    self.fs = int.from_bytes(header[24:28], "little")
                    print(f"[NETWORK AUDIO] Connected! Channels: {self.channels} | Samplerate: {self.fs} Hz")
                else:
                    self.channels = 1
                    self.fs = 44100
                    print("[NETWORK AUDIO] Warning: Unrecognized header, defaulting to Mono 44100 Hz")

                self.connected = True
                bytes_per_sample = 2  # 16-bit PCM
                chunk_samples = 1024
                chunk_bytes = chunk_samples * self.channels * bytes_per_sample

                while self.running:
                    raw = response.read(chunk_bytes)
                    if not raw or len(raw) < chunk_bytes:
                        time.sleep(0.01)
                        continue

                    # Parse signed 16-bit integers to float32 (-1.0 to +1.0)
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                    if self.channels >= 2:
                        stereo = samples.reshape(-1, self.channels)[:, :2]
                    else:
                        stereo = np.column_stack((samples, samples))

                    with self.lock:
                        n_new = len(stereo)
                        self.audio_buffer[:-n_new] = self.audio_buffer[n_new:]
                        self.audio_buffer[-n_new:] = stereo

            except urllib.error.URLError as e:
                self.connected = False
                time.sleep(2.0)
            except Exception as e:
                self.connected = False
                time.sleep(2.0)

    def get_buffer(self) -> Tuple[np.ndarray, int, int]:
        with self.lock:
            return self.audio_buffer.copy(), self.channels, self.fs

    def stop(self):
        self.running = False


class AcousticDoAEngine:
    """
    Real-time Sound Source Localization (SSL) using GCC-PHAT.
    Accepts local stereo microphone OR network phone IP webcam stream.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        samplerate: int = 44100,
        buffer_size: int = 4096,
        mic_distance_m: float = 0.060,  # ASUS TUF Gaming F16 (FX607VU) dual-mic array baseline is 6.0 cm
        device_index: Optional[int] = None,
        speed_of_sound: float = 343.0,
    ):
        self.url = url
        self.fs = samplerate
        self.buffer_size = buffer_size
        self.d = mic_distance_m  # 0.06m on ASUS TUF F16
        self.c = speed_of_sound
        self.max_tau = self.d / self.c

        self.lock = threading.Lock()
        self.audio_buffer = np.zeros((self.buffer_size, 2), dtype=np.float32)
        self.current_target = AcousticTarget(
            is_active=False,
            azimuth_deg=0.0,
            confidence=0.0,
            energy_db=-60.0,
            status_text="INITIALIZING...",
            source_info="STARTING...",
        )
        self.smoothed_angle = 0.0
        self.running = True

        print("=" * 60)
        print("INITIALIZING ACOUSTIC VOICE LOCALIZATION ENGINE (PHASE 2)")
        print("=" * 60)
        print(f"Microphone Baseline (d): {self.d * 100:.1f} cm | Speed of Sound: {self.c} m/s")

        if self.url:
            self.source_mode = "network"
            self.net_streamer = NetworkAudioStreamer(self.url, buffer_size=self.buffer_size)
            self.source_info = f"PHONE IP CAM ({self.url})"
        else:
            self.source_mode = "local"
            self.net_streamer = None
            
            # Prefer laptop hardware stereo mic array (Realtek) for genuine Left/Right separation
            if device_index is None:
                try:
                    for idx, d in enumerate(sd.query_devices()):
                        if "Realtek" in d.get("name", "") and d.get("max_input_channels", 0) >= 2:
                            device_index = idx
                            break
                except Exception:
                    pass

            self.source_info = "LAPTOP STEREO MIC ARRAY"
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
                print(f"[AUDIO] Hardware stereo stream active on device: {self.source_info}")
            except Exception as e:
                print(f"[ERROR] Failed to start local microphone: {e}")
                self.running = False
                return

        # Precompute bandpass filter masks
        n_fft = 2 * self.buffer_size
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.fs)
        self.mode = "beacon"  # "beacon" (for noisy halls/beeps) or "voice" (for human speech)
        self.beacon_mask = (freqs >= 1700.0) & (freqs <= 2700.0)
        self.voice_mask = (freqs >= 250.0) & (freqs <= 3800.0)

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        print(f"[AUDIO] Source: {self.source_info} | Initial Mode: BEACON BEEP (2kHz)")
        print("=" * 60 + "\n")

    def toggle_mode(self) -> str:
        """Toggles between BEACON BEEP mode and VOICE mode."""
        self.mode = "voice" if self.mode == "beacon" else "beacon"
        print(f"[AUDIO] Active Mode Switched -> {self.mode.upper()}")
        return self.mode

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        with self.lock:
            chunk_len = len(indata)
            self.audio_buffer[:-chunk_len] = self.audio_buffer[chunk_len:]
            self.audio_buffer[-chunk_len:] = indata

    def _gcc_phat(self, s1: np.ndarray, s2: np.ndarray, mask: Optional[np.ndarray] = None, interp: int = 32) -> Tuple[float, float]:
        if mask is None:
            mask = self.beacon_mask if self.mode == "beacon" else self.voice_mask

        n = s1.shape[0] + s2.shape[0]
        S1 = np.fft.rfft(s1, n=n) * mask
        S2 = np.fft.rfft(s2, n=n) * mask

        R = S1 * np.conj(S2)
        denom = np.abs(R) + 1e-15
        R_norm = R / denom

        cc = np.fft.irfft(R_norm, n=interp * n)
        max_shift = int(interp * self.fs * self.max_tau)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

        peak_idx = np.argmax(np.abs(cc))
        peak_val = np.abs(cc[peak_idx])
        mean_val = np.mean(np.abs(cc)) + 1e-15
        confidence = float(min(1.0, (peak_val / (mean_val * 4.0))))

        shift = peak_idx - max_shift
        tau = shift / float(interp * self.fs)
        return tau, confidence

    def _process_loop(self):
        n_fft = 2 * self.buffer_size
        while self.running:
            if self.source_mode == "network":
                buf, channels, fs = self.net_streamer.get_buffer()
                self.fs = fs
                is_stereo = (channels >= 2)
            else:
                with self.lock:
                    buf = self.audio_buffer.copy()
                is_stereo = True

            s1 = buf[:, 0]
            s2 = buf[:, 1]

            rms = np.sqrt(np.mean(s1**2 + s2**2) / 2.0) + 1e-12
            db = 20.0 * math.log10(rms)

            if self.mode == "beacon":
                # BEACON BEEP MODE: Rejects library speech chatter, triggers on pure 2kHz beep tones
                S1 = np.abs(np.fft.rfft(s1, n=n_fft))
                in_band_energy = float(np.sum(S1[self.beacon_mask]**2))
                total_energy = float(np.sum(S1**2) + 1e-12)
                tonality = in_band_energy / total_energy
                in_band_rms = np.sqrt(in_band_energy / max(1, np.sum(self.beacon_mask)))
                in_band_db = 20.0 * math.log10(in_band_rms + 1e-12)

                # Pure beep concentrates >8% of total energy in narrow 2kHz bin
                if tonality > 0.08 and in_band_db > -64.0:
                    is_active = True
                    if is_stereo:
                        tau, conf = self._gcc_phat(s1, s2, mask=self.beacon_mask)
                        sin_arg = np.clip((self.c * tau) / self.d, -1.0, 1.0)
                        angle_deg = math.degrees(math.asin(sin_arg))
                        alpha = 0.45 if conf > 0.4 else 0.20
                        self.smoothed_angle = (1.0 - alpha) * self.smoothed_angle + alpha * angle_deg
                        status = f"BEACON LOCKED [2kHz] ({int(tonality*100)}% Tonality)"
                    else:
                        conf = min(1.0, tonality * 2.5)
                        status = "MONO BEACON SPIKE"
                else:
                    is_active = False
                    conf = 0.0
                    status = "SCANNING FOR BEACON (2kHz)..."
            else:
                # VOICE MODE: Broadband speech detection
                if db > -52.0:
                    is_active = True
                    if is_stereo:
                        tau, conf = self._gcc_phat(s1, s2, mask=self.voice_mask)
                        sin_arg = np.clip((self.c * tau) / self.d, -1.0, 1.0)
                        angle_deg = math.degrees(math.asin(sin_arg))
                        alpha = 0.35 if conf > 0.4 else 0.10
                        self.smoothed_angle = (1.0 - alpha) * self.smoothed_angle + alpha * angle_deg
                        status = "VOICE / CRY DETECTED" if conf > 0.45 else "ACOUSTIC ACTIVITY"
                    else:
                        conf = min(1.0, (db + 52.0) / 30.0)
                        status = "PHONE MONO: SOUND SPIKE"
                else:
                    is_active = False
                    conf = 0.0
                    status = "MONITORING RUBBLE..."

            src_tag = f"{self.source_info} [{self.mode.upper()}]"

            with self.lock:
                self.current_target = AcousticTarget(
                    is_active=is_active,
                    azimuth_deg=self.smoothed_angle,
                    confidence=conf,
                    energy_db=db,
                    status_text=status,
                    source_info=src_tag,
                )

            time.sleep(0.03)

    def get_target(self) -> AcousticTarget:
        with self.lock:
            return self.current_target

    def stop(self):
        self.running = False
        if self.net_streamer:
            self.net_streamer.stop()
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()


def render_acoustic_compass(target: AcousticTarget, size: int = 420) -> np.ndarray:
    """Renders authentic tactical acoustic radar compass dial."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2 + 10)
    radius = int(size * 0.38)

    # Circular radar background
    cv2.circle(img, center, radius, (20, 20, 20), -1)
    cv2.circle(img, center, radius, (60, 60, 60), 1)
    cv2.circle(img, center, int(radius * 0.66), (40, 40, 40), 1)
    cv2.circle(img, center, int(radius * 0.33), (40, 40, 40), 1)

    # Azimuth Tick Marks
    for ang in range(-90, 91, 15):
        rad = math.radians(ang - 90)
        x_inner = int(center[0] + (radius - 8) * math.cos(rad))
        y_inner = int(center[1] + (radius - 8) * math.sin(rad))
        x_outer = int(center[0] + radius * math.cos(rad))
        y_outer = int(center[1] + radius * math.sin(rad))
        cv2.line(img, (x_inner, y_inner), (x_outer, y_outer), (80, 80, 80), 1)

        if ang in [-90, -45, 0, 45, 90]:
            label = f"{abs(ang)}L" if ang < 0 else (f"{ang}R" if ang > 0 else "0")
            lx = int(center[0] + (radius + 18) * math.cos(rad)) - 10
            ly = int(center[1] + (radius + 18) * math.sin(rad)) + 4
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 140, 140), 1)

    # Center Robot Glyph
    cv2.circle(img, center, 6, (0, 255, 180), -1)
    cv2.line(img, (center[0], center[1] - 12), (center[0], center[1] + 12), (0, 255, 180), 1)
    cv2.line(img, (center[0] - 12, center[1]), (center[0] + 12, center[1]), (0, 255, 180), 1)

    # Direction Vector Beam
    beam_color = (0, 255, 120) if target.is_active else (70, 70, 70)
    beam_rad = math.radians(target.azimuth_deg - 90)
    beam_len = radius if target.is_active else int(radius * 0.5)
    bx = int(center[0] + beam_len * math.cos(beam_rad))
    by = int(center[1] + beam_len * math.sin(beam_rad))

    if target.is_active:
        cone_left = math.radians(target.azimuth_deg - 98)
        cone_right = math.radians(target.azimuth_deg - 82)
        pt_l = (int(center[0] + (radius - 5) * math.cos(cone_left)), int(center[1] + (radius - 5) * math.sin(cone_left)))
        pt_r = (int(center[0] + (radius - 5) * math.cos(cone_right)), int(center[1] + (radius - 5) * math.sin(cone_right)))
        cone_pts = np.array([center, pt_l, pt_r], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [cone_pts], (0, 180, 80))
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        cv2.line(img, center, (bx, by), beam_color, 3, cv2.LINE_AA)
        cv2.circle(img, (bx, by), 6, (0, 255, 255), -1)
    else:
        cv2.line(img, center, (bx, by), (50, 50, 50), 1, cv2.LINE_AA)

    # Telemetry Banner
    cv2.rectangle(img, (10, 10), (size - 10, 55), (15, 15, 15), -1)
    cv2.rectangle(img, (10, 10), (size - 10, 55), (50, 50, 50), 1)
    title = "ACOUSTIC RADAR [BEACON 2kHz]" if "BEACON" in target.source_info else "ACOUSTIC RADAR [VOICE CHATTER]"
    title_col = (0, 255, 255) if "BEACON" in target.source_info else (255, 255, 255)
    cv2.putText(img, title, (18, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, title_col, 1, cv2.LINE_AA)

    stat_col = (0, 255, 180) if target.is_active else (120, 120, 120)
    cv2.putText(img, f"{target.status_text}", (18, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, stat_col, 1, cv2.LINE_AA)

    # Bearing & SNR
    sign = "+" if target.azimuth_deg >= 0 else ""
    bearing_text = f"BEARING: {sign}{target.azimuth_deg:.1f} deg"
    cv2.putText(img, bearing_text, (20, size - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"CONF: {int(target.confidence * 100)}% | PWR: {target.energy_db:.1f} dB",
                (20, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1, cv2.LINE_AA)

    return img


def run_standalone_radar(url: Optional[str] = None, mic_distance: float = 0.16):
    """Runs live acoustic radar with optional network IP stream."""
    engine = AcousticDoAEngine(url=url, mic_distance_m=mic_distance)

    print("\n" + "=" * 65)
    print(" ACOUSTIC RADAR ACTIVE")
    print(f" Source: {engine.source_info}")
    print(" Speak, shout, or tap near the phone or laptop microphones.")
    print(" The acoustic vector needle points directly at the sound source!")
    print(" Press 'q' key on the radar window to stop.")
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
    parser.add_argument("--url", type=str, default=None, help="Wireless Phone IP Webcam URL (e.g. http://192.168.43.1:8080)")
    parser.add_argument("--mic-distance", type=float, default=0.16, help="Microphone spacing in meters (default: 0.16m for Nothing Phone 2a)")
    args = parser.parse_args()

    run_standalone_radar(url=args.url, mic_distance=args.mic_distance)
