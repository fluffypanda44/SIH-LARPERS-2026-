"""
Urban Search & Rescue (USAR) Unified Video Stream & Camera Utilities.
Provides robust threaded frame ingestion with automatic reconnect for wireless streams.
"""

import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


def format_stream_url(src: str) -> str:
    """Standardizes phone IP Webcam / RTSP stream URLs."""
    if not (src.startswith("http://") or src.startswith("https://") or src.startswith("rtsp://")):
        src = f"http://{src}"
    
    # Auto-append endpoint if omitted by user
    if not src.endswith("/video") and not src.endswith(".mjpg") and not src.startswith("rtsp://"):
        src = f"{src.rstrip('/')}/video"
    return src


class ThreadedCamera:
    """
    Low-latency threaded stream reader to eliminate network buffer bloat.
    Maintains a single-frame buffer and handles stream reconnection gracefully.
    """

    def __init__(self, src: Union[str, int], reconnect_interval: float = 2.0):
        self.src = src if isinstance(src, int) else format_stream_url(str(src))
        self.reconnect_interval = reconnect_interval
        self.lock = threading.Lock()
        self.ret = False
        self.frame: Optional[np.ndarray] = None
        self.running = True
        self.is_connected = False

        print(f"[STREAM] Connecting to video stream: {self.src}")
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            self.is_connected = True
            print("[STREAM] Stream opened successfully.")
        else:
            print(f"[STREAM] Warning: Initial connection failed to {self.src}. Will retry in background.")

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if not self.cap.isOpened():
                self.is_connected = False
                time.sleep(self.reconnect_interval)
                try:
                    self.cap.open(self.src)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.cap.isOpened():
                        self.is_connected = True
                        print(f"[STREAM] Successfully reconnected to {self.src}")
                except Exception as e:
                    print(f"[STREAM] Reconnect error: {e}")
                continue

            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.is_connected = False
                    time.sleep(0.02)
                    continue

                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    self.is_connected = True
            except Exception as e:
                print(f"[STREAM] Frame capture exception: {e}")
                time.sleep(0.05)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        if hasattr(self, "cap"):
            self.cap.release()
        self.is_connected = False

    def isOpened(self) -> bool:
        return self.running and (self.is_connected or self.cap.isOpened())
