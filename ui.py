"""
HUD — Glassmorphism + gradient overlay for OpenCV rendering.
Optimized for real-time performance and dynamic layouts.
"""
from __future__ import annotations

import time
import cv2
import numpy as np


# ── Colours (BGR) ─────────────────────────────────
WHITE   = (255, 255, 255)
GRAY    = (160, 160, 160)
DARK    = (15, 15, 15)
ACCENT  = (255, 200, 0)
ACCENT2 = (0, 120, 255)


class HUD:
    def __init__(self) -> None:
        self._times = []
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    # ── FPS (optimized) ────────────────────────────

    def tick(self) -> None:
        now = time.monotonic()
        self._times.append(now)

        # remove old timestamps in-place (no new list)
        cutoff = now - 1.0
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        return float(len(self._times))

    # ── Background ─────────────────────────────────

    def _gradient_bg(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        ramp = np.linspace(20, 60, h, dtype=np.uint8)
        overlay = np.repeat(ramp[:, None], w, axis=1)
        overlay = cv2.merge([overlay, overlay, overlay])
        return cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    def _glass_rect(self, frame: np.ndarray, y1: int, y2: int, alpha: float = 0.4) -> None:
        region = frame[y1:y2]
        blurred = cv2.GaussianBlur(region, (35, 35), 0)
        frame[y1:y2] = cv2.addWeighted(region, 1.0 - alpha, blurred, alpha, 0)

    def _accent_line(self, frame: np.ndarray, y: int) -> None:
        h, w = frame.shape[:2]
        cv2.line(frame, (20, y), (w - 20, y), ACCENT, 1, cv2.LINE_AA)

    # ── Draw ──────────────────────────────────────

    def draw(self, frame: np.ndarray, b: float, zoom: float) -> np.ndarray:
        h, w = frame.shape[:2]

        frame = self._gradient_bg(frame)

        # ── Top panel ──────────────────────────────
        TOP = 100
        self._glass_rect(frame, 0, TOP)

        cv2.putText(frame, "THOMAS ATTRACTOR",
                    (20, 38), self._font, 0.85, WHITE, 2, cv2.LINE_AA)

        cv2.putText(frame, "Chaotic System Visualization",
                    (22, 68), self._font, 0.48, GRAY, 1, cv2.LINE_AA)

        self._accent_line(frame, TOP - 10)

        # ── Bottom panel ───────────────────────────
        BOT = 70
        self._glass_rect(frame, h - BOT, h)

        # FPS badge
        cv2.rectangle(frame, (20, h - 55), (110, h - 18), DARK, -1)
        cv2.putText(frame, f"{int(self.fps)} FPS",
                    (28, h - 28), self._font, 0.50, ACCENT, 1, cv2.LINE_AA)

        # Parameters
        cv2.putText(frame, f"b : {b:.4f}",
                    (130, h - 28), self._font, 0.52, ACCENT, 1, cv2.LINE_AA)

        cv2.putText(frame, f"zoom : {zoom:.2f}",
                    (265, h - 28), self._font, 0.52, ACCENT2, 1, cv2.LINE_AA)

        # Responsive right-aligned hint
        hint = "Q Quit | R Reset | SPACE Pause | W Cam | S Save"
        (tw, _), _ = cv2.getTextSize(hint, self._font, 0.40, 1)
        cv2.putText(frame, hint,
                    (w - tw - 20, h - 28),
                    self._font, 0.40, GRAY, 1, cv2.LINE_AA)

        return frame