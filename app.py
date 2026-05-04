from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st
from av import VideoFrame
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from attractor import AttractorSimulation
from hand_tracker import HandTracker


# ── PAGE ───────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Thomas Attractor", page_icon="🌀")


# ── METRICS ────────────────────────────────────────
@dataclass
class _M:
    b: float = 0.18
    zoom: float = 1.0
    fps: float = 0.0
    hands: int = 0
    active: bool = False
    particles: int = 700


_metrics = _M()
_lock = threading.Lock()


# ── ROTATION ───────────────────────────────────────
def _rot(rx: float, ry: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], np.float32)

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], np.float32)

    return Ry @ Rx


# ── RENDER ─────────────────────────────────────────
def _render(sim, b, zoom, rx, ry, W, H):
    sim.set_b(b)
    sim.step(2)

    buf = np.zeros((H, W, 3), np.float32)

    R = _rot(rx, ry)
    scale = 85.0 * zoom
    cx, cy = W // 2, H // 2

    for age, pts, cidx in sim.get_trail_points():
        alpha = age ** 2
        if alpha < 0.01:
            continue

        pts = pts @ R.T

        px = (pts[:, 0] * scale + cx).astype(np.int32)
        py = (pts[:, 1] * scale + cy).astype(np.int32)

        mask = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not mask.any():
            continue

        np.add.at(buf, (py[mask], px[mask]),
                  sim.PALETTE[cidx[mask]] * alpha)

    buf = np.nan_to_num(buf, nan=0.0, posinf=1e6, neginf=0.0)
    glow = np.log1p(buf * 6.0) / math.log1p(6.0)

    return (np.clip(glow, 0, 1) * 255).astype(np.uint8)


# ── HUD ────────────────────────────────────────────
def _hud(frame, b, zoom, fps, hands, active):
    f = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, 130), (20, 20, 30), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "THOMAS ATTRACTOR", (20, 35),
                f, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"b: {b:.4f}", (20, 60),
                f, 0.45, (0, 200, 255), 1)

    cv2.putText(frame, f"zoom: {zoom:.2f}", (20, 80),
                f, 0.45, (0, 200, 255), 1)

    cv2.putText(frame, f"fps: {fps:.0f}", (20, 100),
                f, 0.45, (0, 255, 100), 1)

    col = (0, 255, 100) if active else (120, 120, 120)
    cv2.putText(frame, f"hands: {hands}", (20, 120),
                f, 0.45, col, 1)

    return frame


# ── VIDEO PROCESSOR ────────────────────────────────
class _Proc(VideoProcessorBase):

    def __init__(self):
        self.particles = 700
        self.sim = AttractorSimulation(self.particles, 60, 0.18)
        self.tracker = HandTracker()

        self.rx, self.ry = 0.4, -0.4
        self.b, self.zoom = 0.18, 1.0

        self._fps = 0.0
        self._last = time.monotonic()

        self.hand_ctrl = True
        self.show_lm = False
        self.b_ov = 0.18
        self.zoom_ov = 1.0

    def set_cfg(self, hand_ctrl, show_lm, b_ov, zoom_ov, particles):
        self.hand_ctrl = hand_ctrl
        self.show_lm = show_lm
        self.b_ov = b_ov
        self.zoom_ov = zoom_ov

        
        if particles != self.particles:
            self.particles = particles
            self.sim = AttractorSimulation(self.particles, 60, self.b)

    def recv(self, frame: VideoFrame) -> VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")

        bgr = cv2.flip(bgr, 1)
        bgr = cv2.resize(bgr, (640, 480))

        H, W = bgr.shape[:2]

        # FPS
        now = time.monotonic()
        dt = max(now - self._last, 1e-6)
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._last = now

        # Hand tracking
        try:
            self.tracker.update(bgr)
        except Exception:
            pass

        hands = len(self.tracker.hands_data)
        active = self.tracker.active

        # Controls
        if self.hand_ctrl and active:
            pinch_active = (
                abs(self.tracker.zoom - 1.0) > 0.12 and
                abs(self.tracker.b - 0.18) > 0.08
            )

            if pinch_active:
                # Smooth + dead zone
                if abs(self.tracker.b - self.b) > 0.02:
                    # Normalize pinch → stable b mapping
                    raw = self.tracker.b   # coming from tracker

                    # Clamp raw input
                    raw = np.clip(raw, 0.0, 1.0)

                    # Map to desired range
                    mapped_b = 0.05 + raw * (0.28 - 0.05)

                    # Smooth
                    self.b = 0.97 * self.b + 0.03 * mapped_b

                self.zoom = 0.95 * self.zoom + 0.05 * self.tracker.zoom

            # Smooth rotation
            self.rx = 0.92 * self.rx + self.tracker.rot_x
            self.ry = 0.92 * self.ry + self.tracker.rot_y
        else:
            self.b = 0.92 * self.b + 0.08 * self.b_ov
            self.zoom = 0.92 * self.zoom + 0.08 * self.zoom_ov

        self.b = float(np.clip(self.b, -0.10, 0.28))
        self.zoom = float(np.clip(self.zoom, 0.3, 3.0))

        # Render
        try:
            att = _render(self.sim, self.b, self.zoom, self.rx, self.ry, W, H)
        except Exception:
            att = np.zeros((H, W, 3), np.uint8)

        # ── FINAL UI (MODEL + CAMERA) ──
        out = att.copy()

        cam_h, cam_w = 140, 180
        cam_small = cv2.resize(bgr, (cam_w, cam_h))

        y1, y2 = H - cam_h - 20, H - 20
        x1, x2 = W - cam_w - 20, W - 20
        # Safety check before placing camera window
        if y1 >= 0 and x1 >= 0 and y2 <= H and x2 <= W:

            out[y1:y2, x1:x2] = cam_small
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)

        if self.show_lm:
            cam_draw = self.tracker.draw_landmarks(cam_small.copy())
            out[y1:y2, x1:x2] = cam_draw

        out = _hud(out, self.b, self.zoom, self._fps, hands, active)

        # Metrics
        with _lock:
            _metrics.b = self.b
            _metrics.zoom = self.zoom
            _metrics.fps = self._fps
            _metrics.hands = hands
            _metrics.active = active
            _metrics.particles = self.particles

        return VideoFrame.from_ndarray(out, format="bgr24")


# ── SIDEBAR ────────────────────────────────────────
st.sidebar.title("🌀 Thomas Attractor")

hand_ctrl = st.sidebar.toggle("Hand Control", True)
show_lm = st.sidebar.toggle("Show Fingers (Pinch Only)", False)

b_ov = st.sidebar.slider("b", -0.28, 0.28, 0.18)
zoom_ov = st.sidebar.slider("Zoom", 0.3, 3.0, 1.0)
particles = st.sidebar.slider("Particles", 300, 3000, 700)

# 🔥 NEW: Camera Selection
cam_choice = st.sidebar.selectbox(
    "🎥 Select Camera",
    ["Laptop Camera", "DroidCam"]
)

# 🎯 Dynamic constraints
video_constraints = {
    "width": 640,
    "height": 480,
    "frameRate": 15
}


# ── STREAM ─────────────────────────────────────────
ctx = webrtc_streamer(
    key="attractor",
    video_processor_factory=_Proc,
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    media_stream_constraints={
        "video": video_constraints,   
        "audio": False,
    },
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.set_cfg(hand_ctrl, show_lm, b_ov, zoom_ov, particles)


# ── METRICS ────────────────────────────────────────
st.subheader("📊 Metrics")

with _lock:
    m = _metrics

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("b", f"{m.b:.4f}")
c2.metric("Zoom", f"{m.zoom:.2f}")
c3.metric("FPS", f"{m.fps:.0f}")
c4.metric("Hands", f"{m.hands}")
c5.metric("Particles", f"{m.particles}")
