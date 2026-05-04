"""
Renderer — High-performance additive glow renderer
Optimized for real-time attractor visualization (30–60 FPS)
"""
from __future__ import annotations

import math
import cv2
import numpy as np


# ── Rotation helpers ─────────────────────────────────────────────────────────

def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


# ── Renderer ─────────────────────────────────────────────────────────────────

class Renderer:
    """
    Projects 3D attractor trails → 2D screen with HDR glow.
    Fully optimized for real-time interaction.
    """

    BASE_SCALE = 90.0
    FOG_POWER  = 1.8

    def __init__(self, w: int, h: int) -> None:
        self.w = int(w)
        self.h = int(h)

        self.cx = self.w // 2
        self.cy = self.h // 2

        self.rot_x = 0.4
        self.rot_y = -0.4
        self.rot_z = 0.0

        self.zoom = 1.0
        self._zoom_target = 1.0
        self._zoom_speed = 0.08

        # Pre-allocated buffer (CRITICAL for performance)
        self._buf = np.zeros((self.h, self.w, 3), dtype=np.float32)

    # ── Rotation matrix ──────────────────────────────────────────────────────

    @property
    def R(self) -> np.ndarray:
        return _rz(self.rot_z) @ _ry(self.rot_y) @ _rx(self.rot_x)

    # ── Projection ───────────────────────────────────────────────────────────

    def _project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotated = pts @ self.R.T
        scale = self.BASE_SCALE * self.zoom

        px = (rotated[:, 0] * scale + self.cx).astype(np.int32)
        py = (rotated[:, 1] * scale + self.cy).astype(np.int32)

        return px, py

    # ── Render ───────────────────────────────────────────────────────────────

    def render(
        self,
        sim,
        camera_frame: np.ndarray | None = None,
    ) -> np.ndarray:

        # Smooth zoom
        self.zoom += (self._zoom_target - self.zoom) * self._zoom_speed

        buf = self._buf
        buf.fill(0.0)  # reuse memory (IMPORTANT)

        # ── Render trail (NO list conversion) ────────────────────────────────
        for age, pts3d, cidx in sim.get_trail_points():

            alpha = (age ** self.FOG_POWER)
            if alpha < 0.01:
                continue

            px, py = self._project(pts3d)

            mask = (
                (px >= 0) & (px < self.w) &
                (py >= 0) & (py < self.h)
            )
            if not mask.any():
                continue

            px_m = px[mask]
            py_m = py[mask]

            col = sim.PALETTE[cidx[mask]] * alpha

            np.add.at(buf, (py_m, px_m), col)

        # ── Glow pass ────────────────────────────────────────────────────────
        buf = cv2.GaussianBlur(buf, (0, 0), 1.2)

        # ── Tone mapping ─────────────────────────────────────────────────────
        glow = np.log1p(buf * 6.0) / math.log1p(6.0)

        # Safety clamp
        if not np.isfinite(glow).all():
            glow = np.zeros_like(glow)

        layer = (np.clip(glow, 0, 1) * 255).astype(np.uint8)

        # ── Camera compositing ───────────────────────────────────────────────
        if camera_frame is not None:
            bg = cv2.resize(camera_frame, (self.w, self.h)).astype(np.float32)

            # Brightness mask (stable)
            bright = glow.max(axis=2, keepdims=True)
            alpha_mask = np.clip(bright * 2.5, 0.0, 1.0)

            out = (
                bg * (1.0 - alpha_mask) * 0.4 +
                layer.astype(np.float32) * (0.6 + 0.4 * alpha_mask)
            )

            return np.clip(out, 0, 255).astype(np.uint8)

        return layer

    # ── Controls ─────────────────────────────────────────────────────────────

    def add_rotation(self, drx: float, dry: float) -> None:
        self.rot_x += float(drx)
        self.rot_y += float(dry)

    def set_zoom(self, z: float) -> None:
        self._zoom_target = float(np.clip(z, 0.2, 5.0))

    def set_center(self, cx: int, cy: int) -> None:
        self.cx = int(cx)
        self.cy = int(cy)