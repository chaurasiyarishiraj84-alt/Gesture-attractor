from __future__ import annotations

import math
import numpy as np
import cv2

# ── MediaPipe (stable import) ─────────────────────
try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    _MP_OK = True

except Exception as e:
    _MP_OK = False
    print(f"[HandTracker] MediaPipe unavailable: {e}")


# ── Constants ─────────────────────────────────────

B_MIN, B_MAX, B_INIT = -0.35, 0.35, 0.18
ZOOM_MIN, ZOOM_MAX, ZOOM_INIT = 0.30, 4.00, 1.00

_DEAD_ZONE   = 0.04

_SMOOTH_B    = 0.08
_SMOOTH_ZOOM = 0.06
_SMOOTH_ROT  = 0.05

WHITE  = (255, 255, 255)
GRAY   = (140, 140, 140)
YELLOW = (40, 200, 255)
GREEN  = (40, 230, 80)
RED    = (40, 40, 230)
_FONT  = cv2.FONT_HERSHEY_SIMPLEX


# ── EMA ───────────────────────────────────────────

class _EMA:
    def __init__(self, init, speed, dead=0.0):
        self.value = init
        self.speed = speed
        self.dead = dead

    def update(self, target):
        delta = target - self.value
        if abs(delta) > self.dead:
            self.value += delta * self.speed
        return self.value

    def set(self, v):
        self.value = v


# ── Hand Tracker ──────────────────────────────────

class HandTracker:

    def __init__(self, max_hands=2, min_detect=0.7, min_track=0.6):

        self._b_sm    = _EMA(B_INIT, _SMOOTH_B, _DEAD_ZONE)
        self._zoom_sm = _EMA(ZOOM_INIT, _SMOOTH_ZOOM, _DEAD_ZONE)
        self._rx_sm   = _EMA(0.0, _SMOOTH_ROT)
        self._ry_sm   = _EMA(0.0, _SMOOTH_ROT)

        self.b = B_INIT
        self.zoom = ZOOM_INIT
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.active = False

        self._b_norm = (B_INIT - B_MIN) / (B_MAX - B_MIN)
        self._zoom_norm = (ZOOM_INIT - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN)

        self._hands_data = []
        self._pinch_data = []

        
        self._last_landmarks = None

        self._hands = None
        if _MP_OK:
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=min_detect,
                min_tracking_confidence=min_track,
            )
            print("[HandTracker] Ready ✓")
        else:
            print("[HandTracker] MediaPipe unavailable")

    # ── Update ─────────────────────────────────────

    def update(self, bgr_frame):

        self._hands_data.clear()
        self._pinch_data.clear()
        self.active = False

        if self._hands is None:
            return

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        lm_list = getattr(result, "multi_hand_landmarks", None)
        hd_list = getattr(result, "multi_handedness", None)

        
        self._last_landmarks = lm_list

        if lm_list is None or hd_list is None:
            return

        if len(lm_list) == 0 or len(hd_list) == 0:
            return

        self.active = True

        for lm, handed in zip(lm_list, hd_list):

            raw = handed.classification[0].label
            label = "Right" if raw == "Left" else "Left"

            # ── Reject far hands ──
            hand_size = abs(lm.landmark[0].y - lm.landmark[9].y)
            if hand_size < 0.05:
                continue

            # ── Only index finger must be extended ──
            index_tip = lm.landmark[8]
            index_pip = lm.landmark[6]
            is_index_extended = index_tip.y < index_pip.y

            pinch = self._pinch_norm(lm)

            if not is_index_extended:
                pinch = 0.0

            cx = (lm.landmark[0].x + lm.landmark[9].x) / 2
            cy = (lm.landmark[0].y + lm.landmark[9].y) / 2

            mid_x = (lm.landmark[4].x + lm.landmark[8].x) / 2
            mid_y = (lm.landmark[4].y + lm.landmark[8].y) / 2

            self._hands_data.append((label, cx, cy))
            self._pinch_data.append((label, pinch, mid_x, mid_y))

            if label == "Left":
                self._zoom_norm = pinch
                target_zoom = ZOOM_MIN + pinch * (ZOOM_MAX - ZOOM_MIN)
                self.zoom = self._zoom_sm.update(target_zoom)

                rx = (cy - 0.5) * 0.08
                ry = (cx - 0.5) * 0.08

                self.rot_x = self._rx_sm.update(rx)
                self.rot_y = self._ry_sm.update(ry)

            elif label == "Right":
                self._b_norm = pinch
                target_b = B_MIN + pinch * (B_MAX - B_MIN)
                self.b = self._b_sm.update(target_b)

    # ── Pinch ──────────────────────────────────────

    def _pinch_norm(self, lm):

        thumb = lm.landmark[4]
        index = lm.landmark[8]
        base  = lm.landmark[5]
        wrist = lm.landmark[0]

        pinch_dist = math.hypot(
            thumb.x - index.x,
            thumb.y - index.y
        )

        hand_scale = math.hypot(
            base.x - wrist.x,
            base.y - wrist.y
        )

        if hand_scale < 1e-5:
            return 0.0

        ratio = pinch_dist / hand_scale

        if ratio > 0.55:
            return 0.0

        return float(np.clip(
            (ratio - 0.12) / (0.55 - 0.12),
            0.0, 1.0
        ))

    # ── UI ─────────────────────────────────────────

    def draw_landmarks(self, frame):
        
        if self._last_landmarks is None:
            return frame

        h, w = frame.shape[:2]

        for lm in self._last_landmarks:
            thumb = lm.landmark[4]
            index = lm.landmark[8]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)

            cv2.circle(frame, (tx, ty), 8, (0, 255, 0), -1)
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)

            cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 255), 2)

        return frame

    # ── Controls ───────────────────────────────────

    def close(self):
        if self._hands is not None:
            self._hands.close()

    @property
    def hands_data(self):
        return list(self._hands_data)
