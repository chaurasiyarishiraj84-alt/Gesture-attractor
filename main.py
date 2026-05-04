"""
main.py — Strange Attractor Desktop Visualiser
===============================================
Industry-grade · Python 3.11 · venv311

Controls
  Q / ESC     quit                SPACE   pause / resume
  R           reset sim & camera  S       screenshot
  W           webcam on / off     H       hand landmarks
  B / N       b up / down (+0.014 per press)
  + / -       zoom in / out
  Arrow keys  rotate attractor

Gesture (MediaPipe):
  Left  hand pinch → zoom
  Right hand pinch → b parameter
  Palm position    → rotation
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from attractor    import AttractorSimulation
from hand_tracker import HandTracker, B_MIN, B_MAX, ZOOM_MIN, ZOOM_MAX
from renderer     import Renderer
from ui           import HUD

# ── Window name ───────────────────────────────────────────────────────────────
WIN = "Strange Attractor"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_W         = 1280
DEFAULT_H         = 720
DEFAULT_PARTICLES = 1000
DEFAULT_TRAIL     = 90
DEFAULT_B         = 0.18
DEFAULT_CAMERA    = 0
TARGET_FPS        = 30
STEPS_PER_FRAME   = 2

# Auto-rotate speed when no hands detected (radians per frame)
AUTO_ROTATE_SPEED = 0.003


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Thomas Attractor Visualiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--width",     type=int,   default=DEFAULT_W)
    p.add_argument("--height",    type=int,   default=DEFAULT_H)
    p.add_argument("--particles", type=int,   default=DEFAULT_PARTICLES)
    p.add_argument("--trail",     type=int,   default=DEFAULT_TRAIL)
    p.add_argument("--camera",    type=int,   default=DEFAULT_CAMERA,
                   help="-1 = black background")
    p.add_argument("--b",         type=float, default=DEFAULT_B)
    p.add_argument("--fps",       type=int,   default=TARGET_FPS,
                   help="Target FPS cap")
    p.add_argument("--no-auto-rotate", action="store_true",
                   help="Disable auto-rotation when no hands detected")
    return p.parse_args()


# ── Camera helper ─────────────────────────────────────────────────────────────

def _open_camera(idx: int, W: int, H: int, fps: int):
    """Open camera with CAP_DSHOW on Windows for fastest startup."""
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[Camera] Device {idx} not found — using black background.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)      # minimise latency
    print(f"[Camera] Opened device {idx} at {W}x{H} @ {fps}fps")
    return cap


# ── Frame grab helper ─────────────────────────────────────────────────────────

def _grab(cap, W: int, H: int) -> np.ndarray:
    """Read one frame, flip mirror, resize to (W, H). Returns BGR uint8."""
    if cap is None:
        return np.zeros((H, W, 3), dtype=np.uint8)
    ret, bgr = cap.read()
    if not ret:
        return np.zeros((H, W, 3), dtype=np.uint8)
    return cv2.flip(cv2.resize(bgr, (W, H)), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse()
    W, H = args.width, args.height
    frame_budget = 1.0 / max(args.fps, 1)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap        = None
    use_webcam = args.camera >= 0
    if use_webcam:
        cap = _open_camera(args.camera, W, H, args.fps)
        if cap is None:
            use_webcam = False

    # ── Core objects ──────────────────────────────────────────────────────────
    sim      = AttractorSimulation(args.particles, args.trail, args.b)
    renderer = Renderer(W, H)
    tracker  = HandTracker()
    hud      = HUD()

    # ── Window ────────────────────────────────────────────────────────────────
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    # ── State ─────────────────────────────────────────────────────────────────
    paused         = False
    show_webcam    = use_webcam
    show_landmarks = False

    # last known values — hold when hands leave frame
    _last_b    = args.b
    _last_zoom = 1.0

    print(f"[Ready] {W}x{H} | {args.particles} particles | b={args.b}")
    print("[Keys]  Q=quit  SPACE=pause  R=reset  W=webcam  H=landmarks")
    print("[Keys]  B/N=b  +/-=zoom  ARROWS=rotate  S=screenshot")

    while True:
        t_start = time.monotonic()

        # ── Grab frame ────────────────────────────────────────────────────────
        bgr = _grab(cap if show_webcam else None, W, H)

        # ── Hand tracking ─────────────────────────────────────────────────────
        tracker.update(bgr)

        if tracker.active:
            # Hands detected — use live gesture values
            _last_b    = tracker.b
            _last_zoom = tracker.zoom
            renderer.add_rotation(tracker.rot_x, tracker.rot_y)
        else:
            # No hands — hold last known values + optional auto-rotate
            if not args.no_auto_rotate:
                renderer.add_rotation(0.0, AUTO_ROTATE_SPEED)

        # Always apply values (live or held)
        sim.set_b(_last_b)
        renderer.set_zoom(_last_zoom)

        # ── Landmarks overlay ─────────────────────────────────────────────────
        if show_landmarks and tracker.active:
            bgr = tracker.draw_landmarks(bgr)

        # ── Simulate ──────────────────────────────────────────────────────────
        if not paused:
            sim.step(STEPS_PER_FRAME)

        # ── Render ────────────────────────────────────────────────────────────
        frame = renderer.render(sim, bgr)
        tracker.draw_ui(frame)
        hud.tick()
        hud.draw(frame, sim.b, renderer.zoom)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow(WIN, frame)

        # ── FPS cap (prevents CPU spinning) ───────────────────────────────────
        elapsed = time.monotonic() - t_start
        wait_ms = max(1, int((frame_budget - elapsed) * 1000))

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break

        elif key == ord('r'):
            sim = AttractorSimulation(args.particles, args.trail, _last_b)
            renderer.rot_x = 0.4
            renderer.rot_y = -0.4
            print("[Reset] Simulation and camera reset.")

        elif key in (ord('+'), ord('=')):
            tracker.nudge_zoom( 0.03)
            _last_zoom = tracker.zoom

        elif key in (ord('-'), ord('_')):
            tracker.nudge_zoom(-0.03)
            _last_zoom = tracker.zoom

        elif key == 82:  renderer.add_rotation(-0.05,  0.00)   # up
        elif key == 84:  renderer.add_rotation( 0.05,  0.00)   # down
        elif key == 81:  renderer.add_rotation( 0.00, -0.05)   # left
        elif key == 83:  renderer.add_rotation( 0.00,  0.05)   # right

        elif key == ord('b'):
            tracker.nudge_b( 0.014)
            _last_b = tracker.b
            print(f"[b] {tracker.b:+.4f}")

        elif key == ord('n'):
            tracker.nudge_b(-0.014)
            _last_b = tracker.b
            print(f"[b] {tracker.b:+.4f}")

        elif key == ord('w'):
            show_webcam = not show_webcam and use_webcam
            print(f"[Webcam] {'ON' if show_webcam else 'OFF'}")

        elif key == ord('h'):
            show_landmarks = not show_landmarks
            print(f"[Landmarks] {'ON' if show_landmarks else 'OFF'}")

        elif key == ord(' '):
            paused = not paused
            print(f"[{'Paused' if paused else 'Resumed'}]")

        elif key == ord('s'):
            fname = f"attractor_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[Screenshot] Saved → {fname}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[Quit] Cleaning up...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("[Done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())