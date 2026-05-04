"""
Thomas Attractor — vectorised RK4 with smooth b interpolation.
Optimized for real-time rendering (30–60 FPS).
"""
from __future__ import annotations
import numpy as np


class AttractorSimulation:
    """
    Thomas' cyclically symmetric attractor.

        dx/dt = sin(y) - b*x
        dy/dt = sin(z) - b*y
        dz/dt = sin(x) - b*z
    """

    # ── Color palette (RGB float) ────────────────────────────────────────────
    PALETTE: np.ndarray = np.array([
        [0.95, 0.25, 0.10],  # red-orange
        [0.10, 0.45, 0.95],  # blue
        [0.85, 0.10, 0.85],  # magenta
        [0.10, 0.92, 0.35],  # green
        [0.10, 0.85, 0.95],  # cyan
        [0.95, 0.92, 0.10],  # yellow
    ], dtype=np.float32)

    # ── Init ────────────────────────────────────────────────────────────────
    def __init__(
        self,
        n_particles: int = 3000,
        trail_length: int = 100,
        b: float = 0.18,
        dt: float = 0.05,
    ) -> None:

        self.n = int(n_particles)
        self.trail = int(trail_length)
        self.dt = float(dt)

        # b parameter (smoothed externally)
        self.b = float(b)
        self.b_target = float(b)
        self.b_speed = 0.05  # smoothing factor

        # RNG
        self._rng = np.random.default_rng(42)

        # Particle states (float64 for stability)
        self.states = self._rng.normal(0.0, 0.5, (self.n, 3)).astype(np.float64)

        # Trail buffer (float32 for rendering speed)
        self.history = np.zeros((self.trail, self.n, 3), dtype=np.float32)
        self.head = 0

        # Per-particle color
        self.color_idx = self._rng.integers(0, len(self.PALETTE), self.n)

        # Warm-up to reach attractor manifold
        self._warmup(300)

    # ── Internal: warmup ─────────────────────────────────────────────────────
    def _warmup(self, steps: int) -> None:
        for _ in range(steps):
            self._rk4()

    # ── Dynamics ─────────────────────────────────────────────────────────────
    def _deriv(self, x: np.ndarray) -> np.ndarray:
        b = self.b
        return np.column_stack((
            np.sin(x[:, 1]) - b * x[:, 0],
            np.sin(x[:, 2]) - b * x[:, 1],
            np.sin(x[:, 0]) - b * x[:, 2],
        ))

    def _rk4(self) -> None:
        s, dt = self.states, self.dt

        k1 = self._deriv(s)
        k2 = self._deriv(s + 0.5 * dt * k1)
        k3 = self._deriv(s + 0.5 * dt * k2)
        k4 = self._deriv(s + dt * k3)

        new_s = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # ── Stability guard ────────────────────────────────────────────────
        if np.isfinite(new_s).all():
            self.states = new_s
        else:
            # Reset unstable particles only (better than skipping whole frame)
            mask = ~np.isfinite(new_s).any(axis=1)
            if mask.any():
                self.states[mask] = self._rng.normal(0.0, 0.5, (mask.sum(), 3))

    # ── Public API ───────────────────────────────────────────────────────────

    def set_b(self, b: float) -> None:
        """Set target b (clamped)."""
        self.b_target = float(np.clip(b, -0.5, 0.5))

    def step(self, n: int = 2) -> None:
        """Advance simulation and store history."""

        # Smooth b transition
        self.b += (self.b_target - self.b) * self.b_speed

        for _ in range(n):
            self._rk4()

        # Store snapshot
        self.history[self.head] = self.states.astype(np.float32, copy=False)
        self.head = (self.head + 1) % self.trail

    def get_trail_points(self):
        """
        Generator:
            yields (age_norm, points[N,3], color_idx[N])
        """

        # Oldest → newest
        for i in range(self.trail):
            idx = (self.head - self.trail + i) % self.trail
            age = i / max(self.trail - 1, 1)
            yield age, self.history[idx], self.color_idx

    def reset(self, b: float | None = None) -> None:
        """Reinitialize simulation safely."""

        self.states = self._rng.normal(0.0, 0.5, (self.n, 3)).astype(np.float64)
        self.history.fill(0.0)
        self.head = 0

        if b is not None:
            b = float(b)
            self.b = b
            self.b_target = b

        self._warmup(300)