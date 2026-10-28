"""B1 landmark ratios and time-weighted PERCLOS; thresholds require validation tuning."""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


def _points(points, count):
    pts = np.asarray(points, dtype=float)
    if pts.shape != (count, 2) or not np.isfinite(pts).all():
        raise ValueError(f"Expected {count} finite 2D points")
    return pts


def eye_aspect_ratio(points):
    p = _points(points, 6)
    width = np.linalg.norm(p[0] - p[3])
    if width <= 0:
        raise ValueError("Degenerate eye width")
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * width))


def mouth_aspect_ratio(points):
    # Ordered corners, then 3 upper points, right corner, 3 lower points reversed.
    p = _points(points, 8)
    width = np.linalg.norm(p[0] - p[4])
    if width <= 0:
        raise ValueError("Degenerate mouth width")
    return float(sum(np.linalg.norm(p[i] - p[8 - i]) for i in (1, 2, 3)) / (3 * width))


def pitch_degrees(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite MediaPipe 4x4 face transform")
    return math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))


@dataclass(frozen=True)
class Thresholds:
    ear: float = 0.20
    mar: float = 0.50
    pitch: float = 20.0
    perclos: float = 0.40
    window_seconds: float = 10.0
    persist_seconds: float = 1.0
    max_gap_seconds: float = 0.5
    minimum_coverage: float = 0.8


class TemporalRules:
    def __init__(self, thresholds=Thresholds()):
        self.t = thresholds
        if (
            thresholds.window_seconds <= 0
            or thresholds.persist_seconds < 0
            or thresholds.max_gap_seconds <= 0
            or not 0 < thresholds.minimum_coverage <= 1
            or not 0 <= thresholds.perclos <= 1
        ):
            raise ValueError("Invalid temporal thresholds")
        self.history = deque()
        self.last_time = None
        self.starts = {1: None, 2: None, 3: None}

    def update(self, timestamp, ear=None, mar=None, pitch=None):
        if not math.isfinite(timestamp) or (
            self.last_time is not None and timestamp <= self.last_time
        ):
            raise ValueError("Timestamps must be finite and strictly increasing")
        gap = self.last_time is not None and timestamp - self.last_time > self.t.max_gap_seconds
        if gap:
            self.starts = {1: None, 2: None, 3: None}
        self.last_time = timestamp
        valid = all(v is not None and math.isfinite(v) for v in (ear, mar, pitch))
        if valid:
            active = {1: ear < self.t.ear, 2: mar > self.t.mar, 3: abs(pitch) > self.t.pitch}
        else:
            active = {1: False, 2: False, 3: False}
        for state, on in active.items():
            self.starts[state] = (
                (timestamp if self.starts[state] is None else self.starts[state]) if on else None
            )
        self.history.append((timestamp, active[1] if valid else None))
        cutoff = timestamp - self.t.window_seconds
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.popleft()
        observed, closed = 0.0, 0.0
        for (a, state), (b, _) in zip(self.history, list(self.history)[1:]):
            if state is not None and b - a <= self.t.max_gap_seconds:
                dt = max(0.0, min(b, timestamp) - max(a, cutoff))
                observed += dt
                closed += dt * state
        coverage = observed / self.t.window_seconds
        perclos = closed / observed if observed else None
        state = None if not valid else 0
        if valid:
            for candidate in (1, 2, 3):
                start = self.starts[candidate]
                sustained = start is not None and timestamp - start >= self.t.persist_seconds
                if candidate == 1:
                    sustained |= (
                        coverage >= self.t.minimum_coverage
                        and perclos is not None
                        and perclos >= self.t.perclos
                    )
                if sustained:
                    state = candidate
                    break
        return {"state": state, "perclos": perclos, "coverage": coverage}
