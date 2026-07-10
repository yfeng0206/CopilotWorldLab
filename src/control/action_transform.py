"""Fixed action-interface transforms used by controller calibration experiments."""
from __future__ import annotations

import numpy as np


def rotate_action_xy(action, degrees: float):
    """Return a copy of a 7-D action with its world/base-frame XY translation rotated."""
    value = float(degrees)
    if not np.isfinite(value):
        raise ValueError("action XY rotation must be finite")
    transformed = np.asarray(action, dtype=np.float64).copy()
    if transformed.shape != (7,):
        raise ValueError(f"expected a 7-D action, got shape {transformed.shape}")
    if value == 0.0:
        return transformed
    theta = np.radians(value)
    cosine, sine = np.cos(theta), np.sin(theta)
    x, y = transformed[0].copy(), transformed[1].copy()
    transformed[0] = cosine * x - sine * y
    transformed[1] = sine * x + cosine * y
    return transformed
