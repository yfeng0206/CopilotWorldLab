from __future__ import annotations

import numpy as np
import pytest

from src.control.action_transform import rotate_action_xy


def test_rotate_action_xy_returns_copy_and_preserves_other_axes():
    action = np.array([1, 0, 0, 0, 0, 0, 0])
    rotated = rotate_action_xy(action, 90.0)

    np.testing.assert_allclose(rotated[:2], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(rotated[2:], action[2:])
    assert np.issubdtype(rotated.dtype, np.floating)
    np.testing.assert_allclose(action[:2], [1.0, 0.0])


def test_rotate_action_xy_validates_input():
    with pytest.raises(ValueError, match="7-D"):
        rotate_action_xy(np.zeros(6), 10.0)
    with pytest.raises(ValueError, match="finite"):
        rotate_action_xy(np.zeros(7), float("nan"))
