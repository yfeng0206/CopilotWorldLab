"""Tests for the pure utility helpers (no MuJoCo, no GPU, no network)."""
import numpy as np
import pytest

from src.utils.config import Config, load_config
from src.world_model.vjepa2_wrapper import latent_energy


def test_latent_energy_zero_for_identical():
    x = np.random.default_rng(0).standard_normal((4, 8))
    assert latent_energy(x, x) == 0.0


def test_latent_energy_matches_mean_abs_diff():
    pred = np.zeros((2, 3))
    goal = np.full((2, 3), 2.0)
    assert latent_energy(pred, goal) == pytest.approx(2.0)


def test_latent_energy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        latent_energy(np.zeros((2, 3)), np.zeros((3, 2)))


def test_load_config_nested_attr_and_item(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "env:\n  render_width: 256\nplanner:\n  samples: 800\n", encoding="utf-8"
    )
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, Config)
    assert cfg.env.render_width == 256        # attribute access, nested
    assert cfg["planner"]["samples"] == 800   # item access
