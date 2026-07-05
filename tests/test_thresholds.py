"""Regression tests for the closed-loop benchmark success rule (src/bench/thresholds.py).

The key invariant: a pick_place trial can only count as success if it was actually grasped -- a
composite that never grasped the cube but happens to leave it near the zone must NOT pass.
"""
from src.bench.thresholds import GATE_SPEC, THRESHOLDS, success_at


def test_pick_place_requires_grasped_gate():
    """GATE_SPEC['pick_place'] must include 'grasped' so placement gates alone cannot pass."""
    assert "grasped" in GATE_SPEC["pick_place"]


def test_not_grasped_never_succeeds_even_at_zero_error():
    """A pick_place trial with a perfect placement error but grasped=False fails at every threshold."""
    gates = {"grasped": False, "upright": True, "stable": True, "released": True}
    for thr in THRESHOLDS["pick_place"]:
        assert success_at(0.0, gates, thr) is False


def test_grasped_and_gates_pass_within_threshold():
    """With all gates true and error under a threshold, the trial succeeds at that threshold."""
    gates = {"grasped": True, "upright": True, "stable": True, "released": True}
    assert success_at(0.05, gates, 0.10) is True
    assert success_at(0.05, gates, 0.03) is False  # error above the tighter threshold


def test_any_false_gate_blocks_success():
    """Any single failing physical gate blocks success regardless of precision."""
    for bad in ("grasped", "upright", "stable", "released"):
        gates = {"grasped": True, "upright": True, "stable": True, "released": True}
        gates[bad] = False
        assert success_at(0.0, gates, 0.10) is False


def test_reach_has_no_gates():
    """Reach has no physical gates: success depends only on the precision threshold."""
    assert GATE_SPEC["reach"] == []
    assert success_at(0.02, {}, 0.05) is True
    assert success_at(0.06, {}, 0.05) is False
