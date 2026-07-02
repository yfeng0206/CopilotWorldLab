"""V-JEPA 2-AC world-model wrapper (scaffold - no inference this session).

This module defines the interface the pilot will use to drive the coarse
end-effector motion with Meta's action-conditioned V-JEPA 2 latent world model,
and records the exact, primary-source-verified facts needed to wire it up
tomorrow. Nothing here downloads or runs a network; the heavy imports (torch)
happen lazily inside functions so importing this module stays cheap.

Verified facts (arXiv:2506.09985 and facebookresearch/vjepa2)
------------------------------------------------------------
- Checkpoint: the action-conditioned model is NOT on HuggingFace. It ships as a
  single ``.pt`` at
  ``https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`` containing both an
  ``encoder`` and a ``predictor`` state dict, trained from the ViT-g encoder on
  ~62 h of DROID robot video. Load via
  ``torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')``.
- Action: a real-valued 7-D end-effector delta -- 3 position, 3 extrinsic Euler
  orientation, 1 gripper (matches ``MujocoPilotEnv``'s state/action layout).
- Planning: encode current frame -> ``z_k`` and goal image -> ``z_g`` with the
  video encoder, then Cross-Entropy-Method MPC minimising the latent energy
  ``E(a; z_k, s_k, z_g) = || P(a; s_k, z_k) - z_g ||_1`` over sampled action
  sequences (paper uses 800 samples, 10 CEM iterations, horizon 1; receding
  horizon re-planning). Reported latency ~16 s / action on an RTX 4090.
- The same predictive energy ``E`` is the candidate confidence / hand-off signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

ACTION_DIM = 7  # [dx, dy, dz, droll, dpitch, dyaw, dgripper]

# Official action-conditioned checkpoint (direct download; not on HuggingFace).
VJEPA2_AC_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
VJEPA2_AC_HUB = ("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")

# HuggingFace encoder-only checkpoints (MIT / Apache-2.0), 64 frames, patch 16.
VJEPA2_ENCODERS = {
    "vitl": "facebook/vjepa2-vitl-fpc64-256",
    "vith": "facebook/vjepa2-vith-fpc64-256",
    "vitg": "facebook/vjepa2-vitg-fpc64-256",
    "vitg384": "facebook/vjepa2-vitg-fpc64-384",
}


@dataclass
class PlannerConfig:
    """Cross-Entropy-Method MPC settings (defaults follow the V-JEPA 2-AC paper)."""

    samples: int = 800
    iterations: int = 10
    horizon: int = 1
    top_k: int = 10
    action_low: Optional[np.ndarray] = None
    action_high: Optional[np.ndarray] = None


def latent_energy(predicted: np.ndarray, goal: np.ndarray) -> float:
    """Mean L1 latent energy ``|| predicted - goal ||_1``.

    This is the scalar the planner minimises and the candidate hand-off gate. It
    is a pure array op so it can be reasoned about and tested without the model.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if predicted.shape != goal.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {goal.shape}")
    return float(np.abs(predicted - goal).mean())


class VJEPA2ACWorldModel:
    """Interface scaffold around the loaded V-JEPA 2-AC encoder + predictor.

    Instantiating with ``encoder=None, predictor=None`` yields a disabled model
    whose ``encode``/``predict``/``plan_action`` raise a clear error. Real loading
    and inference are wired in a later session (this session is set-up only).
    """

    def __init__(self, encoder=None, predictor=None, device: str = "cuda"):
        self.encoder = encoder
        self.predictor = predictor
        self.device = device

    @property
    def enabled(self) -> bool:
        return self.encoder is not None and self.predictor is not None

    def _require(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "V-JEPA 2-AC weights are not loaded. Run "
                "scripts/download_checkpoints.py, then load with "
                "load_vjepa2_ac(); inference is intentionally disabled this session."
            )

    def encode(self, frames):  # pragma: no cover - requires weights
        self._require()
        raise NotImplementedError("Encoder forward pass is wired up in a later session.")

    def predict(self, latent, state, actions):  # pragma: no cover - requires weights
        self._require()
        raise NotImplementedError("Action-conditioned predictor is wired up later.")

    def plan_action(self, observation, goal_image, config: Optional[PlannerConfig] = None):
        """CEM MPC to a goal image (see module docstring). Not run this session."""
        self._require()  # pragma: no cover - requires weights
        raise NotImplementedError(
            "CEM planning follows facebookresearch/vjepa2 notebooks/utils/mpc_utils.py; "
            "wired up in a later session."
        )


def load_vjepa2_ac(device: str = "cuda", source: str = "hub"):  # pragma: no cover
    """Load the action-conditioned encoder + predictor (NOT called this session).

    Kept for tomorrow's wiring. Imports torch lazily and would trigger a multi-GB
    download on first use, so it is never invoked by tests or module import.
    """
    import torch  # noqa: F401  (lazy; avoids importing torch on module load)

    raise NotImplementedError(
        "Enable in a later session. Intended path:\n"
        "  import torch\n"
        "  enc, pred = torch.hub.load(*VJEPA2_AC_HUB)\n"
        "  return VJEPA2ACWorldModel(enc, pred, device)"
    )
