from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DQNHyperParams:
    gamma: float = 0.99
    max_grad_norm: float = 10.0


if nn is not None:
    class TinyDQN(nn.Module):  # type: ignore[misc]
        def __init__(self, obs_dim: int, num_actions: int):
            assert torch is not None, "torch required for DQN"
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
            )
            self.q_head = nn.Linear(64, num_actions)

        def forward(self, x: Any) -> Any:  # (B, obs_dim)
            z = self.encoder(x)
            q = self.q_head(z)
            return q

        def encode(self, x: Any) -> Any:
            return self.encoder(x)
else:
    # Fallback shim to fail fast if used without torch
    class TinyDQN:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("TinyDQN requires torch to be installed")

        def forward(self, x: Any) -> Any:  # pragma: no cover
            raise RuntimeError("TinyDQN requires torch to be installed")

        def encode(self, x: Any) -> Any:  # pragma: no cover
            raise RuntimeError("TinyDQN requires torch to be installed")


def double_dqn_targets(
    online: TinyDQN,
    target: TinyDQN,
    next_obs: Any,
    reward: Any,
    done: Any,
    gamma: float,
) -> Any:
    assert torch is not None, "torch required for DQN"
    with torch.no_grad():
        q_next_online = online(next_obs)
        next_actions = torch.argmax(q_next_online, dim=1)
        q_next_target = target(next_obs)
        q_next = q_next_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
        td_target = reward + (1.0 - done) * gamma * q_next
        return td_target


