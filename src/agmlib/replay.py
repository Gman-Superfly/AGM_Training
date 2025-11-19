from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import math
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """Simple prioritized replay buffer using proportional prioritization.

    This is a lightweight, dependency-free implementation suitable for demos.
    For production, consider a segment tree for O(log N) updates/samples.
    """

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        beta0: float = 0.4,
    ) -> None:
        assert capacity > 0, "capacity must be > 0"
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        assert 0.0 <= beta0 <= 1.0, "beta0 must be in [0,1]"
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta0)
        self.pos = 0
        self.size = 0
        self.buffer: List[Transition] = [
            Transition(np.zeros(1), 0, 0.0, np.zeros(1), True) for _ in range(capacity)
        ]
        self.priorities = np.zeros((capacity,), dtype=np.float64)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.size

    def push(self, transition: Transition, *, priority: float | None = None) -> None:
        assert isinstance(transition, Transition), "transition must be Transition"
        p = float(priority) if priority is not None else (self.priorities.max() if self.size > 0 else 1.0)
        p = max(1e-6, p)
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, beta: float | None = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        assert batch_size >= 1, "batch_size must be >= 1"
        assert self.size >= batch_size, "not enough samples in buffer"
        beta_eff = float(self.beta if beta is None else beta)
        assert 0.0 <= beta_eff <= 1.0, "beta must be in [0,1]"

        # Proportional prioritized sampling
        scaled = self.priorities[: self.size] ** self.alpha
        probs = scaled / scaled.sum()
        idxs = np.random.choice(self.size, size=batch_size, replace=False, p=probs)

        # Importance-sampling weights
        N = self.size
        weights = (N * probs[idxs]) ** (-beta_eff)
        weights = weights / weights.max()

        batch = {
            "state": np.stack([self.buffer[i].state for i in idxs], axis=0),
            "action": np.asarray([self.buffer[i].action for i in idxs], dtype=np.int64),
            "reward": np.asarray([self.buffer[i].reward for i in idxs], dtype=np.float32),
            "next_state": np.stack([self.buffer[i].next_state for i in idxs], axis=0),
            "done": np.asarray([self.buffer[i].done for i in idxs], dtype=np.float32),
        }
        return batch, idxs.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        assert indices.ndim == 1 and priorities.ndim == 1, "indices/priorities must be 1D"
        assert indices.size == priorities.size, "indices/priorities size mismatch"
        for i, p in zip(indices, priorities):
            pi = float(max(1e-6, p))
            self.priorities[int(i) % self.capacity] = pi


class _SumTree:
    """Classic sum-tree supporting O(log N) updates and prefix-sum sampling."""

    def __init__(self, capacity: int) -> None:
        assert capacity > 0
        # Next power-of-two sized tree for simpler indexing
        size = 1
        while size < capacity:
            size <<= 1
        self._size = size
        self._capacity = capacity
        self._tree = np.zeros(2 * size, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self._tree[1])

    def update(self, index: int, value: float) -> None:
        assert 0 <= index < self._capacity
        pos = index + self._size
        delta = value - self._tree[pos]
        self._tree[pos] = value
        pos //= 2
        while pos >= 1:
            self._tree[pos] += delta
            pos //= 2

    def find_prefixsum_idx(self, mass: float) -> int:
        # Descend the tree to find the leaf where prefix sum crosses mass
        pos = 1
        while pos < self._size:
            left = 2 * pos
            if self._tree[left] >= mass:
                pos = left
            else:
                mass -= self._tree[left]
                pos = left + 1
        return min(pos - self._size, self._capacity - 1)


class SegmentTreePER:
    """Segment-tree prioritized replay with proportional prioritization.

    - O(log N) updates and samples
    - Importance-sampling weights based on current probabilities
    """

    def __init__(self, capacity: int, *, alpha: float = 0.6, beta0: float = 0.4) -> None:
        assert capacity > 0, "capacity must be > 0"
        assert 0.0 <= alpha <= 1.0
        assert 0.0 <= beta0 <= 1.0
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta0)
        self.pos = 0
        self.size = 0
        self.buffer: List[Transition] = [
            Transition(np.zeros(1), 0, 0.0, np.zeros(1), True) for _ in range(capacity)
        ]
        self.tree = _SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.size

    def push(self, transition: Transition, *, priority: float | None = None) -> None:
        assert isinstance(transition, Transition)
        p = float(priority) if priority is not None else self.max_priority
        p = max(1e-6, p)
        self.buffer[self.pos] = transition
        self.tree.update(self.pos, p ** self.alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.max_priority = max(self.max_priority, p)

    def sample(self, batch_size: int, *, beta: float | None = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        assert batch_size >= 1 and self.size >= batch_size
        beta_eff = float(self.beta if beta is None else beta)
        assert 0.0 <= beta_eff <= 1.0
        total = self.tree.total
        assert total > 0.0, "sum-tree total must be > 0"
        # Stratified sampling across mass segments
        seg = total / batch_size
        idxs = []
        probs = []
        for i in range(batch_size):
            a = seg * i
            b = seg * (i + 1)
            m = random.uniform(a, b)
            idx = self.tree.find_prefixsum_idx(m)
            idxs.append(idx)
            p = self.tree._tree[idx + self.tree._size] / total
            probs.append(p)
        indices = np.asarray(idxs, dtype=np.int64)
        probs_arr = np.asarray(probs, dtype=np.float64)
        # IS weights
        N = self.size
        weights = (N * probs_arr) ** (-beta_eff)
        weights = (weights / weights.max()).astype(np.float32)

        batch = {
            "state": np.stack([self.buffer[i].state for i in indices], axis=0),
            "action": np.asarray([self.buffer[i].action for i in indices], dtype=np.int64),
            "reward": np.asarray([self.buffer[i].reward for i in indices], dtype=np.float32),
            "next_state": np.stack([self.buffer[i].next_state for i in indices], axis=0),
            "done": np.asarray([self.buffer[i].done for i in indices], dtype=np.float32),
        }
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        assert indices.ndim == 1 and priorities.ndim == 1
        for i, p in zip(indices, priorities):
            p_use = float(max(1e-6, p))
            self.tree.update(int(i) % self.capacity, p_use ** self.alpha)
            if p_use > self.max_priority:
                self.max_priority = p_use


