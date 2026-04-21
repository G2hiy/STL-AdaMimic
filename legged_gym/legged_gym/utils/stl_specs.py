"""Signal Temporal Logic (STL) utilities for AdaMimic reward shaping (创新点②).

Implements a minimal, pure-torch STL DSL with differentiable robustness:
    φ ::= μ | ¬φ | φ ∧ φ | φ ∨ φ | G_[a,b] φ | F_[a,b] φ | φ U_[a,b] ψ

Temporal operators (G / F) are evaluated in an *online* fashion via
`EventWindowAccumulator`: at each env.step(), for each window [a,b] the
accumulator keeps a running soft-reduce (softmin for G, softmax for F) of the
inner predicate's robustness over all motion_time steps that fell inside
[a,b]. This avoids maintaining per-env sliding history buffers while still
producing a differentiable ρ that reflects the whole temporal window at the
moment it closes.

References:
  - Donzé & Maler (2010)       "Robust Satisfaction of Temporal Logic over Real-Valued Signals"
  - Leung et al. (2025)        "STLCG++: A Masking Approach for Differentiable STL"
  - Balakrishnan & Deshmukh (2019) "Structured Reward Shaping using STL"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Smooth min / max helpers
# ---------------------------------------------------------------------------
def softmin(x: torch.Tensor, beta: float, dim: int = -1) -> torch.Tensor:
    """Smooth (differentiable) min via -LogSumExp(-beta x). Larger beta → harder min."""
    if beta <= 0:
        return x.min(dim=dim).values
    return -torch.logsumexp(-beta * x, dim=dim) / beta


def softmax(x: torch.Tensor, beta: float, dim: int = -1) -> torch.Tensor:
    """Smooth (differentiable) max via LogSumExp(beta x)."""
    if beta <= 0:
        return x.max(dim=dim).values
    return torch.logsumexp(beta * x, dim=dim) / beta


def _pairwise_softmin(a: torch.Tensor, b: torch.Tensor, beta: float) -> torch.Tensor:
    """Pairwise softmin for two same-shape tensors. Handles ±inf by routing through hard branch."""
    if beta <= 0 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return torch.minimum(a, b)
    stacked = torch.stack([a, b], dim=0)
    return softmin(stacked, beta=beta, dim=0)


def _pairwise_softmax(a: torch.Tensor, b: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return torch.maximum(a, b)
    stacked = torch.stack([a, b], dim=0)
    return softmax(stacked, beta=beta, dim=0)


def body_pos_tracking_dist(
    dif_global_body_pos: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """Per-env scalar tracking distance from body-position difference tensor.

    Args:
        dif_global_body_pos: (N, B, 3)
        weights: (B,) optional per-body weights
        reduce: 'mean' | 'max' over bodies
    Returns:
        dist: (N,)
    """
    norms = dif_global_body_pos.norm(dim=-1)  # (N, B)
    if weights is not None:
        w = (weights / weights.sum().clamp_min(1e-8)).to(norms.device)
        if reduce == "mean":
            return (norms * w).sum(dim=-1)
        if reduce == "max":
            return (norms * w).max(dim=-1).values
    if reduce == "mean":
        return norms.mean(dim=-1)
    if reduce == "max":
        return norms.max(dim=-1).values
    raise ValueError(f"Unknown reduce: {reduce!r}")


# ---------------------------------------------------------------------------
# STL DSL
# ---------------------------------------------------------------------------
class STLNode:
    """Abstract base. robustness() returns (N,)-shaped ρ for per-env evaluation."""

    def robustness(
        self,
        signals: Dict[str, torch.Tensor],
        motion_time: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def accumulators(self) -> List["EventWindowAccumulator"]:
        """Return any EventWindowAccumulators nested inside this subtree (for lifecycle mgmt)."""
        return []


class Predicate(STLNode):
    """Atomic predicate μ(x) ≥ 0 (sense='ge') or μ(x) ≤ 0 (sense='le').

    fn(signals) must return a (N,) tensor of the raw predicate value
    (value − threshold for 'ge', threshold − value for 'le' — the caller
    builds the sign so that ρ ≥ 0 ⇔ predicate satisfied).
    """

    def __init__(self, fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor], name: str = ""):
        self.fn = fn
        self.name = name

    def robustness(self, signals, motion_time, beta):
        return self.fn(signals)


class Not(STLNode):
    def __init__(self, child: STLNode):
        self.child = child

    def robustness(self, signals, motion_time, beta):
        return -self.child.robustness(signals, motion_time, beta)

    def accumulators(self):
        return self.child.accumulators()


class And(STLNode):
    def __init__(self, *children: STLNode):
        assert len(children) >= 1
        self.children = list(children)

    def robustness(self, signals, motion_time, beta):
        rhos = torch.stack(
            [c.robustness(signals, motion_time, beta) for c in self.children], dim=-1
        )  # (N, K)
        return softmin(rhos, beta=beta, dim=-1)

    def accumulators(self):
        out: List[EventWindowAccumulator] = []
        for c in self.children:
            out.extend(c.accumulators())
        return out


class Or(STLNode):
    def __init__(self, *children: STLNode):
        assert len(children) >= 1
        self.children = list(children)

    def robustness(self, signals, motion_time, beta):
        rhos = torch.stack(
            [c.robustness(signals, motion_time, beta) for c in self.children], dim=-1
        )
        return softmax(rhos, beta=beta, dim=-1)

    def accumulators(self):
        out: List[EventWindowAccumulator] = []
        for c in self.children:
            out.extend(c.accumulators())
        return out


class EventWindowAccumulator:
    """Online running soft-reduce of child-ρ over a temporal window [a, b].

    For *each* env we maintain a running scalar `value`:
      - kind='G'  → running softmin  (worst-case ρ inside window)
      - kind='F'  → running softmax  (best-case  ρ inside window)

    State machine (per env, driven by motion_time):
      motion_time <  a        → value = neutral (will be (re)initialised on first in-window step)
      motion_time ∈ [a, b]    → if first time in window: value = ρ_now
                                else:                    value = softreduce(value, ρ_now)
      motion_time >  b        → window closed; `value` is final (read by robustness()).

    `reset()` is called per env on episode reset; `step()` is called every env.step()
    with the current motion_time and current-step signals.
    """

    def __init__(
        self,
        child: STLNode,
        a: float,
        b: float,
        kind: str,
        num_envs: int,
        device: torch.device | str,
    ):
        assert kind in ("G", "F")
        assert b >= a, f"invalid window [{a}, {b}]"
        self.child = child
        self.a = float(a)
        self.b = float(b)
        self.kind = kind
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        # Neutral for softmin is +inf; for softmax is -inf. We store a sentinel that
        # gets replaced on the first in-window step.
        self._neutral = float("inf") if kind == "G" else float("-inf")
        self.value = torch.full((num_envs,), self._neutral, device=self.device, dtype=torch.float)
        self.seen = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    # ---- lifecycle ---------------------------------------------------------
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.value.fill_(self._neutral)
            self.seen.zero_()
        else:
            self.value[env_ids] = self._neutral
            self.seen[env_ids] = False

    def step(
        self,
        motion_time: torch.Tensor,
        signals: Dict[str, torch.Tensor],
        beta: float,
    ) -> None:
        """Update running reduce for envs whose motion_time is currently inside [a, b]."""
        in_window = (motion_time >= self.a) & (motion_time <= self.b)
        if not in_window.any():
            return
        rho_now = self.child.robustness(signals, motion_time, beta)  # (N,)
        # Envs entering the window for the first time: overwrite the +/-inf sentinel.
        first = in_window & (~self.seen)
        later = in_window & self.seen
        if first.any():
            self.value = torch.where(first, rho_now, self.value)
            self.seen = self.seen | first
        if later.any():
            if self.kind == "G":
                reduced = _pairwise_softmin(self.value, rho_now, beta=beta)
            else:
                reduced = _pairwise_softmax(self.value, rho_now, beta=beta)
            self.value = torch.where(later, reduced, self.value)

    def read(self, rho_now: Optional[torch.Tensor] = None, beta: float = 10.0) -> torch.Tensor:
        """Return the current running ρ.

        If `self.seen` is False for an env (window hasn't opened yet), fall back to
        `rho_now` when provided (otherwise 0) so we still emit a meaningful gradient
        pre-window; post-window, `self.value` is the closed soft-reduce.
        """
        if rho_now is None:
            rho_now = torch.zeros_like(self.value)
        return torch.where(self.seen, self.value, rho_now)


class Always(STLNode):
    """G_[a,b] φ — window minimum of child ρ over [a, b]."""

    def __init__(self, child: STLNode, a: float, b: float, acc: EventWindowAccumulator):
        assert acc.kind == "G" and acc.child is child
        self.child = child
        self.a = float(a)
        self.b = float(b)
        self.acc = acc

    def robustness(self, signals, motion_time, beta):
        rho_now = self.child.robustness(signals, motion_time, beta)
        return self.acc.read(rho_now=rho_now, beta=beta)

    def accumulators(self):
        return [self.acc, *self.child.accumulators()]


class Eventually(STLNode):
    """F_[a,b] φ — window maximum of child ρ over [a, b]."""

    def __init__(self, child: STLNode, a: float, b: float, acc: EventWindowAccumulator):
        assert acc.kind == "F" and acc.child is child
        self.child = child
        self.a = float(a)
        self.b = float(b)
        self.acc = acc

    def robustness(self, signals, motion_time, beta):
        rho_now = self.child.robustness(signals, motion_time, beta)
        return self.acc.read(rho_now=rho_now, beta=beta)

    def accumulators(self):
        return [self.acc, *self.child.accumulators()]


class Until(STLNode):
    """φ U_[a,b] ψ — not needed for far_jump; stub with clear NotImplementedError."""

    def __init__(self, left: STLNode, right: STLNode, a: float, b: float):
        self.left, self.right, self.a, self.b = left, right, float(a), float(b)

    def robustness(self, signals, motion_time, beta):
        raise NotImplementedError(
            "Until operator is not implemented for the online accumulator path. "
            "Add it when a task spec requires it."
        )


# ---------------------------------------------------------------------------
# Helpers to build Always/Eventually together with their accumulator
# ---------------------------------------------------------------------------
def make_always(
    child: STLNode, a: float, b: float, num_envs: int, device: torch.device | str
) -> Always:
    acc = EventWindowAccumulator(child, a, b, kind="G", num_envs=num_envs, device=device)
    return Always(child, a, b, acc)


def make_eventually(
    child: STLNode, a: float, b: float, num_envs: int, device: torch.device | str
) -> Eventually:
    acc = EventWindowAccumulator(child, a, b, kind="F", num_envs=num_envs, device=device)
    return Eventually(child, a, b, acc)


@dataclass
class STLContext:
    """Aggregates the compiled spec, its accumulators, and task-specific events/meta."""

    spec_root: STLNode
    accumulators: List[EventWindowAccumulator]
    events: Dict[str, float]
    meta: Dict[str, float]
