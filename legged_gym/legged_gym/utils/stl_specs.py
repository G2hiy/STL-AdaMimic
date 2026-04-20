"""Signal Temporal Logic (STL) utilities for AdaMimic reward shaping (创新点②).

References:
  - Balakrishnan & Deshmukh (2019) "Structured Reward Shaping using STL"
  - Varnai & Dimarogonas (2023) "Funnel-based Reward Shaping for STL Tasks"
  - Leung et al. (2025) "STLCG++: Masking Approach for Differentiable STL"
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch


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


class KeyframeFunnelSTL:
    """Funnel-based robustness for an 'eventually reach reference within ε near each keyframe' spec.

        ρ(t) = ψ(t) − d(t)
          ψ(t) = eps_min + (eps_max − eps_min) · clip(Δt(t) / T_funnel, 0, 1)
          Δt(t) = softmin_k |t − t_k|   (smoothed min over K keyframe deadlines)
          d(t)  = scalar tracking distance at time t

    ψ shrinks to eps_min as t approaches any deadline and widens to eps_max between them.
    ρ > 0 iff the current tracking distance is inside the funnel at time t.
    """

    def __init__(
        self,
        keyframe_deadlines: Iterable[float],
        T_funnel: float,
        eps_min: float,
        eps_max: float,
        beta_time: float = 20.0,
        device: torch.device | str = "cuda",
    ):
        self.deadlines = torch.as_tensor(list(keyframe_deadlines), dtype=torch.float, device=device).view(-1)
        assert self.deadlines.numel() > 0, "keyframe_deadlines must be non-empty"
        assert T_funnel > 0, "T_funnel must be positive"
        assert eps_max >= eps_min > 0, "require 0 < eps_min <= eps_max"
        self.T_funnel = float(T_funnel)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.beta_time = float(beta_time)
        self.device = self.deadlines.device

    def funnel_width(self, motion_time: torch.Tensor) -> torch.Tensor:
        delta = (motion_time.unsqueeze(-1) - self.deadlines.unsqueeze(0)).abs()  # (N, K)
        closest = softmin(delta, beta=self.beta_time, dim=-1).clamp_min(0.0)     # (N,)
        ratio = (closest / self.T_funnel).clamp(0.0, 1.0)
        return self.eps_min + (self.eps_max - self.eps_min) * ratio              # (N,)

    def robustness(self, motion_time: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """Funnel robustness ρ(t) = ψ(t) − d(t)."""
        return self.funnel_width(motion_time) - dist


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
