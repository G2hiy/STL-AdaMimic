"""Sanity checks for legged_gym.utils.stl_specs.KeyframeFunnelSTL.

Runs locally (CPU) without isaacgym. Verifies:
  1. forward shape / dtype
  2. no NaN/Inf
  3. ψ monotonicity:   t 远离 keyframe → ψ 变大
  4. ρ 单调性:          d 增大 → ρ 变小（对固定 t）
  5. autograd 通过:     ρ.sum().backward() 产生 d 的非零梯度
"""
import importlib.util
import sys
from pathlib import Path

import torch

# 直接 spec-load stl_specs.py，避开 legged_gym/utils/__init__.py 的 isaacgym 依赖
_STL_PATH = Path(__file__).resolve().parents[1] / "legged_gym" / "legged_gym" / "utils" / "stl_specs.py"
_spec = importlib.util.spec_from_file_location("stl_specs", _STL_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["stl_specs"] = _mod
_spec.loader.exec_module(_mod)
KeyframeFunnelSTL = _mod.KeyframeFunnelSTL
body_pos_tracking_dist = _mod.body_pos_tracking_dist


def main():
    torch.manual_seed(0)
    device = "cpu"
    N, B = 64, 24
    deadlines = [0.6, 1.8, 3.0]  # 3 keyframe deadlines in seconds

    spec = KeyframeFunnelSTL(
        keyframe_deadlines=deadlines,
        T_funnel=0.3,
        eps_min=0.05,
        eps_max=0.30,
        beta_time=20.0,
        device=device,
    )

    # ---- 1. forward shape ---------------------------------------------------
    t = torch.linspace(0.0, 4.0, N, device=device)
    dif = torch.randn(N, B, 3, device=device) * 0.1
    dist = body_pos_tracking_dist(dif, reduce="mean")
    assert dist.shape == (N,), dist.shape
    psi = spec.funnel_width(t)
    rho = spec.robustness(t, dist)
    assert psi.shape == (N,) and rho.shape == (N,)
    assert torch.isfinite(psi).all() and torch.isfinite(rho).all()

    # ---- 2. ψ monotonicity near deadline ------------------------------------
    t_near = torch.tensor([0.6, 0.6 + 0.15, 0.6 + 0.30, 0.6 + 0.60], device=device)
    psi_near = spec.funnel_width(t_near)
    diffs = psi_near[1:] - psi_near[:-1]
    assert (diffs >= -1e-6).all(), f"ψ not non-decreasing: {psi_near.tolist()}"
    assert psi_near[0].item() < spec.eps_min + 0.02, f"ψ@deadline太大: {psi_near[0].item()}"
    assert psi_near[-1].item() > spec.eps_max - 0.02, f"ψ 远端未饱和: {psi_near[-1].item()}"

    # ---- 3. ρ vs dist monotonicity ------------------------------------------
    t_fix = torch.full((5,), 0.6, device=device)
    dist_sweep = torch.tensor([0.0, 0.05, 0.10, 0.20, 0.50], device=device)
    rho_sweep = spec.robustness(t_fix, dist_sweep)
    assert (rho_sweep[1:] - rho_sweep[:-1] <= 1e-6).all(), f"ρ 应随 d 单调下降: {rho_sweep.tolist()}"

    # ---- 4. autograd --------------------------------------------------------
    dif_g = (torch.randn(8, B, 3, device=device) * 0.1).clone().detach().requires_grad_(True)
    t_g = torch.full((8,), 0.7, device=device)
    d_g = body_pos_tracking_dist(dif_g, reduce="mean")
    rho_g = spec.robustness(t_g, d_g)
    rho_g.sum().backward()
    assert dif_g.grad is not None and torch.isfinite(dif_g.grad).all()
    assert dif_g.grad.abs().sum().item() > 0, "梯度全零，STL 不可微"

    print("[OK] STL specs self-check passed.")
    print(f"  ψ@0.6s = {psi_near[0].item():.4f} | ψ@1.2s = {psi_near[-1].item():.4f}")
    print(f"  ρ(d=0) = {rho_sweep[0].item():.4f} | ρ(d=0.5) = {rho_sweep[-1].item():.4f}")


if __name__ == "__main__":
    main()
