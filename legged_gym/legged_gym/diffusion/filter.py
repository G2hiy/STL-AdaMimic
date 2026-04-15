"""Kinematic 过滤 (创新点① MVP C3).

约束目标: 生成的 base_position 序列应满足 G1 物理可行性 (不做 IsaacGym rollout,
仅基于一阶/二阶差分和地面约束). 物理 rollout 过滤移至 B 阶段.
"""

import torch


def kinematic_filter(
    base_pos: torch.Tensor,         # (M, T, 3)
    fps: float,
    max_speed: float = 4.0,         # m/s
    max_accel: float = 30.0,        # m/s^2
    min_height: float = 0.05,       # m, 与 motionlib.py:166 的 clamp 对齐
    max_height: float = 3.0,        # m, 粗筛
) -> torch.BoolTensor:
    """
    返回 (M,) 布尔 mask, True = 通过过滤.

    具体判据:
      - 每帧根速度 |v| <= max_speed
      - 每帧根加速度 |a| <= max_accel
      - 每帧高度 z ∈ [min_height, max_height]
    """
    assert base_pos.dim() == 3 and base_pos.shape[-1] == 3, base_pos.shape
    dt = 1.0 / fps
    v = (base_pos[:, 1:] - base_pos[:, :-1]) / dt                  # (M, T-1, 3)
    a = (v[:, 1:] - v[:, :-1]) / dt                                # (M, T-2, 3)

    speed = v.norm(dim=-1)                                         # (M, T-1)
    accel = a.norm(dim=-1)                                         # (M, T-2)
    z = base_pos[..., 2]                                           # (M, T)

    ok_speed  = speed.max(dim=-1).values <= max_speed
    ok_accel  = accel.max(dim=-1).values <= max_accel
    ok_height = (z.min(dim=-1).values >= min_height) & (z.max(dim=-1).values <= max_height)
    return ok_speed & ok_accel & ok_height
