"""Joint+Root Diffusion Model (创新点① 完整版).

扩散目标: [joint_27 | base_pos_3 | base_rpy_3] 联合轨迹  形状 (B, T, 33)
条件 ψ:    关键帧 base_pos 位移 Δp ∈ R^3 (归一化空间)，通过 AdaLN 注入 Transformer 每层
调度器:    diffusers.DDPMScheduler (cosine, T=1000)

下游 generate 脚本用 SDEdit 采样得到全 33 维变体, 再用 pytorch_kinematics FK 从
(joint_27, base_pos, base_rpy) 重算 link_pose. 不再有 "joint frozen / inpaint" 假设.
"""

import math
import torch
import torch.nn as nn


def _sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:  # 正弦位置编码
    """Diffusion timestep / 标量条件的 sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=values.device, dtype=torch.float32) / half
    )
    args = values.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class AdaLN(nn.Module):
    """Adaptive LayerNorm: cond → (scale, shift)."""

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  cond: (B, cond_dim)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_attn = AdaLN(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp = AdaLN(d_model, cond_dim)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x, cond)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm_mlp(x, cond))
        return x


class RootDiffusionModel(nn.Module):
    """1D Transformer 噪声预测网络, 接口与 diffusers 兼容 (输入加噪 x_t, 输出 ε̂)."""

    def __init__(
        self,
        traj_dim: int = 33,
        max_seq_len: int = 512,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        cond_dim: int = 3,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(traj_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        fused_cond_dim = d_model  # t_emb + cond_emb 相加后维度仍是 d_model
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, fused_cond_dim) for _ in range(n_layers)]
        )
        self.out_norm = AdaLN(d_model, fused_cond_dim)
        self.out_proj = nn.Linear(d_model, traj_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.t_embed_dim = t_embed_dim

    def forward(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x_t:       (B, T, traj_dim)
        timesteps: (B,) int
        cond:      (B, cond_dim)
        returns:   ε̂  同 x_t 形状
        """
        B, T, _ = x_t.shape
        assert T <= self.pos_embed.shape[1], f"seq len {T} exceeds max {self.pos_embed.shape[1]}"

        h = self.input_proj(x_t) + self.pos_embed[:, :T]
        t_emb = self.t_mlp(_sinusoidal_embedding(timesteps, self.t_embed_dim))
        c_emb = self.cond_mlp(cond)
        fused = t_emb + c_emb

        for blk in self.blocks:
            h = blk(h, fused)
        h = self.out_norm(h, fused)
        return self.out_proj(h)
