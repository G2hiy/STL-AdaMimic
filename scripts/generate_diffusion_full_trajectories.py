"""生成 D_ref^diff (创新点① root-only).

核心: **SDEdit** 风格采样 — 把 ref 的 base_pos 归一化后加噪到 t₀, 再从 t₀ denoise 到 0,
    仅 base_pos 自由演化. 条件 Δp 控制偏离方向.

设计 (AdaMimic 论文 Eq.3):
    - joint_position / base_pose / link_orientation 直接复制 ref (q_local 不变)
    - base_position 由扩散生成
    - link_position 由 Δbase_pos 解析平移得到 (无需 FK)
    速度场全部按 fps 有限差分.

输入:
    --ckpt_path   train_diffusion_full 产物 (含 model_config / scheduler_config / norm_stats)
    --ref_path    基准 data.pt
    --output      .pt 输出

输出 (与 motionlib.load_diffusion_variants 对齐):
    {
      "variants": [
        dict(base_position, base_pose, base_velocity, base_angular_velocity,
             joint_position, joint_velocity,
             link_position, link_orientation,
             link_velocity, link_angular_velocity),
        ...
      ],
      "meta": {...}
    }
"""

import argparse
import os

import torch
from diffusers import DDPMScheduler

from legged_gym.diffusion.root_mdm import RootDiffusionModel
from legged_gym.diffusion.filter import kinematic_filter


TRAJ_DIM = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--ref_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_variants", type=int, default=32)
    p.add_argument("--delta_p_range", default="x=-0.3,0.3 y=-0.3,0.3 z=-0.1,0.1",
                   help="条件 Δp 采样范围 (原始空间, 米)")
    p.add_argument("--overgenerate_ratio", type=float, default=4.0)
    p.add_argument("--max_rounds", type=int, default=5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--sdedit_t_start", type=int, default=500,
                   help="SDEdit 加噪起点 t₀ ∈ [1, num_train_timesteps-1]. 500/1000 ≈ 中等保留 ref 结构.")
    p.add_argument("--max_speed", type=float, default=4.0)
    p.add_argument("--max_accel", type=float, default=30.0)
    p.add_argument("--min_height", type=float, default=0.05)
    p.add_argument("--fps", type=float, default=50.0,
                   help="data.pt 真实 50fps; 用于 base_velocity / link_velocity 有限差分")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_delta_p_range(s: str) -> torch.Tensor:
    low, high = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    idx = {"x": 0, "y": 1, "z": 2}
    for tok in s.split():
        key, vals = tok.split("=")
        a, b = vals.split(",")
        low[idx[key]] = float(a)
        high[idx[key]] = float(b)
    return torch.tensor([low, high], dtype=torch.float32)


def sample_conditions(n: int, rng: torch.Tensor) -> torch.Tensor:
    u = torch.rand(n, 3)
    return rng[0] + u * (rng[1] - rng[0])


def load_model(ckpt_path: str, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device)
    assert "norm_stats" in blob, "ckpt 缺 norm_stats; 需由 train_diffusion_full.py 产出"
    model = RootDiffusionModel(**blob["model_config"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    scheduler = DDPMScheduler.from_config(blob["scheduler_config"])
    return model, scheduler, blob


@torch.no_grad()
def sample_sdedit(
    model: RootDiffusionModel,
    scheduler: DDPMScheduler,
    conds: torch.Tensor,           # (M, cond_dim) 归一化空间
    x_ref_norm: torch.Tensor,      # (T, 3) 归一化空间 ref base_pos
    t_start: int,
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """SDEdit: 加噪 ref 到 t_start, 再逐训练步 denoise 到 0.

    用完整训练 timestep, 保证 t_start→0 有足够反向步数 (避免 base_pos 高频抖动).
    """
    M = conds.shape[0]

    scheduler.set_timesteps(
        int(scheduler.config.num_train_timesteps),
        device=device,
    )
    timesteps = scheduler.timesteps  # [999, 998, ..., 0]

    start_mask = timesteps <= t_start
    start_idx = int(start_mask.nonzero()[0].item()) if start_mask.any() else 0
    t0 = int(timesteps[start_idx].item())

    x_ref_dev = x_ref_norm.to(device).unsqueeze(0).expand(M, -1, -1).contiguous()
    ts_add = torch.full((M,), t0, dtype=torch.long, device=device)

    noise = torch.randn_like(x_ref_dev)
    x = scheduler.add_noise(x_ref_dev, noise, ts_add)

    conds_dev = conds.to(device)

    for t in timesteps[start_idx:]:
        ts = torch.full((M,), int(t.item()), dtype=torch.long, device=device)
        eps = model(x, ts, conds_dev)
        x = scheduler.step(eps, t, x).prev_sample

    return x


def diff_last_repeat(x: torch.Tensor, fps: float) -> torch.Tensor:
    """有限差分, 末尾复制最后一帧保持 T."""
    d = (x[1:] - x[:-1]) * fps
    return torch.cat([d, d[-1:]], dim=0)


def build_variant(
    ref: dict,
    new_base_pos: torch.Tensor,    # (T, 3) world-space
    fps: float,
) -> dict:
    """按 AdaMimic Eq.3 构造变体: 仅平移根, 全身姿态/几何相对不变.

    - joint_position / base_pose / link_orientation / joint_velocity / base_angular_velocity
      / link_angular_velocity 复制 ref
    - link_position[t] = ref_link_position[t] + Δbase_pos[t]
    - base_velocity / link_velocity 由新位置有限差分得到
    """
    ref_base_pos = ref["base_position"].float()
    ref_link_pos = ref["link_position"].float()

    delta = (new_base_pos - ref_base_pos)            # (T, 3)
    new_link_pos = ref_link_pos + delta.unsqueeze(1)  # (T, L, 3)

    new_base_vel = diff_last_repeat(new_base_pos, fps)
    new_link_vel = diff_last_repeat(new_link_pos, fps)

    return {
        "base_position":         new_base_pos.clone(),
        "base_pose":             ref["base_pose"].float().clone(),
        "base_velocity":         new_base_vel,
        "base_angular_velocity": ref["base_angular_velocity"].float().clone(),
        "joint_position":        ref["joint_position"].float().clone(),
        "joint_velocity":        ref["joint_velocity"].float().clone(),
        "link_position":         new_link_pos,
        "link_orientation":      ref["link_orientation"].float().clone(),
        "link_velocity":         new_link_vel,
        "link_angular_velocity": ref["link_angular_velocity"].float().clone(),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- load model + stats ---
    model, scheduler, blob = load_model(args.ckpt_path, device)
    mean = blob["norm_stats"]["mean"].float().cpu()      # (3,)
    std = blob["norm_stats"]["std"].float().cpu()        # (3,)
    assert mean.shape == (TRAJ_DIM,) and std.shape == (TRAJ_DIM,), (
        f"norm_stats 维度异常: mean={tuple(mean.shape)} std={tuple(std.shape)}; "
        f"需 root-only 训练产物 (3-dim)"
    )
    num_train_ts = int(scheduler.config.num_train_timesteps)
    assert 1 <= args.sdedit_t_start <= num_train_ts - 1, (
        f"--sdedit_t_start={args.sdedit_t_start} 越界 [1, {num_train_ts-1}]"
    )
    print(f"[ckpt] loaded {args.ckpt_path}")
    print(f"       scheduler.num_train_timesteps={num_train_ts}  "
          f"sdedit_t_start={args.sdedit_t_start}")
    print(f"       norm_stats mean={mean.tolist()}  std={std.tolist()}")

    # --- load ref ---
    ref = torch.load(args.ref_path)
    ref_base_pos = ref["base_position"].float()
    T_ref = ref_base_pos.shape[0]
    L = ref["link_position"].shape[1]
    print(f"[ref]  T={T_ref}  fps={args.fps}  N_links={L}")

    # 训练集 base_pos 是相对窗口首帧的位移; 这里 ref 也减去自己的首帧再归一化
    ref_base_model = ref_base_pos - ref_base_pos[0:1, :]
    x_ref_norm = (ref_base_model - mean[None, :]) / std[None, :]

    # --- 条件采样范围 (原始 → 归一化) ---
    rng_raw = parse_delta_p_range(args.delta_p_range)                   # (2, 3)
    assert (std > 1e-6).all(), f"std degenerate: {std.tolist()}"
    rng_norm = rng_raw / std[None, :]
    print(f"[cond] Δp range raw  low={rng_raw[0].tolist()} high={rng_raw[1].tolist()}")
    print(f"       Δp range norm low={rng_norm[0].tolist()} high={rng_norm[1].tolist()}")

    # --- 生成循环 ---
    n_target = args.num_variants
    kept_variants: list = []
    kept_cond: list = []

    for rd in range(args.max_rounds):
        n_gen = int(n_target * args.overgenerate_ratio) - len(kept_variants)
        if n_gen <= 0:
            break

        conds_norm = sample_conditions(n_gen, rng_norm)                 # (n_gen, 3)
        x_norm = sample_sdedit(
            model, scheduler, conds_norm, x_ref_norm,
            t_start=args.sdedit_t_start,
            num_inference_steps=args.num_inference_steps,
            device=device,
        ).cpu()                                                         # (n_gen, T, 3)

        x_denorm = x_norm * std[None, None, :] + mean[None, None, :]    # (n_gen, T, 3) 相对位移
        # 恢复成 world-space: 让生成轨迹的首帧对齐 ref 首帧
        gen_base_pos = (
            x_denorm
            - x_denorm[:, 0:1, :]
            + ref_base_pos[0:1, :].unsqueeze(0)
        )

        # 诊断: speed/accel/height
        dt = 1.0 / args.fps
        v_diag = (gen_base_pos[:, 1:] - gen_base_pos[:, :-1]) / dt
        a_diag = (v_diag[:, 1:] - v_diag[:, :-1]) / dt
        speed_max = v_diag.norm(dim=-1).max(dim=-1).values
        accel_max = a_diag.norm(dim=-1).max(dim=-1).values
        ref_v = (ref_base_pos[1:] - ref_base_pos[:-1]) / dt
        ref_a = (ref_v[1:] - ref_v[:-1]) / dt
        z_min = gen_base_pos[:, :, 2].min().item()
        z_max = gen_base_pos[:, :, 2].max().item()
        print(f"[round {rd}] n_gen={n_gen}  base_z=[{z_min:.3f},{z_max:.3f}]m")
        print(f"           speed_max: p50={speed_max.median():.2f} p95={speed_max.quantile(0.95):.2f} "
              f"| ref_max={ref_v.norm(dim=-1).max():.2f} m/s  (thr={args.max_speed})")
        print(f"           accel_max: p50={accel_max.median():.1f} p95={accel_max.quantile(0.95):.1f} "
              f"| ref_max={ref_a.norm(dim=-1).max():.1f} m/s^2  (thr={args.max_accel})")

        mask = kinematic_filter(
            gen_base_pos, fps=args.fps,
            max_speed=args.max_speed, max_accel=args.max_accel,
            min_height=args.min_height,
        )
        n_pass = int(mask.sum().item())
        print(f"           kinematic_filter passed {n_pass}/{n_gen}")
        if n_pass == 0:
            continue

        idx_pass = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
        for k in idx_pass:
            if len(kept_variants) >= n_target:
                break
            v = build_variant(ref=ref, new_base_pos=gen_base_pos[k], fps=args.fps)
            kept_variants.append(v)
            kept_cond.append(conds_norm[k])
        if len(kept_variants) >= n_target:
            break

    assert len(kept_variants) > 0, (
        "no variants survived; 放宽 --max_speed/--max_accel, 或降低 --sdedit_t_start"
    )
    kept_variants = kept_variants[:n_target]
    kept_cond = kept_cond[:n_target]
    print(f"[final] kept {len(kept_variants)} variants")

    # self-test: Eq.3 不变量 (joint/rpy/相对几何) 应严格相等
    ref_joint = ref["joint_position"].float()
    ref_rpy = ref["base_pose"].float()
    ref_link_rel = ref["link_position"].float() - ref["base_position"].float().unsqueeze(1)
    for i, v in enumerate(kept_variants):
        assert torch.equal(v["joint_position"], ref_joint), f"variant {i}: joint_position 偏离 ref"
        assert torch.equal(v["base_pose"], ref_rpy), f"variant {i}: base_pose 偏离 ref"
        var_link_rel = v["link_position"] - v["base_position"].unsqueeze(1)
        assert torch.allclose(var_link_rel, ref_link_rel, atol=1e-5), (
            f"variant {i}: link 相对 base 的几何不变性破坏 "
            f"(max err={ (var_link_rel - ref_link_rel).abs().max().item():.2e})"
        )
    print(f"[self-test] Eq.3 invariants OK on {len(kept_variants)} variants")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "variants": kept_variants,
        "meta": dict(
            ckpt_path=args.ckpt_path, ref_path=args.ref_path,
            delta_p_range=args.delta_p_range,
            conds_norm=torch.stack(kept_cond, dim=0),
            max_speed=args.max_speed, max_accel=args.max_accel,
            min_height=args.min_height, fps=args.fps, T=T_ref,
            sdedit_t_start=args.sdedit_t_start,
            num_inference_steps=args.num_inference_steps,
            link_pose_via_translation=True,
            constraints=dict(
                max_speed=args.max_speed,
                max_accel=args.max_accel,
                min_height=args.min_height,
            ),
            num_variants_requested=args.num_variants,
            num_variants_kept=len(kept_variants),
            version="root_only_sdedit_v1",
        ),
    }, args.output)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
