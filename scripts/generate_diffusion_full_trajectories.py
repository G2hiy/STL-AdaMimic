"""生成 D_ref^diff (创新点① 完整版).

核心: **SDEdit** 风格采样 — 把 ref 归一化后加噪到 t₀, 再从 t₀ denoise 到 0,
    三路通道 (joint_27 | base_pos | base_rpy) 同时自由演化. 条件 Δp 控制偏离方向.

与旧"纯自由+ref joint inpaint"路线的差异:
    - 不再用 ref joint 做 inpaint (AMASS vs task-ref OOD 会破坏 base_pos 预测,
      实测 speed 73 m/s → 自由生成 4.5 m/s).
    - 从 ref 出发加噪到中等 t₀, 保留 ref 语义 (跳远姿态) 的同时让 AMASS 先验
      引导 joint/base 协同变化.

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
import math
import os

import torch
from diffusers import DDPMScheduler

from legged_gym.diffusion.root_mdm import RootDiffusionModel
from legged_gym.diffusion.filter import kinematic_filter
from legged_gym.diffusion.fk import G1FK


TRAJ_DIM = 33
JOINT_SLICE = slice(0, 27)
BASE_POS_SLICE = slice(27, 30)
BASE_RPY_SLICE = slice(30, 33)


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
                   help="data.pt 真实 50fps; 用于 link_velocity/base_velocity 有限差分")
    p.add_argument("--urdf_path", default=None,
                   help="G1 URDF; 默认用 legged_gym.diffusion.fk 里的 DEFAULT_URDF")
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


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """角度差值 wrap 到 (-π, π]."""
    return (x + math.pi) % (2 * math.pi) - math.pi


@torch.no_grad()
def sample_sdedit(
    model: RootDiffusionModel,
    scheduler: DDPMScheduler,
    conds: torch.Tensor,           # (M, cond_dim) 归一化空间
    x_ref_norm: torch.Tensor,      # (T, 33) 归一化空间 ref
    t_start: int,
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """SDEdit: 把 ref 加噪到 t₀, 再 denoise 到 0. 返回 (M, T, 33) 归一化空间."""
    M = conds.shape[0]
    T = x_ref_norm.shape[0]
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps  # 降序 [T_max-1, …, 0]

    # 找到 <= t_start 的第一个 timestep (denoise 起点)
    start_mask = timesteps <= t_start
    start_idx = int(start_mask.nonzero()[0].item()) if start_mask.any() else 0
    t0 = int(timesteps[start_idx].item())

    x_ref_dev = x_ref_norm.to(device).unsqueeze(0).expand(M, -1, -1).contiguous()  # (M, T, 33)
    ts_add = torch.full((M,), t0, dtype=torch.long, device=device)
    noise = torch.randn_like(x_ref_dev)
    x = scheduler.add_noise(x_ref_dev, noise, ts_add)

    conds_dev = conds.to(device)
    for t in timesteps[start_idx:]:
        ts = torch.full((M,), int(t.item()), dtype=torch.long, device=device)
        eps = model(x, ts, conds_dev)
        x = scheduler.step(eps, t, x).prev_sample
    return x


def build_variant(
    ref: dict,
    new_joint_27: torch.Tensor,    # (T, 27) 原始空间
    new_base_pos: torch.Tensor,    # (T, 3)
    new_base_rpy: torch.Tensor,    # (T, 3)
    fk: G1FK,
    fps: float,
    link_names: list,
) -> dict:
    """用 FK 重算 link_pose, 派生速度场全部有限差分."""
    T = new_joint_27.shape[0]
    # FK (1, T, ...)
    link_pos, link_rpy, _ = fk(
        new_base_pos.unsqueeze(0), new_base_rpy.unsqueeze(0), new_joint_27.unsqueeze(0),
        target_link_names=link_names,
    )
    link_pos = link_pos.squeeze(0).to(new_base_pos.dtype).cpu()    # (T, L, 3)
    link_rpy = link_rpy.squeeze(0).to(new_base_pos.dtype).cpu()    # (T, L, 3)

    # 速度场 (有限差分, 末尾复制最后一帧保持 T)
    def diff_last_repeat(x, fps_):
        d = (x[1:] - x[:-1]) * fps_
        return torch.cat([d, d[-1:]], dim=0)

    def rpy_diff_last_repeat(x, fps_):
        d = wrap_to_pi(x[1:] - x[:-1]) * fps_
        return torch.cat([d, d[-1:]], dim=0)

    new_joint_vel = diff_last_repeat(new_joint_27, fps)
    new_base_vel = diff_last_repeat(new_base_pos, fps)
    new_base_ang_vel = rpy_diff_last_repeat(new_base_rpy, fps)
    new_link_vel = diff_last_repeat(link_pos, fps)
    new_link_ang_vel = rpy_diff_last_repeat(link_rpy, fps)

    return {
        "base_position":         new_base_pos.clone(),
        "base_pose":             new_base_rpy.clone(),
        "base_velocity":         new_base_vel,
        "base_angular_velocity": new_base_ang_vel,
        "joint_position":        new_joint_27.clone(),
        "joint_velocity":        new_joint_vel,
        "link_position":         link_pos,
        "link_orientation":      link_rpy,
        "link_velocity":         new_link_vel,
        "link_angular_velocity": new_link_ang_vel,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- load model + stats ---
    model, scheduler, blob = load_model(args.ckpt_path, device)
    mean = blob["norm_stats"]["mean"].float().cpu()      # (33,)
    std = blob["norm_stats"]["std"].float().cpu()        # (33,)
    assert mean.shape == (TRAJ_DIM,) and std.shape == (TRAJ_DIM,)
    num_train_ts = int(scheduler.config.num_train_timesteps)
    assert 1 <= args.sdedit_t_start <= num_train_ts - 1, (
        f"--sdedit_t_start={args.sdedit_t_start} 越界 [1, {num_train_ts-1}]"
    )
    print(f"[ckpt] loaded {args.ckpt_path}")
    print(f"       scheduler.num_train_timesteps={num_train_ts}  "
          f"sdedit_t_start={args.sdedit_t_start}")
    print(f"       norm_stats mean[base_pos]={mean[BASE_POS_SLICE].tolist()}  "
          f"std[base_pos]={std[BASE_POS_SLICE].tolist()}")

    # --- load ref ---
    ref = torch.load(args.ref_path)
    ref_base_pos = ref["base_position"].float()
    ref_rpy = ref["base_pose"].float()
    ref_joint = ref["joint_position"].float()
    T_ref = ref_base_pos.shape[0]
    assert ref_joint.shape[1] == 27, f"ref joint_position shape={ref_joint.shape}"
    print(f"[ref]  T={T_ref}  fps={args.fps}")

    # --- FK (CPU 足够快, 避免 CUDA 同步) ---
    urdf_kwargs = {"urdf_path": args.urdf_path} if args.urdf_path else {}
    fk = G1FK(device="cpu", **urdf_kwargs)
    # 与 motion_tracking.py:2124 keyframe_names 过滤一致 (只取含 'keyframe' 子串的 link)
    link_names = list(fk.keyframe_link_names)
    assert len(link_names) == ref["link_position"].shape[1], (
        f"FK keyframe links ({len(link_names)}) != ref link_position N_bodies "
        f"({ref['link_position'].shape[1]}); 检查 URDF 与 data.pt 是否匹配"
    )
    print(f"[fk]   N_keyframe_links={len(link_names)}  names={link_names}")

    # 与新版 S1 保持一致：base_pos 三维全部相对首帧
    ref_base_model = ref_base_pos - ref_base_pos[0:1, :]

    ref_stack = torch.cat([ref_joint, ref_base_model, ref_rpy], dim=-1)
    x_ref_norm = (ref_stack - mean[None, :]) / std[None, :]

    # --- 条件采样范围 (原始 → 归一化) ---
    rng_raw = parse_delta_p_range(args.delta_p_range)                   # (2, 3)
    base_pos_std = std[BASE_POS_SLICE]
    assert (base_pos_std > 1e-6).all(), \
        f"base_pos_std degenerate: {base_pos_std.tolist()}; ckpt 的 norm_stats 异常"
    rng_norm = rng_raw / base_pos_std[None, :]
    print(f"[cond] Δp range raw low={rng_raw[0].tolist()} high={rng_raw[1].tolist()}")
    print(f"       Δp range norm low={rng_norm[0].tolist()} high={rng_norm[1].tolist()}")

    # --- 生成循环 ---
    n_target = args.num_variants
    kept_variants: list = []
    kept_cond: list = []
    joint_dev_log: list = []

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
        ).cpu()                                                         # (n_gen, T, 33)

        x_denorm = x_norm * std[None, None, :] + mean[None, None, :]    # (n_gen, T, 33)
        gen_joint = x_denorm[:, :, JOINT_SLICE]
        gen_base_rel = x_denorm[:, :, BASE_POS_SLICE]
        gen_base_rpy = x_denorm[:, :, BASE_RPY_SLICE]

        # 新版 S1/S2 的 base_pos 是相对位移，所以这里恢复成 data.pt / motionlib 需要的 world-space
        gen_base_pos = (
            gen_base_rel
            - gen_base_rel[:, 0:1, :]
            + ref_base_pos[0:1, :].unsqueeze(0)
        )

        # joint/rpy 继续首帧对齐，避免 reset 瞬移
        gen_joint = gen_joint - gen_joint[:, 0:1, :] + ref_joint[0:1, :].unsqueeze(0)
        gen_base_rpy = gen_base_rpy - gen_base_rpy[:, 0:1, :] + ref_rpy[0:1, :].unsqueeze(0)

        joint_dev = (gen_joint - ref_joint.unsqueeze(0)).norm(dim=-1).mean().item()
        joint_dev_log.append(joint_dev)

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
        print(f"[round {rd}] n_gen={n_gen}  joint_dev_to_ref={joint_dev:.3f} (rad/frame·sqrt27)  "
              f"base_z=[{z_min:.3f},{z_max:.3f}]m")
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
            v = build_variant(
                ref=ref,
                new_joint_27=gen_joint[k],
                new_base_pos=gen_base_pos[k],
                new_base_rpy=gen_base_rpy[k],
                fk=fk,
                fps=args.fps,
                link_names=link_names,
            )
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
            joint_dev_log=joint_dev_log,
            link_names=link_names,
            constraints=dict(
                max_speed=args.max_speed,
                max_accel=args.max_accel,
                min_height=args.min_height,
            ),
            num_variants_requested=args.num_variants,
            num_variants_kept=len(kept_variants),
            version="full_sdedit_v1",
        ),
    }, args.output)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
