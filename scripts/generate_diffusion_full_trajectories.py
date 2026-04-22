"""生成 D_ref^diff (创新点① 完整版 S3).

核心创新: inpainting-guided sampling.
    在去噪循环的每一步, 把已知通道 (joint_27, 可选 base_rpy) 替换为
    "加噪到当前 t 的 ref", 让模型在未知通道 (base_pos) 上演化,
    同时被真实 AMASS 学到的"关节-根耦合先验"约束.

输入:
    --ckpt_path   train_diffusion_full.py 的产物 (含 model_config / scheduler_config / norm_stats)
    --ref_path    基准 data.pt (提供 joint_position / base_pose / link_*)
    --output      .pt 输出路径

输出 (与 motionlib.load_diffusion_variants 约定一致):
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

第一版默认: --inpaint_rpy=True --freeze_rpy=True
    (rpy 完全冻结; 与 MVP 做严格单变量 A/B: 唯一差异是 AMASS 真实耦合先验)

第二版 (独立立项): --no-inpaint_rpy --no-freeze_rpy + pytorch_kinematics FK 重算 link_orientation
"""

import argparse
import os
import torch
from diffusers import DDPMScheduler

from legged_gym.diffusion.root_mdm import RootDiffusionModel
from legged_gym.diffusion.filter import kinematic_filter


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
    p.add_argument("--max_speed", type=float, default=4.0)
    p.add_argument("--max_accel", type=float, default=30.0)
    p.add_argument("--min_height", type=float, default=0.05)
    p.add_argument("--fps", type=float, default=50.0,
                   help="data.pt 真实 50fps; 用于 link_velocity/base_velocity 有限差分")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    # 第一版 gates (均默认 True)
    p.add_argument("--inpaint_rpy", action="store_true", default=True,
                   help="ref_rpy 作为 known 通道参与 inpainting (第一版默认 True)")
    p.add_argument("--no-inpaint_rpy", dest="inpaint_rpy", action="store_false")
    p.add_argument("--freeze_rpy", action="store_true", default=True,
                   help="后处理强制 base_rpy=ref (第一版默认 True)")
    p.add_argument("--no-freeze_rpy", dest="freeze_rpy", action="store_false")
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


def build_known_template(
    ref_joint_27: torch.Tensor,         # (T, 27)
    ref_rpy: torch.Tensor,              # (T, 3)
    mean: torch.Tensor,                 # (33,)
    std: torch.Tensor,                  # (33,)
) -> torch.Tensor:
    """构造 (T, 33) 归一化空间 known 模板; base_pos 通道填 0 (不 inpaint)."""
    T = ref_joint_27.shape[0]
    known = torch.zeros(T, TRAJ_DIM, dtype=torch.float32)
    known[:, JOINT_SLICE] = (ref_joint_27 - mean[JOINT_SLICE]) / std[JOINT_SLICE]
    known[:, BASE_RPY_SLICE] = (ref_rpy - mean[BASE_RPY_SLICE]) / std[BASE_RPY_SLICE]
    return known


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """角度差值 wrap 到 (-π, π], 用于 rpy 有限差分."""
    import math
    return (x + math.pi) % (2 * math.pi) - math.pi


@torch.no_grad()
def sample_inpainting(
    model: RootDiffusionModel,
    scheduler: DDPMScheduler,
    conds: torch.Tensor,                # (M, 3) 归一化空间
    x_known_norm: torch.Tensor,         # (T, 33) 归一化空间
    mask_known: torch.Tensor,           # (33,) bool
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Inpainting 去噪: 已知通道在每一步替换为加噪 ref, 未知通道自由演化.

    返回 (M, T, 33) 归一化空间.
    """
    M = conds.shape[0]
    T = x_known_norm.shape[0]
    x = torch.randn(M, T, TRAJ_DIM, device=device)
    x_known_dev = x_known_norm.to(device).unsqueeze(0).expand(M, -1, -1)
    mask_dev = mask_known.to(device)

    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in scheduler.timesteps:
        ts = torch.full((M,), int(t.item()), dtype=torch.long, device=device)
        # 已知通道 = 加噪到 t 的 ref
        noise_ref = torch.randn_like(x_known_dev)
        x_known_t = scheduler.add_noise(x_known_dev, noise_ref, ts)
        x = torch.where(mask_dev[None, None, :], x_known_t, x)
        eps = model(x, ts, conds.to(device))
        x = scheduler.step(eps, t, x).prev_sample
    # 最后一步也投影一次, 保证输出上已知通道严格等于 ref (归一化空间)
    x = torch.where(mask_dev[None, None, :], x_known_dev, x)
    return x


def build_variant(
    ref: dict,
    new_base_pos: torch.Tensor,         # (T, 3) 原始空间
    new_base_rpy: torch.Tensor,         # (T, 3) 原始空间
    fps: float,
) -> dict:
    """式 3 + 第一版保守近似.

    第一版 (freeze_rpy=True):
        new_base_rpy ≡ ref_rpy → link_orientation 零误差; link_position 平移零误差
    第二版 (freeze_rpy=False):
        link_orientation = ref["link_orientation"] 是保守近似 (此处不 FK 重算)
    """
    ref_base = ref["base_position"].float()
    delta = new_base_pos - ref_base                                        # (T, 3)
    new_link_pos = ref["link_position"].float() + delta[:, None, :]        # (T, N_bodies, 3)

    # base_angular_velocity: rpy 差分 + wrap
    drpy = wrap_to_pi(new_base_rpy[1:] - new_base_rpy[:-1]) * fps          # (T-1, 3)
    new_base_ang_vel = torch.cat([drpy, drpy[-1:]], dim=0)                 # (T, 3)

    # link_velocity / base_velocity: 有限差分
    v_link = (new_link_pos[1:] - new_link_pos[:-1]) * fps
    new_link_vel = torch.cat([v_link, v_link[-1:]], dim=0)
    v_base = (new_base_pos[1:] - new_base_pos[:-1]) * fps
    new_base_vel = torch.cat([v_base, v_base[-1:]], dim=0)

    variant = {
        "base_position":         new_base_pos.clone(),
        "base_pose":             new_base_rpy.clone(),
        "base_velocity":         new_base_vel,
        "base_angular_velocity": new_base_ang_vel,
        "joint_position":        ref["joint_position"].clone(),     # 式 3: 完全拷贝
        "joint_velocity":        ref["joint_velocity"].clone(),
        "link_position":         new_link_pos,
        "link_orientation":      ref["link_orientation"].clone(),   # 第一版零误差 / 第二版保守近似
        "link_velocity":         new_link_vel,
        "link_angular_velocity": ref["link_angular_velocity"].clone(),
    }
    return variant


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- load model + stats ---
    model, scheduler, blob = load_model(args.ckpt_path, device)
    mean = blob["norm_stats"]["mean"].float().cpu()                # (33,)
    std = blob["norm_stats"]["std"].float().cpu()                  # (33,)
    assert mean.shape == (TRAJ_DIM,) and std.shape == (TRAJ_DIM,)
    print(f"[ckpt] loaded {args.ckpt_path}")
    print(f"       norm_stats mean[base_pos]={mean[BASE_POS_SLICE].tolist()}  "
          f"std[base_pos]={std[BASE_POS_SLICE].tolist()}")

    # --- load ref ---
    ref = torch.load(args.ref_path)
    ref_base_pos = ref["base_position"].float()                    # (T, 3)
    ref_rpy = ref["base_pose"].float()                             # (T, 3)
    ref_joint = ref["joint_position"].float()                      # (T, 27) — 已是 27-DoF
    T_ref = ref_base_pos.shape[0]
    assert ref_joint.shape[1] == 27, \
        f"ref['joint_position'] 需为 27-DoF, 实际 {ref_joint.shape[1]}"
    print(f"[ref]  T={T_ref}  fps={args.fps}")

    # --- inpainting 模板 + mask ---
    x_known_norm = build_known_template(ref_joint, ref_rpy, mean, std)     # (T, 33)
    mask_known = torch.zeros(TRAJ_DIM, dtype=torch.bool)
    mask_known[JOINT_SLICE] = True
    if args.inpaint_rpy:
        mask_known[BASE_RPY_SLICE] = True
    print(f"[inpaint] known channels: joint_27 = True, base_rpy = {args.inpaint_rpy}, "
          f"base_pos = False (free)")
    print(f"[post]    freeze_rpy = {args.freeze_rpy} (第一版默认 True)")

    # --- 条件采样范围 (原始 → 归一化) ---
    rng_raw = parse_delta_p_range(args.delta_p_range)                      # (2, 3) 原始空间
    # Δp 条件是归一化空间 base_pos 位移 → 归一化系数 = 1 / std[base_pos]
    base_pos_std = std[BASE_POS_SLICE]
    rng_norm = rng_raw / base_pos_std[None, :]
    print(f"[cond] Δp range raw low={rng_raw[0].tolist()} high={rng_raw[1].tolist()}")
    print(f"       Δp range norm low={rng_norm[0].tolist()} high={rng_norm[1].tolist()}")

    # --- 生成 + 过滤循环 ---
    n_target = args.num_variants
    kept_variants: list[dict] = []
    kept_cond: list[torch.Tensor] = []
    inpaint_err_log: list[float] = []

    for rd in range(args.max_rounds):
        n_gen = int(n_target * args.overgenerate_ratio) - len(kept_variants)
        if n_gen <= 0:
            break
        conds_norm = sample_conditions(n_gen, rng_norm)                    # (n_gen, 3) 归一化空间
        x_norm = sample_inpainting(
            model, scheduler, conds_norm, x_known_norm, mask_known,
            args.num_inference_steps, device,
        ).cpu()                                                            # (n_gen, T, 33)

        # 反归一化
        x_denorm = x_norm * std[None, None, :] + mean[None, None, :]       # (n_gen, T, 33)
        gen_joint = x_denorm[:, :, JOINT_SLICE]                            # (n_gen, T, 27)
        gen_base_pos = x_denorm[:, :, BASE_POS_SLICE]                      # (n_gen, T, 3)
        gen_base_rpy = x_denorm[:, :, BASE_RPY_SLICE]                      # (n_gen, T, 3)

        # World frame 映射: AMASS retargeted 数据的 base_z 基准 ≈ 0, 和 G1 ref 绝对高度 (~0.75m) 错位.
        # 每条 variant 起点归零再加 ref 起点, 把轨迹平移到 ref 所在 world frame.
        # 这也是 MVP 做法 (generate_diffusion_trajectories.py:174-175).
        gen_base_pos = gen_base_pos - gen_base_pos[:, 0:1, :] + ref_base_pos[0:1, :].unsqueeze(0)

        # 诊断: inpaint 后 joint 应 ≈ ref (硬投影前)
        inpaint_err = (gen_joint - ref_joint.unsqueeze(0)).norm(dim=-1).mean().item()
        inpaint_err_log.append(inpaint_err)
        z_min = gen_base_pos[:, :, 2].min().item()
        z_max = gen_base_pos[:, :, 2].max().item()
        print(f"[round {rd}] n_gen={n_gen}  inpaint_err(joint)={inpaint_err:.4f}  "
              f"base_z range=[{z_min:.3f}, {z_max:.3f}]m")

        # 诊断: 分别检查 speed/accel/height, 定位是哪一项卡住
        dt = 1.0 / args.fps
        v_diag = (gen_base_pos[:, 1:] - gen_base_pos[:, :-1]) / dt              # (n_gen, T-1, 3)
        a_diag = (v_diag[:, 1:] - v_diag[:, :-1]) / dt                          # (n_gen, T-2, 3)
        speed_max = v_diag.norm(dim=-1).max(dim=-1).values                      # (n_gen,)
        accel_max = a_diag.norm(dim=-1).max(dim=-1).values                      # (n_gen,)
        ref_v = (ref_base_pos[1:] - ref_base_pos[:-1]) / dt
        ref_a = (ref_v[1:] - ref_v[:-1]) / dt
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
            new_base_pos = gen_base_pos[k]                                 # (T, 3)
            new_base_rpy = ref_rpy.clone() if args.freeze_rpy else gen_base_rpy[k].clone()
            v = build_variant(ref, new_base_pos, new_base_rpy, args.fps)
            # 式 3 断言
            assert torch.equal(v["joint_position"], ref["joint_position"]), \
                "joint_position 未保持不变!"
            if args.freeze_rpy:
                assert torch.equal(v["base_pose"], ref["base_pose"]), \
                    "freeze_rpy=True 但 base_pose 与 ref 不一致!"
            kept_variants.append(v)
            kept_cond.append(conds_norm[k])
        if len(kept_variants) >= n_target:
            break

    assert len(kept_variants) > 0, \
        "no variants survived; 放宽 --max_speed / --max_accel 或检查 norm_stats"
    kept_variants = kept_variants[:n_target]
    kept_cond = kept_cond[:n_target]
    print(f"[final] kept {len(kept_variants)} variants")

    # --- 保存 ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "variants": kept_variants,
        "meta": dict(
            ckpt_path=args.ckpt_path, ref_path=args.ref_path,
            delta_p_range=args.delta_p_range,
            conds_norm=torch.stack(kept_cond, dim=0),
            max_speed=args.max_speed, max_accel=args.max_accel,
            min_height=args.min_height, fps=args.fps, T=T_ref,
            inpaint_rpy=args.inpaint_rpy, freeze_rpy=args.freeze_rpy,
            inpaint_err_log=inpaint_err_log,
            version="full_v1",   # 第一版 = rpy 冻结 + 无 FK 重算
        ),
    }, args.output)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
