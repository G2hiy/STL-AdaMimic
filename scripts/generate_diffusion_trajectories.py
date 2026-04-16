"""生成 D_ref^diff (创新点① MVP C3).

输入:
    --ckpt_path          root_mdm 训练产物 (含 model_config / scheduler_config)
    --ref_path           基准 data.pt (提供 joint_position / link_orientation 等不变量)
    --delta_p_range      条件 Δp 采样范围, 格式 "x=-0.3,0.3 y=-0.3,0.3 z=-0.1,0.1"
    --num_variants (M)   最终需要保留的 variant 数; 过滤后不足则重采样补足
    --output             .pt 输出路径

输出格式 (与 motionlib.load_diffusion_variants 约定一致):
    {
      "variants": [ dict(base_position, base_pose, base_velocity, base_angular_velocity,
                          joint_position, joint_velocity,
                          link_position, link_orientation,
                          link_velocity, link_angular_velocity), ... ],
      "meta": {...}
    }

注: data.pt 不含 framerate 字段, 帧率统一由 cfg.dataset.frame_rate (base.yaml 默认 30)
    提供, 本脚本通过 --fps 参数传入用于 link_velocity / base_velocity 的有限差分重算.

约束 (论文式 3):
    - joint_position       = ref["joint_position"]           (完全拷贝, 关节角不变)
    - joint_velocity       = ref["joint_velocity"]           (同上)
    - base_pose            = ref["base_pose"]                (冻结根姿态, D2)
    - base_angular_velocity= ref["base_angular_velocity"]    (同上)
    - link_orientation     = ref["link_orientation"]         (关节+根姿态不变 → 朝向不变)
    - link_angular_velocity= ref["link_angular_velocity"]
    - link_position[t, k]  = ref_link_position[t, k] + (base_pos_new[t] - ref_base_pos[t])
    - link_velocity        = 有限差分于新 link_position (fps 来自 --fps)
    - base_velocity        = 有限差分于新 base_position   (fps 来自 --fps)
"""

import argparse
import os
import torch
from diffusers import DDPMScheduler

from legged_gym.diffusion.root_mdm import RootDiffusionModel
from legged_gym.diffusion.filter import kinematic_filter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--ref_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_variants", type=int, default=32)
    p.add_argument("--delta_p_range", default="x=-0.3,0.3 y=-0.3,0.3 z=-0.1,0.1")
    p.add_argument("--overgenerate_ratio", type=float, default=4.0,
                   help="生成 num_variants * ratio 条再做过滤")
    p.add_argument("--max_rounds", type=int, default=5, help="过滤不足时的重采样上限")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="DDIM-like 快速采样; DDPMScheduler 会自动 set_timesteps")
    p.add_argument("--max_speed", type=float, default=4.0)
    p.add_argument("--max_accel", type=float, default=30.0)
    p.add_argument("--min_height", type=float, default=0.05)
    p.add_argument("--fps", type=float, default=30.0,
                   help="data.pt 没有 framerate 字段；取自 cfg.dataset.frame_rate (base.yaml 默认 30)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_delta_p_range(s: str):
    """'x=-0.3,0.3 y=-0.3,0.3 z=-0.1,0.1' → Tensor[2, 3] (low/high)."""
    low, high = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    idx = {"x": 0, "y": 1, "z": 2}
    for tok in s.split():
        key, vals = tok.split("=")
        a, b = vals.split(",")
        low[idx[key]] = float(a)
        high[idx[key]] = float(b)
    return torch.tensor([low, high], dtype=torch.float32)


def sample_conditions(n: int, rng: torch.Tensor):
    u = torch.rand(n, 3)
    return rng[0] + u * (rng[1] - rng[0])


def load_model(ckpt_path: str, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device)
    model = RootDiffusionModel(**blob["model_config"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    scheduler = DDPMScheduler.from_config(blob["scheduler_config"])
    return model, scheduler, blob


@torch.no_grad()
def sample_trajectories(
    model: RootDiffusionModel,
    scheduler: DDPMScheduler,
    conds: torch.Tensor,            # (M, 3)
    seq_len: int,
    num_inference_steps: int,
    device: torch.device,
):
    M = conds.shape[0]
    x = torch.randn(M, seq_len, 3, device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in scheduler.timesteps:
        ts = torch.full((M,), int(t.item()), dtype=torch.long, device=device)
        eps = model(x, ts, conds.to(device))
        x = scheduler.step(eps, t, x).prev_sample
    return x  # (M, T, 3)


def build_variant(ref: dict, new_base_pos: torch.Tensor, fps: float) -> dict:
    """根据论文式 3 约束, 由新 base_position 派生其他字段.

    ref 字段（data.pt 实际 schema）:
      base_position         (T, 3)
      base_pose             (T, 3)   RPY
      base_velocity         (T, 3)   由 base_position 差分得到
      base_angular_velocity (T, 3)
      joint_position        (T, N_src)
      joint_velocity        (T, N_src)
      link_position         (T, N_bodies, 3)
      link_orientation      (T, N_bodies, 3)  RPY
      link_velocity         (T, N_bodies, 3)
      link_angular_velocity (T, N_bodies, 3)
    """
    ref_base = ref["base_position"].float()
    delta = new_base_pos - ref_base                           # (T, 3)
    new_link_pos = ref["link_position"].float() + delta[:, None, :]
    # 有限差分重算 link_velocity 和 base_velocity (末尾填最后一帧保持形状)
    v_link = (new_link_pos[1:] - new_link_pos[:-1]) * fps
    new_link_vel = torch.cat([v_link, v_link[-1:]], dim=0)
    v_base = (new_base_pos[1:] - new_base_pos[:-1]) * fps
    new_base_vel = torch.cat([v_base, v_base[-1:]], dim=0)
    variant = {
        "base_position":         new_base_pos.clone(),
        "base_pose":             ref["base_pose"].clone(),
        "base_velocity":         new_base_vel,
        "base_angular_velocity": ref["base_angular_velocity"].clone(),
        "joint_position":        ref["joint_position"].clone(),
        "joint_velocity":        ref["joint_velocity"].clone(),
        "link_position":         new_link_pos,
        "link_orientation":      ref["link_orientation"].clone(),
        "link_velocity":         new_link_vel,
        "link_angular_velocity": ref["link_angular_velocity"].clone(),
    }
    return variant


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ref = torch.load(args.ref_path)
    ref_base = ref["base_position"].float()
    T_ref = ref_base.shape[0]
    fps_ref = float(args.fps)
    print(f"[ref] T={T_ref}  fps={fps_ref} (from --fps)")

    model, scheduler, _ = load_model(args.ckpt_path, device)
    rng = parse_delta_p_range(args.delta_p_range)
    print(f"[cond] Δp range low={rng[0].tolist()}  high={rng[1].tolist()}")

    kept_root: list[torch.Tensor] = []
    kept_cond: list[torch.Tensor] = []
    n_target = args.num_variants
    for rd in range(args.max_rounds):
        n_gen = int(n_target * args.overgenerate_ratio) - len(kept_root)
        if n_gen <= 0:
            break
        conds = sample_conditions(n_gen, rng)
        traj = sample_trajectories(model, scheduler, conds, T_ref,
                                   args.num_inference_steps, device).cpu()

        # 模型学的是起点归零的相对轨迹; 这里把 ref 起点作为绝对基准叠加
        traj_abs = traj + ref_base[:1]

        mask = kinematic_filter(
            traj_abs, fps=fps_ref,
            max_speed=args.max_speed, max_accel=args.max_accel,
            min_height=args.min_height,
        )
        n_pass = int(mask.sum().item())
        print(f"[round {rd}] generated={n_gen} passed={n_pass}/{n_gen}")
        if n_pass == 0:
            continue
        kept_root.extend(list(traj_abs[mask]))
        kept_cond.extend(list(conds[mask]))
        if len(kept_root) >= n_target:
            break

    assert len(kept_root) > 0, "no variants survived kinematic filter; relax thresholds"
    kept_root = kept_root[:n_target]
    kept_cond = kept_cond[:n_target]
    print(f"[final] kept {len(kept_root)} variants")

    variants = []
    for i, (new_base, cond) in enumerate(zip(kept_root, kept_cond)):
        v = build_variant(ref, new_base, fps_ref)
        # 关键断言: 式 3
        assert torch.equal(v["joint_position"], ref["joint_position"]), "joint_position 未保持不变!"
        variants.append(v)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "variants": variants,
        "meta": dict(
            ckpt_path=args.ckpt_path, ref_path=args.ref_path,
            delta_p_range=args.delta_p_range,
            conds=torch.stack(kept_cond, dim=0),
            max_speed=args.max_speed, max_accel=args.max_accel,
            min_height=args.min_height, fps=fps_ref, T=T_ref,
        ),
    }, args.output)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
