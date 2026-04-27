"""AMASS (G1-retargeted) → 全身扩散预训练集 (创新点① 完整版 S1).

输入源: HuggingFace fleaven/Retargeted_AMASS_for_robotics/g1/CMU
    每个 .npy shape=(T_src, 36), 列布局:
        [0:3]   = base_pos           (m, G1 坐标系)
        [3:7]   = base_quat_xyzw     (已 retargeted 到 G1 约定)
        [7:36]  = joint_29           (弧度)
    src_fps = 120

目标维度: (T, 33) = [joint_27, base_pos(3), base_rpy(3)]
    - joint_27 = np.delete(joint_29, [13, 14], axis=1)  (丢 waist_roll / waist_pitch)
    - base_rpy = scipy Rotation.from_quat(q_xyzw).as_euler('xyz')  ← S0 实证确认
    - 目标 fps = 50 (data.pt 真实帧率)

管线:
    1. 递归扫 .npy
    2. 拆列 → quat→rpy → joint_29→27 → 拼 33-dim
    3. 120→50 重采样
    4. 起点对齐 (只归零 base_pos 的 xy; 保留绝对 z 和绝对关节/rpy)
    5. 滑窗 seq_len=ref_T, stride=0.5*seq_len
    6. 过滤静止片段 (base_z range < 0.01m)
    7. 全量计算 per-channel mean/std, 归一化存盘
    8. 输出 {trajectories[N,T,33], norm_stats{mean,std}, source_ids, meta}

用法:
    python scripts/prepare_diffusion_full_training_set.py \\
        --amass_dir legged_gym/resources/dataset/amass_g1/g1/CMU \\
        --ref_path  legged_gym/resources/dataset/g1_dof27_data/high_jump/output/data.pt \\
        --output    resources/dataset/diffusion_train/amass_g1_full.pt
"""

import argparse
import glob
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation


JOINT_DROP_INDICES = [13, 14]  # waist_roll, waist_pitch (G1 29→27)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--amass_dir", required=True,
                   help="递归扫 .npy 的根目录 (e.g. .../amass_g1/g1/CMU)")
    p.add_argument("--ref_path", required=True,
                   help="基准 data.pt; 取其 T 作为 seq_len, 并透传 ref_path 到 meta")
    p.add_argument("--output", required=True)
    p.add_argument("--src_fps", type=float, default=120.0)
    p.add_argument("--target_fps", type=float, default=50.0,
                   help="data.pt 真实 50fps; 不要写 30")
    p.add_argument("--seq_len", type=int, default=-1,
                   help="-1: 跟随 ref 的 T; 否则固定长度")
    p.add_argument("--window_stride_ratio", type=float, default=0.5)
    p.add_argument("--min_z_range", type=float, default=0.01,
                   help="base_z 跨度小于该值视为静止, 丢弃 (m)")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="上限 (调试用, -1=不限)")
    p.add_argument("--dump_preview", action="store_true",
                   help="保存前 2 条样本 (反归一化) 到 .npy 供目视检查")
    return p.parse_args()


def resample_linear(x: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """线性插值重采样 (T_src, D) → (T_tgt, D)."""
    T_src = x.shape[0]
    duration = T_src / src_fps
    T_tgt = max(int(round(duration * tgt_fps)), 2)
    t_src = np.linspace(0, 1, T_src)
    t_tgt = np.linspace(0, 1, T_tgt)
    out = np.stack([np.interp(t_tgt, t_src, x[:, d]) for d in range(x.shape[1])], axis=1)
    return out.astype(np.float32)


def quat_xyzw_to_rpy(q_xyzw: np.ndarray) -> np.ndarray:
    """scipy 'xyz' extrinsic. S0 脚本已证其与 G1 euler_xyz_to_quat 互逆 (tol<1e-4).

    输入 (T, 4) xyzw; 输出 (T, 3) rpy 弧度.
    """
    norms = np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
    q_xyzw = q_xyzw / np.clip(norms, 1e-8, None)
    return Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=False).astype(np.float32)


def unwrap_rpy(rpy: np.ndarray) -> np.ndarray:
    """对 yaw 做 ±π wrap 展开, 防止滑窗跨越 ±π 时出现跳变."""
    out = rpy.copy()
    for d in range(3):
        out[:, d] = np.unwrap(out[:, d])
    return out


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    ref = torch.load(args.ref_path)
    ref_base = ref["base_position"].float()
    T_ref = ref_base.shape[0]
    seq_len = T_ref if args.seq_len <= 0 else args.seq_len
    print(f"[ref] T_ref={T_ref}  seq_len={seq_len}  target_fps={args.target_fps}")

    npy_files = sorted(glob.glob(os.path.join(args.amass_dir, "**/*.npy"), recursive=True))
    print(f"[scan] found {len(npy_files)} .npy under {args.amass_dir}")
    assert len(npy_files) > 0, "no .npy found; check --amass_dir"

    trajectories: list[np.ndarray] = []
    source_ids: list[str] = []
    n_bad_shape = 0
    n_dropped_short = 0
    n_dropped_static = 0

    for i, fp in enumerate(npy_files):
        if args.max_samples > 0 and len(trajectories) >= args.max_samples:
            break
        try:
            arr = np.load(fp)
        except Exception as e:
            print(f"  skip {fp}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] != 36 or arr.shape[0] < 10:
            n_bad_shape += 1
            continue

        arr = arr.astype(np.float64)
        base_pos = arr[:, 0:3]
        quat_xyzw = arr[:, 3:7]
        joint_29 = arr[:, 7:36]

        rpy = quat_xyzw_to_rpy(quat_xyzw)            # (T_src, 3)
        rpy = unwrap_rpy(rpy)
        joint_27 = np.delete(joint_29, JOINT_DROP_INDICES, axis=1)  # (T_src, 27)

        traj = np.concatenate([
            joint_27.astype(np.float32),             # (T_src, 27)
            base_pos.astype(np.float32),             # (T_src, 3)
            rpy.astype(np.float32),                  # (T_src, 3)
        ], axis=1)                                   # (T_src, 33)

        traj = resample_linear(traj, args.src_fps, args.target_fps)
        T_src = traj.shape[0]
        if T_src < seq_len:
            n_dropped_short += 1
            continue

        stride = max(int(seq_len * args.window_stride_ratio), 1)
        for start in range(0, T_src - seq_len + 1, stride):
            win = traj[start:start + seq_len].copy()  # (T, 33)
            z = win[:, 27 + 2]                        # base_z
            if float(z.max() - z.min()) < args.min_z_range:
                n_dropped_static += 1
                continue
            # base_pos 三维全部转成相对窗口首帧
            # 训练模型只学习 [Δx, Δy, Δz]，不学习 AMASS/仿真世界坐标的绝对高度
            base0 = win[0, 27:30].copy()
            win[:, 27:30] -= base0

            trajectories.append(win)
            source_ids.append(f"{os.path.relpath(fp, args.amass_dir)}::{start}")

        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(npy_files)}  kept={len(trajectories)}  "
                  f"bad_shape={n_bad_shape}  short={n_dropped_short}  static={n_dropped_static}")

    assert len(trajectories) > 0, "all trajectories filtered; relax --min_z_range"
    traj_np = np.stack(trajectories, axis=0)         # (N, T, 33)
    N, T, D = traj_np.shape
    print(f"[done] N={N}  T={T}  D={D}")

    # 全量 per-channel mean/std (over N*T)
    mean = traj_np.reshape(-1, D).mean(axis=0).astype(np.float32)   # (33,)
    std = traj_np.reshape(-1, D).std(axis=0).astype(np.float32)     # (33,)
    std = np.clip(std, 1e-6, None)                                  # 防除 0
    traj_norm = ((traj_np - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

    # 反归一化后 std per-dim 应约为原 std; 归一化后应约为 1
    print(f"[norm] mean[joint]   : min={mean[0:27].min():+.3f}  max={mean[0:27].max():+.3f}")
    print(f"       std[joint]    : min={std[0:27].min():.3f}   max={std[0:27].max():.3f}")
    print(f"       mean[base_pos]: {mean[27:30].tolist()}")
    print(f"       std [base_pos]: {std[27:30].tolist()}")
    print(f"       mean[base_rpy]: {mean[30:33].tolist()}")
    print(f"       std [base_rpy]: {std[30:33].tolist()}")
    check_std = traj_norm.reshape(-1, D).std(axis=0)
    print(f"[sanity] normalized std: min={check_std.min():.3f}  max={check_std.max():.3f}  "
          f"(应 ≈ 1.0)")

    blob = {
        "trajectories": torch.from_numpy(traj_norm),
        "norm_stats": {
            "mean": torch.from_numpy(mean),
            "std":  torch.from_numpy(std),
        },
        "source_ids": source_ids,
        "meta": dict(
            seq_len=T, target_fps=args.target_fps, src_fps=args.src_fps,
            joint_drop_indices=JOINT_DROP_INDICES,
            dim_layout="joint_27 | base_pos(3) | base_rpy(3)",
            quat_convention="scipy_xyz_extrinsic (verified against G1 euler_xyz_to_quat in S0)",
            ref_path=args.ref_path, ref_T=T_ref,
            n_bad_shape=n_bad_shape, n_dropped_short=n_dropped_short,
            n_dropped_static=n_dropped_static,
        ),
    }
    torch.save(blob, args.output)
    print(f"[save] {args.output}")

    if args.dump_preview:
        preview = traj_np[:2]                         # 反归一化之前 = 原值
        preview_path = args.output.replace(".pt", "_preview.npy")
        np.save(preview_path, preview)
        print(f"[preview] saved 2 samples (denormalized) to {preview_path}")


if __name__ == "__main__":
    main()
