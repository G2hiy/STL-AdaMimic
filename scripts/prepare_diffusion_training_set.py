"""AMASS → 根轨迹预训练集 (创新点① MVP C1).

管线:
    1. 递归收集 --amass_dir 下所有 .npz
    2. 取 trans 字段 [T_src, 3] + mocap_framerate
    3. 坐标轴对齐: AMASS Y-up → G1 Z-up (可配置 axis_perm / axis_sign)
    4. 尺度对齐: 以 --ref_path (e.g. high_jump data.pt) 的 base_position 量级为基准
    5. framerate 对齐: 线性重采样到 --target_fps
    6. 滑窗切分到固定长度 T = --seq_len
    7. 过滤: (z_max - z_min) < --min_z_range 的静止片段丢弃
    8. 输出: {trajectories: Tensor[N, T, 3], source_ids: List[str], meta: dict}

用法:
    python scripts/prepare_diffusion_training_set.py \\
        --amass_dir resources/dataset/amass_raw/CMU \\
        --ref_path  legged_gym/resources/dataset/g1_dof27_data/high_jump/output/data.pt \\
        --output    resources/dataset/diffusion_train/amass_root.pt \\
        --seq_len   -1   # -1 表示按 ref 的 T 自动取
"""

import argparse
import glob
import os
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--amass_dir", required=True, help="递归扫描 .npz 的根目录")
    p.add_argument("--ref_path", required=True, help="基准 data.pt（取 base_position 做尺度/长度对齐）")
    p.add_argument("--output", required=True)
    p.add_argument("--target_fps", type=float, default=30.0)
    p.add_argument("--seq_len", type=int, default=-1, help="-1: 跟随 ref 的 T；否则固定长度")
    p.add_argument("--window_stride_ratio", type=float, default=0.5,
                   help="滑窗步长 = seq_len * ratio")
    p.add_argument("--min_z_range", type=float, default=0.01,
                   help="z 跨度小于该值视为静止片段, 丢弃 (m)")
    p.add_argument("--axis_perm", default="0,2,1",
                   help="逗号分隔三个 index; 默认 AMASS(x,y,z)→G1(x,z,y)")
    p.add_argument("--axis_sign", default="1,1,1",
                   help="逗号分隔三个 sign; 与 axis_perm 配合翻转方向")
    p.add_argument("--scale", type=float, default=-1.0,
                   help="手动缩放; <=0 表示用 ref 身高/SMPL 身高自动估计 (默认 0.85)")
    p.add_argument("--max_samples", type=int, default=-1, help="上限 (调试用)")
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


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    ref = torch.load(args.ref_path)
    ref_base = ref["base_position"].float()  # (T_ref, 3)
    T_ref = ref_base.shape[0]
    seq_len = T_ref if args.seq_len <= 0 else args.seq_len
    z_range_ref = float((ref_base[:, 2].max() - ref_base[:, 2].min()).item())
    xy_std_ref = float(ref_base[:, :2].std().item())
    print(f"[ref] T={T_ref}  z_range={z_range_ref:.3f}m  xy_std={xy_std_ref:.3f}m")

    axis_perm = [int(x) for x in args.axis_perm.split(",")]
    axis_sign = np.array([float(x) for x in args.axis_sign.split(",")], dtype=np.float32)
    assert len(axis_perm) == 3 and len(axis_sign) == 3

    scale = args.scale if args.scale > 0 else 0.85  # G1 身高 ~1.3m, SMPL 平均 ~1.7m
    print(f"[align] axis_perm={axis_perm}  axis_sign={axis_sign.tolist()}  scale={scale}")

    npz_files = glob.glob(os.path.join(args.amass_dir, "**/*.npz"), recursive=True)
    print(f"[scan] found {len(npz_files)} .npz under {args.amass_dir}")
    assert len(npz_files) > 0, "no AMASS files found; check --amass_dir"

    trajectories, source_ids = [], []
    n_dropped_static, n_dropped_short = 0, 0

    for i, fp in enumerate(npz_files):
        if args.max_samples > 0 and len(trajectories) >= args.max_samples:
            break
        try:
            data = np.load(fp)
        except Exception as e:
            print(f"  skip {fp}: {e}")
            continue
        if "trans" not in data:
            continue
        trans = np.asarray(data["trans"], dtype=np.float32)  # (T_src, 3)
        if trans.ndim != 2 or trans.shape[1] != 3 or trans.shape[0] < 10:
            n_dropped_short += 1
            continue

        src_fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else 120.0
        trans = resample_linear(trans, src_fps, args.target_fps)

        # axis align + scale
        trans = trans[:, axis_perm] * axis_sign[None, :]
        trans = trans * scale

        # 起点归零 (与 motionlib.py:160 的 base_pos 起点归零策略一致)
        trans = trans - trans[:1]

        stride = max(int(seq_len * args.window_stride_ratio), 1)
        T_src = trans.shape[0]
        if T_src < seq_len:
            n_dropped_short += 1
            continue
        for start in range(0, T_src - seq_len + 1, stride):
            win = trans[start:start + seq_len]
            z_range = win[:, 2].max() - win[:, 2].min()
            if z_range < args.min_z_range:
                n_dropped_static += 1
                continue
            # 窗内再次起点归零, 让模型学习相对位移形状
            win = win - win[:1]
            trajectories.append(torch.from_numpy(win.copy()))
            source_ids.append(f"{os.path.relpath(fp, args.amass_dir)}::{start}")

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(npz_files)}  kept {len(trajectories)}  "
                  f"static_dropped {n_dropped_static}  short_dropped {n_dropped_short}")

    assert len(trajectories) > 0, "all trajectories filtered out; relax --min_z_range"
    trajectories = torch.stack(trajectories, dim=0)  # (N, T, 3)
    print(f"[done] N={trajectories.shape[0]}  shape={tuple(trajectories.shape)}")
    print(f"       xy_std={trajectories[:, :, :2].std().item():.3f}m  "
          f"z_std={trajectories[:, :, 2].std().item():.3f}m")

    torch.save({
        "trajectories": trajectories,
        "source_ids": source_ids,
        "meta": dict(
            seq_len=seq_len, target_fps=args.target_fps,
            axis_perm=axis_perm, axis_sign=axis_sign.tolist(),
            scale=scale, min_z_range=args.min_z_range,
            ref_path=args.ref_path, ref_T=T_ref,
        ),
    }, args.output)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
