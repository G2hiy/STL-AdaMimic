"""Quat→RPY 轴约定实证 (创新点① 完整版 S0 前置阻塞项).

目的: 在写大规模 AMASS→训练集预处理之前, 实证验证 scipy.spatial.transform.Rotation
的哪种 Euler 约定与 AdaMimic 内部 `legged_gym.utils.math.euler_xyz_to_quat` 一致.

设计:
    A. 等价性测试 (强证据, 自动判定)
       - 从 AMASS 样本抽 quat (xyzw), 用 scipy 'xyz' 转 rpy
       - 用 G1 内部 euler_xyz_to_quat 把 rpy 转回 quat
       - 与原 quat 比较 (处理 q / -q 双重覆盖)
       - 若 max |Δq| < 1e-4 -> scipy 'xyz' ≡ G1 约定 -> C1 脚本使用 'xyz'

    B. AMASS 与 G1 ref 物理合理性 (弱证据, 人工辅助)
       - 打印 AMASS rpy 的统计, 对照 ref["base_pose"] 的统计
       - 量级应一致 (roll/pitch 通常小, yaw 可能覆盖 ±π)

    C. 可视化 (可选, 人工目视)
       - 画 AMASS rpy 与 ref rpy 时间曲线, 保存 PNG

用法:
    python scripts/verify_amass_g1_rpy_convention.py \
        --amass_sample legged_gym/resources/dataset/amass_g1/g1/CMU/01/01_01_poses_120_jpos.npy \
        --ref_path     legged_gym/resources/dataset/g1_dof27_data/high_jump/output/data.pt
"""

import argparse
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation


G1_QUAT_ORDER = "xyzw"  # legged_gym.utils.math: euler_xyz_to_quat 返回 qx,qy,qz,qw


def g1_euler_xyz_to_quat(rpy_np: np.ndarray) -> np.ndarray:
    """复刻 legged_gym.utils.math.euler_xyz_to_quat, 返回 (T, 4) qx,qy,qz,qw.

    来源: legged_gym/legged_gym/utils/math.py:247
    """
    roll, pitch, yaw = rpy_np[..., 0], rpy_np[..., 1], rpy_np[..., 2]
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    q = np.stack([qx, qy, qz, qw], axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def quat_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """双重覆盖距离: min(||q1-q2||, ||q1+q2||). Shape: (..., 4) -> (...)."""
    d_pos = np.linalg.norm(q1 - q2, axis=-1)
    d_neg = np.linalg.norm(q1 + q2, axis=-1)
    return np.minimum(d_pos, d_neg)


def load_amass_sample(path: str) -> dict:
    arr = np.load(path)
    assert arr.ndim == 2 and arr.shape[1] == 36, \
        f"expected (T, 36) retargeted AMASS, got {arr.shape}"
    return {
        "base_pos":  arr[:, 0:3].astype(np.float64),
        "quat_xyzw": arr[:, 3:7].astype(np.float64),
        "joint_29":  arr[:, 7:36].astype(np.float64),
        "T": arr.shape[0],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--amass_sample", required=True, help="单个 AMASS .npy 路径 (36 列)")
    p.add_argument("--ref_path", required=True, help="data.pt 路径, 用于读 base_pose")
    p.add_argument("--out_png", default="verify_rpy.png", help="时间曲线对照图")
    p.add_argument("--tol", type=float, default=1e-4, help="等价性判定阈值")
    return p.parse_args()


def main():
    args = parse_args()

    # --- 加载 ---
    print(f"[amass] loading {args.amass_sample}")
    amass = load_amass_sample(args.amass_sample)
    q_xyzw = amass["quat_xyzw"]
    T = amass["T"]
    q_norms = np.linalg.norm(q_xyzw, axis=-1)
    print(f"        T={T}  quat_norm: mean={q_norms.mean():.4f}  "
          f"min={q_norms.min():.4f}  max={q_norms.max():.4f}")

    print(f"[ref]   loading {args.ref_path}")
    ref = torch.load(args.ref_path)
    ref_rpy = ref["base_pose"].float().numpy() if torch.is_tensor(ref["base_pose"]) \
        else np.asarray(ref["base_pose"], dtype=np.float64)
    print(f"        T_ref={ref_rpy.shape[0]}  keys={list(ref.keys())}")

    # --- A. 等价性测试 ---
    print("\n=== A. Scipy 'xyz' vs G1 euler_xyz_to_quat 等价性 ===")
    candidates = [
        ("xyz (extrinsic, 小写)", lambda q: Rotation.from_quat(q).as_euler("xyz", degrees=False)),
        ("XYZ (intrinsic, 大写)", lambda q: Rotation.from_quat(q).as_euler("XYZ", degrees=False)),
        ("zyx reversed",           lambda q: Rotation.from_quat(q).as_euler("zyx", degrees=False)[..., ::-1]),
        ("ZYX reversed",           lambda q: Rotation.from_quat(q).as_euler("ZYX", degrees=False)[..., ::-1]),
    ]
    best = None
    for name, fn in candidates:
        try:
            rpy = fn(q_xyzw)                             # (T, 3)
            q_back = g1_euler_xyz_to_quat(rpy)           # (T, 4) 走 G1 convention
            d = quat_distance(q_xyzw, q_back)
            max_d, mean_d = d.max(), d.mean()
            status = "PASS" if max_d < args.tol else "FAIL"
            print(f"  [{status}] {name:30s}  max|Δq|={max_d:.2e}  mean|Δq|={mean_d:.2e}")
            if best is None or max_d < best[2]:
                best = (name, fn, max_d, rpy)
        except Exception as e:
            print(f"  [ERR ] {name:30s}  {e}")

    assert best is not None and best[2] < args.tol, \
        f"NO candidate passed tol={args.tol}; 检查 quat 顺序 (xyzw vs wxyz) 或符号翻转"
    print(f"\n  >>> 推荐使用: {best[0]}  (max|Δq|={best[2]:.2e})")

    # --- B. AMASS 与 ref RPY 物理合理性 ---
    print("\n=== B. AMASS rpy 与 ref base_pose 统计对照 ===")
    amass_rpy = best[3]

    def stats(name, arr):
        for i, axis in enumerate(["roll", "pitch", "yaw"]):
            col = arr[:, i]
            print(f"  {name:12s}/{axis}: mean={col.mean():+.3f}  std={col.std():.3f}  "
                  f"min={col.min():+.3f}  max={col.max():+.3f}")

    stats("AMASS", amass_rpy)
    stats("ref (G1)", ref_rpy)

    print("\n  [期望] yaw 范围可能覆盖 ±π (全向运动); roll/pitch 通常在 ±π/2 内;")
    print("         若 AMASS 某轴 |max|>3 且 ref 同轴 |max|<1, 可能是 AMASS quat 归一化问题,")
    print("         或源数据起始姿态定义差异 (AMASS 站立 vs G1 站立 可能不同基准).")

    # --- C. 可视化 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        for i, (axis, ax) in enumerate(zip(["roll", "pitch", "yaw"], axes)):
            ax.plot(amass_rpy[:, i], label=f"AMASS  {axis}", lw=1.0)
            ax.plot(ref_rpy[:, i], label=f"ref    {axis}", lw=1.0, alpha=0.8)
            ax.set_ylabel(f"{axis} (rad)")
            ax.axhline(0, color="k", lw=0.3)
            ax.axhline(np.pi, color="r", lw=0.3, ls="--")
            ax.axhline(-np.pi, color="r", lw=0.3, ls="--")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("frame")
        fig.suptitle(f"RPY 时间曲线: AMASS ({T} frames) vs ref ({ref_rpy.shape[0]} frames)")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=120)
        print(f"\n[save] {args.out_png}")
    except ImportError:
        print("\n[warn] matplotlib 未装, 跳过可视化; pip install matplotlib 可启用")


if __name__ == "__main__":
    main()
