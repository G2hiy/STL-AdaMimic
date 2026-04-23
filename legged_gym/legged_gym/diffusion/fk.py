"""G1 floating-base Forward Kinematics (创新点① 完整版).

把 (base_pos, base_rpy, joint_27) 映射回世界系下的 link_position / link_orientation,
替代 generate_diffusion_full_trajectories 里"link_pose ≡ ref + base 平移"的保守近似.

Convention:
    - rpy: scipy 'xyz' intrinsic (与 prepare_diffusion_full_training_set.py:82
      quat_xyzw_to_rpy 对齐, S0 已实证).
    - R = Rx(r) @ Ry(p) @ Rz(y).

依赖: pytorch_kinematics (pip install pytorch_kinematics).

CLI 自洽测试:
    python -m legged_gym.diffusion.fk \\
        --ref_path legged_gym/resources/dataset/g1_dof27_data/far_jump/output/data.pt
"""

import argparse
import os
from typing import Optional

import torch

try:
    import pytorch_kinematics as pk
except ImportError:
    pk = None


_THIS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.normpath(
    os.path.join(_THIS, "..", "..", "resources", "robots", "g1", "urdf", "g1_27dof.urdf")
)

# G1 27-DoF joint 顺序 — 与 configs/robot/g1_dof27.yaml 的 all_default_joint_angles
# 以及 prepare_diffusion_full_training_set.py (HuggingFace joint_29 删 waist_roll/pitch)
# 保持一致.
JOINT_NAMES_27 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def intrinsic_xyz_to_rotmat(rpy: torch.Tensor) -> torch.Tensor:
    """scipy 'xyz' intrinsic Euler → rotation matrix. R = Rx(r) @ Ry(p) @ Rz(y)."""
    r = rpy[..., 0]
    p = rpy[..., 1]
    y = rpy[..., 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)

    m = torch.empty(*rpy.shape[:-1], 3, 3, dtype=rpy.dtype, device=rpy.device)
    m[..., 0, 0] = cp * cy
    m[..., 0, 1] = -cp * sy
    m[..., 0, 2] = sp
    m[..., 1, 0] = sr * sp * cy + cr * sy
    m[..., 1, 1] = -sr * sp * sy + cr * cy
    m[..., 1, 2] = -sr * cp
    m[..., 2, 0] = -cr * sp * cy + sr * sy
    m[..., 2, 1] = cr * sp * sy + sr * cy
    m[..., 2, 2] = cr * cp
    return m


def rotmat_to_intrinsic_xyz(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → scipy 'xyz' intrinsic Euler (r, p, y)."""
    sp = m[..., 0, 2].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    p = torch.asin(sp)
    cp = torch.cos(p)
    r_main = torch.atan2(-m[..., 1, 2], m[..., 2, 2])
    y_main = torch.atan2(-m[..., 0, 1], m[..., 0, 0])
    # 奇异 (cp≈0): 固定 r=0, 把全部绕 z 吃到 y
    singular = cp.abs() < 1e-6
    if singular.any():
        r_alt = torch.zeros_like(r_main)
        y_alt = torch.atan2(m[..., 1, 0], m[..., 1, 1])
        r = torch.where(singular, r_alt, r_main)
        y = torch.where(singular, y_alt, y_main)
    else:
        r = r_main
        y = y_main
    return torch.stack([r, p, y], dim=-1)


class G1FK:
    """G1 floating-base FK.

    forward(base_pos, base_rpy, joint_27)
        -> link_pos (B, T, L, 3), link_rpy (B, T, L, 3), link_names: list[str]

    target_link_names=None 时输出 chain.get_link_names() 全量 (含 fixed-joint 的 child
    link, 与 Isaac Gym collapse_fixed_joints=false 默认对齐).
    """

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF,
        joint_names: Optional[list] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if pk is None:
            raise ImportError(
                "pytorch_kinematics not installed. Install with `pip install pytorch_kinematics`."
            )
        with open(urdf_path, "rb") as f:
            chain = pk.build_chain_from_urdf(f.read())
        self.chain = chain.to(device=device, dtype=dtype)
        self.joint_names = list(joint_names) if joint_names is not None else list(JOINT_NAMES_27)
        self.pk_joint_names = list(self.chain.get_joint_parameter_names())
        missing = [n for n in self.joint_names if n not in self.pk_joint_names]
        assert not missing, (
            f"joint names not found in URDF chain: {missing}\n"
            f"URDF actuated joints: {self.pk_joint_names}"
        )
        # pk 可能额外要求某些 joint 不在 joint_names_27 (比如 waist_roll/pitch 已锁).
        # 这些没传入的 joint 填 0.
        self.link_names = list(self.chain.get_link_names())
        self.device = torch.device(device)
        self.dtype = dtype

    def __call__(
        self,
        base_pos: torch.Tensor,        # (B, T, 3)
        base_rpy: torch.Tensor,        # (B, T, 3)
        joint_vals: torch.Tensor,      # (B, T, n_joints) 按 self.joint_names 顺序
        target_link_names: Optional[list] = None,
    ):
        B, T, _ = base_pos.shape
        BT = B * T

        bp = base_pos.reshape(BT, 3).to(self.device, self.dtype)
        br = base_rpy.reshape(BT, 3).to(self.device, self.dtype)
        jv = joint_vals.reshape(BT, -1).to(self.device, self.dtype)
        assert jv.shape[1] == len(self.joint_names), (
            f"joint_vals last dim {jv.shape[1]} != len(joint_names) {len(self.joint_names)}"
        )

        # Reorder / zero-pad to pk 的 joint 顺序
        idx_in_ours = {n: i for i, n in enumerate(self.joint_names)}
        cols = []
        for n in self.pk_joint_names:
            if n in idx_in_ours:
                cols.append(jv[:, idx_in_ours[n]])
            else:
                cols.append(torch.zeros(BT, device=self.device, dtype=self.dtype))
        jv_pk = torch.stack(cols, dim=-1)  # (BT, n_pk_joints)

        fk_out = self.chain.forward_kinematics(jv_pk)
        # fk_out: {link_name: Transform3d} (pk >=0.5) 或 dict of mat (老版)

        R_wb = intrinsic_xyz_to_rotmat(br)    # (BT, 3, 3) world ← base
        links = list(target_link_names) if target_link_names is not None else list(self.link_names)

        pos_list, rot_list = [], []
        for name in links:
            assert name in fk_out, f"link '{name}' not in chain; available: {self.link_names}"
            T_bl = fk_out[name].get_matrix()  # (BT, 4, 4) base-frame transform of link
            R_bl = T_bl[:, :3, :3]
            t_bl = T_bl[:, :3, 3]
            R_wl = R_wb @ R_bl
            t_wl = (R_wb @ t_bl.unsqueeze(-1)).squeeze(-1) + bp
            pos_list.append(t_wl)
            rot_list.append(R_wl)

        link_pos = torch.stack(pos_list, dim=1)   # (BT, L, 3)
        link_R = torch.stack(rot_list, dim=1)     # (BT, L, 3, 3)
        link_rpy = rotmat_to_intrinsic_xyz(link_R)  # (BT, L, 3)

        link_pos = link_pos.reshape(B, T, len(links), 3)
        link_rpy = link_rpy.reshape(B, T, len(links), 3)
        return link_pos, link_rpy, links


def _self_consistency_main():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf_path", default=DEFAULT_URDF)
    p.add_argument("--ref_path", required=True)
    p.add_argument("--max_pos_err", type=float, default=1e-2, help="m")
    p.add_argument("--max_rpy_err", type=float, default=1e-2, help="rad")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--dump_reorder", default=None,
        help="若 link 顺序与 ref 不一致, 尝试首帧最近邻匹配并保存 perm 到此路径."
    )
    args = p.parse_args()

    ref = torch.load(args.ref_path)
    base_pos = ref["base_position"].float()
    base_rpy = ref["base_pose"].float()
    joint_27 = ref["joint_position"].float()
    T = base_pos.shape[0]
    assert joint_27.shape[1] == 27, f"ref joint_position shape {joint_27.shape}"

    fk = G1FK(urdf_path=args.urdf_path, device=args.device)
    print(f"[fk] joint_names ({len(fk.joint_names)}): {fk.joint_names}")
    print(f"[fk] pk_joint_names ({len(fk.pk_joint_names)}): {fk.pk_joint_names}")
    print(f"[fk] link_names ({len(fk.link_names)}): {fk.link_names}")

    link_pos, link_rpy, names = fk(
        base_pos.unsqueeze(0), base_rpy.unsqueeze(0), joint_27.unsqueeze(0)
    )
    link_pos = link_pos.squeeze(0)  # (T, L, 3)
    link_rpy = link_rpy.squeeze(0)  # (T, L, 3)

    ref_link_pos = ref["link_position"].float()
    ref_link_rpy = ref["link_orientation"].float()
    print(f"[ref] link_position shape={tuple(ref_link_pos.shape)}  "
          f"link_orientation shape={tuple(ref_link_rpy.shape)}")
    print(f"[fk ] link_pos      shape={tuple(link_pos.shape)}")

    L, N_ref = len(names), ref_link_pos.shape[1]

    if L == N_ref:
        diff_pos = (link_pos - ref_link_pos).norm(dim=-1)   # (T, L)
        diff_rpy = (link_rpy - ref_link_rpy).norm(dim=-1)
        per_link_pos = diff_pos.mean(dim=0)                 # (L,)
        per_link_rpy = diff_rpy.mean(dim=0)
        print(f"[direct] max_pos_err={diff_pos.max().item():.4f}m  "
              f"mean={diff_pos.mean().item():.4f}m")
        print(f"         max_rpy_err={diff_rpy.max().item():.4f}rad "
              f"mean={diff_rpy.mean().item():.4f}rad")
        if (diff_pos.max().item() <= args.max_pos_err and
                diff_rpy.max().item() <= args.max_rpy_err):
            print("[pass] direct-order FK self-consistency OK")
            return
        print("[warn] direct order failed; per-link pos err (mean):")
        for i, (n, e) in enumerate(zip(names, per_link_pos.tolist())):
            print(f"  {i:3d} {n:<40s} {e:.4f} m")

    if args.dump_reorder is None:
        print("[fail] direct order mismatch and --dump_reorder not set; abort.")
        raise SystemExit(1)

    # 首帧最近邻匹配 (ref_index → fk_index)
    print("[warn] trying greedy frame-0 permutation match …")
    f0_ref = ref_link_pos[0]              # (N_ref, 3)
    f0_fk = link_pos[0]                   # (L, 3)
    dist = torch.cdist(f0_ref, f0_fk)     # (N_ref, L)
    # 每个 ref link 选最近的 fk link
    ref_to_fk = dist.argmin(dim=1).tolist()
    unique = len(set(ref_to_fk))
    print(f"[perm] ref_to_fk ({len(ref_to_fk)} entries, {unique} unique)")
    if unique != min(L, N_ref):
        print("[fail] permutation not injective; abort")
        raise SystemExit(1)

    link_pos_perm = link_pos[:, ref_to_fk]
    link_rpy_perm = link_rpy[:, ref_to_fk]
    diff_pos = (link_pos_perm - ref_link_pos).norm(dim=-1)
    diff_rpy = (link_rpy_perm - ref_link_rpy).norm(dim=-1)
    print(f"[perm]   max_pos_err={diff_pos.max().item():.4f}m  "
          f"mean={diff_pos.mean().item():.4f}m")
    print(f"         max_rpy_err={diff_rpy.max().item():.4f}rad "
          f"mean={diff_rpy.mean().item():.4f}rad")
    if not (diff_pos.max().item() <= args.max_pos_err and
            diff_rpy.max().item() <= args.max_rpy_err):
        print("[fail] even with permutation, errors exceed threshold — check URDF / joint order")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.dump_reorder)), exist_ok=True)
    torch.save({
        "ref_to_fk": torch.tensor(ref_to_fk, dtype=torch.long),
        "fk_link_names": names,
        "ref_N_bodies": N_ref,
        "urdf_path": args.urdf_path,
    }, args.dump_reorder)
    print(f"[pass] perm saved to {args.dump_reorder}")


if __name__ == "__main__":
    _self_consistency_main()
