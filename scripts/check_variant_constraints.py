"""检查 variants.pt 的物理合理性 (创新点① 完整版).

校验项:
    1. FK self-consistency: link_position / link_orientation ≈ FK(joint, base_pos, base_rpy)
    2. 速度上限: base speed ≤ max_speed, |joint_velocity| ≤ max_joint_vel
    3. 加速度上限: base accel ≤ max_accel
    4. base_z ≥ min_height

用法:
    python scripts/check_variant_constraints.py \\
        --variants_path resources/dataset/g1_dof27_data_diff/far_jump/variants_joint.pt \\
        --ref_path      legged_gym/resources/dataset/g1_dof27_data/far_jump/output/data.pt
"""

import argparse
import torch

from legged_gym.diffusion.fk import G1FK


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants_path", required=True)
    p.add_argument("--ref_path", required=True)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--max_pos_err", type=float, default=1e-3, help="m, FK self-consistency")
    p.add_argument("--max_rpy_err", type=float, default=1e-3, help="rad, FK self-consistency")
    p.add_argument("--max_speed", type=float, default=4.0, help="m/s, base")
    p.add_argument("--max_accel", type=float, default=30.0, help="m/s^2, base")
    p.add_argument("--max_joint_vel", type=float, default=20.0, help="rad/s")
    p.add_argument("--min_height", type=float, default=0.0, help="m, base_z 下限")
    p.add_argument("--urdf_path", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    blob = torch.load(args.variants_path)
    variants = blob["variants"] if isinstance(blob, dict) and "variants" in blob else blob
    ref = torch.load(args.ref_path)

    urdf_kwargs = {"urdf_path": args.urdf_path} if args.urdf_path else {}
    fk = G1FK(device=args.device, **urdf_kwargs)
    fk_link_names = blob.get("meta", {}).get("link_names") if isinstance(blob, dict) else None
    if fk_link_names is None:
        fk_link_names = list(fk.link_names)

    print(f"checking {len(variants)} variants against ref {args.ref_path}")
    print(f"[fk] using {len(fk_link_names)} links (meta={fk_link_names is not None})")

    dt = 1.0 / args.fps
    n_ok = 0
    for i, v in enumerate(variants):
        errs = []

        # shape 一致
        if v["base_position"].shape != ref["base_position"].shape:
            errs.append(
                f"base_position shape {tuple(v['base_position'].shape)} "
                f"!= ref {tuple(ref['base_position'].shape)}"
            )

        # base_z 下限
        z_min = v["base_position"][:, 2].min().item()
        if z_min < args.min_height:
            errs.append(f"base_z min={z_min:.3f} < min_height={args.min_height}")

        # base 速度 / 加速度
        bp = v["base_position"].float()
        v_base = (bp[1:] - bp[:-1]) / dt
        a_base = (v_base[1:] - v_base[:-1]) / dt
        s_max = v_base.norm(dim=-1).max().item()
        a_max = a_base.norm(dim=-1).max().item()
        if s_max > args.max_speed:
            errs.append(f"base speed max={s_max:.2f} > {args.max_speed}")
        if a_max > args.max_accel:
            errs.append(f"base accel max={a_max:.2f} > {args.max_accel}")

        # joint 速度
        jp = v["joint_position"].float()
        j_vel = (jp[1:] - jp[:-1]) / dt
        jv_max = j_vel.abs().max().item()
        if jv_max > args.max_joint_vel:
            errs.append(f"joint_vel max={jv_max:.2f} > {args.max_joint_vel}")

        # FK self-consistency
        link_pos, link_rpy, _ = fk(
            v["base_position"].float().unsqueeze(0),
            v["base_pose"].float().unsqueeze(0),
            v["joint_position"].float().unsqueeze(0),
            target_link_names=fk_link_names,
        )
        link_pos = link_pos.squeeze(0).to(v["link_position"].dtype)
        link_rpy = link_rpy.squeeze(0).to(v["link_orientation"].dtype)
        if link_pos.shape != v["link_position"].shape:
            errs.append(
                f"FK link_pos shape {tuple(link_pos.shape)} "
                f"!= variant {tuple(v['link_position'].shape)}"
            )
        else:
            dp = (link_pos - v["link_position"]).norm(dim=-1).max().item()
            dr = (link_rpy - v["link_orientation"]).norm(dim=-1).max().item()
            if dp > args.max_pos_err:
                errs.append(f"FK link_pos err={dp:.4f} > {args.max_pos_err}")
            if dr > args.max_rpy_err:
                errs.append(f"FK link_rpy err={dr:.4f} > {args.max_rpy_err}")

        if errs:
            print(f"  [FAIL] variant {i}: {errs}")
        else:
            n_ok += 1

    print(f"[result] {n_ok}/{len(variants)} variants passed")
    if n_ok < len(variants):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
