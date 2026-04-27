"""检查 variants.pt 的物理合理性 (创新点① root-only).

校验项 (基于 Eq.3 root-only 约束):
    1. Eq.3 不变量: joint_position / base_pose 与 ref 严格相等
    2. 相对几何不变性: link_position - base_position 与 ref 一致 (容差 1e-5 m)
    3. 速度上限: base speed ≤ max_speed; |joint_velocity| ≤ max_joint_vel (joint_velocity 复制自 ref)
    4. 加速度上限: base accel ≤ max_accel
    5. base_z ≥ min_height

用法:
    python scripts/check_variant_constraints.py \\
        --variants_path resources/dataset/g1_dof27_data_diff/far_jump/variants.pt \\
        --ref_path      legged_gym/resources/dataset/g1_dof27_data/far_jump/output/data.pt
"""

import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants_path", required=True)
    p.add_argument("--ref_path", required=True)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--max_rel_geom_err", type=float, default=1e-5,
                   help="m, |link_pos - base_pos - (ref_link_pos - ref_base_pos)| 上限")
    p.add_argument("--max_speed", type=float, default=4.0, help="m/s, base")
    p.add_argument("--max_accel", type=float, default=30.0, help="m/s^2, base")
    p.add_argument("--max_joint_vel", type=float, default=20.0, help="rad/s")
    p.add_argument("--min_height", type=float, default=0.0, help="m, base_z 下限")
    args = p.parse_args()

    blob = torch.load(args.variants_path)
    variants = blob["variants"] if isinstance(blob, dict) and "variants" in blob else blob
    ref = torch.load(args.ref_path)
    meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
    fps = float(args.fps) if args.fps is not None else float(meta.get("fps", 50.0))

    ref_base_pos = ref["base_position"].float()
    ref_joint = ref["joint_position"].float()
    ref_rpy = ref["base_pose"].float()
    ref_link_rel = ref["link_position"].float() - ref_base_pos.unsqueeze(1)

    print(f"checking {len(variants)} variants against ref {args.ref_path}")
    print(f"[meta] version={meta.get('version', 'n/a')} fps={fps} "
          f"sdedit_t_start={meta.get('sdedit_t_start', 'n/a')}")
    if "constraints" in meta:
        print(f"[meta] constraints={meta['constraints']}")

    dt = 1.0 / fps
    n_ok = 0
    for i, v in enumerate(variants):
        errs = []

        if v["base_position"].shape != ref_base_pos.shape:
            errs.append(
                f"base_position shape {tuple(v['base_position'].shape)} "
                f"!= ref {tuple(ref_base_pos.shape)}"
            )

        # Eq.3 不变量
        if not torch.equal(v["joint_position"].float(), ref_joint):
            errs.append("joint_position ≠ ref (Eq.3 violated)")
        if not torch.equal(v["base_pose"].float(), ref_rpy):
            errs.append("base_pose ≠ ref (Eq.3 violated)")

        # 相对几何不变性
        var_link_rel = v["link_position"].float() - v["base_position"].float().unsqueeze(1)
        if var_link_rel.shape == ref_link_rel.shape:
            rel_err = (var_link_rel - ref_link_rel).abs().max().item()
            if rel_err > args.max_rel_geom_err:
                errs.append(f"link rel-geom err={rel_err:.2e} > {args.max_rel_geom_err}")
        else:
            errs.append(
                f"link_position shape {tuple(v['link_position'].shape)} "
                f"!= ref {tuple(ref['link_position'].shape)}"
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

        # joint 速度 (joint_position 等于 ref, 这里更多是 sanity)
        jp = v["joint_position"].float()
        j_vel = (jp[1:] - jp[:-1]) / dt
        jv_max = j_vel.abs().max().item()
        if jv_max > args.max_joint_vel:
            errs.append(f"joint_vel max={jv_max:.2f} > {args.max_joint_vel}")

        if errs:
            print(f"  [FAIL] variant {i}: {errs}")
        else:
            n_ok += 1

    print(f"[result] {n_ok}/{len(variants)} variants passed")
    if n_ok < len(variants):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
