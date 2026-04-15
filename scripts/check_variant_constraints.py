"""检查 variants.pt 是否满足论文式 3 约束 (创新点① MVP C5).

用法:
    python scripts/check_variant_constraints.py \\
        --variants_path resources/dataset/g1_dof27_data_diff/high_jump/variants.pt \\
        --ref_path      legged_gym/resources/dataset/g1_dof27_data/high_jump/output/data.pt
"""

import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants_path", required=True)
    p.add_argument("--ref_path", required=True)
    p.add_argument("--min_height", type=float, default=0.0)
    args = p.parse_args()

    ref = torch.load(args.ref_path)
    blob = torch.load(args.variants_path)
    variants = blob["variants"] if isinstance(blob, dict) and "variants" in blob else blob

    print(f"checking {len(variants)} variants against ref {args.ref_path}")
    n_ok = 0
    for i, v in enumerate(variants):
        errs = []
        if not torch.equal(v["joint_position"], ref["joint_position"]):
            errs.append("joint_position differs from ref")
        if not torch.equal(v["base_pose"], ref["base_pose"]):
            errs.append("base_pose differs from ref")
        if v["base_position"][:, 2].min().item() < args.min_height:
            errs.append(f"base z min={v['base_position'][:, 2].min().item():.3f} < {args.min_height}")
        if v["base_position"].shape != ref["base_position"].shape:
            errs.append(f"base_position shape {v['base_position'].shape} != ref {ref['base_position'].shape}")
        # link_position delta 一致性
        delta = v["base_position"] - ref["base_position"]
        reconstructed = ref["link_position"] + delta[:, None, :]
        if not torch.allclose(v["link_position"], reconstructed, atol=1e-5):
            errs.append("link_position inconsistent with base_position offset")

        if errs:
            print(f"  [FAIL] variant {i}: {errs}")
        else:
            n_ok += 1
    print(f"[result] {n_ok}/{len(variants)} variants passed")
    if n_ok < len(variants):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
