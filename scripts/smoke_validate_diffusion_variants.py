"""CPU-only smoke validation for diffusion-generated variants.pt.

This is the local pre-training check for 创新点①:
1. Load the bundle through motionlib's diffusion loader contract
2. Verify required metadata exists (`meta.fps`, `meta.link_names`, `meta.version`, `meta.sdedit_t_start`)
3. Instantiate MotionLib to confirm the variants can enter Stage 1 without schema/shape errors
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def _spec_load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_motionlib_module():
    utils_root = Path(__file__).resolve().parents[1] / "legged_gym" / "legged_gym" / "utils"
    # Isaac Gym must be imported before any torch import hidden inside math.py / motionlib.py.
    import isaacgym  # noqa: F401
    math_mod = _spec_load("motionlib_math", utils_root / "math.py")

    legged_gym_pkg = types.ModuleType("legged_gym")
    legged_gym_pkg.__path__ = []
    utils_pkg = types.ModuleType("legged_gym.utils")
    utils_pkg.__path__ = []
    sys.modules.setdefault("legged_gym", legged_gym_pkg)
    sys.modules.setdefault("legged_gym.utils", utils_pkg)
    sys.modules["legged_gym.utils.math"] = math_mod

    return _spec_load("motionlib_smoke", utils_root / "motionlib.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants_path", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--fps", type=float, default=None, help="Stage 1 cfg.dataset.frame_rate to test against")
    parser.add_argument("--height_offset", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    motionlib = _load_motionlib_module()
    variants, summary = motionlib.inspect_diffusion_variant_bundle(args.variants_path)
    meta = summary["meta"]

    required_meta = ("fps", "link_names", "version", "sdedit_t_start")
    missing_meta = [key for key in required_meta if key not in meta]
    if missing_meta:
        raise ValueError(f"diffusion bundle missing required meta keys: {missing_meta}")

    lines = Path(args.mapping_path).read_text().splitlines()
    mapping = {}
    for line in lines:
        idx, name = line.split(" ")
        mapping[name] = int(idx)

    dof_names = [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    body_names = list(meta["link_names"])
    fps = args.fps if args.fps is not None else float(meta["fps"])

    motion_lib = motionlib.MotionLib(
        variants,
        mapping,
        dof_names,
        body_names,
        fps=fps,
        min_dt=0.1,
        device="cpu",
        height_offset=args.height_offset,
    )

    print(f"[load] {summary['path']}")
    print(f"[variants] count={summary['num_variants']} version={meta['version']} fps_meta={meta['fps']}")
    print(f"[motionlib] num_motion={motion_lib.num_motion} total_length={int(motion_lib.total_length)} fps={fps}")
    print(f"[variant0] shapes={summary['variant0_shapes']}")
    constraints = meta.get("constraints")
    if constraints is not None:
        print(f"[constraints] {constraints}")
    print("[ok] diffusion variants passed local MotionLib smoke validation.")


if __name__ == "__main__":
    main()
