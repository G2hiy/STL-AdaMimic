"""Evaluate a trained policy and dump per-episode metrics for ablation analysis.

Typical usage (run on GPU server; same Hydra config stack as play.py/train.py):

  # 1. baseline (no diffusion, no stl)
  python scripts/eval_rollout.py \\
      task_id=far_jump \\
      dataset=g1_dof27/far_jump \\
      algorithm=adamimic/stage2 \\
      resume_path=<baseline_ckpt.pt> \\
      eval_rollout.tag=baseline

  # 2. +diffusion
  python scripts/eval_rollout.py \\
      task_id=far_jump \\
      dataset=g1_dof27/far_jump \\
      algorithm=adamimic/stage2 \\
      algorithm.algorithm.use_diffusion_ref=true \\
      algorithm.algorithm.diffusion_ref_path=<variants.pt> \\
      resume_path=<diff_ckpt.pt> \\
      eval_rollout.tag=diff

  # 3. +stl
  python scripts/eval_rollout.py \\
      ... algorithm.algorithm.use_stl_reward=true resume_path=<stl_ckpt.pt> \\
      eval_rollout.tag=stl

  # 4. +diff+stl
  python scripts/eval_rollout.py \\
      ... algorithm.algorithm.use_diffusion_ref=true algorithm.algorithm.use_stl_reward=true \\
      resume_path=<diff_stl_ckpt.pt> eval_rollout.tag=diff_stl

Outputs (per run, under eval_rollout.out_root):
    <out_root>/<task>/<tag>_seed<seed>_<timestamp>/
        episodes.csv   – per-episode metrics (one row per episode)
        summary.json   – mean±std aggregates
        traces.pt      – first K episodes' per-step traces (for paper figures)
        config.yaml    – fully resolved Hydra config (for reproducibility)
"""
import os
import time

import isaacgym  # noqa: F401  MUST precede torch
import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403  register tasks
from legged_gym.utils import task_registry, AttrDict
from legged_gym.utils.eval_metrics import RolloutMetricCollector


def _freeze_eval_variance(cfg: DictConfig) -> None:
    """Disable curriculum/RSI/noise/domain-rand so all variants face the same env."""
    cfg.env.terrain.curriculum = False
    cfg.env.termination_curriculum.terminate_when_motion_far_curriculum = False
    cfg.env.termination_curriculum.terminate_when_motion_far_initial_threshold = 1000
    cfg.env.termination.height_termination = False
    cfg.env.termination.rot_termination = False
    cfg.env.termination.dof_termination = False
    cfg.env.algorithm.rsi = False
    cfg.env.noise.add_noise = False
    cfg.env.domain_rand.use_random = False


@hydra.main(config_path="../legged_gym/legged_gym/configs", config_name="eval", version_base="1.1")
def main(cfg: DictConfig):
    _freeze_eval_variance(cfg)
    ev = cfg.eval_rollout
    num_envs = int(ev.num_envs)
    num_rollouts = int(ev.num_rollouts)
    target_episodes = num_envs * num_rollouts
    seed = int(ev.seed)
    tag = str(ev.tag)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(cfg, "algo") and hasattr(cfg.algo, "seed"):
        cfg.algo.seed = seed

    cfg.num_envs = num_envs
    cfg.env.env.num_envs = num_envs
    cfg.env.terrain.num_rows = int(ev.terrain_rows)
    cfg.env.terrain.num_cols = int(ev.terrain_cols)
    cfg.env.env.test = True
    cfg.algo.policy.checkpoint_path = None

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_attr = AttrDict(cfg_dict)
    cfg_attr.run_dir = HydraConfig.get().runtime.output_dir

    env, env_cfg = task_registry.make_env_hydra(cfgs=cfg_attr)
    obs = env.get_observations()

    cfg_attr.algo.runner.resume = True
    cfg_attr.algo.policy.resume = False
    ppo_runner, _ = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=cfg_attr)
    policy = ppo_runner.get_inference_policy(device=env.device)

    stl_events = None
    ds = cfg.get("dataset", None)
    if ds is not None and "stl_events" in ds:
        stl_events = {k: float(v) for k, v in OmegaConf.to_container(ds.stl_events, resolve=True).items()}
    collector = RolloutMetricCollector(env, stl_events=stl_events, save_trace_k=int(ev.save_trace_k))

    env.reset()

    task_id = str(cfg.dataset.task_id)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        str(ev.out_root), task_id, f"{tag}_seed{seed}_{stamp}"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)
    print(f"[eval] tag={tag} task={task_id} target_episodes={target_episodes} → {out_dir}")

    max_episode_steps = int(env.max_episode_length)
    hard_cap = int(max_episode_steps * num_rollouts * 3)  # safety

    with torch.inference_mode():
        step = 0
        while not collector.done(target_episodes) and step < hard_cap:
            actions = policy(obs.detach())
            # env.step returns either 8 items (no amp) or 9 (amp); we only need
            # obs + infos. unpacking pattern copied from play.py.
            if not env.amp:
                obs, _, _, _, _, infos, _, _ = env.step(actions.detach())
            else:
                obs, _, _, _, _, infos, _, _, _ = env.step(actions.detach())
            collector.step(env, infos)
            step += 1

    meta = {
        "seed": seed,
        "num_envs": num_envs,
        "num_rollouts": num_rollouts,
        "use_diffusion_ref": bool(cfg.env.algorithm.get("use_diffusion_ref", False)),
        "use_stl_reward": bool(cfg.env.algorithm.get("use_stl_reward", False)),
        "resume_path": str(cfg.get("resume_path", "")),
        "steps_executed": step,
    }
    collector.save(out_dir, tag=tag, extra_meta=meta)


if __name__ == "__main__":
    main()
