from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, AttrDict, export_jit_to_onnx
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(cfg):
    cfg.env.terrain.curriculum = False
    cfg.env.termination_curriculum.terminate_when_motion_far_curriculum = False
    cfg.env.termination_curriculum.terminate_when_motion_far_initial_threshold = 1000
    cfg.env.termination.height_termination = False
    cfg.env.termination.rot_termination = False
    cfg.env.termination.dof_termination = False
    cfg.env.algorithm.rsi = False
    cfg.env.noise.add_noise = False
    cfg.env.domain_rand.use_random = False

    cfg = AttrDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.run_dir = HydraConfig.get().runtime.output_dir
    
    cfg.env.terrain.num_rows = 4
    cfg.env.terrain.num_cols = 2
    cfg.env.env.test = True
    cfg.algo.policy.checkpoint_path = None

    env, env_cfg = task_registry.make_env_hydra(cfgs=cfg)

    obs = env.get_observations()

    cfg.algo.runner.resume = True
    cfg.algo.policy.resume = False

    ppo_runner, train_cfg = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if cfg.export_policy:
        ppo_runner.export_policy_as_jit(ppo_runner.get_actor_critic(), cfg.export_policy_path, cfg.export_policy_name)
        jit_model = torch.jit.load(os.path.join(cfg.export_policy_path, cfg.export_policy_name + '.pt')).to('cuda:0')
        dummy_obs = env.get_observations()
        export_onnx_path = os.path.join(cfg.export_policy_path, cfg.export_policy_name + '.onnx')
        export_jit_to_onnx(jit_model, export_onnx_path, dummy_obs)
        print('successful export to onnx')

        info = {            
            "STIFFNESS": env.cfg.control.stiffness,
            "DAMPING": env.cfg.control.damping,
            "ACTION SCALE": env.cfg.control.action_scale,
            "DEFAULT JOINT ANGLES": env.cfg.init_state.actutaed_default_joint_angles,
            "DOF NAMES": env.dof_names,
            "KEYFRAME NAMES": env.keyframe_names,
            "DEFAULT DOF POS": env.default_dof_pos.cpu().numpy().tolist(),
            "TORQUE LIMITS": env.torque_limits.cpu().numpy().tolist(),
            "MOTION LENGTH": int(env.motions.length[0]),
            "Difficulty MIN": float(env.terrain_difficulty.min().cpu().item()),
            "Difficulty MAX": float(env.terrain_difficulty.max().cpu().item()),
        }
        with open(f'{cfg.export_policy_path}/info' + ".json", "w") as f:
            import json
            json.dump(info, f, indent=2)

    _, _ = env.reset()

    for i in range(100000*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, critic_obs, obs_high, rews, _ , dones, infos, _ = env.step(actions.detach())
        

if __name__ == '__main__':
    main()
