# AGENTS.md

## 项目背景
基于 AdaMimic (motion_tracking.py) 做三处改进：
① Diffusion 替代 fedit  ② STL 连续奖励  ③ STL 引导 Phase Adapter

## 环境
- 本地 WSL：代码编辑，Codex 运行于此
- 远程服务器：GPU 训练，通过 VSCode SSH 连接操作
- conda 环境名：adamimic（服务器上）
- 代码同步：git push origin main → 服务器 git pull

## 核心文件
- 主改动文件: legged_gym/envs/base/motion_tracking.py（已加 diffusion 分支 + 拼 variants）
- Diffusion 模块: legged_gym/legged_gym/diffusion/{root_mdm.py, filter.py, fk.py}（fk.py 封装 pytorch_kinematics 做 G1 floating-base FK）
- STL 工具模块: legged_gym/utils/stl_specs.py（待创建，创新点② 用）
- 离线脚本（位于顶层 scripts/）:
  - prepare_diffusion_full_training_set.py  HuggingFace-retargeted AMASS → (T,33) [joint_27|base_pos|base_rpy]
  - train_diffusion_full.py                 diffusers DDPMScheduler 训练 33-dim 联合扩散
  - generate_diffusion_full_trajectories.py SDEdit 采样 + FK 重算 link_pose
  - check_variant_constraints.py            FK self-consistency + 速度上限
- 控制开关: configs/algorithm/adamimic/stage1.yaml 的 algorithm.use_diffusion_ref / diffusion_ref_path

## 当前阶段
Phase: far_jump基线复现完成，正在实现创新点①和创新点②
Status: 创新点① 切换到 SDEdit + 联合生成 joint_pos+base_pos+base_rpy + FK 重算；待服务器 FK self-consistency 验证

## 代码约束
- 所有新功能用 cfg flag 控制开关（use_stl_reward / use_diffusion_ref）
- 同时生成 joint_pos + base_pos + base_rpy（joint 冻结的 MVP 分支已删除，git 历史保留作事后对照）
- link_pose 用 pytorch_kinematics FK 从 (base_pos, base_rpy, joint_27) 重算，与 data.pt 的 link_position/orientation 应 self-consistent

## 已知问题 / 注意事项
- AdaMimic里对关键帧的修改是手工编辑，但是原文的假设是假定joint_pos保持不动的情况下对base_pos作一些修改，但是我认为这是违背物理直觉的，因为对于跳远这个动作来说，不同的跳远距离对应的joint_pos是不一样的，所以我的想法是利用diffusion扩散出符合物理直觉的关键帧的joint_pos和base_pos
- 运动数据格式（远程实测）：resources/dataset/g1_dof27_data/{task}/output/data.pt，dict 字段为 base_position / base_pose(RPY) / base_velocity / base_angular_velocity / joint_position / joint_velocity / link_position(T, N_bodies, 3) / link_orientation(RPY) / link_velocity / link_angular_velocity；**不含 framerate**，帧率由 cfg.dataset.frame_rate (base.yaml 默认 30) 提供。Explore agent 早期报告里的 framerate 字段不准确，已在脚本和 motionlib.load_diffusion_variants 中修正
- AMASS 轴映射 axis_perm=0,2,1 / scale=0.85 是旧 MVP (原始 AMASS .npz) 管线的经验值；完整版改用 HuggingFace 预 retargeted .npy (base_pos+quat_xyzw+joint_29)，已在 G1 坐标系下，无需轴重映射
- diffusion_sample_policy=curriculum 目前未实现；MotionLib 自带 visit/completion 加权采样会自然混采 variants，先用 random 即可
- Hydra 配置路径：stage1.yaml 中 `use_diffusion_ref` 等字段嵌套在 `algorithm:` key 下，而 Hydra config group 名也叫 `algorithm`，所以命令行 override 需要写 `algorithm.algorithm.use_diffusion_ref=true`（两层 algorithm）。同理 `algorithm.algorithm.diffusion_ref_path=...`
- 29 → 27 的由来G1 的 29 DoF 分布是：6（每条腿）× 2 + 7（每条手臂）× 2 + 3（腰部）= 29
其中腰部那 3 个自由度分别是 waist_yaw / waist_pitch / waist_roll。论文在附录 C 明确写到：
"We use the Unitree 29-DoF G1 humanoid robot (6 per leg, 7 per arm, and 1 in the waist). Waist roll and pitch joints are locked for stability and safety."
以及附录 E 再次强调：
"For hardware stability, the waist pitch and roll joints are locked."
所以 29 − 2（锁定的 waist_pitch + waist_roll）= 27，这就是训练时 num_dofs: 27 的来源
