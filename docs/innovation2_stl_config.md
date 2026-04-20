# 创新点② STL 连续奖励 —— 配置速查

所有新增配置字段、默认值、启用命令都在这里；后续跑实验只看这张表就够。

---

## 1. 涉及文件清单

| 文件 | 角色 | 字段位置 |
|------|------|----------|
| `legged_gym/legged_gym/configs/algorithm/adamimic/stage1.yaml` | 开关 + 漏斗超参 | `algorithm.use_stl_reward`, `algorithm.stl_funnel.*` |
| `legged_gym/legged_gym/configs/dataset/base.yaml` | 奖励权重 | `rewards.scales.dense_stl_keyframe`（默认 0，自动剔除） |
| `legged_gym/legged_gym/utils/stl_specs.py` | 鲁棒性 ρ 实现 | — |
| `legged_gym/legged_gym/envs/base/motion_tracking.py` | `__init__` 构建 `self.stl_spec` + `_reward_stl_keyframe` | — |
| `scripts/check_stl_specs.py` | 本地自检 | 运行一次确保 ρ/ψ 逻辑正确 |

---

## 2. 配置字段（含默认值）

### 2.1 `algorithm:` 下（stage1.yaml）

| 字段 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `use_stl_reward` | bool | `false` | 总开关；false 时完全不走 STL 分支，零开销 |
| `stl_funnel.T` | float (s) | `0.3` | 漏斗过渡时长；越小则只有非常靠近 keyframe 时 ψ 才收紧 |
| `stl_funnel.eps_min` | float (m) | `0.05` | keyframe 处容差下界（漏斗最窄处） |
| `stl_funnel.eps_max` | float (m) | `0.30` | 远离 keyframe 的松弛容差上界 |
| `stl_funnel.beta_time` | float | `20.0` | 时间域 softmin 锐度，越大越接近硬 min；≤0 则走硬 min |
| `stl_funnel.reduce` | `mean`\|`max` | `mean` | body-位置 L2 距离的聚合方式 |

> **漏斗公式**：`ψ(t) = eps_min + (eps_max − eps_min) · clip(Δt(t)/T, 0, 1)`，其中 `Δt(t) = softmin_k |t − t_k|`，`t_k` 是 `cfg.dataset.keyframe_times[cfg.dataset.keyframe_pos_index]`。
>
> **鲁棒性**：`ρ(t) = ψ(t) − d(t)`，`d(t)` 是当前步 `dif_global_body_pos` 的 L2 均值（或最大值）。

### 2.2 `rewards.scales:` 下（base.yaml）

| 字段 | 默认 | 说明 |
|------|------|------|
| `dense_stl_keyframe` | `0.0` | 0 会被 `_prepare_reward_function` 剔除，**启用时必须覆盖成非零** |

> 命名规则：前缀 `dense_` → 归入 `dense` reward group，被 `apply_reward_scale` 按 `infer_dt/dt` 自动缩放；方法名自动对应 `_reward_stl_keyframe`。

---

## 3. 启用方式（Hydra 命令行 override）

注意 **两层 `algorithm`**：外层是 config group 名，内层是 YAML 里的 key。

```bash
# 基础启用（权重先从小往大试 0.5 → 1.0 → 2.0）
python train.py \
  algorithm.algorithm.use_stl_reward=true \
  rewards.scales.dense_stl_keyframe=0.5
```

### 常见调参 override

```bash
# 收紧漏斗（更严格）
algorithm.algorithm.stl_funnel.eps_min=0.03 \
algorithm.algorithm.stl_funnel.eps_max=0.20

# 放宽漏斗过渡（远离 keyframe 也保留一些 shaping）
algorithm.algorithm.stl_funnel.T=0.5

# 改用最大 body 距离（更敏感于 worst-case body）
algorithm.algorithm.stl_funnel.reduce=max

# 硬 min（调试用，失去可微性）
algorithm.algorithm.stl_funnel.beta_time=0
```

### 关闭（回到 baseline）

```bash
algorithm.algorithm.use_stl_reward=false
# 或者 rewards.scales.dense_stl_keyframe=0.0（等价，scale=0 会被剔除）
```

---

## 4. 默认实验矩阵（建议顺序）

| ID | use_stl_reward | dense_stl_keyframe | eps_min | eps_max | T | 目的 |
|----|----------------|--------------------|---------|---------|---|------|
| S0 | false | 0 | — | — | — | baseline（创新点② 关） |
| S1 | true | 0.5 | 0.05 | 0.30 | 0.3 | 默认 STL，小权重，看是否稳定 |
| S2 | true | 1.0 | 0.05 | 0.30 | 0.3 | 加大权重 |
| S3 | true | 1.0 | 0.03 | 0.20 | 0.3 | 收紧漏斗 |
| S4 | true | 1.0 | 0.05 | 0.30 | 0.5 | 放宽过渡时长 |
| S5 | true | 1.0 | 0.05 | 0.30 | 0.3 | `reduce=max` 变体 |

> 对比指标参考 `query_pack.md`：Easy Success ≥98%、Hard ≥70%、E^dense_l-bpe ≈30mm；wandb 里额外看 `episode_sums/dense_stl_keyframe` 曲线走势。

---

## 5. Sanity-check 与回归

启用前先跑：
```bash
/home/gzy/miniconda3/envs/robojudo/bin/python scripts/check_stl_specs.py
```
预期输出：
```
[OK] STL specs self-check passed.
  ψ@0.6s = 0.0500 | ψ@1.2s = 0.3000
  ρ(d=0) = 0.0500 | ρ(d=0.5) = -0.4500
```

在服务器端首次启用训练后，前 200 iter 观察：
- `episode_sums/dense_stl_keyframe` 应逐步从负转正
- 若一直是大负数 → `eps_max` 太小 或权重太大，先降 `dense_stl_keyframe`
- 若接近 0 无区分度 → `eps_max` 太大，降到 0.20

---

## 6. 与创新点①（Diffusion）的关系

两者**完全正交**，flag 分别是 `use_diffusion_ref` / `use_stl_reward`，可任意组合：

| 组合 | 语义 |
|------|------|
| 00 | 原 AdaMimic baseline |
| 10 | 仅 Diffusion 参考增强 |
| 01 | 仅 STL 连续奖励 |
| 11 | 两者同时启用（最终目标组合） |
